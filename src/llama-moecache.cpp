#include "llama-moecache.h"

#include <algorithm>

#include "llama-impl.h"
#include "llama-model.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cinttypes>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <functional>
#include <sstream>
#ifdef _WIN32
#include <process.h>
#define getpid _getpid
#else
#include <unistd.h>
#include <sys/mman.h>
#endif

namespace {

// deprecated_name is the pre-release German name, kept as an alias
static inline const char * fle_getenv_compat(const char * name, const char * deprecated_name) {
    const char * e = std::getenv(name);
    return e ? e : std::getenv(deprecated_name);
}

struct layer_state {
    llama_moe_cache_layer pub;

    // LRU bookkeeping (host side; the tables mirror expert_slot)
    std::vector<int32_t>  slot_expert;   // slot -> expert id, -1 when empty
    std::vector<int32_t>  expert_slot;   // expert id -> slot, -1 when uncached
    std::vector<uint64_t> slot_last_use; // slot -> lamport clock of last hit
    std::vector<int32_t>  pending;       // uncached ids observed since last step (dedup, obs order)
    // FLE: how often this step requested the expert. `pending` is deduplicated, so the
    // request count would be lost, and with it any basis for choosing among several
    // candidates. Same length and order as `pending`.
    std::vector<int32_t>  pending_count;

    std::vector<bool>     slot_in_flight; // slot has an upload pending

    // FLE_MOE_ADMIT_FREQ / FLE_MOE_EVICT_OLDEST: aged request frequency per expert of this layer (observations, halved every
    // fle_admit_freq_h observations). Decides WHICH miss is admitted and WHO is evicted; it does not change the layer's
    // results (only which experts run on the GPU instead of the CPU).
    std::vector<float>    freq;
    uint32_t              freq_obs = 0;

    uint64_t n_hit  = 0;
    uint64_t n_miss = 0;

    // Patch 44a (FLE_MOE_PROFILE): lifetime routing counts of this layer, saved to / seeded from the profile file
    std::vector<uint64_t> prof_cnt;
    uint64_t              prof_total  = 0;   // sum of prof_cnt (aged: halved when it reaches prof_max_obs)
    int32_t               n_prefilled = 0;   // slots filled from the profile at start

    // Patch 44b: slot counts of THIS layer. Without FLE_MOE_CACHE_BUDGET_MIB every layer carries the global values
    // (mc->n_slots / mc->n_slots_total); with it they differ per layer. The zero slot sits at index n_total.
    int32_t n_use = 0;   // usable slots
    int32_t n_total = 0;   // usable + shadow slots

    // Patch 47 (FLE_MOE_PCIE_FRAC): ring slots after the zero slot (index n_total + 1 + r). Not part of the LRU above:
    // admission never uses them, and an expert served from the ring keeps expert_slot -1, so the observer still counts it
    // as a miss and the admission sees it. A ring entry stays valid until its slot is reused.
    std::vector<int32_t>  ring_expert;   // ring slot -> expert whose weights it holds, -1 when empty
    std::vector<uint64_t> ring_last;     // ring slot -> hook clock of the last use (LRU among ring slots)
    int8_t                ring_dev_ok = -1;   // -1 unknown, 0 cache not on the hit chain's device (hook skips), 1 ok

    // Patch 48 (FLE_UBATCH_LEND): with lending every layer owns its device buffer, so whole layers can be lent to the
    // compute buffer of a large prefill ubatch and refilled from the host afterwards. While lent the layer has no cache:
    // llama_moe_cache_lookup() returns nullptr and the CPU computes all of its experts.
    ggml_context *             lctx  = nullptr;
    ggml_backend_buffer_t      lbuf  = nullptr;
    ggml_backend_buffer_type_t lbuft = nullptr;
    bool                       lent  = false;
};

struct upload_job {
    size_t  layer_idx;
    int32_t expert;
    int32_t slot;
    // Shadow mode: the victim is only released when the upload is published, not when it
    // is scheduled. -1 means "no victim" (a free slot was available anyway).
    int32_t victim_slot = -1;
    bool    done = false;
};

// Patch 44a: one layer of the usage profile (file content or live counters)
struct prof_layer {
    int                   il         = -1;
    int64_t               n_expert   = 0;
    size_t                slot_bytes = 0;   // bytes of one expert over up + gate + down
    std::vector<uint64_t> cnt;
};

struct moe_cache {
    int32_t n_slots     = 0;   // USABLE slots (without shadow slots, without the zero slot)
    int32_t n_slots_total = 0;   // n_slots + shadow slots; the zero slot sits at n_slots_total
    bool    fle_shadow = false;
    bool    fle_admit_frequent = false;  // pick the most-requested miss instead of the latest
    bool    fle_budget_global = false;      // unused insert budget carries over to the next layer
    int32_t fle_admit_freq_h = 0;           // > 0: admission by aged frequency (halved every this many observations per layer)
    int32_t fle_evict_oldest = 0;           // >= 2: eviction = least frequent of the N oldest slots (instead of the oldest)
    uint64_t fle_budget_used = 0;        // how many insertions were actually scheduled
    uint64_t fle_budget_possible = 0;       // reference count: how many would have been possible
    uint64_t fle_choice_differs = 0;           // how often the choice differs from rbegin()
    uint64_t fle_choice_total = 0;           // reference count: without it a 0 cannot be interpreted
    int32_t max_inserts = 2;

    // Patch 44a: expert-usage profile (FLE_MOE_PROFILE, default off; empty path = off)
    std::string             prof_path;
    int64_t                 prof_save_sec   = 600;        // FLE_MOE_PROFILE_SAVE_SEC; 0 = only at shutdown
    uint64_t                prof_max_obs    = 1ULL << 27; // FLE_MOE_PROFILE_MAX_OBS: per-layer total at which counts are halved
    bool                    prof_dirty      = false;      // observations since the last save
    bool                    prof_loaded     = false;
    uint64_t                prof_base_steps = 0;          // steps recorded by earlier runs
    uint64_t                prof_saves      = 0;
    std::chrono::steady_clock::time_point prof_last_save{};
    std::vector<prof_layer> prof_keep;                    // layers of the loaded file without a cache in this run
    std::string             prof_model_desc;              // guard: the file must come from this model
    uint64_t                prof_model_size = 0;
    uint32_t                prof_n_layer    = 0;
    uint64_t                prefilled_slots = 0;
    uint64_t                prefill_cap     = 0;
    double                  prefill_mass    = 0.0;        // hit rate the profile predicts for the pre-filled slots

    // Patch 44b: global byte budget (FLE_MOE_CACHE_BUDGET_MIB, default off)
    int64_t                 budget_mib      = 0;          // > 0: slots per layer come from plan_budget_all, not from n_slots
    int32_t                 shadow_n        = 0;          // shadow slots per layer (0 without FLE_MOECACHE_SHADOW)

    // Patch 47: PCIe share of the misses (FLE_MOE_PCIE_FRAC, default 0 = off; FLE_MOE_PCIE_RING ring slots per layer)
    float                   pcie_frac       = 0.0f;
    int32_t                 pcie_ring       = 0;
    double                  pcie_acc        = 0.0;        // fractional carry: share * misses, summed over hooks
    uint64_t                pcie_clock      = 0;
    uint64_t                pcie_hooks      = 0;          // hook calls that reached a cache layer
    uint64_t                pcie_skipped    = 0;          // hook calls without effect (other device, ids not on the host)
    uint64_t                pcie_misses     = 0;          // distinct uncached experts seen by the hook (reference count)
    uint64_t                pcie_loaded     = 0;          // experts copied into a ring slot
    uint64_t                pcie_reused     = 0;          // misses mapped back onto a ring slot that still held them (no copy)
    uint64_t                pcie_served     = 0;          // distinct experts served from the ring (already mapped, reused or loaded)
    uint64_t                pcie_ring_full  = 0;          // wanted copies dropped because every ring slot was in use
    uint64_t                pcie_bytes      = 0;

    // Patch 48: cache lending for large prefill ubatches (FLE_UBATCH_LEND, default off)
    bool                    lend_on         = false;
    bool                    wbusy           = false;      // the upload worker holds a job (guarded by wmtx)
    bool                    wpaused         = false;      // the upload worker takes no job (guarded by wmtx)
    ggml_backend_t          refill_backend  = nullptr;    // own stream for the refill after a lend
    uint64_t                lend_n          = 0;          // lends
    uint64_t                lend_layers     = 0;          // layers lent, summed
    uint64_t                lend_bytes      = 0;          // bytes lent, summed
    uint64_t                lend_short      = 0;          // lends that freed less than asked
    double                  refill_ms       = 0.0;        // refill time, summed
    uint64_t                refill_failed   = 0;          // layers that could not get their memory back (stay without cache)
    size_t                  lend_cur        = 0;          // bytes lent right now (0 = no active lend)
    size_t                  lend_asked      = 0;          // bytes the active lend was asked for
    uint64_t                lend_epoch      = 0;          // bumped on lend and return: graphs built before must not be reused

    uint64_t clock   = 0;
    uint64_t n_steps = 0;

    std::mutex mtx; // guards pending lists + clock (observe runs during graph exec)

    std::vector<layer_state> layers;
    std::map<const ggml_tensor *, size_t> by_up_src;

    std::vector<ggml_context *>         ctxs;
    std::vector<ggml_backend_buffer_t>  bufs;

    // async upload worker: slices are copied to the device off the decode
    // thread; the new table mapping is only published at a later step() once
    // the upload has completed, so a running graph never reads a torn slot
    std::thread              worker;
    std::mutex               wmtx;
    std::condition_variable  wcv;
    std::deque<upload_job>   todo;
    // FLE counters that separate the three explanations for the gap to LRU.
    // All three are candidates, and all three look the same in the hit rate.
    uint64_t fle_double_planned = 0;   // expert already in todo -> second slot
    uint64_t fle_no_free_slot  = 0;   // all slots in flight -> layer skipped
    uint64_t fle_dropped       = 0;   // pending longer than max_inserts
    std::vector<upload_job>  done;
    bool                     stop = false;
};

moe_cache * g_cache = nullptr;
std::mutex g_init_mtx;
bool g_init_done = false;

int parse_layer_from_name(const char * name) {
    // "blk.<il>.ffn_gate_exps.weight"
    if (strncmp(name, "blk.", 4) != 0) {
        return -1;
    }
    return atoi(name + 4);
}

// ---- FLE_MOE_WILLNEED_NOW: request the pages of the experts the CPU is about to read --------------------------------
// When the CPU mul_mat_id of gate_exps(L) starts, the experts of layer L are known (ids on the host). For those not in
// the GPU cache, madvise(WILLNEED) on the whole gate/up/down slices, on a worker thread: the NVMe reads them in large
// parallel chunks while the CPU starts on gate, instead of page fault by page fault. No prediction, no upload.
struct wn_range { void * a; size_t n; };
static std::mutex              g_wn_mtx;
static std::condition_variable g_wn_cv;
static std::deque<wn_range>    g_wn_q;

static void wn_worker() {
    for (;;) {
        wn_range r;
        {
            std::unique_lock<std::mutex> lk(g_wn_mtx);
            g_wn_cv.wait(lk, [] { return !g_wn_q.empty(); });
            r = g_wn_q.front(); g_wn_q.pop_front();
        }
#ifndef _WIN32
        madvise(r.a, r.n, MADV_WILLNEED);
#endif
    }
}

static void wn_willneed(const ggml_tensor * src, int32_t expert) {
#ifndef _WIN32
    if (!src || !src->data) { return; }
    static const uintptr_t pg = (uintptr_t) sysconf(_SC_PAGESIZE);
    const uintptr_t a  = (uintptr_t) src->data + (uintptr_t) expert * src->nb[2];
    const uintptr_t a0 = a & ~(pg - 1);
    static std::once_flag started;
    std::call_once(started, [] { std::thread(wn_worker).detach(); });
    { std::lock_guard<std::mutex> lk(g_wn_mtx); g_wn_q.push_back({ (void *) a0, (size_t) (a + src->nb[2] - a0) }); }
    g_wn_cv.notify_one();
#else
    (void) src; (void) expert;
#endif
}

void moe_obs_cb(const char * name, const struct ggml_tensor * ids, void * ud) {
    moe_cache * mc = (moe_cache *) ud;

    static const bool wn_on = std::getenv("FLE_MOE_WILLNEED_NOW") != nullptr;
    if (wn_on && ids->ne[1] <= 4) {
        const int il0 = parse_layer_from_name(name);
        layer_state * l0 = nullptr;
        for (auto & l : mc->layers) { if (l.pub.il == il0) { l0 = &l; break; } }
        if (l0) {
            std::lock_guard<std::mutex> lock(mc->mtx);
            for (int64_t t = 0; t < ids->ne[1]; ++t) {
                for (int64_t i = 0; i < ids->ne[0]; ++i) {
                    const int32_t e = *(const int32_t *) ((const char *) ids->data + t*ids->nb[1] + i*ids->nb[0]);
                    if (e < 0 || e >= (int32_t) l0->expert_slot.size() || l0->expert_slot[e] >= 0) { continue; }
                    wn_willneed(l0->pub.up_src,   e);
                    wn_willneed(l0->pub.down_src, e);
                    wn_willneed(l0->pub.gate_src, e);
                }
            }
        }
    }

    const int64_t n_ids    = ids->ne[0];
    const int64_t n_tokens = ids->ne[1];
    static const int obs_max = [] { const char * e = std::getenv("FLE_MOE_OBS_MAXTOK"); return (e && *e) ? std::max(1, atoi(e)) : 4; }();
    if (n_tokens > obs_max) {
        return; // batch/prefill: the cache graph is not built there, don't pollute the LRU (FLE_MOE_OBS_MAXTOK for measurements)
    }

    const int il = parse_layer_from_name(name);
    if (il < 0) {
        return;
    }

    layer_state * ls = nullptr;
    for (auto & l : mc->layers) {
        if (l.pub.il == il) { ls = &l; break; }
    }
    if (!ls) {
        return;
    }

    std::lock_guard<std::mutex> lock(mc->mtx);
    for (int64_t t = 0; t < n_tokens; ++t) {
        for (int64_t i = 0; i < n_ids; ++i) {
            const int32_t id = *(const int32_t *) ((const char *) ids->data + t*ids->nb[1] + i*ids->nb[0]);
            if (id < 0 || id >= (int32_t) ls->expert_slot.size()) {
                continue;
            }
            if (!ls->prof_cnt.empty()) {
                ls->prof_cnt[id]++;
                mc->prof_dirty = true;
                if (++ls->prof_total >= mc->prof_max_obs) {   // age: old runs must not dominate forever
                    ls->prof_total = 0;
                    for (auto & c : ls->prof_cnt) { c >>= 1; ls->prof_total += c; }
                }
            }
            if (!ls->freq.empty()) {
                ls->freq[id] += 1.0f;
                if (mc->fle_admit_freq_h > 0 && ++ls->freq_obs >= (uint32_t) mc->fle_admit_freq_h) {
                    ls->freq_obs = 0;
                    for (auto & f : ls->freq) { f *= 0.5f; }
                }
            }
            const int32_t slot = ls->expert_slot[id];
            if (slot >= 0) {
                ls->n_hit++;
                ls->slot_last_use[slot] = ++mc->clock;
            } else {
                ls->n_miss++;
                bool dup = false;
                for (size_t p = 0; p < ls->pending.size(); ++p) {
                    if (ls->pending[p] == id) { ls->pending_count[p]++; dup = true; break; }
                }
                if (!dup) {
                    ls->pending.push_back(id);
                    ls->pending_count.push_back(1);
                }
            }
        }
    }
}

// ---- Patch 47: PCIe share of the misses (FLE_MOE_PCIE_FRAC, default off) ------------------------------------------
// Strata-style idea, own implementation: a miss costs the CPU ~68 us per expert (DIMM bandwidth), a copy of the same
// expert over PCIe ~43-90 us, and the two run in parallel (the GPU stream is idle while the CPU computes the misses).
// So a share of the misses of every layer and step is copied into a small ring of extra slots per layer and computed by
// the GPU hit chain; the CPU skips them through the host table like any cached expert.
//   FLE_MOE_PCIE_FRAC=<0..1>  share of the misses (summed with a fractional carry over all hooks), 0 = off
//   FLE_MOE_PCIE_RING=<n>     ring slots per layer (default 2, 1..16); they come out of the cache budget
// Needs FLE_MC_OVERLAP: the scheduler calls pcie_hook() for the marked hit chain once the routed ids are on the host
// and before the hit chain is enqueued (see ggml_backend_set_fle_mc_hook). Order on the GPU stream: table, weights,
// then the hit chain with its get_rows over the table (marked instead of gate_g, see llama-graph.cpp).
// The ring is not part of the LRU: admission never fills it, and an expert served from the ring keeps expert_slot -1, so
// the observer still counts it as a miss and the admission still sees it. A ring entry stays valid until its slot is
// reused; then it is unmapped only if the table still points at that ring slot (the step may have published the expert
// into a regular slot meanwhile).
struct pcie_env {
    float   frac = 0.0f;
    int32_t ring = 0;
};

const pcie_env & pcie_config() {
    static const pcie_env cfg = [] {
        pcie_env c;
        const char * f  = std::getenv("FLE_MOE_PCIE_FRAC");
        const char * ov = std::getenv("FLE_MC_OVERLAP");
        if (!f || !*f) {
            return c;
        }
        const float v = std::min(1.0f, std::max(0.0f, (float) atof(f)));
        if (v <= 0.0f) {
            return c;
        }
        if (!ov || atoi(ov) == 0) {
            LLAMA_LOG_WARN("[fle-pcie] FLE_MOE_PCIE_FRAC=%s needs FLE_MC_OVERLAP - PCIe share OFF\n", f);
            return c;
        }
        c.frac = v;
        c.ring = 2;
        if (const char * r = std::getenv("FLE_MOE_PCIE_RING")) {
            c.ring = std::min(16, std::max(1, atoi(r)));
        }
        return c;
    }();
    return cfg;
}

void pcie_hook(ggml_tensor * node, ggml_backend_t backend, void * ud) {
    moe_cache * mc = (moe_cache *) ud;
    if (!mc || mc->pcie_ring <= 0 || !node || !backend) {
        return;
    }
    std::lock_guard<std::mutex> lock(mc->mtx);

    layer_state * ls = nullptr;
    auto it = mc->by_up_src.find(node->src[0]);
    if (it != mc->by_up_src.end() && mc->layers[it->second].pub.host_table == node->src[3]) {
        ls = &mc->layers[it->second];
    } else {
        for (auto & l : mc->layers) {
            if (l.pub.host_table == node->src[3]) { ls = &l; break; }
        }
    }
    if (!ls || ls->lent) {   // Patch 48: a lent layer has no device tensors (its graph has no cache chain either)
        mc->pcie_skipped++;
        return;
    }
    const ggml_tensor * ids = node->src[2];
    if (!ids || ids->type != GGML_TYPE_I32 || !ids->data || !ids->buffer || !ggml_backend_buffer_is_host(ids->buffer)) {
        mc->pcie_skipped++;
        return;
    }
    if (ls->ring_dev_ok < 0) {
        // ggml-cuda's set_tensor_async only takes the device's own default buffer type
        const ggml_backend_buffer_type_t bt = ls->pub.up_c && ls->pub.up_c->buffer ? ggml_backend_buffer_get_type(ls->pub.up_c->buffer) : nullptr;
        ls->ring_dev_ok = bt && bt == ggml_backend_get_default_buffer_type(backend) ? 1 : 0;
        if (!ls->ring_dev_ok) {
            LLAMA_LOG_WARN("[fle-pcie] layer %d: cache buffer %s is not the default buffer of backend %s - no PCIe share here\n",
                    ls->pub.il, bt ? ggml_backend_buft_name(bt) : "?", ggml_backend_name(backend));
        }
    }
    if (!ls->ring_dev_ok) {
        mc->pcie_skipped++;
        return;
    }
    mc->pcie_hooks++;

    int32_t *     tbl      = (int32_t *) ls->pub.host_table->data;
    const int32_t dummy    = ls->n_total;
    const int32_t base     = ls->n_total + 1;
    const int32_t R        = mc->pcie_ring;
    const int64_t n_expert = ls->pub.up_src->ne[2];

    // distinct experts of this call with their token counts (at most n_expert_used * max tokens entries)
    std::vector<std::pair<int32_t, int32_t>> seen;
    for (int64_t t = 0; t < ids->ne[1]; ++t) {
        for (int64_t i = 0; i < ids->ne[0]; ++i) {
            const int32_t id = *(const int32_t *) ((const char *) ids->data + t*ids->nb[1] + i*ids->nb[0]);
            if (id < 0 || id >= n_expert) {
                continue;
            }
            bool dup = false;
            for (auto & s : seen) {
                if (s.first == id) { s.second++; dup = true; break; }
            }
            if (!dup) {
                seen.push_back({id, 1});
            }
        }
    }

    std::vector<char> busy((size_t) R, 0);
    std::vector<std::pair<int32_t, int32_t>> miss;   // uncached and not on the ring: (expert, tokens)
    for (const auto & s : seen) {
        const int32_t v = tbl[s.first];
        if (v == dummy) {
            miss.push_back(s);
        } else if (v >= base && v < base + R) {
            busy[(size_t) (v - base)] = 1;
            ls->ring_last[(size_t) (v - base)] = ++mc->pcie_clock;
            mc->pcie_served++;
            mc->pcie_misses++;
        }
    }
    mc->pcie_misses += miss.size();

    bool table_changed = false;
    // 1) a ring slot that still holds the expert (unmapped after a publish/evict cycle): map it back, no copy
    std::vector<std::pair<int32_t, int32_t>> rest;
    for (const auto & m : miss) {
        int32_t r_hit = -1;
        for (int32_t r = 0; r < R; ++r) {
            if (!busy[(size_t) r] && ls->ring_expert[(size_t) r] == m.first) { r_hit = r; break; }
        }
        if (r_hit < 0) {
            rest.push_back(m);
            continue;
        }
        busy[(size_t) r_hit] = 1;
        ls->ring_last[(size_t) r_hit] = ++mc->pcie_clock;
        tbl[m.first] = base + r_hit;
        table_changed = true;
        mc->pcie_reused++;
        mc->pcie_served++;
    }

    // 2) the share: how many of the remaining misses go over PCIe, with a fractional carry so small layers count too
    mc->pcie_acc += (double) rest.size() * mc->pcie_frac;
    int32_t want = (int32_t) mc->pcie_acc;
    mc->pcie_acc -= want;
    // most requested first (stable: equal counts keep the routing order)
    std::stable_sort(rest.begin(), rest.end(), [](const std::pair<int32_t, int32_t> & a, const std::pair<int32_t, int32_t> & b) {
        return a.second > b.second;
    });
    std::vector<std::pair<int32_t, int32_t>> loads;   // (expert, ring slot)
    for (size_t k = 0; k < rest.size() && want > 0; ++k) {
        int32_t r_new = -1;
        uint64_t best = UINT64_MAX;
        for (int32_t r = 0; r < R; ++r) {
            if (!busy[(size_t) r] && ls->ring_last[(size_t) r] < best) { best = ls->ring_last[(size_t) r]; r_new = r; }
        }
        if (r_new < 0) {
            mc->pcie_ring_full += (uint64_t) want;
            break;
        }
        const int32_t e = rest[k].first;
        bool fits = true;
        for (const auto & p : { std::make_pair(ls->pub.up_c, ls->pub.up_src), std::make_pair(ls->pub.gate_c, ls->pub.gate_src),
                                std::make_pair(ls->pub.down_c, ls->pub.down_src) }) {
            const size_t sz = p.second->nb[2];
            fits = fits && (size_t) (base + r_new)*p.first->nb[2] + sz <= ggml_nbytes(p.first) && (size_t) e*sz + sz <= ggml_nbytes(p.second);
        }
        if (!fits) {
            LLAMA_LOG_ERROR("moe-cache: bad PCIe copy layer %d expert=%d ring=%d - ring off for this layer\n", ls->pub.il, e, r_new);
            ls->ring_dev_ok = 0;
            break;
        }
        const int32_t old = ls->ring_expert[(size_t) r_new];
        if (old >= 0 && tbl[old] == base + r_new) {
            tbl[old] = dummy;
        }
        ls->ring_expert[(size_t) r_new] = e;
        ls->ring_last[(size_t) r_new]   = ++mc->pcie_clock;
        busy[(size_t) r_new] = 1;
        tbl[e] = base + r_new;
        table_changed = true;
        loads.push_back({e, r_new});
        mc->pcie_loaded++;
        mc->pcie_served++;
        --want;
    }
    if (!table_changed) {
        return;
    }

    // Table first: its source is pageable, so the copy is staged right away while the stream is still idle. The two tables
    // always hold the same values (set_table_entry writes both), so the whole host table is the new device table.
    ggml_backend_tensor_set_async(backend, ls->pub.dev_table, tbl, 0, ggml_nbytes(ls->pub.dev_table));
    for (const auto & l : loads) {
        const ggml_tensor * srcs[3] = { ls->pub.up_src, ls->pub.gate_src, ls->pub.down_src };
        ggml_tensor *       dsts[3] = { ls->pub.up_c,   ls->pub.gate_c,   ls->pub.down_c   };
        for (int k = 0; k < 3; ++k) {   // bounds checked above, before the table pointed at the ring slot
            const size_t sz   = srcs[k]->nb[2];
            const size_t doff = (size_t) (base + l.second) * dsts[k]->nb[2];
            ggml_backend_tensor_set_async(backend, dsts[k], (const char *) srcs[k]->data + (size_t) l.first*sz, doff, sz);
            mc->pcie_bytes += sz;
        }
    }
}

// ---- FLE upload counters (default off) ---------------------------------------
// Measures the path the scheduler counter in ggml-backend.cpp does NOT see: that counter
// stays at 0 because this cache took over the experts and loads them through a different
// function. The setter used below is SYNCHRONOUS (ggml-cuda), so every insertion blocks
// until it has completed.
// FLE_UPLOAD_ASYNC: batch the slot copies. Measured: 5.44 ms per step in blocking copies,
// 25.9 % of the token budget, at 10.97 of 28.54 possible GB/s. The setter synchronizes
// after EVERY copy.
// FLE_SLOWDOWN_UPLOAD_US: artificially LENGTHEN a time to see whether it is in series with
// the rest. Without this check a build was wasted: a factor 231 less copy time gave
// -2.35 t/s, because the upload thread runs alongside decoding.
//   throughput drops with it -> in SERIES, a lever there works
//   throughput stays         -> PARALLEL, a lever there does not work
static long long fle_slowdown_us = [] {
    const char * e = fle_getenv_compat("FLE_SLOWDOWN_UPLOAD_US", "FLE_BREMSE");
    return e ? atoll(e) : 0LL;
}();
static long long fle_slowdown_n = 0;      // reference count: how often slowed down
static double    fle_slowdown_ms = 0.0;   // reference count: how much delay was ACTUALLY inserted

// FLE_SLOWDOWN_TABLE_US: the same question for the TABLE instead of the slot copy.
// `FLE_SLOWDOWN_UPLOAD_US` sits in `upload_slice`; that path is parallel to decoding
// (a factor 231 less copy time gave -2.35 t/s). The table sync is a different place, and
// a measured DURATION does not answer whether it is in series; only artificial
// lengthening does.
static long long fle_slowdown_tab_us = [] {
    const char * e = fle_getenv_compat("FLE_SLOWDOWN_TABLE_US", "FLE_BREMSE_TABELLE");
    return e ? atoll(e) : 0LL;
}();
static long long fle_slowdown_tab_n = 0;
static double    fle_slowdown_tab_ms = 0.0;

static bool fle_up_async = [] {
    const char * e = getenv("FLE_UPLOAD_ASYNC");
    return e && atoi(e) != 0;
}();
static ggml_backend_t fle_up_backend = nullptr;   // dedicated backend for the upload thread
static long long fle_up_sync_points = 0;          // reference count: how often we wait
static long long fle_up_pieces = 0, fle_up_bytes = 0, fle_up_table_sets = 0;
static double    fle_up_ms = 0.0, fle_up_tab_ms = 0.0;
static long long fle_up_every = [] {
    const char * e = fle_getenv_compat("FLE_UPLOAD_COUNTERS", "FLE_UPLOAD_ZAEHLER");
    return e ? atoll(e) : 0LL;
}();

struct fle_up_timer {
    // Times the ONE sync point separately. Without it, after the rework one could no longer
    // tell whether the copies got faster or were merely counted differently.
    std::chrono::steady_clock::time_point t0;
    fle_up_timer() : t0(std::chrono::steady_clock::now()) {}
    ~fle_up_timer() {
        if (fle_up_every) {
            fle_up_ms += std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t0).count();
        }
    }
};

void upload_slice(ggml_tensor * dst_c, const ggml_tensor * src, int32_t expert, int32_t slot) {
    const size_t sz = src->nb[2];
    if ((size_t) slot*dst_c->nb[2] + sz > ggml_nbytes(dst_c) || (size_t) expert*sz + sz > ggml_nbytes(src)) {
        LLAMA_LOG_ERROR("moe-cache: bad upload %s <- %s expert=%d slot=%d sz=%zu dst_nb2=%zu dst_bytes=%zu src_bytes=%zu\n",
                dst_c->name, src->name, expert, slot, sz, dst_c->nb[2], ggml_nbytes(dst_c), ggml_nbytes(src));
        return;
    }
    if (fle_slowdown_us > 0) {
        // Not `usleep`: it sleeps imprecisely and gives no feedback. Busy-wait with
        // measurement: the report line states the delay ACTUALLY inserted, otherwise the
        // slowdown would be an unverified switch.
        const auto b0 = std::chrono::steady_clock::now();
        const auto deadline = b0 + std::chrono::microseconds(fle_slowdown_us);
        while (std::chrono::steady_clock::now() < deadline) { }
        fle_slowdown_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - b0).count();
        fle_slowdown_n++;
    }
    if (fle_up_every) {
        const auto t0 = std::chrono::steady_clock::now();
        ggml_backend_tensor_set(dst_c, (const char *) src->data + (size_t) expert*sz, (size_t) slot*dst_c->nb[2], sz);
        fle_up_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        fle_up_pieces++;
        fle_up_bytes += (long long) sz;
    } else {
        ggml_backend_tensor_set(dst_c, (const char *) src->data + (size_t) expert*sz, (size_t) slot*dst_c->nb[2], sz);
    }
}

void upload_slice_async(ggml_tensor * dst_c, const ggml_tensor * src, int32_t expert, int32_t slot) {
    // Like upload_slice, but WITHOUT waiting. The caller synchronizes once at the end of
    // the batch. Same bounds checks: an asynchronous path must not check less than the
    // synchronous one, otherwise the rework turns an error into a silent one.
    const size_t sz = src->nb[2];
    if ((size_t) slot*dst_c->nb[2] + sz > ggml_nbytes(dst_c) || (size_t) expert*sz + sz > ggml_nbytes(src)) {
        LLAMA_LOG_ERROR("moe-cache: bad async upload %s <- %s expert=%d slot=%d\n",
                dst_c->name, src->name, expert, slot);
        return;
    }
    if (fle_slowdown_us > 0) {
        const auto b0 = std::chrono::steady_clock::now();
        const auto deadline = b0 + std::chrono::microseconds(fle_slowdown_us);
        while (std::chrono::steady_clock::now() < deadline) { }
        fle_slowdown_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - b0).count();
        fle_slowdown_n++;
    }
    ggml_backend_tensor_set_async(fle_up_backend, dst_c,
            (const char *) src->data + (size_t) expert*sz, (size_t) slot*dst_c->nb[2], sz);
    if (fle_up_every) {
        fle_up_pieces++;
        fle_up_bytes += (long long) sz;
    }
}

float g_cache_bias = 0.0f; // FLE: additive routing bonus for resident experts (0 = off)

// FLE_MOE_CACHE_BIAS_LAYERS="0-11:0.002,12-47:0.0005": bonus per layer range (range:value, comma separated); layers without an
// entry keep the global value. Only for measuring which layers tolerate the bonus. A value of 0 turns it off there.
float layer_cache_bias(int il) {
    static const std::vector<std::pair<std::pair<int, int>, float>> table = [] {
        std::vector<std::pair<std::pair<int, int>, float>> t;
        const char * e = std::getenv("FLE_MOE_CACHE_BIAS_LAYERS");
        if (!e || !*e) { return t; }
        std::string s(e);
        size_t pos = 0;
        while (pos < s.size()) {
            size_t end = s.find(',', pos);
            if (end == std::string::npos) { end = s.size(); }
            const std::string item = s.substr(pos, end - pos);
            const size_t colon = item.find(':');
            if (colon != std::string::npos) {
                const std::string range = item.substr(0, colon);
                const float val = (float) atof(item.c_str() + colon + 1);
                const size_t dash = range.find('-');
                const int lo = atoi(range.c_str());
                const int hi = dash == std::string::npos ? lo : atoi(range.c_str() + dash + 1);
                t.push_back({{lo, hi}, val});
            }
            pos = end + 1;
        }
        return t;
    }();
    for (const auto & it : table) {
        if (il >= it.first.first && il <= it.first.second) { return it.second; }
    }
    return g_cache_bias;
}

void set_table_entry(llama_moe_cache_layer & pub, int32_t expert, int32_t slot_or_dummy) {
    const int32_t v = slot_or_dummy;
    if (pub.dev_bias) {
        // uploaded once per layer and step (llama_moe_cache_step), not per entry
        pub.bias_host[expert] = slot_or_dummy == pub.n_slots ? 0.0f : pub.bias_val;
        pub.bias_dirty = true;
    }
    if (fle_slowdown_tab_us > 0) {
        // Busy-wait, not `usleep`: it sleeps imprecisely and gives no feedback.
        const auto b0 = std::chrono::steady_clock::now();
        const auto deadline = b0 + std::chrono::microseconds(fle_slowdown_tab_us);
        while (std::chrono::steady_clock::now() < deadline) { }
        fle_slowdown_tab_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - b0).count();
        fle_slowdown_tab_n++;
    }
    // TWO synchronous 4-byte copies per call. At 4 bytes this is pure latency, not
    // bandwidth, and the call happens on EVERY eviction and EVERY publication.
    if (fle_up_every) {
        const auto t0 = std::chrono::steady_clock::now();
        ggml_backend_tensor_set(pub.dev_table,  &v, (size_t) expert*sizeof(int32_t), sizeof(int32_t));
        ggml_backend_tensor_set(pub.host_table, &v, (size_t) expert*sizeof(int32_t), sizeof(int32_t));
        fle_up_tab_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        fle_up_table_sets++;
    } else {
        ggml_backend_tensor_set(pub.dev_table,  &v, (size_t) expert*sizeof(int32_t), sizeof(int32_t));
        ggml_backend_tensor_set(pub.host_table, &v, (size_t) expert*sizeof(int32_t), sizeof(int32_t));
    }
}

// ---- Patch 44a: expert-usage profile (warm start) -------------------------------------------
// FLE_MOE_PROFILE=<file> (default off): record how often every (layer, expert) is routed to, save the table
// atomically (tmp + rename) every FLE_MOE_PROFILE_SAVE_SEC seconds (default 600, 0 = only at shutdown) and at
// context teardown, and on the next start pre-fill each cache layer with its most frequent experts and seed the
// aged-frequency counters (FLE_MOE_ADMIT_FREQ / FLE_MOE_EVICT_OLDEST) from the same counts. The admission and
// eviction policy itself is untouched; only the starting state differs. Own implementation, no Strata code.
//
// File format (text, human readable, "v1"):
//   FLE-MOE-PROFILE v1 / model_size / model_desc / n_layer / steps / layers N
//   then per layer: "L <il> <n_expert> <slot_bytes>" and one line with n_expert counts
//   END <fnv1a-64 of everything before "END">
// Guards: model file size, model description and layer count must equal; per layer expert count and bytes per
// expert must equal. Anything else -> warning, cold start, the file is replaced at the next save.
// (struct prof_layer is declared above moe_cache, which keeps the layers of a loaded file that have no cache)
struct prof_file {
    bool                    ok    = false;
    uint64_t                steps = 0;
    std::vector<prof_layer> layers;

    const prof_layer * find(int il) const {
        for (const auto & l : layers) {
            if (l.il == il) { return &l; }
        }
        return nullptr;
    }
};

uint64_t prof_fnv1a(const char * p, size_t n) {
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; ++i) {
        h ^= (unsigned char) p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

size_t prof_slot_bytes(const llama_moe_cache_layer & pub) {
    return pub.up_src->nb[2] + pub.gate_src->nb[2] + pub.down_src->nb[2];
}

bool prof_read(const std::string & path, const moe_cache & mc, prof_file & out, std::string & why) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        why = "not found";
        return false;
    }
    std::string txt;
    {
        char buf[65536];
        size_t n;
        while ((n = fread(buf, 1, sizeof(buf), f)) > 0) {
            txt.append(buf, n);
            if (txt.size() > ((size_t) 64 << 20)) {
                fclose(f);
                why = "file larger than 64 MiB";
                return false;
            }
        }
    }
    fclose(f);

    const size_t e = txt.rfind("END ");
    if (e == std::string::npos || (e > 0 && txt[e - 1] != '\n')) {
        why = "no END line (truncated?)";
        return false;
    }
    if (prof_fnv1a(txt.data(), e) != (uint64_t) strtoull(txt.c_str() + e + 4, nullptr, 16)) {
        why = "checksum mismatch";
        return false;
    }

    std::istringstream is(txt.substr(0, e));
    std::string line;
    if (!std::getline(is, line) || line != "FLE-MOE-PROFILE v1") {
        why = "unknown format";
        return false;
    }
    unsigned long long model_size = 0, steps = 0, n_layer = 0, n_layers_file = 0;
    std::string model_desc;
    bool have_size = false, have_desc = false, have_nl = false, have_steps = false, have_layers = false;
    while (!have_layers && std::getline(is, line)) {
        if (sscanf(line.c_str(), "model_size %llu", &model_size) == 1) { have_size = true; }
        else if (line.compare(0, 11, "model_desc ") == 0)             { model_desc = line.substr(11); have_desc = true; }
        else if (sscanf(line.c_str(), "n_layer %llu", &n_layer) == 1)  { have_nl = true; }
        else if (sscanf(line.c_str(), "steps %llu", &steps) == 1)      { have_steps = true; }
        else if (sscanf(line.c_str(), "layers %llu", &n_layers_file) == 1) { have_layers = true; }
    }
    if (!(have_size && have_desc && have_nl && have_steps && have_layers)) {
        why = "incomplete header";
        return false;
    }
    if (model_size != mc.prof_model_size || model_desc != mc.prof_model_desc || n_layer != mc.prof_n_layer) {
        why = "written for a different model (size/description/layer count differ)";
        return false;
    }

    prof_file pf;
    pf.steps = steps;
    for (unsigned long long li = 0; li < n_layers_file; ++li) {
        int il = -1;
        long long n_expert = 0;
        unsigned long long slot_bytes = 0;
        if (!std::getline(is, line) || sscanf(line.c_str(), "L %d %lld %llu", &il, &n_expert, &slot_bytes) != 3 ||
                n_expert <= 0 || n_expert > (1 << 20)) {
            why = "bad layer header";
            return false;
        }
        prof_layer pl;
        pl.il         = il;
        pl.n_expert   = n_expert;
        pl.slot_bytes = (size_t) slot_bytes;
        pl.cnt.resize((size_t) n_expert);
        if (!std::getline(is, line)) {
            why = "missing counts line";
            return false;
        }
        const char * p = line.c_str();
        for (long long i = 0; i < n_expert; ++i) {
            char * endp = nullptr;
            const unsigned long long v = strtoull(p, &endp, 10);
            if (endp == p) {
                why = "short counts line";
                return false;
            }
            pl.cnt[(size_t) i] = v;
            p = endp;
        }
        pf.layers.push_back(std::move(pl));
    }
    pf.ok = true;
    out = std::move(pf);
    return true;
}

// Every layer that is in both the file and the cache must agree on shape. One mismatch rejects the whole file.
template <typename Groups>
bool prof_validate(prof_file & pf, const Groups & groups, std::string & why) {
    for (const auto & g : groups) {
        for (const auto & c : g.second) {
            const prof_layer * pl = pf.find(c.il);
            if (!pl) {
                continue;
            }
            if (pl->n_expert != c.l->ffn_up_exps->ne[2] ||
                pl->slot_bytes != c.l->ffn_up_exps->nb[2] + c.l->ffn_gate_exps->nb[2] + c.l->ffn_down_exps->nb[2]) {
                why = "layer " + std::to_string(c.il) + ": expert count or expert size differs";
                pf.ok = false;
                return false;
            }
        }
    }
    return true;
}

// Caller holds mc->mtx. Writes live counters of the cache layers plus the layers of the loaded file that have no
// cache in this run (e.g. after a fit that left fewer layers on the host), so a smaller run does not drop them.
bool prof_write(moe_cache * mc, const char * reason) {
    std::vector<prof_layer> out;
    uint64_t obs = 0;
    for (const auto & ls : mc->layers) {
        if (ls.prof_cnt.empty()) {
            continue;
        }
        prof_layer pl;
        pl.il         = ls.pub.il;
        pl.n_expert   = (int64_t) ls.prof_cnt.size();
        pl.slot_bytes = prof_slot_bytes(ls.pub);
        pl.cnt        = ls.prof_cnt;
        for (auto c : pl.cnt) { obs += c; }
        out.push_back(std::move(pl));
    }
    const size_t n_live = out.size();
    for (const auto & pl : mc->prof_keep) {
        out.push_back(pl);
    }
    std::sort(out.begin(), out.end(), [](const prof_layer & a, const prof_layer & b) { return a.il < b.il; });

    std::string txt = "FLE-MOE-PROFILE v1\n";
    txt += "model_size " + std::to_string(mc->prof_model_size) + "\n";
    txt += "model_desc " + mc->prof_model_desc + "\n";
    txt += "n_layer " + std::to_string(mc->prof_n_layer) + "\n";
    txt += "steps " + std::to_string(mc->prof_base_steps + mc->n_steps) + "\n";
    txt += "layers " + std::to_string(out.size()) + "\n";
    for (const auto & pl : out) {
        txt += "L " + std::to_string(pl.il) + " " + std::to_string(pl.n_expert) + " " + std::to_string(pl.slot_bytes) + "\n";
        for (size_t i = 0; i < pl.cnt.size(); ++i) {
            if (i) { txt += ' '; }
            txt += std::to_string(pl.cnt[i]);
        }
        txt += '\n';
    }
    char endl_buf[48];
    snprintf(endl_buf, sizeof(endl_buf), "END %016llx\n", (unsigned long long) prof_fnv1a(txt.data(), txt.size()));
    txt += endl_buf;

    const std::string tmp = mc->prof_path + ".tmp." + std::to_string((long long) getpid());
    FILE * f = fopen(tmp.c_str(), "wb");
    if (!f) {
        LLAMA_LOG_WARN("[fle-moe-profile] cannot write %s - profile not saved (%s)\n", tmp.c_str(), reason);
        return false;
    }
    bool ok = fwrite(txt.data(), 1, txt.size(), f) == txt.size() && fflush(f) == 0;
#ifndef _WIN32
    ok = ok && fsync(fileno(f)) == 0;
#endif
    ok = (fclose(f) == 0) && ok;
    if (!ok || std::rename(tmp.c_str(), mc->prof_path.c_str()) != 0) {
        LLAMA_LOG_WARN("[fle-moe-profile] writing %s failed - profile not saved (%s)\n", mc->prof_path.c_str(), reason);
        std::remove(tmp.c_str());
        return false;
    }
    mc->prof_dirty      = false;
    mc->prof_last_save  = std::chrono::steady_clock::now();
    mc->prof_saves++;
    LLAMA_LOG_INFO("[fle-moe-profile] saved %s (%s): %zu live + %zu kept layers, %" PRIu64 " observations, %zu bytes\n",
            mc->prof_path.c_str(), reason, n_live, out.size() - n_live, obs, txt.size());
    return true;
}

// One slot copy at start-up: same bounds check as upload_slice, but no brake and no counters.
void prefill_slice(ggml_tensor * dst_c, const ggml_tensor * src, int32_t expert, int32_t slot) {
    const size_t sz = src->nb[2];
    if ((size_t) slot*dst_c->nb[2] + sz > ggml_nbytes(dst_c) || (size_t) expert*sz + sz > ggml_nbytes(src)) {
        LLAMA_LOG_ERROR("moe-cache: bad prefill %s <- %s expert=%d slot=%d\n", dst_c->name, src->name, expert, slot);
        return;
    }
    ggml_backend_tensor_set(dst_c, (const char *) src->data + (size_t) expert*sz, (size_t) slot*dst_c->nb[2], sz);
}

// Start-up fill, runs inside llama_moe_cache_init after the tables exist and before the worker thread and the
// observer start, so nothing runs concurrently. The expert weights are read from the host tensors (already loaded:
// the init above skips layers whose `data` is null); with mmap the pages are faulted in by this read, with
// `--load-mode none` they are plain host memory. Slots are copied one by one, synchronously (a few seconds for
// ~20 GB); the decode path never sees a half-filled slot because no graph has run yet.
void prof_prefill(moe_cache * mc, const prof_file & pf) {
    const auto t0 = std::chrono::steady_clock::now();
    uint64_t n_fill = 0, n_cap = 0, n_lay = 0, n_lay_prof = 0;
    double   mass_sum = 0.0;
    for (auto & ls : mc->layers) {
        const int64_t n_expert = ls.pub.up_src->ne[2];
        n_cap += (uint64_t) ls.n_use;
        const prof_layer * pl = pf.find(ls.pub.il);
        if (!pl || pl->cnt.size() != (size_t) n_expert) {
            continue;
        }
        n_lay_prof++;
        ls.prof_cnt = pl->cnt;
        uint64_t total = 0;
        for (auto c : pl->cnt) { total += c; }
        ls.prof_total = total;
        if (total == 0) {
            continue;
        }

        std::vector<int32_t> order((size_t) n_expert);
        for (int32_t e = 0; e < (int32_t) n_expert; ++e) { order[(size_t) e] = e; }
        std::stable_sort(order.begin(), order.end(), [&](int32_t a, int32_t b) { return pl->cnt[(size_t) a] > pl->cnt[(size_t) b]; });
        int32_t k = 0;
        while (k < ls.n_use && k < (int32_t) n_expert && pl->cnt[(size_t) order[(size_t) k]] > 0) { k++; }

        // seed the aged frequencies so the admission/eviction rules see a history instead of zeros; the scale
        // matches the steady state of the halving (total between H and 2H)
        if (!ls.freq.empty()) {
            const double target = mc->fle_admit_freq_h > 0 ? 1.5 * mc->fle_admit_freq_h : 3000.0;
            for (int64_t e = 0; e < n_expert; ++e) {
                ls.freq[(size_t) e] = (float) ((double) pl->cnt[(size_t) e] * target / (double) total);
            }
            if (mc->fle_admit_freq_h > 0) { ls.freq_obs = (uint32_t) mc->fle_admit_freq_h / 2; }
        }

        std::vector<int32_t> tbl((size_t) n_expert, ls.n_total);
        uint64_t covered = 0;
        for (int32_t r = k - 1; r >= 0; --r) {      // lowest rank first: the hottest expert gets the newest clock
            const int32_t e = order[(size_t) r];
            const int32_t s = r;
            prefill_slice(ls.pub.up_c,   ls.pub.up_src,   e, s);
            prefill_slice(ls.pub.gate_c, ls.pub.gate_src, e, s);
            prefill_slice(ls.pub.down_c, ls.pub.down_src, e, s);
            ls.slot_expert[(size_t) s]    = e;
            ls.expert_slot[(size_t) e]    = s;
            ls.slot_last_use[(size_t) s]  = ++mc->clock;
            tbl[(size_t) e]               = s;
            if (ls.pub.dev_bias) { ls.pub.bias_host[(size_t) e] = ls.pub.bias_val; }
            covered += pl->cnt[(size_t) e];
        }
        if (k > 0) {
            ggml_backend_tensor_set(ls.pub.dev_table,  tbl.data(), 0, (size_t) n_expert*sizeof(int32_t));
            ggml_backend_tensor_set(ls.pub.host_table, tbl.data(), 0, (size_t) n_expert*sizeof(int32_t));
            if (ls.pub.dev_bias) {
                ggml_backend_tensor_set(ls.pub.dev_bias, ls.pub.bias_host.data(), 0, ls.pub.bias_host.size()*sizeof(float));
            }
            n_lay++;
            mass_sum += (double) covered / (double) total;
        }
        ls.n_prefilled = k;
        n_fill += (uint64_t) k;
    }
    mc->prefilled_slots = n_fill;
    mc->prefill_cap     = n_cap;
    mc->prefill_mass    = n_lay ? mass_sum / (double) n_lay : 0.0;
    // count next to sum: slots filled of slots available, layers with a profile of layers with a cache, and the hit
    // rate the profile itself predicts for the filled slots (compare with the measured first-512-steps line)
    LLAMA_LOG_INFO("[fle-moe-profile] EVIDENCE pre-filled %" PRIu64 " of %" PRIu64 " slots (%.1f %%) in %" PRIu64 " of %zu layers "
                   "(%" PRIu64 " with profile), %.0f ms; profile predicts %.1f %% hits for the filled slots\n",
            n_fill, n_cap, n_cap ? 100.0 * (double) n_fill / (double) n_cap : 0.0, n_lay, mc->layers.size(), n_lay_prof,
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(),
            100.0 * mc->prefill_mass);
}

// ---- Patch 44b: global byte budget for the expert cache -------------------------------------
// FLE_MOE_CACHE_BUDGET_MIB=<n> (default off = the fixed slot count of --moe-expert-cache for every layer): the cache gets
// a byte budget instead, spread over ALL layers with host-resident experts. A layer's cost per slot is the size of one
// expert in up + gate + down, so layers with larger experts cost more. Every layer pays first for its fixed part (the
// zero slot and, with FLE_MOECACHE_SHADOW, the shadow slots), then gets a floor of FLE_MOE_CACHE_FLOOR_PCT percent of
// the uniform slot count (default 10, at least one slot while the budget allows), and the rest goes to the experts with the
// most routed hits per byte (profile from 44a; layers without counts are treated as flat). Without any profile the rest
// is spread evenly over the layers. FLE_MOE_CACHE_SLOTS_MAX caps the usable slots of a layer (default: the expert count).
// The budget counts the slot tensors up_c + gate_c + down_c (the "device memory" figure of the start-up log); the I32
// tables and the bias vector (4 or 8 bytes per expert and layer) come on top.
struct plan_in {
    int                           il;
    int64_t                       n_expert;
    size_t                        slot_bytes;   // one slot over up + gate + down
    const std::vector<uint64_t> * cnt;          // profile counts of this layer or nullptr
};

struct plan_out {
    std::map<int, int32_t> use;                 // il -> usable slots; layers that got 0 are absent
    uint64_t fixed_bytes   = 0;                 // zero slots + shadow slots of the planned layers
    uint64_t planned_bytes = 0;                 // everything the plan allocates (fixed + usable slots)
    int32_t  floor_slots   = 0;
    int32_t  min_use = 0, max_use = 0;
    double   mean_use = 0.0;
    bool     profile = false;                   // at least one layer was ranked from counts
    double   hit_planned = 0.0;                 // hit rate the counts predict for this plan (mean over layers)
    double   hit_equal   = 0.0;                 // ... and for the same budget spread as equal slot counts
    size_t   n_dropped   = 0;                   // layers without a single slot
};

plan_out plan_budget(const std::vector<plan_in> & in, uint64_t budget, int32_t shadow, int32_t floor_pct, int32_t cap_max) {
    plan_out po;
    const size_t L = in.size();
    uint64_t fixed = 0, sum_b = 0;
    for (const auto & p : in) {
        fixed += (uint64_t) (1 + shadow) * p.slot_bytes;
        sum_b += p.slot_bytes;
    }
    if (L == 0 || budget <= fixed) {
        po.n_dropped = L;
        return po;
    }
    const uint64_t avail = budget - fixed;
    const double   u     = (double) avail / (double) sum_b;   // slots per layer if every layer got the same count

    // per layer: slot cap and the probability of the k-th best expert (smoothed, sums to 1)
    std::vector<int32_t>             cap(L);
    std::vector<std::vector<double>> gain(L);
    std::vector<std::vector<double>> gain_cum(L);   // prefix sums, for the predicted hit rate
    bool any_profile = false;
    for (size_t l = 0; l < L; ++l) {
        const int64_t n = in[l].n_expert;
        cap[l] = (int32_t) std::min<int64_t>(n, cap_max > 0 ? cap_max : n);
        uint64_t total = 0;
        if (in[l].cnt && (int64_t) in[l].cnt->size() == n) {
            for (auto c : *in[l].cnt) { total += c; }
        }
        gain[l].resize((size_t) n);
        if (total > 0) {
            any_profile = true;
            for (int64_t e = 0; e < n; ++e) {
                gain[l][(size_t) e] = ((double) (*in[l].cnt)[(size_t) e] + 0.5) / ((double) total + 0.5 * (double) n);
            }
            std::sort(gain[l].begin(), gain[l].end(), std::greater<double>());
        } else {
            std::fill(gain[l].begin(), gain[l].end(), 1.0 / (double) n);
        }
        gain_cum[l].assign((size_t) n + 1, 0.0);
        for (int64_t k = 0; k < n; ++k) { gain_cum[l][(size_t) k + 1] = gain_cum[l][(size_t) k] + gain[l][(size_t) k]; }
    }
    po.profile = any_profile;

    // floor, then the rest
    int32_t floor_s = (int32_t) std::floor(u * (double) std::max(0, floor_pct) / 100.0);
    if (floor_pct > 0) { floor_s = std::max(floor_s, 1); }
    floor_s = std::min(floor_s, (int32_t) std::floor(u));
    po.floor_slots = floor_s;
    std::vector<int32_t> slots(L, 0);
    uint64_t rest = avail;
    for (size_t l = 0; l < L; ++l) {
        slots[l] = std::min(cap[l], floor_s);
        rest -= (uint64_t) slots[l] * in[l].slot_bytes;   // floor_s <= floor(u), so the floors never exceed `avail`
    }

    if (any_profile) {
        // greedy on the marginal hit probability per byte; a layer that cannot afford its next slot is closed for good
        // (its cost is constant and `rest` only shrinks). Ties go to the lower layer index.
        std::vector<char> closed(L, 0);
        for (;;) {
            int    best = -1;
            double best_key = -1.0;
            for (size_t l = 0; l < L; ++l) {
                if (closed[l]) { continue; }
                if (slots[l] >= cap[l] || in[l].slot_bytes > rest) { closed[l] = 1; continue; }
                const double key = gain[l][(size_t) slots[l]] / (double) in[l].slot_bytes;
                if (key > best_key) { best_key = key; best = (int) l; }
            }
            if (best < 0) { break; }
            slots[(size_t) best]++;
            rest -= in[(size_t) best].slot_bytes;
        }
    } else {
        // no counts anywhere: equal slot counts, remainder one slot at a time from the lowest layer up
        bool progress = true;
        while (progress) {
            progress = false;
            for (size_t l = 0; l < L; ++l) {
                if (slots[l] < cap[l] && in[l].slot_bytes <= rest) {
                    slots[l]++;
                    rest -= in[l].slot_bytes;
                    progress = true;
                }
            }
        }
    }

    // predicted hit rate for the plan and for equal slot counts at the same budget (same caps)
    const int32_t eq = (int32_t) std::floor(u);
    double hit_p = 0.0, hit_e = 0.0;
    int32_t mn = INT32_MAX, mx = 0;
    uint64_t sum_use = 0;
    for (size_t l = 0; l < L; ++l) {
        hit_p += gain_cum[l][(size_t) slots[l]];
        hit_e += gain_cum[l][(size_t) std::min(cap[l], eq)];
        if (slots[l] > 0) {
            po.use[in[l].il] = slots[l];
            po.fixed_bytes += (uint64_t) (1 + shadow) * in[l].slot_bytes;
            po.planned_bytes += (uint64_t) (1 + shadow + slots[l]) * in[l].slot_bytes;
            mn = std::min(mn, slots[l]);
            mx = std::max(mx, slots[l]);
            sum_use += (uint64_t) slots[l];
        } else {
            po.n_dropped++;
        }
    }
    po.min_use     = po.use.empty() ? 0 : mn;
    po.max_use     = mx;
    po.mean_use    = po.use.empty() ? 0.0 : (double) sum_use / (double) po.use.size();
    po.hit_planned = hit_p / (double) L;
    po.hit_equal   = hit_e / (double) L;
    return po;
}

// A layer that ends with no slot would only waste its fixed part (zero + shadow slots): drop it and plan again. Layers
// without a slot count as 0 in the predicted hit rates, which are averaged over all layers that were offered.
plan_out plan_budget_all(std::vector<plan_in> in, uint64_t budget, int32_t shadow, int32_t floor_pct, int32_t cap_max) {
    const size_t n_total = in.size();
    plan_out po;
    for (int iter = 0; iter < 4; ++iter) {
        po = plan_budget(in, budget, shadow, floor_pct, cap_max);
        if (po.use.empty() || po.use.size() == in.size()) {
            break;
        }
        std::vector<plan_in> keep;
        for (const auto & p : in) {
            if (po.use.count(p.il)) { keep.push_back(p); }
        }
        in = keep;
    }
    const double scale = n_total ? (double) in.size() / (double) n_total : 0.0;
    po.hit_planned *= scale;
    po.hit_equal   *= scale;
    po.n_dropped    = n_total - po.use.size();
    return po;
}

} // namespace

int llama_moe_cache_max_tokens() {
    const char * m = std::getenv("FLE_MOE_CACHE_BATCH");
    if (m && (std::string(m) == "off" || std::string(m) == "aus")) {   // "aus": deprecated German value, kept as an alias
        return 1;
    }
    const char * e = std::getenv("FLE_MOE_CACHE_MAXTOK");
    return (e && *e) ? std::max(1, atoi(e)) : 8;
}

void llama_moe_cache_init(const llama_model & model, int32_t n_slots, int32_t max_inserts, float cache_bias) {
    std::lock_guard<std::mutex> init_lock(g_init_mtx);
    if (g_init_done) {
        return;
    }
    [&]() {
        // Patch 44b: a byte budget alone (without --moe-expert-cache) also enables the cache
        int64_t budget_mib = 0;
        if (const char * e = std::getenv("FLE_MOE_CACHE_BUDGET_MIB")) { budget_mib = std::max<int64_t>(0, atoll(e)); }
        if (n_slots <= 0 && budget_mib <= 0) {
            g_init_done = true;
            return;
        }

        auto * mc = new moe_cache();
        mc->n_slots = n_slots;
        g_cache_bias = cache_bias > 0.0f ? cache_bias : 0.0f;
        // FLE_MOECACHE_SHADOW: max_inserts additional slots per layer, so that an upload
        // never goes into an occupied slot and the victim can stay valid.
        mc->fle_shadow = fle_getenv_compat("FLE_MOECACHE_SHADOW", "FLE_MOECACHE_SCHATTEN") != nullptr;
        mc->fle_admit_frequent = fle_getenv_compat("FLE_ADMIT_FREQUENT", "FLE_ZULASSUNG_HAEUFIG") != nullptr;
        mc->fle_budget_global     = std::getenv("FLE_BUDGET_GLOBAL") != nullptr;
        if (const char * e = std::getenv("FLE_MOE_ADMIT_FREQ"))   { mc->fle_admit_freq_h = std::max(0, atoi(e)); }
        if (const char * e = std::getenv("FLE_MOE_EVICT_OLDEST")) { mc->fle_evict_oldest = std::max(0, atoi(e)); }
        // Patch 44a: usage profile / warm start
        if (const char * e = std::getenv("FLE_MOE_PROFILE"))          { if (*e) { mc->prof_path = e; } }
        if (const char * e = std::getenv("FLE_MOE_PROFILE_SAVE_SEC")) { mc->prof_save_sec = std::max<int64_t>(0, atoll(e)); }
        if (const char * e = std::getenv("FLE_MOE_PROFILE_MAX_OBS"))  { if (atoll(e) > 0) { mc->prof_max_obs = (uint64_t) atoll(e); } }
        if (!mc->prof_path.empty()) {
            mc->prof_model_desc = model.desc();
            mc->prof_model_size = model.size();
            mc->prof_n_layer    = (uint32_t) model.layers.size();
            mc->prof_last_save  = std::chrono::steady_clock::now();
        }
        // k = max_inserts + 1, not max_inserts. With exactly max_inserts it only works out
        // as long as EVERY transfer takes exactly one step. If one takes longer, a shadow
        // slot is missing and the `if (slot < 0) break;` path is hit, which triggered the
        // collapse at a cap of 16. Cost: one slot per layer, 44 x 2.655 = 117 MiB.
        mc->n_slots_total  = n_slots + (mc->fle_shadow ? (2 * (max_inserts > 0 ? max_inserts : 2)) : 0);
        if (max_inserts > 0) {
            mc->max_inserts = max_inserts;
        }
        mc->shadow_n   = mc->n_slots_total - mc->n_slots;   // = 2 * inserts with shadow slots, else 0 (before 44b plans anything)
        mc->budget_mib = budget_mib;
        // Patch 47: the ring slots sit after the zero slot and are paid like the shadow slots (fixed part of every layer)
        mc->pcie_frac  = pcie_config().frac;
        mc->pcie_ring  = pcie_config().ring;

        // collect the host-resident expert layers, grouped by the device buffer
        // type of that layer's router (the cache lives next to the router)
        struct cand { int il; const llama_layer * l; };
        std::map<ggml_backend_buffer_type_t, std::vector<cand>> groups;

        for (size_t il = 0; il < model.layers.size(); ++il) {
            const auto & l = model.layers[il];
            if (!l.ffn_up_exps || !l.ffn_gate_exps || !l.ffn_down_exps || !l.ffn_gate_inp) {
                continue;
            }
            if (!l.ffn_up_exps->data || !l.ffn_gate_exps->data || !l.ffn_down_exps->data) {
                continue; // dry-run / memory-estimation model: weights not loaded, don't bind to it
            }
            if (!l.ffn_up_exps->buffer || !ggml_backend_buffer_is_host(l.ffn_up_exps->buffer)) {
                continue; // experts already on a device: nothing to cache
            }
            if (!l.ffn_gate_inp->buffer || ggml_backend_buffer_is_host(l.ffn_gate_inp->buffer)) {
                continue; // no device home for the cache
            }
            groups[ggml_backend_buffer_get_type(l.ffn_gate_inp->buffer)].push_back({(int) il, &l});
        }

        if (groups.empty()) {
            LLAMA_LOG_INFO("%s: LLAMA_MOE_CACHE_SLOTS=%d but no host-resident expert layers found - disabled\n", __func__, n_slots);
            delete mc;
            return;
        }

        // Patch 44a: read the usage profile before anything is allocated (44b plans slot counts from it)
        prof_file pf;
        if (!mc->prof_path.empty()) {
            std::string why;
            if (prof_read(mc->prof_path, *mc, pf, why) && prof_validate(pf, groups, why)) {
                LLAMA_LOG_INFO("[fle-moe-profile] loaded %s: %zu layers, %" PRIu64 " steps recorded by earlier runs\n",
                        mc->prof_path.c_str(), pf.layers.size(), pf.steps);
            } else {
                pf = prof_file();
                if (why == "not found") {
                    LLAMA_LOG_INFO("[fle-moe-profile] %s not found - cold start, the file is created at the first save\n",
                            mc->prof_path.c_str());
                } else {
                    LLAMA_LOG_WARN("[fle-moe-profile] %s rejected: %s - cold start, the file is replaced at the next save\n",
                            mc->prof_path.c_str(), why.c_str());
                }
            }
        }

        // Patch 44b: global byte budget. Plans the usable slots of every layer before any tensor exists.
        std::map<int, int32_t> plan_use;   // empty = every layer gets mc->n_slots, as before
        if (mc->budget_mib > 0) {
            int32_t floor_pct = 10, cap_max = 0;
            if (const char * e = std::getenv("FLE_MOE_CACHE_FLOOR_PCT")) { floor_pct = std::min(100, std::max(0, atoi(e))); }
            if (const char * e = std::getenv("FLE_MOE_CACHE_SLOTS_MAX")) { cap_max   = std::max(0, atoi(e)); }
            std::vector<plan_in> in;
            for (auto & g : groups) {
                for (auto & c : g.second) {
                    const prof_layer * pl = pf.ok ? pf.find(c.il) : nullptr;
                    in.push_back({c.il, c.l->ffn_up_exps->ne[2],
                            c.l->ffn_up_exps->nb[2] + c.l->ffn_gate_exps->nb[2] + c.l->ffn_down_exps->nb[2],
                            pl ? &pl->cnt : nullptr});
                }
            }
            std::sort(in.begin(), in.end(), [](const plan_in & a, const plan_in & b) { return a.il < b.il; });
            const plan_out po = plan_budget_all(in, (uint64_t) mc->budget_mib << 20, mc->shadow_n + mc->pcie_ring, floor_pct, cap_max);
            if (po.use.empty()) {
                LLAMA_LOG_WARN("%s: FLE_MOE_CACHE_BUDGET_MIB=%lld does not cover even one slot per layer for %zu layers - cache disabled\n",
                        __func__, (long long) mc->budget_mib, in.size());
                delete mc;
                g_init_done = true;
                return;
            }
            for (auto it = groups.begin(); it != groups.end();) {
                auto & v = it->second;
                v.erase(std::remove_if(v.begin(), v.end(), [&](const cand & c) { return po.use.count(c.il) == 0; }), v.end());
                it = v.empty() ? groups.erase(it) : std::next(it);
            }
            plan_use    = po.use;
            mc->n_slots     = po.max_use;                    // only for the log lines below; the layers carry their own
            mc->n_slots_total = po.max_use + mc->shadow_n;
            if (n_slots > 0) {
                LLAMA_LOG_INFO("%s: [fle-moe-budget] --moe-expert-cache %d is ignored, the budget decides the slots per layer\n", __func__, n_slots);
            }
            // count next to sum: layers planned of layers offered, MiB planned of the budget, slots min/mean/max
            LLAMA_LOG_INFO("%s: [fle-moe-budget] EVIDENCE plan: %zu of %zu layers, %.1f of %lld MiB (fixed zero/shadow part %.1f MiB, "
                           "floor %d slots), usable slots per layer min/mean/max %d/%.1f/%d, ranked by %s\n", __func__,
                    po.use.size(), in.size(), po.planned_bytes / 1048576.0, (long long) mc->budget_mib,
                    po.fixed_bytes / 1048576.0, po.floor_slots, po.min_use, po.mean_use, po.max_use,
                    po.profile ? "profile hits per byte" : "equal slot counts (no profile)");
            if (po.profile) {
                LLAMA_LOG_INFO("%s: [fle-moe-budget] EVIDENCE profile predicts %.1f %% hits for this plan, %.1f %% for equal slot counts "
                               "at the same budget\n", __func__, 100.0 * po.hit_planned, 100.0 * po.hit_equal);
            }
        }

        // host buffer for the CPU-side tables
        std::vector<cand> all;
        for (auto & g : groups) {
            all.insert(all.end(), g.second.begin(), g.second.end());
        }

        auto alloc_group = [&](ggml_backend_buffer_type_t buft, const std::vector<cand> & cands, bool tables_only) -> bool {
            ggml_init_params ip = {
                /*.mem_size  =*/ ggml_tensor_overhead()*(cands.size()*5 + 8),
                /*.mem_buffer=*/ nullptr,
                /*.no_alloc  =*/ true,
            };
            ggml_context * ctx = ggml_init(ip);
            if (!ctx) {
                return false;
            }
            mc->ctxs.push_back(ctx);

            for (const auto & c : cands) {
                layer_state * ls = nullptr;
                for (auto & l : mc->layers) {
                    if (l.pub.il == c.il) { ls = &l; break; }
                }
                if (!ls) {
                    mc->layers.push_back({});
                    ls = &mc->layers.back();
                    ls->pub.il       = c.il;
                    ls->n_use        = plan_use.empty() ? mc->n_slots : plan_use.at(c.il);
                    ls->n_total        = ls->n_use + mc->shadow_n;
                    ls->pub.n_slots  = ls->n_total;
                    ls->pub.up_src   = c.l->ffn_up_exps;
                    ls->pub.gate_src = c.l->ffn_gate_exps;
                    ls->pub.down_src = c.l->ffn_down_exps;
                }

                if (tables_only) {
                    ls->pub.host_table = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, ls->pub.up_src->ne[2]);
                    ggml_format_name(ls->pub.host_table, "moe_cache_htbl.%d", c.il);
                } else {
                    const ggml_tensor * u = c.l->ffn_up_exps;
                    const ggml_tensor * g = c.l->ffn_gate_exps;
                    const ggml_tensor * d = c.l->ffn_down_exps;
                    // usable + shadow slots, the zero slot at n_total, then the PCIe ring (Patch 47, 0 without it)
                    const int64_t n_c = ls->n_total + 1 + mc->pcie_ring;
                    ls->pub.up_c   = ggml_new_tensor_3d(ctx, u->type, u->ne[0], u->ne[1], n_c);
                    ls->pub.gate_c = ggml_new_tensor_3d(ctx, g->type, g->ne[0], g->ne[1], n_c);
                    ls->pub.down_c = ggml_new_tensor_3d(ctx, d->type, d->ne[0], d->ne[1], n_c);
                    ls->pub.dev_table = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, u->ne[2]);
                    if (g_cache_bias > 0.0f) {
                        ls->pub.dev_bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, u->ne[2]);
                        ggml_format_name(ls->pub.dev_bias, "moe_cache_bias.%d", c.il);
                    }
                    ggml_format_name(ls->pub.up_c,      "moe_cache_up.%d",   c.il);
                    ggml_format_name(ls->pub.gate_c,    "moe_cache_gate.%d", c.il);
                    ggml_format_name(ls->pub.down_c,    "moe_cache_down.%d", c.il);
                    ggml_format_name(ls->pub.dev_table, "moe_cache_tbl.%d",  c.il);
                }
            }

            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
            if (!buf) {
                LLAMA_LOG_WARN("%s: failed to allocate MoE cache buffer on %s - cache disabled\n",
                        __func__, ggml_backend_buft_name(buft));
                return false;
            }
            ggml_backend_buffer_clear(buf, 0);
            mc->bufs.push_back(buf);
            return true;
        };

        // A share of the layers may keep their cache on a SECOND device. The second GPU
        // may hold 16 GiB and compute nothing, while GPU0 sits at 79.7 % with a long
        // context.
        //
        // The distribution is PER LAYER, not per expert. This keeps the invariant the
        // whole cache rests on: every expert contributes from exactly one half, the other
        // half yields zero for it (zero slot here, host_table as src[3] there). A
        // per-expert distribution would need two tables per layer, and an expert in both
        // caches would be counted twice, without a crash and without a message.
        int dev1_share = 0;
        if (const char * e = fle_getenv_compat("FLE_MOECACHE_DEV1_SHARE", "FLE_MOECACHE_DEV1_ANTEIL")) {
            dev1_share = atoi(e);
            if (dev1_share < 0)   { dev1_share = 0;   }
            if (dev1_share > 100) { dev1_share = 100; }
        }

        // The second GPU device in registration order. Device-neutral, so this file does
        // not become CUDA-specific.
        ggml_backend_buffer_type_t buft_dev1 = nullptr;
        if (dev1_share > 0) {
            int seen = 0;
            for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                ggml_backend_dev_t d = ggml_backend_dev_get(i);
                if (ggml_backend_dev_type(d) != GGML_BACKEND_DEVICE_TYPE_GPU) {
                    continue;
                }
                if (++seen == 2) {
                    buft_dev1 = ggml_backend_dev_buffer_type(d);
                    break;
                }
            }
            if (!buft_dev1) {
                LLAMA_LOG_WARN("%s: FLE_MOECACHE_DEV1_SHARE=%d, but no second "
                        "GPU device found - staying with one\n", __func__, dev1_share);
                dev1_share = 0;
            }
        }

        // Patch 48: lending needs one device buffer per layer (and only one device)
        mc->lend_on = llama_moe_cache_lend_enabled();
        if (mc->lend_on && dev1_share > 0) {
            LLAMA_LOG_WARN("%s: [fle-lend] FLE_UBATCH_LEND does not support FLE_MOECACHE_DEV1_SHARE - lending OFF\n", __func__);
            mc->lend_on = false;
        }

        bool ok = alloc_group(ggml_backend_cpu_buffer_type(), all, /*tables_only=*/true);
        for (auto & g : groups) {
            if (!ok) {
                break;
            }
            if (dev1_share == 0 && mc->lend_on) {
                for (const auto & c : g.second) {
                    ok = alloc_group(g.first, std::vector<cand>{c}, /*tables_only=*/false);
                    if (!ok) {
                        break;
                    }
                    for (auto & l : mc->layers) {
                        if (l.pub.il == c.il) {
                            l.lctx  = mc->ctxs.back();
                            l.lbuf  = mc->bufs.back();
                            l.lbuft = g.first;
                            break;
                        }
                    }
                }
                continue;
            }
            if (dev1_share == 0) {
                ok = alloc_group(g.first, g.second, /*tables_only=*/false);
                continue;
            }
            // TWO DISTRIBUTIONS. ROUND-ROBIN was the first version, justified by equal
            // load over the depth. BLOCK puts a contiguous tail on the second device:
            // one layer boundary instead of 24.
            // MEASURED: +0.08 % difference. Without peer access the distribution is
            // IRRELEVANT. The switch stays so the finding can be re-checked.
            std::vector<cand> on_dev0, on_dev1;
            const bool block = std::getenv("FLE_MOECACHE_DEV1_BLOCK") != nullptr;
            if (block) {
                const size_t n  = g.second.size();
                const size_t n1 = (n * (size_t) dev1_share + 99) / 100;
                const size_t first1 = n > n1 ? n - n1 : 0;
                for (size_t i = 0; i < n; ++i) {
                    if (i >= first1) { on_dev1.push_back(g.second[i]); }
                    else         { on_dev0.push_back(g.second[i]); }
                }
            } else {
                int acc = 0;
                for (const auto & c : g.second) {
                    acc += dev1_share;
                    if (acc >= 100) { acc -= 100; on_dev1.push_back(c); }
                    else                {                 on_dev0.push_back(c); }
                }
            }
            LLAMA_LOG_INFO("%s: MoE cache distributed [%s]: %zu layers on %s, %zu on %s "
                    "(FLE_MOECACHE_DEV1_SHARE=%d)\n", __func__, block ? "BLOCK" : "round-robin",
                    on_dev0.size(), ggml_backend_buft_name(g.first),
                    on_dev1.size(), ggml_backend_buft_name(buft_dev1), dev1_share);
            if (!on_dev0.empty()) { ok = alloc_group(g.first,   on_dev0, /*tables_only=*/false); }
            if (ok && !on_dev1.empty()) { ok = alloc_group(buft_dev1, on_dev1, /*tables_only=*/false); }
        }

        if (!ok) {
            for (auto * b : mc->bufs) { ggml_backend_buffer_free(b); }
            for (auto * c : mc->ctxs) { ggml_free(c); }
            delete mc;
            g_init_done = true; // a real model was seen and allocation failed: stay disabled
            return;
        }

        // init LRU state + tables (everything uncached -> dummy slot n_slots)
        size_t vram = 0;
        for (auto & ls : mc->layers) {
            const int64_t n_expert = ls.pub.up_src->ne[2];
            ls.slot_expert.assign(ls.n_total, -1);
            ls.expert_slot.assign(n_expert, -1);
            ls.slot_last_use.assign(ls.n_total, 0);
            ls.slot_in_flight.assign(ls.n_total, false);
            ls.ring_expert.assign((size_t) mc->pcie_ring, -1);
            ls.ring_last.assign((size_t) mc->pcie_ring, 0);
            if (mc->fle_admit_freq_h > 0 || mc->fle_evict_oldest > 1) { ls.freq.assign(n_expert, 0.0f); }
            if (!mc->prof_path.empty()) { ls.prof_cnt.assign((size_t) n_expert, 0); }

            std::vector<int32_t> dummy(n_expert, ls.n_total);
            ggml_backend_tensor_set(ls.pub.dev_table,  dummy.data(), 0, n_expert*sizeof(int32_t));
            ggml_backend_tensor_set(ls.pub.host_table, dummy.data(), 0, n_expert*sizeof(int32_t));
            if (ls.pub.dev_bias) {
                ls.pub.bias_val = layer_cache_bias(ls.pub.il);
                ls.pub.bias_host.assign(n_expert, 0.0f);   // nothing cached -> no bonus
                ggml_backend_tensor_set(ls.pub.dev_bias, ls.pub.bias_host.data(), 0, n_expert*sizeof(float));
            }

            mc->by_up_src[ls.pub.up_src] = &ls - mc->layers.data();
            vram += ggml_nbytes(ls.pub.up_c) + ggml_nbytes(ls.pub.gate_c) + ggml_nbytes(ls.pub.down_c);
            LLAMA_LOG_DEBUG("moe-cache: init layer %d '%s' %zu bytes/expert\n",
                    ls.pub.il, ls.pub.up_src->name, ls.pub.up_src->nb[2]);
        }

        // Patch 44a: warm start. Runs before the upload worker and the observer exist, so nothing is concurrent.
        if (pf.ok) {
            mc->prof_loaded     = true;
            mc->prof_base_steps = pf.steps;
            for (const auto & pl : pf.layers) {
                bool live = false;
                for (const auto & ls : mc->layers) { if (ls.pub.il == pl.il) { live = true; break; } }
                if (!live) { mc->prof_keep.push_back(pl); }
            }
            prof_prefill(mc, pf);
        }

        // A DEDICATED backend for the upload thread. The path goes through the buffer:
        // buffer -> buft -> device -> dev_init. This is not a workaround: the thread
        // has its own stream anyway (cudaStreamPerThread is thread-local), and a
        // dedicated backend makes that separation explicit instead of accidental.
        if (fle_up_async && !mc->layers.empty() && mc->layers[0].pub.up_c
                && mc->layers[0].pub.up_c->buffer) {
            ggml_backend_buffer_type_t bt = ggml_backend_buffer_get_type(mc->layers[0].pub.up_c->buffer);
            ggml_backend_dev_t dev = bt ? ggml_backend_buft_get_device(bt) : nullptr;
            fle_up_backend = dev ? ggml_backend_dev_init(dev, nullptr) : nullptr;
            fprintf(stderr, "[fle-upload-async] EVIDENCE fle-upload-async %s (device %s)\n",
                    fle_up_backend ? "ON" : "OFF - no backend available",
                    dev ? ggml_backend_dev_name(dev) : "?");
        }

        mc->worker = std::thread([mc]() {
            for (;;) {
                upload_job j;
                {
                    std::unique_lock<std::mutex> lk(mc->wmtx);
                    mc->wcv.wait(lk, [mc]() { return mc->stop || (!mc->wpaused && !mc->todo.empty()); });
                    if (mc->stop) {
                        return;
                    }
                    j = mc->todo.front();
                    mc->todo.pop_front();
                    mc->wbusy = true;   // Patch 48: a lend waits until the worker no longer writes into a slot
                }
                auto & ls = mc->layers[j.layer_idx];
                if (fle_up_async && fle_up_backend) {
                    // BATCHED: issue this job and everything else currently waiting
                    // without intermediate waits, then synchronize ONCE.
                    std::vector<upload_job> batch;
                    batch.push_back(j);
                    {
                        std::lock_guard<std::mutex> lk(mc->wmtx);
                        while (!mc->todo.empty()) {
                            batch.push_back(mc->todo.front());
                            mc->todo.pop_front();
                        }
                    }
                    for (auto & q : batch) {
                        auto & qs = mc->layers[q.layer_idx];
                        upload_slice_async(qs.pub.up_c,   qs.pub.up_src,   q.expert, q.slot);
                        upload_slice_async(qs.pub.gate_c, qs.pub.gate_src, q.expert, q.slot);
                        upload_slice_async(qs.pub.down_c, qs.pub.down_src, q.expert, q.slot);
                    }
                    // THE one sync point. It used to be 3 per job.
                    {
                        fle_up_timer u;
                        ggml_backend_synchronize(fle_up_backend);
                    }
                    fle_up_sync_points++;
                    {
                        std::lock_guard<std::mutex> lk(mc->wmtx);
                        for (auto & q : batch) {
                            q.done = true;
                            mc->done.push_back(q);
                        }
                        mc->wbusy = false;
                    }
                    mc->wcv.notify_all();
                } else {
                    upload_slice(ls.pub.up_c,   ls.pub.up_src,   j.expert, j.slot);
                    upload_slice(ls.pub.gate_c, ls.pub.gate_src, j.expert, j.slot);
                    upload_slice(ls.pub.down_c, ls.pub.down_src, j.expert, j.slot);
                    fle_up_sync_points += 3;
                    {
                        std::lock_guard<std::mutex> lk(mc->wmtx);
                        j.done = true;
                        mc->done.push_back(j);
                        mc->wbusy = false;
                    }
                    mc->wcv.notify_all();
                }
            }
        });

        ggml_set_moe_obs_callback(moe_obs_cb, mc);
        if (mc->pcie_ring > 0) {
            ggml_backend_set_fle_mc_hook(pcie_hook, mc);
            const ggml_tensor * u0 = mc->layers.empty() ? nullptr : mc->layers[0].pub.up_src;
            // the source buffer decides whether the weight copies are truly asynchronous (pinned host memory) or staged
            LLAMA_LOG_INFO("%s: [fle-pcie] PCIe share ON: %.2f of the misses, %d ring slots per layer (%.1f MiB), expert source buffer %s\n",
                    __func__, mc->pcie_frac, mc->pcie_ring,
                    u0 ? (double) mc->layers.size() * mc->pcie_ring *
                         (double) (mc->layers[0].pub.up_src->nb[2] + mc->layers[0].pub.gate_src->nb[2] + mc->layers[0].pub.down_src->nb[2]) / 1048576.0 : 0.0,
                    u0 && u0->buffer ? ggml_backend_buffer_name(u0->buffer) : "?");
        }
        g_cache = mc;
        g_init_done = true;

        LLAMA_LOG_INFO("%s: MoE expert cache enabled: %zu layers x %d slots, %d inserts/step, %.1f MiB device memory\n",
                __func__, mc->layers.size(), mc->n_slots, mc->max_inserts, vram/1024.0/1024.0);
        if (mc->budget_mib > 0) {
            int32_t mn = INT32_MAX, mx = 0;
            uint64_t sum = 0;
            for (const auto & ls : mc->layers) { mn = std::min(mn, ls.n_use); mx = std::max(mx, ls.n_use); sum += (uint64_t) ls.n_use; }
            LLAMA_LOG_INFO("%s: [fle-moe-budget] EVIDENCE allocated %.1f MiB of budget %lld MiB (%.1f %%) in %zu layers, "
                           "usable slots per layer min/mean/max %d/%.1f/%d (\"x %d slots\" above is the maximum)\n", __func__,
                    vram / 1048576.0, (long long) mc->budget_mib, 100.0 * (vram / 1048576.0) / (double) mc->budget_mib,
                    mc->layers.size(), mn, (double) sum / (double) mc->layers.size(), mx, mc->n_slots);
        }
        // Evidence lines, visible from -lv 5. Without them "no effect" and "never
        // arrived" look the same.
        if (fle_slowdown_tab_us > 0) {
            LLAMA_LOG_INFO("%s: TABLE SLOWDOWN %lld us per call (FLE_SLOWDOWN_TABLE_US)\n",
                           __func__, fle_slowdown_tab_us);
        }
        LLAMA_LOG_INFO("%s: budget %s (FLE_BUDGET_GLOBAL)\n",
                       __func__, mc->fle_budget_global ? "CARRIED OVER" : "per layer, expires");
        LLAMA_LOG_INFO("%s: admission by %s (FLE_ADMIT_FREQUENT)\n",
                       __func__, mc->fle_admit_frequent ? "FREQUENCY" : "observation order");
        LLAMA_LOG_INFO("%s: shadow slots %s (%d usable + %d shadow = %d total, zero slot at %d)\n",
                __func__, mc->fle_shadow ? "ON" : "off",
                mc->n_slots, mc->n_slots_total - mc->n_slots, mc->n_slots_total, mc->n_slots_total);
        if (!mc->prof_path.empty()) {
            LLAMA_LOG_INFO("%s: usage profile %s (FLE_MOE_PROFILE): %s, saved every %lld s and at shutdown\n", __func__,
                    mc->prof_path.c_str(), mc->prof_loaded ? "warm start" : "cold start", (long long) mc->prof_save_sec);
        }
    }();
}

namespace {

// publish completed uploads; only between graph executions (llama_moe_cache_step, and before a lend)
void publish_done(moe_cache * mc) {
    std::lock_guard<std::mutex> wlk(mc->wmtx);
    std::lock_guard<std::mutex> lk(mc->mtx);
    for (const auto & j : mc->done) {
        auto & ls = mc->layers[j.layer_idx];
        ls.slot_expert[j.slot]     = j.expert;
        ls.expert_slot[j.expert]   = j.slot;
        ls.slot_last_use[j.slot]   = ++mc->clock;
        ls.slot_in_flight[j.slot]  = false;
        set_table_entry(ls.pub, j.expert, j.slot);
        // Only NOW release the victim: until here it was valid and kept producing
        // hits. That is the whole point of shadow mode.
        if (j.victim_slot >= 0) {
            const int32_t victim_expert = ls.slot_expert[j.victim_slot];
            if (victim_expert >= 0) {
                ls.expert_slot[victim_expert]        = -1;
                set_table_entry(ls.pub, victim_expert, ls.n_total);
            }
            ls.slot_expert[j.victim_slot]    = -1;
            ls.slot_in_flight[j.victim_slot] = false;
        }
    }
    mc->done.clear();
}

} // namespace

bool llama_moe_cache_pcie_on() {
    return pcie_config().ring > 0;
}

bool llama_moe_cache_lend_enabled() {
    static const bool on = [] {
        const char * e = std::getenv("FLE_UBATCH_LEND");
        return e && *e && atoi(e) != 0;
    }();
    return on;
}

uint64_t llama_moe_cache_epoch() {
    return g_cache ? g_cache->lend_epoch : 0;
}

size_t llama_moe_cache_lend(size_t bytes) {
    moe_cache * mc = g_cache;
    if (!mc || !mc->lend_on || bytes == 0) {
        return 0;
    }
    if (mc->lend_cur > 0) {
        return mc->lend_cur;
    }

    // no slot may be written while its buffer goes away: let the worker finish what it holds, then pause it
    {
        std::unique_lock<std::mutex> lk(mc->wmtx);
        mc->wcv.wait(lk, [mc]() { return mc->todo.empty() && !mc->wbusy; });
        mc->wpaused = true;
    }
    publish_done(mc);

    std::lock_guard<std::mutex> lock(mc->mtx);
    size_t   freed = 0;
    uint64_t n     = 0;
    for (size_t i = mc->layers.size(); i-- > 0 && freed < bytes; ) {
        auto & ls = mc->layers[i];
        if (ls.lent || !ls.lbuf || !ls.lctx) {
            continue;
        }
        const size_t sz = ggml_backend_buffer_get_size(ls.lbuf);
        for (auto & b : mc->bufs) {
            if (b == ls.lbuf) { b = nullptr; }
        }
        ggml_backend_buffer_free(ls.lbuf);
        for (ggml_tensor * t = ggml_get_first_tensor(ls.lctx); t; t = ggml_get_next_tensor(ls.lctx, t)) {
            t->data   = nullptr;
            t->buffer = nullptr;
        }
        ls.lbuf = nullptr;
        ls.lent = true;
        freed  += sz;
        n++;
    }
    if (n == 0) {
        std::lock_guard<std::mutex> lk(mc->wmtx);
        mc->wpaused = false;
        mc->wcv.notify_all();
        return 0;
    }
    mc->lend_cur    = freed;
    mc->lend_asked  = bytes;
    mc->lend_epoch++;
    mc->lend_n++;
    mc->lend_layers += n;
    mc->lend_bytes  += freed;
    if (freed < bytes) {
        mc->lend_short++;
    }
    return freed;
}

void llama_moe_cache_return() {
    moe_cache * mc = g_cache;
    if (!mc || !mc->lend_on) {
        return;
    }
    const auto t0 = std::chrono::steady_clock::now();
    uint64_t n_back = 0, n_fail = 0, n_slices = 0;
    {
        std::lock_guard<std::mutex> lock(mc->mtx);
        bool any = false;
        for (const auto & ls : mc->layers) {
            if (ls.lent) { any = true; break; }
        }
        if (!any && mc->lend_cur == 0) {
            return;
        }

        // first every allocation, then the copies: all of them on one stream, one synchronize at the end
        std::vector<layer_state *> back;
        for (auto & ls : mc->layers) {
            if (!ls.lent) {
                continue;
            }
            ggml_backend_buffer_t b = ggml_backend_alloc_ctx_tensors_from_buft(ls.lctx, ls.lbuft);
            if (!b) {
                n_fail++;   // stays lent: lookup keeps returning nullptr, the CPU computes this layer
                continue;
            }
            ggml_backend_buffer_clear(b, 0);   // the zero slot
            for (auto & ob : mc->bufs) {
                if (ob == nullptr) { ob = b; break; }
            }
            ls.lbuf = b;
            back.push_back(&ls);
        }

        if (!mc->refill_backend && !back.empty()) {
            ggml_backend_dev_t dev = ggml_backend_buft_get_device(back[0]->lbuft);
            mc->refill_backend = dev ? ggml_backend_dev_init(dev, nullptr) : nullptr;
        }
        ggml_backend_t be = mc->refill_backend;

        for (auto * lp : back) {
            auto & ls = *lp;
            const bool async_ok = be && ggml_backend_get_default_buffer_type(be) == ls.lbuft;
            auto put = [&](ggml_tensor * dst, const void * data, size_t off, size_t size) {
                if (async_ok) { ggml_backend_tensor_set_async(be, dst, data, off, size); }
                else          { ggml_backend_tensor_set(dst, data, off, size); }
            };
            const ggml_tensor * srcs[3] = { ls.pub.up_src, ls.pub.gate_src, ls.pub.down_src };
            ggml_tensor *       dsts[3] = { ls.pub.up_c,   ls.pub.gate_c,   ls.pub.down_c   };
            for (int32_t s = 0; s < ls.n_total; ++s) {
                const int32_t e = ls.slot_expert[(size_t) s];
                if (e < 0) {
                    continue;
                }
                for (int k = 0; k < 3; ++k) {
                    const size_t sz = srcs[k]->nb[2];
                    put(dsts[k], (const char *) srcs[k]->data + (size_t) e*sz, (size_t) s*dsts[k]->nb[2], sz);
                }
                n_slices++;
            }
            // the ring held per-call copies; they are gone, so every ring mapping falls back to the zero slot
            int32_t * tbl = (int32_t *) ls.pub.host_table->data;
            const int64_t n_expert = ls.pub.host_table->ne[1];
            for (int64_t e = 0; e < n_expert; ++e) {
                if (tbl[e] > ls.n_total) { tbl[e] = ls.n_total; }
            }
            std::fill(ls.ring_expert.begin(), ls.ring_expert.end(), -1);
            put(ls.pub.dev_table, tbl, 0, ggml_nbytes(ls.pub.dev_table));
            if (ls.pub.dev_bias) {
                put(ls.pub.dev_bias, ls.pub.bias_host.data(), 0, ls.pub.bias_host.size()*sizeof(float));
                ls.pub.bias_dirty = false;
            }
        }
        if (be) {
            ggml_backend_synchronize(be);
        }
        for (auto * lp : back) {
            lp->lent = false;
            n_back++;
        }
        mc->refill_failed += n_fail;
        mc->refill_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        mc->lend_epoch++;
        const size_t lent_bytes = mc->lend_cur;
        mc->lend_cur = 0;
        LLAMA_LOG_INFO("[fle-lend] EVIDENCE lend %" PRIu64 ": %.1f MiB lent for %.1f MiB asked, %" PRIu64 " layers back "
                       "(%" PRIu64 " slots refilled) in %.1f ms, %" PRIu64 " failed; totals %" PRIu64 " layers, %.2f GB, "
                       "%" PRIu64 " short, %" PRIu64 " failed, refill %.1f ms\n",
                mc->lend_n, lent_bytes / 1048576.0, mc->lend_asked / 1048576.0, n_back, n_slices,
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(), n_fail,
                mc->lend_layers, mc->lend_bytes / 1e9, mc->lend_short, mc->refill_failed, mc->refill_ms);
    }
    {
        std::lock_guard<std::mutex> lk(mc->wmtx);
        mc->wpaused = false;
    }
    mc->wcv.notify_all();
}

const llama_moe_cache_layer * llama_moe_cache_lookup(const ggml_tensor * up_exps) {
    if (!g_cache) {
        return nullptr;
    }
    auto it = g_cache->by_up_src.find(up_exps);
    if (it == g_cache->by_up_src.end()) {
        return nullptr;
    }
    if (g_cache->layers[it->second].lent) {
        return nullptr;   // Patch 48: lent to a prefill compute buffer, the CPU computes all experts of this layer
    }
    return &g_cache->layers[it->second].pub;
}

void llama_moe_cache_step() {
    moe_cache * mc = g_cache;
    if (!mc) {
        return;
    }

    // 1) publish completed uploads (sync point: no graph is executing)
    publish_done(mc);

    std::lock_guard<std::mutex> lock(mc->mtx);
    mc->n_steps++;

    // 2) schedule new uploads: evict at a sync point (clear the victim's table
    //    entry now), then hand the slice copies to the worker
    // FLE_BUDGET_GLOBAL: what one layer does not need is passed on to the next.
    // The budget used to live INSIDE this loop and expired per layer, but about half of
    // all layers have no miss at all in a step (22.8 of 45 used). The total number of
    // possible insertions stays the same, hence also the copy load; only their
    // distribution now follows the misses.
    int carry = 0;
    for (size_t li = 0; li < mc->layers.size(); ++li) {
        auto & ls = mc->layers[li];
        if (ls.lent) {
            // Patch 48: no device memory right now; its misses come back at the next access
            ls.pending.clear();
            ls.pending_count.clear();
            continue;
        }
        if (ls.pending.empty()) {
            // An empty layer also contributes its unused budget.
            if (mc->fle_budget_global) { carry += mc->max_inserts; }
            mc->fle_budget_possible += mc->max_inserts;
            continue;
        }

        int budget = mc->max_inserts;
        if (mc->fle_budget_global) {
            budget += carry;
            carry = 0;
        }
        mc->fle_budget_possible += mc->max_inserts;
        const int budget_start = budget;
        // FLE: the ORDER in which candidates are considered.
        //
        // By default `rbegin()`: the most recently observed miss. `pending` is
        // deduplicated, so the request count was already discarded before anyone chose;
        // the choice followed arrival order in the graph. That there IS something to
        // choose from shows in the run counters: dropped-over-cap 41.09 per step at
        // inserts 1.
        //
        // With FLE_ADMIT_FREQUENT the most-requested expert of this step is taken, at the
        // SAME budget and thus the same copy load. Not more admission (that costs copies,
        // and `inserts 1` is confirmed as a throughput lever), but better admission at
        // the same price.
        std::vector<size_t> ord(ls.pending.size());
        for (size_t k = 0; k < ord.size(); ++k) {
            ord[k] = ls.pending.size() - 1 - k;   // rbegin order, unchanged
        }
        if (ord.size() > 1) {
            mc->fle_choice_total++;
            if (mc->fle_admit_frequent) {
                // stable_sort: at equal frequency the old order is kept, so the
                // switch then changes nothing. Without stable, part of the effect
                // would be mere reordering.
                std::stable_sort(ord.begin(), ord.end(),
                    [&](size_t a, size_t b) { return ls.pending_count[a] > ls.pending_count[b]; });
                if (ord[0] != ls.pending.size() - 1) {
                    mc->fle_choice_differs++;
                }
            }
        }
        if (mc->fle_admit_freq_h > 0 && ord.size() > 1 && !ls.freq.empty()) {
            // stable: at equal frequency the rbegin order is kept
            std::stable_sort(ord.begin(), ord.end(),
                [&](size_t a, size_t b) { return ls.freq[ls.pending[a]] > ls.freq[ls.pending[b]]; });
        }
        for (size_t oi = 0; oi < ord.size() && budget > 0; ++oi) {
            const int32_t id = ls.pending[ord[oi]];
            if (ls.expert_slot[id] >= 0) {
                // Published between observation and planning. The budget is NOT
                // consumed here: the `--budget` check used to sit in the loop header, and
                // at budget 1 such a case cost the whole insertion of the step.
                continue;
            }

            // Find the target slot: empty and not in flight, otherwise the oldest.
            int32_t slot = -1, victim_slot = -1;
            uint64_t best = UINT64_MAX;
            // FLE: the stock of empty slots is COUNTED, not just the find.
            // Without that, normal operation consumes the shadow slots (it used to take
            // the first empty one, shadow or not), and in steady state the shadow branch
            // never finds an empty target -> break -> pending.clear().
            // The reservation ROTATES: which k slots are empty does not matter.
            int32_t n_empty = 0, first_empty = -1, oldest = -1;
            for (int32_t s = 0; s < ls.n_total; ++s) {
                if (ls.slot_in_flight[s]) {
                    continue;
                }
                if (ls.slot_expert[s] < 0) {
                    n_empty++;
                    if (first_empty < 0) { first_empty = s; }
                    continue;
                }
                if (ls.slot_last_use[s] < best) { best = ls.slot_last_use[s]; oldest = s; }
            }
            // With shadow mode off, reserve is 0 -> `n_empty > 0` picks the same slot as
            // the earlier `break` at the first empty one. Unchanged.
            const int32_t reserve = mc->fle_shadow ? 2 * mc->max_inserts : 0;
            if (mc->fle_evict_oldest > 1 && oldest >= 0 && !ls.freq.empty() && !(first_empty >= 0 && n_empty > reserve)) {
                // the N oldest occupied slots (not in flight) by last use, of those the one with the smallest frequency
                std::vector<std::pair<uint64_t, int32_t>> old_slots;
                old_slots.reserve(ls.n_total);
                for (int32_t s = 0; s < ls.n_total; ++s) {
                    if (!ls.slot_in_flight[s] && ls.slot_expert[s] >= 0) { old_slots.push_back({ls.slot_last_use[s], s}); }
                }
                const size_t nn = std::min<size_t>(old_slots.size(), (size_t) mc->fle_evict_oldest);
                std::partial_sort(old_slots.begin(), old_slots.begin() + nn, old_slots.end());
                float best_f = 1e30f;
                for (size_t k = 0; k < nn; ++k) {
                    const float f = ls.freq[ls.slot_expert[old_slots[k].second]];
                    if (f < best_f) { best_f = f; oldest = old_slots[k].second; }   // on a tie the older one stays
                }
            }
            if (first_empty >= 0 && n_empty > reserve) {
                slot = first_empty;
            } else if (oldest >= 0) {
                slot = oldest;
            } else if (first_empty >= 0) {
                slot = first_empty;   // nothing to evict; the reserve does not take precedence over nothing
            }
            if (slot < 0) {
                mc->fle_no_free_slot++;
                break; // every slot is in flight; try again next step
            }
            if (mc->fle_shadow && mc->n_steps % 512 == 0 && li == 0) {
                // Evidence with TWO numbers, and that is the point:
                // `n_empty` counts only AVAILABLE slots (the continue on
                // slot_in_flight comes before it). A line that reports only this one
                // number cannot show a leak: it counts with the same predicate as the
                // allocator and confirms itself. Next to it belongs the OTHER count:
                // empty including those already handed out. If the two diverge, one
                // sees immediately how much of the reserve is currently bound in flight.
                int32_t empty_total = 0;
                for (int32_t s = 0; s < ls.n_total; ++s) {
                    if (ls.slot_expert[s] < 0) { empty_total++; }
                }
                LLAMA_LOG_INFO("[fle-shadow] EVIDENCE reserve: %d available · %d empty in total "
                               "(of which %d already handed out) of %d slots · reserve %d "
                               "· layer 0, step %" PRIu64 "%s\n",
                               n_empty, empty_total, empty_total - n_empty, ls.n_total,
                               reserve, mc->n_steps,
                               n_empty < reserve ? "  << BELOW RESERVE, invariant leaking" : "");
            }

            if (mc->fle_shadow) {
                // SHADOW MODE: only load into an EMPTY slot. If the one found is
                // occupied, it is the victim; it stays VALID until the successor is
                // published. That needs a second, empty slot.
                if (ls.slot_expert[slot] >= 0) {
                    victim_slot = slot;
                    slot = -1;
                    for (int32_t s = 0; s < ls.n_total; ++s) {
                        if (!ls.slot_in_flight[s] && ls.slot_expert[s] < 0) { slot = s; break; }
                    }
                    if (slot < 0) {
                        // No shadow slot free: better not to insert at all than to
                        // discard the victim blindly. Counts as "no free slot".
                        mc->fle_no_free_slot++;
                        break;
                    }
                    // Take the victim out of the eviction selection; the table stays valid.
                    ls.slot_in_flight[victim_slot] = true;
                }
            } else {
                const int32_t victim = ls.slot_expert[slot];
                if (victim >= 0) {
                    ls.expert_slot[victim] = -1;
                    ls.slot_expert[slot]   = -1;
                    set_table_entry(ls.pub, victim, ls.n_total);
                }
            }
            ls.slot_in_flight[slot] = true;

            std::lock_guard<std::mutex> wlk(mc->wmtx);
            // Is the same expert of this layer already in the queue? Then it is
            // transferred a second time and occupies a second slot.
            for (const auto & q : mc->todo) {
                if (q.layer_idx == li && q.expert == id) { mc->fle_double_planned++; break; }
            }
            mc->todo.push_back({li, id, slot, victim_slot});
            --budget;   // only now: an insertion is actually scheduled
        }
        mc->fle_budget_used += (uint64_t) (budget_start - budget);
        if (mc->fle_budget_global && budget > 0) {
            carry = budget;   // unused -> next layer
        }
        // What the budget could no longer take is dropped here (not deferred) and
        // reappears as a miss at the next access.
        if ((int) ls.pending.size() > mc->max_inserts) {
            mc->fle_dropped += ls.pending.size() - mc->max_inserts;
        }
        ls.pending.clear();
        ls.pending_count.clear();
    }
    mc->wcv.notify_all();   // a lend may wait on the same condition variable

    // FLE: bonus tables (cache-aware routing), one copy per changed layer
    for (auto & ls : mc->layers) {
        if (ls.pub.dev_bias && ls.pub.bias_dirty && !ls.lent) {
            ggml_backend_tensor_set(ls.pub.dev_bias, ls.pub.bias_host.data(), 0, ls.pub.bias_host.size()*sizeof(float));
            ls.pub.bias_dirty = false;
        }
    }

    if (mc->n_steps % 512 == 0 && fle_slowdown_tab_n > 0) {
        // Count NEXT TO sum: an inserted time without its count says nothing, and
        // without the actually measured sum the slowdown would be an unverified switch.
        LLAMA_LOG_INFO("[fle-slowdown-tab] EVIDENCE %lld calls slowed down, %.1f ms actually "
                       "inserted (%.1f us each)\n",
                       fle_slowdown_tab_n, fle_slowdown_tab_ms,
                       1000.0 * fle_slowdown_tab_ms / (double) fle_slowdown_tab_n);
    }
    if (mc->n_steps % 512 == 0 && mc->fle_budget_possible > 0) {
        // Two numbers, because one could not be interpreted: only in ratio does it say
        // whether the carry-over had anything to carry.
        LLAMA_LOG_INFO("[fle-budget] EVIDENCE budget: %" PRIu64 " of %" PRIu64
                       " used (%.1f %%)\n",
                       mc->fle_budget_used, mc->fle_budget_possible,
                       100.0 * (double) mc->fle_budget_used / (double) mc->fle_budget_possible);
    }
    if (mc->n_steps % 512 == 0 && mc->fle_choice_total > 0) {
        // Reference count next to it: without `of N` a 0 cannot be interpreted; it would
        // mean either "the switch never changes anything" or "it was never reached".
        LLAMA_LOG_INFO("[fle-admission] EVIDENCE choice differs: %" PRIu64 " of %" PRIu64
                       " opportunities (%.1f %%)\n",
                       mc->fle_choice_differs, mc->fle_choice_total,
                       100.0 * (double) mc->fle_choice_differs / (double) mc->fle_choice_total);
    }
    if (mc->n_steps % 512 == 0) {
        uint64_t h = 0, m = 0;
        for (auto & ls : mc->layers) { h += ls.n_hit; m += ls.n_miss; }
        if (fle_up_every && mc->n_steps % fle_up_every == 0) {
            // REFERENCE COUNTS: pieces next to bytes next to duration. A sum without a
            // count says nothing, and the duration is the actual question here.
            fprintf(stderr, "[fle-upload] EVIDENCE fle-upload after %" PRIu64 " steps: "
                    "%lld slots (%.2f GB, %.1f ms, %.1f us each) · "
                    "table %lld calls (%.1f ms, %.1f us each, TWO syncs each) · "
                    "together %.1f ms · %lld sync points · slowdown %lld x %.1f ms\n",
                    mc->n_steps, fle_up_pieces, fle_up_bytes / 1e9, fle_up_ms,
                    fle_up_pieces ? 1000.0*fle_up_ms/fle_up_pieces : 0.0,
                    fle_up_table_sets, fle_up_tab_ms,
                    fle_up_table_sets ? 1000.0*fle_up_tab_ms/fle_up_table_sets : 0.0,
                    fle_up_ms + fle_up_tab_ms, fle_up_sync_points, fle_slowdown_n, fle_slowdown_ms);
        }
        LLAMA_LOG_INFO("moe-cache: steps=%" PRIu64 " hits=%" PRIu64 " misses=%" PRIu64 " hit-rate=%.1f%%\n",
                mc->n_steps, h, m, h + m ? 100.0*h/(h + m) : 0.0);
        if (mc->pcie_ring > 0) {
            // Patch 47: served of seen (distinct uncached experts per layer call), split into copies and free re-mappings;
            // the hit rate above still counts ring-served experts as misses
            LLAMA_LOG_INFO("[fle-pcie] EVIDENCE %" PRIu64 " hooks (%" PRIu64 " skipped): %" PRIu64 " of %" PRIu64
                           " uncached experts served over the ring (%.1f %%), %" PRIu64 " copied (%.2f GB), %" PRIu64
                           " re-mapped without copy, %" PRIu64 " dropped (ring full); per step %.2f copies\n",
                    mc->pcie_hooks, mc->pcie_skipped, mc->pcie_served, mc->pcie_misses,
                    mc->pcie_misses ? 100.0 * (double) mc->pcie_served / (double) mc->pcie_misses : 0.0,
                    mc->pcie_loaded, mc->pcie_bytes / 1e9, mc->pcie_reused, mc->pcie_ring_full,
                    (double) mc->pcie_loaded / (double) mc->n_steps);
        }
        // The three candidates for the gap to LRU, with their reference size next to
        // them; a sum without a count says nothing.
        LLAMA_LOG_DEBUG("moe-cache: double-planned=%" PRIu64 " no-free-slot=%" PRIu64
                " dropped-over-cap=%" PRIu64 " (per step: %.3f / %.3f / %.3f)\n",
                mc->fle_double_planned, mc->fle_no_free_slot, mc->fle_dropped,
                mc->n_steps ? (double) mc->fle_double_planned / mc->n_steps : 0.0,
                mc->n_steps ? (double) mc->fle_no_free_slot  / mc->n_steps : 0.0,
                mc->n_steps ? (double) mc->fle_dropped       / mc->n_steps : 0.0);
    }

    // Patch 44a: first-512-steps hit rate next to the pre-fill count, then the periodic profile save
    if (!mc->prof_path.empty()) {
        if (mc->n_steps == 512) {
            uint64_t h = 0, m = 0;
            for (auto & ls : mc->layers) { h += ls.n_hit; m += ls.n_miss; }
            LLAMA_LOG_INFO("[fle-moe-profile] EVIDENCE first 512 steps: %" PRIu64 " hits of %" PRIu64 " lookups (%.1f %%); "
                           "pre-filled %" PRIu64 " of %" PRIu64 " slots, profile %s, profile predicted %.1f %%\n",
                    h, h + m, h + m ? 100.0 * (double) h / (double) (h + m) : 0.0,
                    mc->prefilled_slots, mc->prefill_cap, mc->prof_loaded ? "loaded" : "none", 100.0 * mc->prefill_mass);
        }
        if (mc->prof_save_sec > 0 && (mc->n_steps & 63) == 0 && mc->prof_dirty &&
                std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - mc->prof_last_save).count() >= mc->prof_save_sec) {
            prof_write(mc, "periodic");
        }
    }
}

void llama_moe_cache_profile_save() {
    moe_cache * mc = g_cache;
    if (!mc || mc->prof_path.empty()) {
        return;
    }
    std::lock_guard<std::mutex> lock(mc->mtx);
    if (mc->prof_dirty) {   // nothing new since the last save (or since start): leave the file alone
        prof_write(mc, "shutdown");
    }
}
