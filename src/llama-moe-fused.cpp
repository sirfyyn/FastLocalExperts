#include "llama-moe-fused.h"
#include "llama-moecache.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <mutex>
#include <vector>
#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif
#include <cstdlib>
#include <cstdio>
#include <thread>
#include <functional>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define FLE_PAUSE() _mm_pause()
#else
#define FLE_PAUSE() ((void) 0)
#endif

// ggml_get_type_traits_cpu lives in the CPU backend; with GGML_BACKEND_DL that is a module, so look it up at run time
typedef const struct ggml_type_traits_cpu * (*fle_traits_fn_t)(enum ggml_type);
static const struct ggml_type_traits_cpu * fle_traits(ggml_type t) {
    static fle_traits_fn_t fn = [] {
        ggml_backend_reg_t reg = ggml_backend_reg_by_name("CPU");
        return reg ? (fle_traits_fn_t) ggml_backend_reg_get_proc_address(reg, "ggml_get_type_traits_cpu") : nullptr;
    }();
    return fn ? fn(t) : nullptr;
}

namespace {

// ---- FLE_MOE_POOL=1: own pinned worker pool for the fused CPU MoE op (after Strata's lane pool, ds4_pool.hpp, MIT) ----------------
// ggml runs the op on ONE thread; that thread hands the work to N workers pinned to their own physical cores (FLE_MOE_POOL_CPUS,
// default 2-15: two cores stay free for the threads that feed the GPU), spinning between layers instead of sleeping. Jobs are
// (expert pair, 64-row block), claimed with one atomic ticket, so a layer with several CPU experts spreads evenly over all cores.
struct fle_pool {
    std::vector<std::thread> th;
    std::atomic<uint64_t> gen{0};      // bumped for every parallel section
    std::atomic<int64_t>  next{0};     // next job index of the running section
    std::atomic<int>      done{0};     // workers finished with the running section
    int64_t n_jobs = 0;
    std::function<void(int64_t)> fn;
    int n = 0;
    std::atomic<bool> stop{false};

    static std::vector<int> cpus_from_env() {
        std::vector<int> c;
        const char * e = std::getenv("FLE_MOE_POOL_CPUS");
        int lo = 2, hi = 15;
        if (e && *e) { if (std::sscanf(e, "%d-%d", &lo, &hi) != 2) { lo = 2; hi = 15; } }
        for (int i = lo; i <= hi; ++i) { c.push_back(i); }
        return c;
    }
    void work() {
        for (;;) {
            const int64_t j = next.fetch_add(1, std::memory_order_acq_rel);
            if (j >= n_jobs) { break; }
            fn(j);
        }
    }
    void start() {
        const auto cpus = cpus_from_env();
        n = (int) cpus.size();
        for (int w = 0; w < n; ++w) {
            th.emplace_back([this, w, cpu = cpus[(size_t) w]] {
#if defined(__linux__)
                cpu_set_t set; CPU_ZERO(&set); CPU_SET(cpu, &set);
                pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
#else
                (void) cpu;
#endif
                uint64_t seen = 0;
                for (;;) {
                    // spin on the generation (a layer is ~0.5 ms away), yield after a while
                    int spins = 0;
                    while (gen.load(std::memory_order_acquire) == seen) {
                        if (stop.load(std::memory_order_relaxed)) { return; }
                        if (++spins < 200000) { FLE_PAUSE(); } else { std::this_thread::yield(); }
                    }
                    seen = gen.load(std::memory_order_acquire);
                    work();
                    done.fetch_add(1, std::memory_order_acq_rel);
                }
                (void) w;
            });
        }
    }
    // run fn(0..n_jobs-1) on the workers and the caller; returns when every job is done
    void run(int64_t jobs, std::function<void(int64_t)> f) {
        if (th.empty()) { start(); }
        fn = std::move(f); n_jobs = jobs;
        next.store(0, std::memory_order_release);
        done.store(0, std::memory_order_release);
        gen.fetch_add(1, std::memory_order_acq_rel);
        work();
        while (done.load(std::memory_order_acquire) < n) { FLE_PAUSE(); }
    }
};
static fle_pool & fle_pool_get() { static fle_pool * p = new fle_pool(); return *p; }

struct pair_t { int32_t t, slot, e; };

struct fused_ud {
    int   il    = -1;
    int   dummy = 0;
    float limit = 0.0f;
    // shared across the threads of one call (written by thread 0 before the first barrier)
    std::vector<pair_t> pairs;
    std::vector<float>  act;        // [n_pairs][n_ff]
    std::atomic<int>    arrive{0};
    std::atomic<int>    gen{0};
};

static void fused_barrier(fused_ud * u, int nth) {
    const int g = u->gen.load(std::memory_order_acquire);
    if (u->arrive.fetch_add(1, std::memory_order_acq_rel) + 1 == nth) {
        u->arrive.store(0, std::memory_order_relaxed);
        u->gen.fetch_add(1, std::memory_order_acq_rel);
    } else {
        while (u->gen.load(std::memory_order_acquire) == g) { FLE_PAUSE(); }
    }
}

static const char * expert_data(const ggml_tensor * w, int32_t e) {
    void * ud = nullptr;
    const ggml_moe_data_cb_t cb = ggml_get_moe_data_callback(&ud);
    if (cb) {
        const void * p = cb(w, e, ud);
        if (p) { return (const char *) p; }
    }
    return (const char *) w->data + (size_t) e * w->nb[2];
}

static void fused_fn(ggml_tensor * dst, int ith, int nth, void * userdata) {
    fused_ud * u = (fused_ud *) userdata;
    const ggml_tensor * cur   = dst->src[0];
    const ggml_tensor * ids   = dst->src[1];
    const ggml_tensor * gate  = dst->src[2];
    const ggml_tensor * up    = dst->src[3];
    const ggml_tensor * down  = dst->src[4];
    const ggml_tensor * table = dst->src[5];
    const ggml_tensor * pred  = dst->src[6];

    const int64_t n_embd = cur->ne[0];
    const int64_t n_tok  = ids->ne[1];
    const int64_t n_used = ids->ne[0];
    const int64_t n_ff   = gate->ne[1];
    const int32_t * tbl  = (const int32_t *) table->data;

    // phase 0 (thread 0): routing observation (expert cache, expert store, prediction) and the list of CPU experts
    if (ith == 0) {
        void * oud = nullptr;
        ggml_moe_obs_cb_t obs = ggml_get_moe_obs_callback(&oud);
        if (obs) { obs(gate->name, ids, oud); }
        void * pud = nullptr;
        ggml_moe_pred_cb_t pcb = ggml_get_moe_pred_callback(&pud);
        if (pcb && pred) { pcb(gate->name, pred, pud); }
        u->pairs.clear();
        for (int64_t t = 0; t < n_tok; ++t) {
            for (int64_t i = 0; i < n_used; ++i) {
                const int32_t e = *(const int32_t *) ((const char *) ids->data + t*ids->nb[1] + i*ids->nb[0]);
                if (e < 0 || e >= gate->ne[2]) { GGML_ABORT("fused moe: expert id %d outside [0, %lld)", e, (long long) gate->ne[2]); }
                if (tbl[e] != u->dummy) { continue; }   // the GPU cache computes it
                u->pairs.push_back({ (int32_t) t, (int32_t) i, e });
            }
        }
        u->act.resize(u->pairs.size() * (size_t) n_ff);
    }
    // FLE_MOE_POOL: ggml runs this op on one thread (n_tasks 1); the work goes to the pinned pool, jobs of 64 rows
    static const bool use_pool = [] { const char * e = std::getenv("FLE_MOE_POOL"); return e && atoi(e) != 0; }();
    if (use_pool) {
        static bool said = false;
        if (!said) { said = true; std::fprintf(stderr, "[fle-moe-pool] fused CPU MoE op on the pool (ggml nth=%d)\n", nth); }
        if (ith != 0) { return; }   // the pool does the work; with nth > 1 the other ggml threads only wait at the next barrier
        memset(dst->data, 0, ggml_nbytes(dst));
        const int64_t np = (int64_t) u->pairs.size();
        if (np == 0) { return; }
        // experts whose bytes are there first (asynchronous expert store), the ones still being read last: the tickets hand out
        // the resident work while the reads land, instead of a worker waiting on a missing expert early in the layer
        {
            void * rud = nullptr;
            ggml_moe_ready_cb_t rcb = ggml_get_moe_ready_callback(&rud);
            if (rcb && np > 1) {
                std::stable_partition(u->pairs.begin(), u->pairs.end(), [&](const pair_t & q) { return rcb(gate, q.e, rud); });
            }
        }
        const auto * tg = fle_traits(gate->type);
        const auto * tu = fle_traits(up->type);
        const auto * td = fle_traits(down->type);
        GGML_ASSERT(tg && tu && td && tg->vec_dot_type == tu->vec_dot_type);
        const auto * tq  = fle_traits(tg->vec_dot_type);
        const size_t xrs = ggml_row_size(tg->vec_dot_type, n_embd);
        static std::vector<uint8_t> xq, aq;
        xq.resize((size_t) n_tok * xrs);
        for (int64_t t = 0; t < n_tok; ++t) {
            tq->from_float((const float *) ((const char *) cur->data + t*cur->nb[2]), xq.data() + t*xrs, n_embd);
        }
        constexpr int64_t RB = 64;
        const int64_t nb_ff = (n_ff + RB - 1) / RB, nb_e = (n_embd + RB - 1) / RB;
        const float lim = u->limit;
        fle_pool & pool = fle_pool_get();
        // jobs per (expert, 64-row block): several tokens routed to the same expert (DSpark verification, prefill-sized batches)
        // read its rows once and apply them to every such token; the per-row arithmetic is the same as one pair at a time
        std::vector<int32_t> grp_beg, grp_pair;   // groups of pair indices with the same expert (pairs are kept in their order)
        {
            std::vector<int32_t> order((size_t) np);
            for (int64_t q = 0; q < np; ++q) { order[(size_t) q] = (int32_t) q; }
            std::stable_sort(order.begin(), order.end(), [&](int32_t a, int32_t b) { return u->pairs[(size_t) a].e < u->pairs[(size_t) b].e; });
            // keep the resident-first order of the groups: a group starts where its first pair stood
            std::vector<std::pair<int32_t, int32_t>> starts;   // (first position in pairs, expert)
            for (size_t k = 0; k < order.size(); ++k) {
                if (k == 0 || u->pairs[(size_t) order[k]].e != u->pairs[(size_t) order[k-1]].e) { starts.push_back({ order[k], u->pairs[(size_t) order[k]].e }); }
            }
            std::sort(starts.begin(), starts.end());
            for (const auto & st : starts) {
                grp_beg.push_back((int32_t) grp_pair.size());
                for (int64_t q = 0; q < np; ++q) { if (u->pairs[(size_t) q].e == st.second) { grp_pair.push_back((int32_t) q); } }
            }
            grp_beg.push_back((int32_t) grp_pair.size());
        }
        const int64_t ng = (int64_t) grp_beg.size() - 1;
        pool.run(ng * nb_ff, [&](int64_t j) {
            const int64_t g0 = j / nb_ff;
            const pair_t & p0 = u->pairs[(size_t) grp_pair[(size_t) grp_beg[(size_t) g0]]];
            const int64_t r0 = (j % nb_ff) * RB, r1 = std::min(r0 + RB, n_ff);
            const char * wg = expert_data(gate, p0.e), * wu = expert_data(up, p0.e);
            for (int64_t row = r0; row < r1; ++row) {
                for (int32_t k = grp_beg[(size_t) g0]; k < grp_beg[(size_t) g0 + 1]; ++k) {
                    const int32_t q = grp_pair[(size_t) k];
                    const pair_t & pp = u->pairs[(size_t) q];
                    float g = 0.0f, v = 0.0f;
                    tg->vec_dot((int) n_embd, &g, 0, wg + row*gate->nb[1], 0, xq.data() + pp.t*xrs, 0, 1);
                    tu->vec_dot((int) n_embd, &v, 0, wu + row*up->nb[1],   0, xq.data() + pp.t*xrs, 0, 1);
                    const float gc = std::min(g, lim);
                    const float uc = std::clamp(v, -lim, lim);
                    u->act[(size_t) q * n_ff + row] = gc / (1.f + expf(-gc)) * uc;
                }
            }
        });
        const auto * tqd = fle_traits(td->vec_dot_type);
        const size_t ars = ggml_row_size(td->vec_dot_type, n_ff);
        aq.resize((size_t) np * ars);
        for (int64_t q = 0; q < np; ++q) { tqd->from_float(u->act.data() + q*n_ff, aq.data() + q*ars, n_ff); }
        pool.run(ng * nb_e, [&](int64_t j) {
            const int64_t g0 = j / nb_e;
            const pair_t & p0 = u->pairs[(size_t) grp_pair[(size_t) grp_beg[(size_t) g0]]];
            const int64_t r0 = (j % nb_e) * RB, r1 = std::min(r0 + RB, n_embd);
            const char * wd = expert_data(down, p0.e);
            for (int64_t row = r0; row < r1; ++row) {
                for (int32_t k = grp_beg[(size_t) g0]; k < grp_beg[(size_t) g0 + 1]; ++k) {
                    const int32_t q = grp_pair[(size_t) k];
                    const pair_t & pp = u->pairs[(size_t) q];
                    float * o = (float *) ((char *) dst->data + pp.slot*dst->nb[1] + pp.t*dst->nb[2]);
                    td->vec_dot((int) n_ff, o + row, 0, wd + row*down->nb[1], 0, aq.data() + (size_t) q*ars, 0, 1);
                }
            }
        });
        if (n_tok == 1) { llama_moe_cpu_predict_after(u->il, (const float *) cur->data, n_embd); }   // FLE_MOE_CPU_PRED
        return;
    }

    // zero this thread's share of the output (the GPU-cached slots stay 0, the CPU ones are written in phase 2)
    {
        const int64_t n_rows = n_used * n_tok;
        for (int64_t r = ith; r < n_rows; r += nth) {
            memset((char *) dst->data + (r % n_used)*dst->nb[1] + (r / n_used)*dst->nb[2], 0, n_embd * sizeof(float));
        }
    }
    fused_barrier(u, nth);

    const int64_t np = (int64_t) u->pairs.size();
    if (np == 0) {
        return;
    }

    // the activation quantized once per token, as mul_mat_id does (from_float of the weights' vec_dot_type)
    const auto * tg = fle_traits(gate->type);
    const auto * tu = fle_traits(up->type);
    const auto * td = fle_traits(down->type);
    GGML_ASSERT(tg && tu && td && tg->vec_dot_type == tu->vec_dot_type);
    const auto * tq  = fle_traits(tg->vec_dot_type);
    const size_t xrs = ggml_row_size(tg->vec_dot_type, n_embd);
    thread_local std::vector<uint8_t> xq;
    xq.resize((size_t) n_tok * xrs);
    for (int64_t t = 0; t < n_tok; ++t) {
        tq->from_float((const float *) ((const char *) cur->data + t*cur->nb[2]), xq.data() + t*xrs, n_embd);
    }

    // phase 1: gate and up rows of every CPU expert, clamped SwiGLU on the spot (ggml_swiglu_clamp's arithmetic)
    {
        const int64_t n_rows = np * n_ff;
        const int64_t r0 = n_rows * ith / nth, r1 = n_rows * (ith + 1) / nth;
        const float lim = u->limit;
        for (int64_t r = r0; r < r1; ++r) {
            const pair_t & p = u->pairs[(size_t) (r / n_ff)];
            const int64_t row = r % n_ff;
            float g = 0.0f, v = 0.0f;
            tg->vec_dot((int) n_embd, &g, 0, expert_data(gate, p.e) + row*gate->nb[1], 0, xq.data() + p.t*xrs, 0, 1);
            tu->vec_dot((int) n_embd, &v, 0, expert_data(up,   p.e) + row*up->nb[1],   0, xq.data() + p.t*xrs, 0, 1);
            const float gc = std::min(g, lim);
            const float uc = std::clamp(v, -lim, lim);
            u->act[(size_t) r] = gc / (1.f + expf(-gc)) * uc;
        }
    }
    fused_barrier(u, nth);

    // phase 2: down rows; every thread quantizes the activations of all CPU experts for itself (small)
    {
        const auto * tqd = fle_traits(td->vec_dot_type);
        const size_t ars = ggml_row_size(td->vec_dot_type, n_ff);
        thread_local std::vector<uint8_t> aq;
        aq.resize((size_t) np * ars);
        for (int64_t p = 0; p < np; ++p) {
            tqd->from_float(u->act.data() + p*n_ff, aq.data() + p*ars, n_ff);
        }
        const int64_t n_rows = np * n_embd;
        const int64_t r0 = n_rows * ith / nth, r1 = n_rows * (ith + 1) / nth;
        for (int64_t r = r0; r < r1; ++r) {
            const pair_t & p = u->pairs[(size_t) (r / n_embd)];
            const int64_t row = r % n_embd;
            float * o = (float *) ((char *) dst->data + p.slot*dst->nb[1] + p.t*dst->nb[2]) + row;
            td->vec_dot((int) n_ff, o, 0, expert_data(down, p.e) + row*down->nb[1], 0, aq.data() + (r / n_embd)*ars, 0, 1);
        }
    }
}

} // namespace

ggml_tensor * llama_moe_fused_cpu(ggml_context * ctx, ggml_tensor * cur, ggml_tensor * ids,
        ggml_tensor * gate_exps, ggml_tensor * up_exps, ggml_tensor * down_exps,
        ggml_tensor * table, int32_t dummy, ggml_tensor * pred, float clamp_limit, int il) {
    static std::mutex mtx;
    static std::vector<fused_ud *> uds;   // one per layer, never freed (graphs are rebuilt per batch shape)
    fused_ud * u = nullptr;
    {
        std::lock_guard<std::mutex> lk(mtx);
        if ((size_t) (il + 1) > uds.size()) { uds.resize((size_t) il + 1, nullptr); }
        if (!uds[(size_t) il]) { uds[(size_t) il] = new fused_ud(); }
        u = uds[(size_t) il];
    }
    u->il = il; u->dummy = dummy; u->limit = clamp_limit;
    ggml_tensor * args[7] = { cur, ids, gate_exps, up_exps, down_exps, table, pred };
    static const bool use_pool = [] { const char * e = std::getenv("FLE_MOE_POOL"); return e && atoi(e) != 0; }();
    return ggml_custom_4d(ctx, GGML_TYPE_F32, cur->ne[0], ids->ne[0], ids->ne[1], 1, args, pred ? 7 : 6, fused_fn,
            use_pool ? 1 : GGML_N_TASKS_MAX, u);
}
