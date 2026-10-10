// pager_buffer.cu — expert pager, phase 1 (P3+P4): custom backend buffer and staging kernel for *_exps tensors.
// Adapted from FreeToken (Apache-2.0), https://github.com/FlashML-org/FreeToken
// ============================================================================================
// P3 (buffer): the stacked expert tensor (~80 GB) lives entirely in pinned HOST RAM
//   (host_backing, tier 1); a COMPACT VRAM POOL (vram_pool, tier 0) is the staging target.
//   get_base = host_backing -> tensor->data points to the host (pinned = UVA-readable by kernels;
//   the zero-copy fallback is therefore ALWAYS correct, just PCIe-slow).
//
// P4 (staging, pager_stage_and_remap): for every mul_mat_id call the expert slices selected by
//   `ids` are staged host->pool (LRU eviction, slots of the current call pinned via stamp) and
//   the ids are remapped to slot indices -> mul_mat_id runs UNCHANGED against the pool
//   (compact-pool invariant). Every abort path returns nullptr = zero-copy fallback.
//
// Shared-buffer reality: llama.cpp puts ALL tensors of a buft into ONE buffer (offsets).
//   Slot ownership is therefore keyed by (tensor data ptr, expert), not by expert alone.
//
// v1 = correctness first: synchronous ids D2H (async P2b later), residency LRU kept local
//   (predictive prefetch attaches to exactly these slot maps later).
// ============================================================================================
#ifdef _OPENMP
#include <omp.h>   // omp_get_max_threads (only used inside the OpenMP pragmas below)
#endif
#include <unordered_set>
#include "pager_wire.h"

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <string>
#include <algorithm>
#include <chrono>
#include <cstdio>
#ifndef _WIN32
#include <sys/mman.h>  // FLE_PAGER_MMAP_FILE cold-tier first-cut
#include <fcntl.h>
#include <unistd.h>
#else
#include <malloc.h>    // _aligned_malloc / _aligned_free (the FLE_PAGER_MMAP_FILE tier does not exist on Windows)
#endif
#include <cmath>
#include <mutex>
#include <unordered_map>
#include <map>
#include <vector>

#include "ggml.h"
#include "ggml-backend-impl.h"
#include "ggml-cuda.h"
#include "ggml-cpu.h"   // S3: ggml_get_type_traits_cpu (vec_dot/from_float) for CPU miss compute

#if defined(_WIN32)
// Windows is built with GGML_BACKEND_DL only: libllama / the tools reach the functions through
// ggml_backend_reg_get_proc_address (ggml_backend_cuda_reg_get_proc_address), nothing is exported by name.
#define FLE_EXPORT
#else
#define FLE_EXPORT __attribute__((visibility("default")))
#endif

namespace fle {
namespace pager {

// Environment lookup with a deprecated alias: the new name wins, the old (German) name still works.
static const char * getenv_alias(const char * name, const char * deprecated_name) {
    const char * e = std::getenv(name);
    return e ? e : std::getenv(deprecated_name);
}

// ggml_get_type_traits_cpu lives in the CPU backend. With GGML_BACKEND_DL that backend is a separate module that this
// one cannot link to: the function is looked up in the loaded CPU backend instead (ggml_backend_reg_get_proc_address).
// nullptr = no CPU backend loaded; the callers handle that (the CPU miss compute is then skipped for the type).
static const struct ggml_type_traits_cpu * pager_type_traits_cpu(enum ggml_type type) {
#ifdef GGML_BACKEND_DL
    typedef const struct ggml_type_traits_cpu * (*traits_fn_t)(enum ggml_type);
    static std::atomic<traits_fn_t> cached{nullptr};
    traits_fn_t fn = cached.load(std::memory_order_acquire);
    if (!fn) {
        ggml_backend_reg_t reg = ggml_backend_reg_by_name("CPU");
        fn = reg ? (traits_fn_t) ggml_backend_reg_get_proc_address(reg, "ggml_get_type_traits_cpu") : nullptr;
        if (fn) {
            cached.store(fn, std::memory_order_release);
        }
    }
    return fn ? fn(type) : nullptr;
#else
    return ggml_get_type_traits_cpu(type);
#endif
}

// One stage/remap/finalize at a time (slot/graph parallelism; shared by stage_and_remap and cpu_miss_finalize).
static std::mutex g_pager_mtx;

// ── S2 telemetry: bytes and time (default OFF via FLE_PAGER_BW) ──────────────────────────────
// The existing S1 telemetry counts events (hit/miss/reuse-dist/freq-skew) and answers "how often?",
// not "how fast?". Choosing between the CPU branch and the PCIe branch depends on bandwidth:
// FreeToken's rule cuts over at CPU_BW > 2 x PCIe_BW (arXiv 2608.16157, `benchbw.py:recommend`).
// These counters supply the two missing quantities.
//
// File scope because BOTH paths write: the stage path (PCIe) and pager_cpu_miss_finalize (CPU).
// Both run under g_pager_mtx, so plain fields suffice without atomics.
static const bool g_bw_on = [] { const char* e = std::getenv("FLE_PAGER_BW"); return e && *e && *e != '0'; }();
struct BwStats {
    unsigned long long h2d_bytes = 0, d2d_bytes = 0;    // kept separate: D2D is intra-VRAM, NOT the bus
    double             pcie_ms   = 0.0;
    unsigned long long pcie_samples = 0, pcie_skipped = 0;   // control number: a sample, not a full survey
    unsigned long long cpu_bytes = 0;
    double             cpu_ms    = 0.0;
    unsigned long long cpu_calls = 0;
    cudaEvent_t ev_a = nullptr, ev_b = nullptr;         // WITH timing — rt.stage_ev is DisableTiming
    bool        pending = false;
    unsigned long long pending_bytes = 0;
};
// Fix (2026-08-26): this used to be a single `static BwStats g_bw;` shared by ALL devices.
// Two problems in one:
//
//  1. `ev_a`/`ev_b` are CUDA events and belong to the device that was current when they were
//     created. `cudaEventRecord(bw_cur().ev_a, sstream)` with a stream of a DIFFERENT device
//     fails with `invalid resource handle` (the crash seen with two pagers in
//     `ggml_backend_cuda_graph_compute`).
//
//  2. Even without the crash the numbers would be wrong: two pagers would mix their
//     bandwidths into one counter, and `26.4 GB/s` would be an average over two different links.
//
// Same class of bug as the other per-device fixes in this file: a CUDA object without a
// device binding, here one level higher as a global variable.
static BwStats g_bw_dev[16];

// Access goes through the CURRENT device, not through a passed-in context:
// the place that measures is the place that computes.
// `cudaGetDevice` reads thread-locally and costs nothing measurable.
static inline BwStats & bw_cur() {
    int d = 0;
    cudaGetDevice(&d);
    if (d < 0 || d >= 16) d = 0;
    return g_bw_dev[d];
}

// q*-split (default OFF): the binary choice wastes the other branch although both
// run concurrently.
static const bool g_split_on = [] { const char* e = std::getenv("FLE_PAGER_SPLIT"); return e && *e && *e != '0'; }();
// Below this number of samples per side NO split is done — better the old threshold
// than a quota derived from two uncertain numbers.
static const unsigned long long g_split_min_n = [] {
    const char* e = std::getenv("FLE_PAGER_SPLIT_MIN_N"); return (unsigned long long)(e && *e ? std::atoll(e) : 64);
}();


// ── Runtime: slot management + scratch (lazy, lives on the buffer) ──────────────────────────
struct SlotInfo {
    const void * owner  = nullptr;  // tensor data ptr of the owner (shared-buffer-safe)
    int          expert = -1;
    uint64_t     last   = 0;        // LRU clock
    uint64_t     stamp  = 0;        // pin marker of the current call
};

// Per-tensor pool region: slot IDs are LOCAL [0, n_slots) and therefore guaranteed < n_expert
// (downstream indexes expert_bounds with the remapped IDs — global slot IDs > n_expert
// were the illegal-access crash of v1). The region pointer is returned as src0->data.
struct TensorPool {
    size_t   base_off   = 0;      // offset of the region in vram_pool
    size_t   slot_bytes = 0;      // slice size of THIS tensor (gate/up IQ2XXS vs down Q2K!)
    int      n_slots    = 0;
    std::vector<SlotInfo> slots;   // owner=unused (region belongs to the tensor)
    std::vector<int32_t>  e2s;     // expert -> local slot
    std::vector<uint64_t> evicted_at; // per expert: clock of the LAST eviction (0=never) -> miss nature: cold vs reuse
    std::vector<uint32_t> freq;    // per expert: routing frequency (reuse-aware eviction, FLE_PAGER_EVICT_FREQ)
    std::vector<int32_t>  last_uniq;  // CLGP prediction: unique experts of this tensor in the LAST decode call
    const void*           self = nullptr; // src0->data key (for prefetch: host weights of this tensor)
    // Working-set measurement (prediction- vs capacity-bound): last-use call per expert of this tensor.
    std::vector<uint64_t> seen_call;  // per expert: tp.calls at last use (0=never)
    uint64_t              calls = 0;  // decode calls of THIS tensor
};

struct PagerRuntime {
    // ids clone scratch — LAYOUT-FAITHFUL (full ggml_nbytes + original strides): mmq reads ids
    // with ids->nb strides; compact packing caused the illegal access at mmq.cu:206.
    uint8_t* h_buf = nullptr;   // pinned host clone of ids
    uint8_t* d_buf = nullptr;   // device target (remapped ids, same strides)
    size_t   cap   = 0;

    // regions
    std::unordered_map<const void*, TensorPool> pools;  // one region per tensor data ptr
    size_t   region_bytes  = 0;    // (vram_bytes - scratch - swap) / FLE_PAGER_REGIONS (default 160)
    size_t   scratch_bytes = 0;    // prefill scratch at pool start (FLE_PAGER_SCRATCH_MB, default 2048)
    size_t   next_off      = 0;    // next free region (starts AFTER the scratch)
    uint64_t clock = 0, stamp = 0;
    // Device this runtime lives on. Since the multi-device pool the context (`PagerBufferContext`)
    // carries it; PagerRuntime did not — which is why `ensure_copy()` created its CUDA objects
    // on whatever device happened to be current. See there.
    int device = -1;
    std::vector<int32_t> uniq_map; // transient expert->scratch slot per prefill call

    // Swap arena (pool 3, stream-transient) at the pool END — a reserved free exchange cache into
    // which the L->L+1 prefetch asynchronously loads the INCOMING (predicted) experts host->swap
    // WITHOUT polluting the resident regions (no speculative prefetch evicts hot residents).
    // Decode then promotes swap->region via a fast D2D copy instead of a slow H2D (hides the fetch
    // behind compute = the L->L+1 prefetch lever). FLE_PAGER_SWAP_MB (0 = off, old in-place prefetch path).
    size_t   swap_off      = 0;    // byte offset of the swap arena in vram_pool (after the regions)
    size_t   swap_bytes    = 0;    // size of the swap arena (0 = off)
    size_t   swap_next     = 0;    // bump allocator in the arena (ring, reset per prefetch round)
    std::unordered_map<const void*, std::unordered_map<int, size_t>> swap_map; // tensor->expert->swap byte offset

    // S3 (CPU miss compute): list of miss positions filled per stage call; finalize consumes it.
    struct CpuMiss { int iex; int token; int expert; };
    std::vector<CpuMiss> cpu_misses;
    std::vector<uint8_t> cpu_xq;    // quantized activation (vec_dot_type) per token, reused
    std::vector<float>   cpu_x;     // D2H src1 column (f32), reused
    std::vector<float>   cpu_out;   // CPU result column (f32), reused
    std::vector<uint8_t> cpu_wasmiss; // per expert: was a miss during counting (before staging) -> CPU compute candidate
    // S3 overlap: activations of the cold misses are D2H-copied and quantized already in stage_and_remap
    // (before mmvq) so that finalize computes without sync (CPU vec_dot overlaps the GPU mmvq kernel).
    std::vector<uint8_t> cpu_xq_pre;  // flat: cpu_misses.size() * pre_stride (quantized activation per miss)
    size_t   cpu_pre_stride = 0;      // xq_bytes per miss (= ggml_row_size(vdt, K))
    int64_t  cpu_pre_K      = 0;      // contraction dim of this call
    int      cpu_pre_vdt    = -1;     // vec_dot_type (for the finalize consistency check)
    float*   cpu_out_pin    = nullptr;// PINNED output (distinct slice per miss -> true async H2D, no reuse race)
    size_t   cpu_out_cap    = 0;      // capacity in floats
    float*   cpu_x_pin      = nullptr;// PINNED activation batch (nm*K) -> all D2H async + ONE sync instead of per-miss
    size_t   cpu_x_cap      = 0;
    bool ensure_out(size_t n) {
        if (n <= cpu_out_cap) return true;
        if (cpu_out_pin) { cudaFreeHost(cpu_out_pin); cpu_out_pin = nullptr; cpu_out_cap = 0; }
        if (cudaMallocHost((void**)&cpu_out_pin, n * sizeof(float)) != cudaSuccess) return false;
        cpu_out_cap = n; return true;
    }
    bool ensure_xin(size_t n) {
        if (n <= cpu_x_cap) return true;
        if (cpu_x_pin) { cudaFreeHost(cpu_x_pin); cpu_x_pin = nullptr; cpu_x_cap = 0; }
        if (cudaMallocHost((void**)&cpu_x_pin, n * sizeof(float)) != cudaSuccess) return false;
        cpu_x_cap = n; return true;
    }

    // S2 substrate: dedicated copy stream + event for miss staging (env FLE_PAGER_ASYNC_STAGE).
    // Miss H2D runs on copy_stream (own copy engine), the compute stream waits via event before
    // mul_mat_id -> ordering enforced (no corruption), transfers can overlap with compute.
    cudaStream_t copy_stream = nullptr;
    cudaEvent_t  stage_ev    = nullptr;  // staging done -> compute waits (RAW before mul_mat_id)

    // FLE_PAGER_BOUNCE — pinned bounce buffer for miss staging (H2D).
    //
    // Measured 2026-08-29 (scattered 2.97 MB slices, GPU0):
    //     MAP_SHARED unpinned    153.2 us   19.8 GB/s   <- what the pager does today
    //     fully pinned           106.2 us   28.6 GB/s   factor 1.44
    //     bounce buffer          107.1 us   28.4 GB/s   factor 1.43, pins 2.9 MB instead of 44 GB
    //
    // A ring, so the copy does not wait on itself: before buffer i is rewritten, its event
    // must have completed. With BOUNCE_N=8 and a slice of ~0.9 MB that is 7 MB pinned — nothing
    // compared to 44 GB.
    static const int BOUNCE_N = 8;
    uint8_t*    bounce[BOUNCE_N]    = {};
    cudaEvent_t bounce_ev[BOUNCE_N] = {};
    size_t      bounce_bytes = 0;
    int         bounce_i     = 0;

    bool ensure_bounce(size_t need) {
        if (bounce_bytes >= need && bounce[0]) return true;
        // Size never shrinks: a larger slice later must not silently copy
        // into a buffer that is too small.
        free_bounce();
        int prev_dev = -1;
        if (device >= 0) {
            if (cudaGetDevice(&prev_dev) != cudaSuccess) return false;
            if (prev_dev != device && cudaSetDevice(device) != cudaSuccess) return false;
        }
        bool ok = true;
        for (int i = 0; i < BOUNCE_N && ok; ++i) {
            ok = cudaMallocHost((void**) &bounce[i], need) == cudaSuccess
              && cudaEventCreateWithFlags(&bounce_ev[i], cudaEventDisableTiming) == cudaSuccess;
        }
        if (ok) { bounce_bytes = need; bounce_i = 0; }
        else    { free_bounce(); }
        // Fail LOUD: a silent failure would be especially costly here because the
        // measurement would then see "no effect" and nobody would know that nothing
        // happened. This is exactly how FLE_PAGER_HOST_REGISTER slipped through.
        std::fprintf(stderr, ok
            ? "[fle-pager] BOUNCE: %d buffers of %zu KB pinned (%.1f MB total)\n"
            : "[fle-pager] BOUNCE FAILED: %d x %zu KB could not be allocated (%.1f MB)\n",
            BOUNCE_N, need >> 10, (double) BOUNCE_N * need / 1048576.0);
        if (prev_dev >= 0 && prev_dev != device) cudaSetDevice(prev_dev);
        return ok;
    }

    void free_bounce() {
        for (int i = 0; i < BOUNCE_N; ++i) {
            if (bounce_ev[i]) { cudaEventDestroy(bounce_ev[i]); bounce_ev[i] = nullptr; }
            if (bounce[i])    { cudaFreeHost(bounce[i]);        bounce[i]    = nullptr; }
        }
        bounce_bytes = 0;
    }
    cudaEvent_t  compute_ev  = nullptr;  // previous compute done -> copy waits (WAR: region reuse across tokens)
    // CLGP predictive prefetch: tensor chain (current->next) + prefetch event.
    std::unordered_map<const void*, const void*> next_ptr;  // tensor order (learned in the 1st token)
    const void*  prev_ptr        = nullptr;
    cudaEvent_t  prefetch_ev     = nullptr;
    bool         prefetch_pending = false;  // a prefetch is in flight -> the next tensor's mmvq waits on it
    // Fix (2026-08-25): stream and events belong to the device that is current at CREATION
    // time — not the one they are later used on. There used to be no `cudaSetDevice` here,
    // while the two sibling places (the destructor and the buffer allocation) have it and
    // state the reason: the destructor MUST run in the right device context (multi-device
    // pool), and cudaMallocHost + pool in the right device context. This third place was
    // missed because `PagerRuntime` did not know the device at all.
    //
    // Consequence: a stream from the context of device 1, used for work on device 0, yields
    // `invalid resource handle` at `MUL_MAT_ID` — the picture of the crash with two pagers.
    // As long as only GPU0 pages it goes unnoticed: the accidentally current device is then
    // always the right one.
    bool ensure_copy() {
        if (copy_stream) return true;

        int prev_dev = -1;
        if (device >= 0) {
            if (cudaGetDevice(&prev_dev) != cudaSuccess) return false;
            if (prev_dev != device && cudaSetDevice(device) != cudaSuccess) return false;
        }

        const bool ok =
               cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking) == cudaSuccess
            && cudaEventCreateWithFlags(&stage_ev,    cudaEventDisableTiming) == cudaSuccess
            && cudaEventCreateWithFlags(&compute_ev,  cudaEventDisableTiming) == cudaSuccess
            && cudaEventCreateWithFlags(&prefetch_ev, cudaEventDisableTiming) == cudaSuccess;

        // Restore the calling device: `ensure_copy` is called in the middle of the graph,
        // not at setup.
        if (device >= 0 && prev_dev >= 0 && prev_dev != device) cudaSetDevice(prev_dev);

        if (!ok) {
            // Second bug at the same spot, found while reading: the old version returned
            // `false` on a failed event but left `copy_stream` SET. The next call hit
            // `if (copy_stream) return true` above and reported success — with a null event
            // in hand. A failure that turns into success on the second attempt is the
            // nastiest kind.
            if (prefetch_ev) { cudaEventDestroy(prefetch_ev); prefetch_ev = nullptr; }
            if (compute_ev)  { cudaEventDestroy(compute_ev);  compute_ev  = nullptr; }
            if (stage_ev)    { cudaEventDestroy(stage_ev);    stage_ev    = nullptr; }
            if (copy_stream) { cudaStreamDestroy(copy_stream); copy_stream = nullptr; }
        }
        return ok;
    }

    ~PagerRuntime() {
        if (h_buf) cudaFreeHost(h_buf);
        if (d_buf) cudaFree(d_buf);
        if (stage_ev)    cudaEventDestroy(stage_ev);
        if (compute_ev)  cudaEventDestroy(compute_ev);
        if (prefetch_ev) cudaEventDestroy(prefetch_ev);
        if (copy_stream) cudaStreamDestroy(copy_stream);
        if (cpu_out_pin) cudaFreeHost(cpu_out_pin);
        if (cpu_x_pin)   cudaFreeHost(cpu_x_pin);
        if (slot_map_dev) cudaFree(slot_map_dev);
    }
    bool ensure(size_t nb) {
        if (nb <= cap) return true;
        if (h_buf) { cudaFreeHost(h_buf); h_buf = nullptr; }
        if (d_buf) { cudaFree(d_buf);     d_buf = nullptr; }
        cap = 0;
        if (cudaMallocHost((void**)&h_buf, nb) != cudaSuccess) return false;
        if (cudaMalloc    ((void**)&d_buf, nb) != cudaSuccess) return false;
        cap = nb;
        return true;
    }

    // CUDA-graph fix (step i): persistent device slot_map (expert->slot) — mirror of tp.e2s.
    // Reusable (calls serialized via g_pager_mtx). The device remap kernel reads it -> graph-legal
    // gather instead of host D2H+sync+value-replace. Gated by FLE_PAGER_DEVICE_REMAP.
    int32_t* slot_map_dev = nullptr;
    size_t   slot_map_cap = 0;
    bool ensure_slotmap(size_t n_expert) {
        if (n_expert <= slot_map_cap) return true;
        if (slot_map_dev) { cudaFree(slot_map_dev); slot_map_dev = nullptr; slot_map_cap = 0; }
        if (cudaMalloc((void**)&slot_map_dev, n_expert * sizeof(int32_t)) != cudaSuccess) return false;
        slot_map_cap = n_expert; return true;
    }
};

// Context of a concrete pager buffer.
struct PagerBufferContext {
    int           device       = 0;
    bool          host_pinned  = true;     // false = FLE_PAGER_NO_PIN (S0 bad-page A/B)
    uint8_t*      host_backing = nullptr;  // full stack, pinned RAM (tier 1)
    size_t        host_registered_bytes = 0;  // >0: pinned via cudaHostRegister, must be released again
    size_t        host_bytes   = 0;
    void*         vram_pool    = nullptr;  // compact pool (tier 0)
    bool          host_mmap    = false;    // NVMe mmap instead of RAM (FLE_PAGER_MMAP_FILE)
    int           mmap_fd      = -1;
    size_t        vram_bytes   = 0;
    PagerRuntime* rt           = nullptr;  // lazy
};

// Context of the buffer TYPE (once per device).
struct PagerBufferTypeContext {
    int         device      = 0;
    size_t      vram_budget = 0;
    std::string name;
};

// ── buffer_i ────────────────────────────────────────────────────────────────────────────────
static void pager_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    auto* ctx = (PagerBufferContext*) buffer->context;
    cudaSetDevice(ctx->device);   // fix (2026-07-18): the rt destructor (stream/event/cudaFree) MUST run in the right device context (multi-device pool)
    delete ctx->rt;
    if (ctx->vram_pool)    { cudaFree(ctx->vram_pool); }
    if (ctx->host_backing) {
        if (ctx->host_registered_bytes) { cudaHostUnregister(ctx->host_backing); ctx->host_registered_bytes = 0; }
#ifndef _WIN32
        if (ctx->host_mmap)        { munmap(ctx->host_backing, ctx->host_bytes); if (ctx->mmap_fd >= 0) close(ctx->mmap_fd); }
        else
#endif
        if (ctx->host_pinned) cudaFreeHost(ctx->host_backing);
#ifdef _WIN32
        else                       _aligned_free(ctx->host_backing);
#else
        else                       free(ctx->host_backing);
#endif
    }
    delete ctx;
}

static void* pager_buffer_get_base(ggml_backend_buffer_t buffer) {
    auto* ctx = (PagerBufferContext*) buffer->context;
    // host backing as base: tensor->data = host ptr (pinned/UVA). The staging path replaces
    // src0->data per call with the pool; without staging, zero-copy is correct.
    return ctx->host_backing;
}

static enum ggml_status pager_buffer_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    (void) buffer; (void) tensor;
    return GGML_STATUS_SUCCESS;
}

static void pager_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor,
                                    const void * data, size_t offset, size_t size) {
    // Model load: weights go into the host backing. tensor->data already points into it (get_base+offset).
    (void) buffer;
    std::memcpy((uint8_t*) tensor->data + offset, data, size);
}

static void pager_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor,
                                    void * data, size_t offset, size_t size) {
    (void) buffer;
    std::memcpy(data, (const uint8_t*) tensor->data + offset, size);
}

static void pager_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto* ctx = (PagerBufferContext*) buffer->context;
    if (ctx->host_backing) std::memset(ctx->host_backing, value, ctx->host_bytes);
}

static const ggml_backend_buffer_i pager_buffer_interface = {
    /* .free_buffer   = */ pager_buffer_free_buffer,
    /* .get_base      = */ pager_buffer_get_base,
    /* .init_tensor   = */ pager_buffer_init_tensor,
    /* .memset_tensor = */ nullptr,
    /* .set_tensor    = */ pager_buffer_set_tensor,
    /* .get_tensor    = */ pager_buffer_get_tensor,
    /* .set_tensor_2d = */ nullptr,
    /* .get_tensor_2d = */ nullptr,
    /* .cpy_tensor    = */ nullptr,
    /* .clear         = */ pager_buffer_clear,
    /* .reset         = */ nullptr,
};

// ── buffer_type_i ───────────────────────────────────────────────────────────────────────────
static const char * pager_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    auto* tctx = (PagerBufferTypeContext*) buft->context;
    return tctx->name.c_str();
}

static ggml_backend_buffer_t pager_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    auto* tctx = (PagerBufferTypeContext*) buft->context;
    auto* ctx  = new PagerBufferContext();
    ctx->device = tctx->device;
    cudaSetDevice(ctx->device);   // fix (2026-07-18): cudaMallocHost + pool in the right device context (multi-device pool)

    // S0 (bad-page-state hypothesis): FLE_PAGER_NO_PIN=1 -> pageable instead of cudaMallocHost.
    // H2D gets slower (driver bounce), but the kernel pinning path (75 GB!) is bypassed.
    const char * mf = std::getenv("FLE_PAGER_MMAP_FILE");
    const char * np = std::getenv("FLE_PAGER_NO_PIN");
    ctx->host_pinned = !(np && *np && *np != '0');
    if (mf && *mf) {
        // NVMe mmap host tier (first-cut cold store): 135 GB fits via the page cache (reclaimable),
        // no pinned-RAM OOM. Staging-only path -> FLE_PAGER_NO_UVA=1 needed (pageable H2D).
#ifdef _WIN32
        // open/ftruncate/mmap are POSIX: no mmap tier on Windows. Fail loud instead of silently running without it.
        std::fprintf(stderr, "[fle-pager] FLE_PAGER_MMAP_FILE is not supported on Windows\n");
        delete ctx;
        return nullptr;
#else
        char mfpath[4096]; snprintf(mfpath, sizeof(mfpath), "%s.dev%d", mf, ctx->device);  // per-device file (multi-device pool 2026-07-18): otherwise O_TRUNC clobber + SIGBUS
        int fd = open(mfpath, O_RDWR | O_CREAT | O_TRUNC, 0644);
        if (fd < 0) { std::fprintf(stderr, "[fle-pager] MMAP open %s failed\n", mfpath); delete ctx; return nullptr; }
        if (ftruncate(fd, (off_t) size) != 0) { close(fd); delete ctx; return nullptr; }
        void* p = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (p == MAP_FAILED) { std::fprintf(stderr, "[fle-pager] mmap %zu MB failed\n", size >> 20); close(fd); delete ctx; return nullptr; }
        ctx->host_backing = (uint8_t*) p;
        ctx->host_mmap = true; ctx->mmap_fd = fd; ctx->host_pinned = false;
        std::fprintf(stderr, "[fle-pager] MMAP-BACKING: %s (%zu MB, NVMe page-cache reclaimable)\n", mfpath, size >> 20);
#endif

        // FLE_PAGER_HOST_REGISTER — pin the mmap'd file after the fact.
        //
        // Measured 2026-08-28 (FLE_PAGER_PHASES=1, 405,504 calls, steady state):
        //     ids D2H + sync    101.2 us/call   33.9 %
        //     staging (H2D)     195.9 us/call   65.5 %
        // In the first statistics window staging was still 1,660 us — it drops by a factor
        // of 8.5 while the sync stays CONSTANT. The falling part is NVMe, the rest is the copy
        // itself: `cudaMemcpyAsync` from PAGEABLE memory is not asynchronous, CUDA copies via
        // an internal pinned buffer and blocks. Measured rate: 2.97 MB per call in 195.9 us =
        // 15.2 GB/s against a 55 GB/s link.
        //
        // `cudaHostRegister` pins the existing mmap range without copying it —
        // occupying 77 GiB a second time would be hopeless with 128 GB RAM anyway.
        //
        // CAUTION, two real costs: pinned pages are no longer reclaimable (the whole point of
        // the mmap tier is lost), and registering has to page in all pages — which takes a
        // while for 28 GB. Hence default OFF, hence timed, and hence `_MB` as a cap to test
        // a subrange.
        //
        // Fail LOUD: if it fails, the reason is printed and the run continues unpinned. A
        // silent failure would be especially costly here because the measurement would then
        // see "no effect" and nobody would know that nothing happened.
        if (const char * hr = std::getenv("FLE_PAGER_HOST_REGISTER")) {
            if (*hr && *hr != '0') {
                size_t reg = size;
                if (const char * cap = std::getenv("FLE_PAGER_HOST_REGISTER_MB")) {
                    const long v = std::atol(cap);
                    if (v > 0 && (size_t) v * 1024ull * 1024ull < reg) reg = (size_t) v * 1024ull * 1024ull;
                }
                const auto t0 = std::chrono::steady_clock::now();
                const cudaError_t rc = cudaHostRegister(ctx->host_backing, reg, cudaHostRegisterDefault);
                const double ms = std::chrono::duration<double, std::milli>(
                                      std::chrono::steady_clock::now() - t0).count();
                if (rc == cudaSuccess) {
                    ctx->host_pinned = true;
                    ctx->host_registered_bytes = reg;
                    std::fprintf(stderr,
                        "[fle-pager] HOST-REGISTER: %zu of %zu MB pinned in %.0f ms (%.2f GB/s paging in)\n"
                        "[fle-pager]   -> H2D can now run truly asynchronously; these pages are NO LONGER reclaimable\n",
                        reg >> 20, size >> 20, ms, (double) reg / 1e9 / (ms / 1000.0));
                } else {
                    (void) cudaGetLastError();
                    std::fprintf(stderr,
                        "[fle-pager] HOST-REGISTER FAILED after %.0f ms: %s (%zu MB requested)\n"
                        "[fle-pager]   -> run continues UNPINNED. Every measurement from here on measures the old state.\n",
                        ms, cudaGetErrorString(rc), reg >> 20);
                }
            }
        }
    } else if (ctx->host_pinned) {
        if (cudaMallocHost((void**)&ctx->host_backing, size) != cudaSuccess) {
            (void) cudaGetLastError();
            delete ctx;
            return nullptr;
        }
    } else {
#ifdef _WIN32
        ctx->host_backing = (uint8_t*) _aligned_malloc(size, 4096);
        if (!ctx->host_backing) {
#else
        if (posix_memalign((void**)&ctx->host_backing, 4096, size) != 0) {
#endif
            delete ctx;
            return nullptr;
        }
        std::fprintf(stderr, "[fle-pager] NO_PIN active: host_backing pageable (%zu MB)\n", size >> 20);
    }
    ctx->host_bytes = size;

    size_t pool = tctx->vram_budget ? tctx->vram_budget : 0;
    if (pool > size) pool = size;
    if (pool > 0 && cudaMalloc(&ctx->vram_pool, pool) == cudaSuccess) {
        ctx->vram_bytes = pool;
    } else {
        cudaError_t pe = cudaGetLastError();
        std::fprintf(stderr, "[fle-pager] POOL-ALLOC FAIL: %zu MB -> %s (VRAM leaked? check nvidia-smi)\n",
                     pool >> 20, cudaGetErrorString(pe));
        ctx->vram_pool = nullptr;  // fail-soft (under NO_UVA the first stage call aborts cleanly)
    }

    return ggml_backend_buffer_init(buft, pager_buffer_interface, ctx, size);
}

static size_t pager_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    (void) buft;
    return 256;
}

static bool pager_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    (void) buft;
    // false: the op should run on the CUDA backend (supports_buft patch), data comes via
    // UVA/staging. is_host=true would pull the scheduler towards the CPU backend.
    return false;
}

static const ggml_backend_buffer_type_i pager_buffer_type_interface = {
    /* .get_name       = */ pager_buffer_type_get_name,
    /* .alloc_buffer   = */ pager_buffer_type_alloc_buffer,
    /* .get_alignment  = */ pager_buffer_type_get_alignment,
    /* .get_max_size   = */ nullptr,
    /* .get_alloc_size = */ nullptr,
    /* .is_host        = */ pager_buffer_type_is_host,
};

// ── Exports (pager_wire.h) ──────────────────────────────────────────────────────────────────
// ── Early-routing predictor ("Procyon"): router registry per layer for early-routing prefetch ──
struct ProcyonRouter {
    const void* gate_exps = nullptr;   // prefetch target (this layer's expert pool key = src0->data)
    const void* w = nullptr;           // gate_inp weight (GPU) - lazy F32 dequant cache (step-ii prerequisite)
    const struct ggml_tensor * gate_inp_t = nullptr;   // original tensor for lazy deref (data still NULL at register time)
    int wtype = 0; long n_embd = 0, n_expert = 0;
    const void* bias = nullptr; int bias_type = 0; int n_used = 0;
    // The cause of the crash with the DeepSeek-V4-Flash model, isolated 2026-08-11.
    //
    // The predictor predicts the routing of layer L+1 at layer L and reads the router AND the
    // bias of L+1 for that. Both belong to tensors of the NEXT layer. `-ot` only matches
    // `*_exps` tensors, not `exp_probs_b` — so the bias lives on the card of its own layer
    // while the kernel runs on the card of the pager buffer. Beyond the layer split these are
    // two different cards.
    //
    // Measured (FLE_PAGER_PROCYON_DEBUG=1, [procyon-dev]):
    //     L28  kernel-on-dev0 | bias_raw=dev0     runs
    //     L29  kernel-on-dev0 | bias_raw=dev1     illegal access
    //
    // With FLE_PROCYON_RAW_BIAS=1 the crash returns reproducibly — so the explanation can
    // reproduce the fault. That is the only reason the switch is still there; in operation
    // it must NOT be set.
    //
    // The original type is kept so the router can be re-unpacked on a device change.
    int qtype = 0;          // type of the original tensor, survives unpacking
    int wdev  = -1;         // device on which `w` lives; -1 = original, not ours
    const void* bias_dev = nullptr;   // device-local copy of the bias
    int bias_devnum = -1;
};
static std::unordered_map<int, ProcyonRouter> g_procyon;   // layer -> router
static std::mutex g_procyon_mtx;

FLE_EXPORT void pager_register_router(const struct ggml_tensor * gate_exps,
                                      const struct ggml_tensor * gate_inp,
                                      const struct ggml_tensor * exp_probs_b,
                                      int n_expert_used) {
    if (!gate_exps || !gate_inp) return;
    // no-op without pager env -> default hot path (per ubatch x layer) has 0 overhead.
    static const bool procyon_on = (std::getenv("FLE_PAGER_VRAM_MB") != nullptr);
    if (!procyon_on) return;
    int layer = -1;
    if (gate_exps->name[0]) { const char* p = std::strstr(gate_exps->name, "blk."); if (p) layer = std::atoi(p + 4); }
    if (layer < 0) return;
    std::lock_guard<std::mutex> lk(g_procyon_mtx);
    ProcyonRouter r;
    r.gate_exps = gate_exps->data;
    r.w = gate_inp->data; r.wtype = (int) gate_inp->type; r.qtype = (int) gate_inp->type; r.gate_inp_t = gate_inp;
    r.n_embd = (long) gate_inp->ne[0]; r.n_expert = (long) gate_inp->ne[1];
    r.bias = exp_probs_b ? exp_probs_b->data : nullptr;
    r.bias_type = exp_probs_b ? (int) exp_probs_b->type : 0;
    r.n_used = n_expert_used;
    g_procyon[layer] = r;
    if (std::getenv("FLE_PAGER_PROCYON_DEBUG"))
        std::fprintf(stderr, "[procyon] register L%d: w=%p wtype=%d n_embd=%ld n_expert=%ld bias=%p n_used=%d\n",
            layer, r.w, r.wtype, r.n_embd, r.n_expert, r.bias, r.n_used);
}

static std::unordered_map<int, std::unordered_set<int>> g_procyon_pred;   // layer -> predicted L experts

// Early-routing GEMV: logits[e] = sum_k w[e*n_embd+k]*x[k] + bias[e]  (router on gate_input)
__global__ void procyon_gemv_kernel(const float* __restrict__ w, const float* __restrict__ x,
                                    const float* __restrict__ bias, float* __restrict__ out,
                                    int n_embd, int n_expert) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_expert) return;
    const float* wr = w + (size_t) e * n_embd;
    float acc = 0.0f;
    for (int k = 0; k < n_embd; ++k) acc += wr[k] * x[k];
    out[e] = 1.0f / (1.0f + __expf(-acc)) + (bias ? bias[e] : 0.0f);   // minimax-m2: sigmoid(logits)+bias
}

FLE_EXPORT bool is_pager_buffer(ggml_backend_buffer_t buffer) {
    return buffer && buffer->iface.get_base == pager_buffer_get_base;
}

FLE_EXPORT bool is_pager_buffer_type(ggml_backend_buffer_type_t buft) {
    return buft && buft->iface.get_name == pager_buffer_type_get_name;
}

FLE_EXPORT ggml_backend_buffer_type_t pager_buffer_type(int device, size_t vram_budget_bytes) {
    static PagerBufferTypeContext s_ctx[16];
    static ggml_backend_buffer_type s_buft[16];
    static bool s_init[16] = { false };
    if (device < 0 || device >= 16) return nullptr;
    if (!s_init[device]) {
        s_ctx[device].device      = device;
        s_ctx[device].vram_budget = vram_budget_bytes;
        s_ctx[device].name        = "PAGER_EXPS_dev" + std::to_string(device);
        s_buft[device].iface   = pager_buffer_type_interface;
        s_buft[device].device  = ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), device);
        s_buft[device].context = &s_ctx[device];
        s_init[device] = true;
    }
    return &s_buft[device];
}

// NO_UVA regime (observed on a test host 2026-07-06): every nullptr return means "caller reads
// host RAM via GPU/UVA" — exactly the crash suspect. FLE_PAGER_NO_UVA=1 turns that into
// a hard, clear abort instead of a silent UVA read.
static bool pager_no_uva() {
    static const bool v = [] {
        const char * e = std::getenv("FLE_PAGER_NO_UVA");
        return e && *e && *e != '0';
    }();
    return v;
}

// S2 substrate A/B: miss staging on a dedicated copy stream (default off = proven path).
// FLE_PAGER_BOUNCE — miss staging through a pinned bounce buffer.
// Without it the H2D copy from the mmap backing is pageable and therefore NOT asynchronous;
// FLE_PAGER_ASYNC_STAGE then runs on its own stream but blocks anyway.
static bool pager_bounce() {
    static const bool v = [] {
        const char * e = std::getenv("FLE_PAGER_BOUNCE");
        const bool enabled = e && *e && *e != '0';
        // Evidence line: without it "no effect" cannot be told apart from "never arrived".
        fprintf(stderr, "[fle-pager] SWITCH BOUNCE = %s\n", enabled ? "ON" : "off");
        return enabled;
    }();
    return v;
}

static bool pager_async_stage() {
    static const bool v = [] {
        const char * e = std::getenv("FLE_PAGER_ASYNC_STAGE");
        const bool enabled = e && *e && *e != '0';
        // Evidence line: without it "no effect" cannot be told apart from "never arrived".
        fprintf(stderr, "[fle-pager] SWITCH ASYNC_STAGE = %s\n", enabled ? "ON" : "off");
        return enabled;
    }();
    return v;
}

// S1 policy A/B: reuse-aware eviction (min freq instead of min last). freq = routes since last stage
// (reset on eviction) -> protects recurring-hot experts, evicts one-hit wonders first.
static bool pager_evict_freq() {
    static const bool v = [] {
        const char * e = std::getenv("FLE_PAGER_EVICT_FREQ");
        const bool enabled = e && *e && *e != '0';
        // Evidence line: without it "no effect" cannot be told apart from "never arrived".
        fprintf(stderr, "[fle-pager] SWITCH EVICT_FREQ = %s\n", enabled ? "ON" : "off");
        return enabled;
    }();
    return v;
}

// S3 A/B: do NOT H2D-stage miss experts but compute them on the CPU (only the activation crosses PCIe).
static bool pager_cpu_miss() {
    static const bool v = [] {
        const char * e = std::getenv("FLE_PAGER_CPU_MISS");
        const bool enabled = e && *e && *e != '0';
        // Evidence line: without it "no effect" cannot be told apart from "never arrived".
        fprintf(stderr, "[fle-pager] SWITCH CPU_MISS = %s\n", enabled ? "ON" : "off");
        return enabled;
    }();
    return v;
}

// CLGP predictive prefetch: predicts the experts of the NEXT tensor (from its last token)
// and prefetches them async during the current compute. default OFF (A/B).
static bool pager_prefetch() {
    static const bool v = [] {
        const char * e = std::getenv("FLE_PAGER_PREFETCH");
        return e && *e && *e != '0';
    }();
    return v;
}

// S3 hybrid: freq threshold. Expert freq>=T -> stage (GPU-resident, hit), freq<T -> CPU compute (cold).
// Default = huge (all cold -> all-CPU, backward compatible). FLE_PAGER_CPU_MISS_FREQ_T=2 = admission after 2 misses.
static int pager_cpu_miss_freq_t() {
    static const int v = [] {
        const char * e = std::getenv("FLE_PAGER_CPU_MISS_FREQ_T");
        int t = (e && *e) ? std::atoi(e) : 0;
        return (t > 0) ? t : (1 << 30);
    }();
    return v;
}

#define PAGER_FALLBACK(reason) do { \
    cudaError_t _pe = cudaGetLastError(); \
    if (pager_no_uva()) { \
        std::fprintf(stderr, "[fle-pager] NO_UVA abort: %s | cuda: %s\n", (reason), cudaGetErrorString(_pe)); \
        std::abort(); \
    } \
    std::fprintf(stderr, "[fle-pager] fallback: %s | cuda: %s\n", (reason), cudaGetErrorString(_pe)); \
    return nullptr; \
} while (0)

// FLE_PAGER_VERIFY: asks about the CAUSE, not the effect.
//
// Whether it flips at 296 or 414 tokens is the same question. The cause is whether the
// pager delivers the RIGHT expert for a given ID.
//
// After staging this check reads back from the GPU buffer and compares byte for byte against
// the source in host_backing. It needs no long prompt and no threshold — a single call
// suffices for a verdict.
//
//   pager_verify_mapping(base, off_in_base, e, slice, src0, label)
//     base         device pointer the path will return (scratch or region)
//     off_in_base  byte offset the remapped ID points to
//     e            the expert that SHOULD be there
static void pager_verify_mapping(const void * base, size_t off_in_base, int e,
                                   size_t slice, const struct ggml_tensor * src0,
                                   const char * label, int layer_hint) {
    static int enabled = -1;
    if (enabled < 0) { const char * v = std::getenv("FLE_PAGER_VERIFY"); enabled = (v && *v) ? 1 : 0; }
    if (!enabled) return;
    static int reported = 0, checked = 0;
    static int cap = [] { const char * v = std::getenv("FLE_PAGER_VERIFY_CAP");
                          return (v && *v) ? std::atoi(v) : 64; }();
    if (checked >= cap) return;
    checked++;

    // Compare only the start of the slice — a wrong expert deviates immediately.
    const size_t probe = slice < 4096 ? slice : 4096;
    std::vector<uint8_t> from_device(probe);
    if (cudaMemcpy(from_device.data(), (const uint8_t *) base + off_in_base,
                   probe, cudaMemcpyDeviceToHost) != cudaSuccess) {
        cudaGetLastError();
        std::fprintf(stderr, "[VERIFY] %s: readback failed (e=%d)\n", label, e);
        return;
    }
    const uint8_t * expected = (const uint8_t *) src0->data + (size_t) e * slice;
    if (std::memcmp(from_device.data(), expected, probe) == 0) return;   // matches

    // Deviation: say WHICH expert lies there instead.
    int found = -1;
    const int64_t n_e = src0->ne[2];
    for (int64_t k = 0; k < n_e; ++k) {
        if (std::memcmp(from_device.data(), (const uint8_t *) src0->data + (size_t) k * slice,
                        probe < 256 ? probe : 256) == 0) { found = (int) k; break; }
    }
    if (reported++ < 12) {
        std::fprintf(stderr,
          "[VERIFY] WRONG MAPPING  %s  layer~%d  tensor=%s\n"
          "[VERIFY]   ID points to offset %zu, expert %d should be there\n"
          "[VERIFY]   but found there:    %s\n",
          label, layer_hint, src0->name ? src0->name : "?",
          off_in_base, e,
          found >= 0 ? "expert " : "no known expert (uninitialized or foreign)");
        if (found >= 0)
            std::fprintf(stderr, "[VERIFY]     -> expert %d (offset %d)\n", found, found - e);
    }
}

// FLE_PAGER_WALKTHROUGH: line-by-line trace with a variable table.
//
// Cap FLE_PAGER_WALKTHROUGH (default 0 = off) = number of prefill calls logged completely.
// After that it stays silent. Without the environment variable it costs one comparison
// per step and writes nothing.
static int pager_walkthrough_n() {
    static int v = -1;
    if (v < 0) { const char * e = getenv_alias("FLE_PAGER_WALKTHROUGH", "FLE_PAGER_DURCHZUG");   // deprecated alias
                 v = (e && *e) ? std::atoi(e) : 0; }
    return v;
}
static int g_walkthrough_runs = 0;     // number of prefill calls logged so far
static bool g_walkthrough_active = false;

// The walkthrough above reaches ONLY the prefill path: g_walkthrough_active is set solely in the branch
// `n_unique > n_slots`. The defect with mixed slice sizes occurs in DECODE — no measurement
// point reached it so far. A tool that does not reach the region counts as absent.
// Separate switch, separate counter: FLE_PAGER_WALKTHROUGH_DECODE=<n> calls.
static int pager_walkthrough_dec_n() {
    static int v = -1;
    if (v < 0) { const char * e = getenv_alias("FLE_PAGER_WALKTHROUGH_DECODE", "FLE_PAGER_DURCHZUG_DECODE");   // deprecated alias
                 v = (e && *e) ? std::atoi(e) : 0; }
    return v;
}
static int  g_walkthrough_dec_runs = 0;
static bool g_walkthrough_dec      = false;

// Phase measurement (FLE_PAGER_PHASES=1). The ~27 ms of pager overhead per run were so far
// INFERRED (80.2 - 36.6 - 16.4 ms), not measured. llama.cpp discussion #18369 shows on SYCL
// what a host-sync-and-loop in mul_mat_id costs: +48.8 % decode when removed
// (43.97 -> 65.41 tg128). CUDA upstream is sync-free (mm_ids_helper) — this pager puts the
// sync back (ids D2H + stream sync below). But WHERE the time goes is told only by a clock, not
// by the source: on 2026-08-28 two causes deduced from the code were wrong.
struct PagerPhases {
    double ids_d2h_sync = 0.0;   // ids D2H + cudaStreamSynchronize
    double host_scan    = 0.0;   // 4D traversal, uniq_map, early-routing prediction
    double staging      = 0.0;   // LRU + H2D of the misses
    double remap_rest   = 0.0;   // remap kernel, prefetch, return
    unsigned long long n = 0;
};
static PagerPhases g_phases;
static bool pager_phases() {
    static const bool v = [] { const char * e = getenv_alias("FLE_PAGER_PHASES", "FLE_PAGER_PHASEN");   // deprecated alias
                               return e && *e && *e != '0'; }();
    return v;
}
#define PH_NOW() std::chrono::steady_clock::now()
#define PH_MS(a,b) (std::chrono::duration<double, std::milli>((b) - (a)).count())
#define SHOW(line, fmt, ...) do { if (g_walkthrough_active) \
    std::fprintf(stderr, "[WALKTHROUGH] L%-5d " fmt "\n", (line), ##__VA_ARGS__); } while (0)

// CUDA-graph fix (step i): FLE_PAGER_DEVICE_REMAP — device kernel replaces host D2H+value-replace+H2D.
static bool pager_device_remap() {
    static int v = -1;
    if (v < 0) { const char* e = std::getenv("FLE_PAGER_DEVICE_REMAP"); v = (e && e[0] && e[0] != '0') ? 1 : 0; }
    return v == 1;
}
// Strided 4D gather (mirrors host PAGER_FOR_EACH_ID: *pe = e2s[e]). ids I32, dbuf = layout-faithful clone (same nb).
// slot_map = tp.e2s (expert->local slot). No host sync -> graph-legal.
__global__ void fle_pager_remap_kernel(const uint8_t* __restrict__ ids, uint8_t* __restrict__ dbuf,
        const int32_t* __restrict__ slot_map, int n_expert,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3,
        int64_t nb0, int64_t nb1, int64_t nb2, int64_t nb3) {
    const int64_t total = ne0 * ne1 * ne2 * ne3;
    for (int64_t gid = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; gid < total; gid += (int64_t) gridDim.x * blockDim.x) {
        int64_t i0 = gid % ne0, t = gid / ne0;
        int64_t i1 = t % ne1;   t /= ne1;
        int64_t i2 = t % ne2;   int64_t i3 = t / ne2;
        const int64_t off = i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3;
        const int32_t e = *(const int32_t*)(ids + off);
        // The guard checks the INPUT, not the OUTPUT. A valid but non-resident expert has
        // slot_map[e] == -1 (set on eviction). The -1 used to go unchecked into the ids;
        // mul_mat_id then computes data + (-1)*nb02 and reads BEFORE the region ->
        // `cuda: an illegal memory access`. The intent (per the S3 header comment) is slot 0:
        // the GPU computes garbage there and pager_cpu_miss_finalize overwrites the column
        // CPU-side.
        const int32_t s = (e >= 0 && e < n_expert) ? slot_map[e] : e;
        *(int32_t*)(dbuf + off) = (s >= 0) ? s : 0;
    }
}

// P4 core: staging + remap. Every abort -> nullptr = zero-copy fallback (always correct).
FLE_EXPORT void * pager_stage_and_remap(ggml_backend_buffer_t buffer,
                                        const struct ggml_tensor * src0,
                                        const struct ggml_tensor * ids,
                                        const int32_t ** ids_remapped_dev,
                                        cudaStream_t stream,
                                        const struct ggml_tensor * src1,
                                        struct ggml_tensor * dst,
        int * n_experts_out) {
    // S3 geometry dump (env FLE_PAGER_DUMP_SHAPES, ONCE): exact ne/nb of all operands -> correct dst override.
    if (std::getenv("FLE_PAGER_DUMP_SHAPES")) {
        static bool dumped = false;
        if (!dumped) {
            dumped = true;
            auto dq = [](const char* n, const struct ggml_tensor* t){
                if (!t) { std::fprintf(stderr, "[fle-pager] SHAPE %s = NULL\n", n); return; }
                std::fprintf(stderr, "[fle-pager] SHAPE %-5s type=%d ne=[%ld,%ld,%ld,%ld] nb=[%zu,%zu,%zu,%zu]\n",
                    n, (int)t->type, (long)t->ne[0],(long)t->ne[1],(long)t->ne[2],(long)t->ne[3],
                    t->nb[0],t->nb[1],t->nb[2],t->nb[3]);
            };
            dq("src0", src0); dq("src1", src1); dq("ids", ids); dq("dst", dst);
        }
    }
    std::lock_guard<std::mutex> lock(g_pager_mtx);
    auto* ctx = (PagerBufferContext*) buffer->context;
    if (!ctx->vram_pool) PAGER_FALLBACK("no vram_pool");

    if (!ctx->rt) { ctx->rt = new PagerRuntime(); ctx->rt->device = ctx->device; }
    PagerRuntime& rt = *ctx->rt;

    const size_t  slice    = src0->nb[2];   // bytes per expert slice (stride)
    const int64_t n_expert = src0->ne[2];
    if (slice == 0 || n_expert <= 0) PAGER_FALLBACK("slice/n_expert invalid");

    if (rt.region_bytes == 0) {
        // Dimensions PER DEVICE (2026-08-26). Until then `FLE_PAGER_VRAM_MB1` was the only
        // variable with a device binding; `REGIONS` and `SCRATCH_MB` applied globally. The
        // second pager inherited the first one's dimensions — and they do not fit, because it
        // has a different tensor count and a different arena.
        //
        // Measured: GPU1 had 33 tensors and got regions=96 from GPU0. Result
        // (8192-2048)/96 = 64.0 MiB per region instead of (8192-1024)/33 = 217.2 MiB — a factor
        // of 3.3 wasted, and the scratch ate 25 % of its arena instead of 13 %. The 23.80 t/s
        // read from that measured a crippled second pager, not the split.
        //
        // It is the same mistake that the set-count measurement exposed for the ONE pager on
        // 2026-08-25 (regions 160 with 129 tensors cost 10.9 %) — just invisible because the
        // second one had no knob.
        //
        // Pattern as for VRAM_MB: `<NAME>` for device 0, `<NAME>1` for device 1.
        // If the second is missing, the first applies — every existing invocation stays valid.
        auto env_per_device = [&](const char * base, const char * base1) -> const char * {
            if (ctx->device == 1) { if (const char * e = std::getenv(base1)) return e; }
            return std::getenv(base);
        };
        size_t regions = 160, scratch_mb = 2048;
        if (const char * e = env_per_device("FLE_PAGER_REGIONS", "FLE_PAGER_REGIONS1"))
            { long v = std::atol(e); if (v > 0) regions = (size_t) v; }
        if (const char * e = env_per_device("FLE_PAGER_SCRATCH_MB", "FLE_PAGER_SCRATCH_MB1"))
            { long v = std::atol(e); if (v >= 0) scratch_mb = (size_t) v; }
        rt.scratch_bytes = scratch_mb * 1024ull * 1024ull;
        if (rt.scratch_bytes > ctx->vram_bytes / 2) rt.scratch_bytes = ctx->vram_bytes / 2;
        // Reserve the swap arena (exchange cache) at the pool end — never assign it to resident regions.
        size_t swap_mb = 0;
        if (const char * e = env_per_device("FLE_PAGER_SWAP_MB", "FLE_PAGER_SWAP_MB1")) { long v = std::atol(e); if (v > 0) swap_mb = (size_t) v; }
        rt.swap_bytes = swap_mb * 1024ull * 1024ull;
        if (rt.swap_bytes > ctx->vram_bytes / 4) rt.swap_bytes = ctx->vram_bytes / 4;   // cap 25%
        rt.swap_off  = ctx->vram_bytes - rt.swap_bytes;                                  // arena at the pool end
        rt.next_off      = rt.scratch_bytes;
        const size_t region_budget = ctx->vram_bytes - rt.scratch_bytes - rt.swap_bytes;
        rt.region_bytes  = (region_budget / regions) & ~(size_t)255;  // 256-aligned
        if (rt.region_bytes == 0) PAGER_FALLBACK("region_bytes==0 (pool too small)");
        if (rt.swap_bytes && std::getenv("FLE_PAGER_PROCYON_DEBUG"))
            std::fprintf(stderr, "[fle-pager] swap-arena %zu MB @off=%zu | region_bytes=%zu regions=%zu\n",
                         rt.swap_bytes >> 20, rt.swap_off, rt.region_bytes, regions);
    }

    // Get/carve the region of this tensor (lazy, first come)
    auto it = rt.pools.find(src0->data);
    if (it == rt.pools.end()) {
        const size_t region_limit = rt.swap_bytes ? rt.swap_off : ctx->vram_bytes;  // regions never extend into the swap arena
        if (rt.next_off + rt.region_bytes > region_limit) PAGER_FALLBACK("pool full (FLE_PAGER_REGIONS too small)");
        TensorPool tp;
        tp.base_off   = rt.next_off;
        tp.slot_bytes = slice;
        tp.n_slots    = (int) (rt.region_bytes / slice);
        if (tp.n_slots > (int) n_expert) tp.n_slots = (int) n_expert;   // slot ID MUST stay < n_expert
        if (tp.n_slots <= 0) PAGER_FALLBACK("region smaller than one slice");
        tp.slots.assign(tp.n_slots, SlotInfo{});
        tp.e2s.assign((size_t) n_expert, -1);
        tp.evicted_at.assign((size_t) n_expert, 0);   // miss-nature telemetry (cold vs reuse)
        tp.freq.assign((size_t) n_expert, 0);         // reuse-aware eviction frequency
        tp.seen_call.assign((size_t) n_expert, 0);     // working-set measurement
        tp.self = src0->data;                          // host weights key (for CLGP prefetch)
        rt.next_off += rt.region_bytes;
        it = rt.pools.emplace(src0->data, std::move(tp)).first;
    }
    TensorPool& tp = it->second;
    if (slice != tp.slot_bytes) PAGER_FALLBACK("slice drift on tensor");

    // CLGP prefetch: (a) learn the tensor chain, (b) wait for the prefetch that targeted THIS tensor
    // -> the mmvq that follows (compute stream) waits for the prefetch H2D (copy stream).
    const bool prefetch_on = pager_prefetch() && rt.ensure_copy();
    if (prefetch_on) {
        if (rt.prev_ptr) rt.next_ptr[rt.prev_ptr] = src0->data;
        rt.prev_ptr = src0->data;
        if (rt.prefetch_pending) { cudaStreamWaitEvent(stream, rt.prefetch_ev, 0); rt.prefetch_pending = false; }
    }

    // ids as a LAYOUT-FAITHFUL clone (full nbytes, original strides): mmq reads ids with
    // ids->nb strides — compact packing caused the illegal access at mmq.cu:206.
    const size_t nbytes = ggml_nbytes(ids);
    if (nbytes == 0 || !rt.ensure(nbytes)) PAGER_FALLBACK("ids clone alloc failed");

    const auto _ph_a = PH_NOW();
    if (cudaMemcpyAsync(rt.h_buf, ids->data, nbytes, cudaMemcpyDeviceToHost, stream) != cudaSuccess) PAGER_FALLBACK("ids D2H failed");
    if (cudaStreamSynchronize(stream) != cudaSuccess) PAGER_FALLBACK("stream sync failed (sticky error from earlier kernel)");

    const auto _ph_b = PH_NOW();   // ids are on the host, sync is done
    rt.clock++; rt.stamp++;
    uint8_t * region = (uint8_t *) ctx->vram_pool + tp.base_off;

    // Generic 4D traversal over ne/nb (ids = I32, possibly a view with non-compact strides)
    const int64_t ne0 = ids->ne[0], ne1 = ids->ne[1], ne2t = ids->ne[2], ne3 = ids->ne[3];
    const size_t  nb0 = ids->nb[0], nb1 = ids->nb[1], nb2t = ids->nb[2], nb3 = ids->nb[3];
    #define PAGER_FOR_EACH_ID(BODY) \
        for (int64_t i3 = 0; i3 < ne3; ++i3) for (int64_t i2 = 0; i2 < ne2t; ++i2) \
        for (int64_t i1 = 0; i1 < ne1; ++i1) for (int64_t i0 = 0; i0 < ne0; ++i0) { \
            int32_t * pe = (int32_t *)(rt.h_buf + i3*nb3 + i2*nb2t + i1*nb1 + i0*nb0); BODY }

    // Collect the unique experts of this call (prefill detection)
    if ((int64_t) rt.uniq_map.size() < n_expert) rt.uniq_map.assign((size_t) n_expert, -1);
    else std::fill(rt.uniq_map.begin(), rt.uniq_map.begin() + n_expert, -1);
    int n_unique = 0;
    PAGER_FOR_EACH_ID({
        const int32_t e = *pe;
        if (e >= 0 && e < n_expert) { if (rt.uniq_map[e] < 0) rt.uniq_map[e] = n_unique++; }   // fix: freq NOT counted here (double counting with the per-call count); reuse-aware eviction got T+1 instead of 1
    })
    // Routing trace dump (2026-07-18, FLE_PAGER_TRACE_FILE): for offline pager simulation.
    // Per call: "<layer> e1 e2 ..." (experts in routing order).
    static FILE* _trace_fp[16] = {};   // per device (function runs under s_mtx -> static is safe), otherwise multi-device clobber
    if (const char* _tf = std::getenv("FLE_PAGER_TRACE_FILE")) {
        const int _d = ctx->device & 15;
        if (!_trace_fp[_d]) { char _tp[4096]; std::snprintf(_tp, sizeof(_tp), "%s.dev%d", _tf, ctx->device); _trace_fp[_d] = std::fopen(_tp, "w"); }
        FILE* _fp = _trace_fp[_d];
        if (_fp) {
            int _tl = -1; if (src0->name[0]) { const char* _p = std::strstr(src0->name, "blk."); if (_p) _tl = std::atoi(_p + 4); }
            std::vector<int> _ord((size_t) (n_unique > 0 ? n_unique : 1), -1);
            for (int _e = 0; _e < (int) n_expert; ++_e) { const int _s = rt.uniq_map[_e]; if (_s >= 0 && _s < n_unique) _ord[_s] = _e; }
            std::fprintf(_fp, "%d", _tl);
            for (int _i = 0; _i < n_unique; ++_i) std::fprintf(_fp, " %d", _ord[_i]);
            std::fprintf(_fp, "\n");
            static int _tflush = 0; if ((++_tflush & 1023) == 0) std::fflush(_fp);
        }
    }

    // ── S1 telemetry: hit/miss per unique expert (whole function under s_mtx -> plain static is safe) ──
    // Fix (2026-08-26): there used to be ONE static struct for ALL devices.
    // The comment above stated the assumption itself — "plain static safe" holds
    // against concurrency, not against two cards. Since then two pagers run,
    // and `hit=%` was a mixed value over both.
    //
    // Same class as g_bw_dev. The `tensors=96`/`tensors=33` rows are NOT
    // affected — they live in `tp`, per tensor.
    //
    // New: the OUTFLOW. So far the pager only counted what came in.
    // `evicts`/`evict_bytes` are the other side — what a slot gives up and
    // has to bring over the bus again at the next need.
    struct PagerStats {
        unsigned long long calls=0, decode=0, prefill=0, routed=0, hits=0, misses=0,
                           reuse_miss=0, cold_miss=0, rd_sum=0, rd_cnt=0, rd_fit=0;
        // ── outflow (2026-08-26) ──
        unsigned long long fills=0,  fill_bytes=0;    // in: an expert takes a slot
        unsigned long long evicts=0, evict_bytes=0;   // out: an expert loses it again
    };
    static PagerStats st_dev[16];
    // As with bw_cur(): the CURRENT device decides, not a passed-in
    // context — counting happens where the computing happens.
    auto st_cur = []() -> PagerStats & {
        int d = 0; cudaGetDevice(&d);
        if (d < 0 || d >= 16) d = 0;
        return st_dev[d];
    };
    PagerStats & st = st_cur();
    static const unsigned long long st_win = [] {
        const char * e = std::getenv("FLE_PAGER_STAT_EVERY"); return (unsigned long long)(e && *e ? std::atoll(e) : 1024);
    }();
    auto stat_log = [&]() {
        if (st_win == 0 || st.calls % st_win != 0) return;
        if (const char* rf = std::getenv("FLE_PAGER_RESET_FILE")) {
            if (std::remove(rf) == 0) {  // file existed -> subtract warm-up: st + all freq to 0
                st = PagerStats{};
                for (auto& kv : rt.pools) std::fill(kv.second.freq.begin(), kv.second.freq.end(), 0u);
                std::fprintf(stderr, "[fle-pager] RESET (warm-up subtracted)\n");
                return;
            }
        }
        double hr = st.routed ? 100.0 * (double) st.hits / (double) st.routed : 0.0;
        double rr = st.misses ? 100.0 * (double) st.reuse_miss / (double) st.misses : 0.0;
        double rd_avg = st.rd_cnt ? (double) st.rd_sum / (double) st.rd_cnt : 0.0;
        double fit_pct = st.rd_cnt ? 100.0 * (double) st.rd_fit / (double) st.rd_cnt : 0.0;
        int _resident=0; for (int _e=0;_e<(int)tp.e2s.size();++_e) if (tp.e2s[_e]>=0) _resident++;
        int _pf=0; for (auto& _kv: rt.pools) for (int _e=0;_e<(int)_kv.second.e2s.size();++_e) if (_kv.second.e2s[_e]>=0) _pf++;
        std::fprintf(stderr, "[fle-pager] RESIDENT this-tensor=%d/%d total-pools=%d\n", _resident, tp.n_slots, _pf);
        // ── Flow ledger (2026-08-26) ─────────────────────────────────────────
        // So far the pager reported what came IN, and hits as a rate. Both leave the
        // decisive question open: does the memory keep what it fetches, or does it carry
        // the same things around in a circle? A cache that loads 40 GiB and throws 38 of
        // them away again reports the same hit rate as one that keeps them.
        //
        // Three pairs, each with its control number. The device number comes first —
        // without it two pagers could not be told apart again.
        {
            int _d = 0; cudaGetDevice(&_d);
            const double _gib = 1024.0*1024.0*1024.0;
            const unsigned long long _in_b = bw_cur().h2d_bytes + bw_cur().d2d_bytes + bw_cur().cpu_bytes;
            // Turnover = what went out again, measured against what came in.
            // Near 0 -> the memory holds. Near 1 -> it carries in circles, and every
            // additional slot pays off immediately.
            // Without inflow NO ratio is printed — a 0 next to it would be a
            // measured value, "n/a" is recognizably a gap (empty instead of broken).
            char _turnover[24];
            if (_in_b) std::snprintf(_turnover, sizeof _turnover, "%.0f%%", 100.0 * (double) st.evict_bytes / (double) _in_b);
            else        std::snprintf(_turnover, sizeof _turnover, "n/a");

            std::fprintf(stderr,
                "[fle-pager] FLOW dev%d bytes    in=%.2f GiB (h2d %.2f + d2d %.2f + cpu %.2f)"
                "  out=%.2f GiB evicted  -> turnover %s\n",
                _d, (double) _in_b / _gib,
                (double) bw_cur().h2d_bytes / _gib, (double) bw_cur().d2d_bytes / _gib,
                (double) bw_cur().cpu_bytes / _gib,
                (double) st.evict_bytes / _gib, _turnover);

            // Fills minus evictions IS the stock — not estimated, but the difference of
            // two counted sides. If it deviates from the counted occupancy (_pf), one of the
            // two sides is wrong; that is why both stand next to each other instead of only
            // the more convenient one.
            std::fprintf(stderr,
                "[fle-pager] FLOW dev%d experts  in=%llu out=%llu -> stock %lld"
                "  (counted occupied: %d)\n",
                _d, st.fills, st.evicts, (long long) st.fills - (long long) st.evicts, _pf);

            // The miss splits into two natures, and only one of them is curable by more
            // slots: "cold" we never had, "repeat" we had and threw away. A high repeat share
            // next to a turnover near 1 is the signature of a set that is too small.
            std::fprintf(stderr,
                "[fle-pager] FLOW dev%d store    hits=%llu misses=%llu (cold %llu / repeat %llu)"
                "  -> repeat share %.0f%% of %llu\n",
                _d, st.hits, st.misses, st.cold_miss, st.reuse_miss, rr, st.misses);

            // The third pool. It has no sets and no turnover — its residency is 100 %
            // because the whole model sits in main memory (measured 2026-08-26: llama-server
            // RSS 93.0 GB of 98 GB model). Its scarce resource is cores, so the thread
            // count is printed here. `cpu=0` means: the path is off, not "there was nothing to do".
            {
                const char * _ct = std::getenv("FLE_PAGER_CPU_THREADS");
                const int _n = (_ct && *_ct) ? std::atoi(_ct) : 0;
                std::fprintf(stderr,
                    "[fle-pager] FLOW host  cores=%s  calls=%llu  bytes=%.2f GiB  time=%.1f ms%s\n",
                    _n > 0 ? _ct : "unlimited", bw_cur().cpu_calls,
                    (double) bw_cur().cpu_bytes / _gib, bw_cur().cpu_ms,
                    bw_cur().cpu_calls == 0 ? "   (CPU path OFF)" : "");
            }
        }
        std::fprintf(stderr, "[fle-pager] stats calls=%llu decode=%llu prefill=%llu | routed=%llu hit=%.1f%% miss=%llu (reuse=%llu=%.0f%% cold=%llu) | n_slots=%d tensors=%d | REUSE-DIST avg=%.1f fit<=slots=%.0f%% (fit=Policy/miss>slots=Capacity)\n",
                     st.calls, st.decode, st.prefill, st.routed, hr, st.misses, st.reuse_miss, rr, st.cold_miss, tp.n_slots, (int) rt.pools.size(), rd_avg, fit_pct);
        if (g_bw_on) {
            // GB/s = bytes / ms * 1e3 / 2^30. Both branches side by side, plus the ratio
            // against which FreeToken decides (threshold 2.0: CPU_BW > 2 x PCIe_BW -> hybrid).
            const double _pg = bw_cur().pcie_ms > 0.0 ? (double) bw_cur().h2d_bytes / bw_cur().pcie_ms * 1e3 / (1024.0*1024.0*1024.0) : 0.0;
            const double _cg = bw_cur().cpu_ms  > 0.0 ? (double) bw_cur().cpu_bytes / bw_cur().cpu_ms  * 1e3 / (1024.0*1024.0*1024.0) : 0.0;
            // No value without its control number: samples/skipped says how reliable the PCIe number is.
            // If one side is missing entirely, NO ratio is printed — a 0 next to it would be a
            // measured value, "n/a" is recognizably a gap.
            std::fprintf(stderr, "[fle-pager] BW pcie-h2d=%.2f GB/s (%llu B / %.1f ms, samples=%llu skipped=%llu) | "
                                 "cpu=%.2f GB/s (%llu B / %.1f ms, calls=%llu) | d2d-swap=%llu B (no bus) | ",
                         _pg, bw_cur().h2d_bytes, bw_cur().pcie_ms, bw_cur().pcie_samples, bw_cur().pcie_skipped,
                         _cg, bw_cur().cpu_bytes, bw_cur().cpu_ms, bw_cur().cpu_calls, bw_cur().d2d_bytes);
            if (_pg > 0.0 && _cg > 0.0) {
                const double _r = _cg / _pg;
                std::fprintf(stderr, "cpu/pcie=%.2f -> %s (threshold 2.0)\n", _r, _r > 2.0 ? "hybrid" : "offload");
            } else {
                std::fprintf(stderr, "cpu/pcie=n/a (one side without a measurement)\n");
            }
        }
        if (std::getenv("FLE_PAGER_FREQ_DUMP")) {
            std::vector<uint32_t> f = tp.freq;
            std::sort(f.begin(), f.end(), std::greater<uint32_t>());
            unsigned long long tot = 0; for (uint32_t v : f) tot += v;
            size_t nz = 0, rare = 0; for (uint32_t v : f) { if (v == 0) nz++; else if (v <= 2) rare++; }
            int c50 = -1, c80 = -1, c95 = -1; unsigned long long acc = 0;
            for (size_t i = 0; i < f.size(); ++i) { acc += f[i];
                if (c50 < 0 && acc * 2 >= tot) c50 = (int)(i + 1);
                if (c80 < 0 && acc * 10 >= tot * 8) c80 = (int)(i + 1);
                if (c95 < 0 && acc * 20 >= tot * 19) c95 = (int)(i + 1); }
            std::fprintf(stderr, "[fle-pager] FREQ-SKEW n_exp=%zu total=%llu top8=[%u,%u,%u,%u,%u,%u,%u,%u] cover50=%d cover80=%d cover95=%d zero=%zu rare<=2=%zu\n",
                f.size(), tot,
                f.size()>0?f[0]:0u, f.size()>1?f[1]:0u, f.size()>2?f[2]:0u, f.size()>3?f[3]:0u,
                f.size()>4?f[4]:0u, f.size()>5?f[5]:0u, f.size()>6?f[6]:0u, f.size()>7?f[7]:0u,
                c50, c80, c95, nz, rare);
        }
    };
    st.calls++;
    // ── Early-routing predictor part 2a: predict early routing (L+1) + accuracy against real routing (env FLE_PAGER_PROCYON_ACC) ──
    // [!] DEPRECATED / MEASURED USELESS (2026-07-18): The whole early-routing prediction path below
    // (GEMV L->L+1, lazy dequant, acc) is empirically dead: prediction accuracy 6.2% (100k+ samples, ~chance),
    // markov prefetch ~= LRU in offline simulation. Expert routing is NOT predictable from cheap signals
    // -> step ii (prediction-driven pager) abandoned. Active ONLY with FLE_PAGER_PROCYON (never in production).
    // Contains known dual-GPU device bugs (dequant malloc + static _dlog on the wrong device) that fire ONLY with
    // the env. NOT removed (entangled with the CLGP prefetch last_uniq) -> marked.
    if ((std::getenv("FLE_PAGER_PROCYON_ACC") || std::getenv("FLE_PAGER_PROCYON")) && src1 && src0->name[0]) {
        const char* _bp = std::strstr(src0->name, "blk.");
        if (_bp && std::strstr(src0->name, "gate_exps")) {
            const int _L = std::atoi(_bp + 4);
            { static int _shc = 0; if (_shc < 16) { _shc++;
                std::fprintf(stderr, "[procyon-shape] %s ids_ne=[%ld,%ld,%ld,%ld] n_unique_slots\n", src0->name,
                    (long)(ids?ids->ne[0]:-1),(long)(ids?ids->ne[1]:-1),(long)(ids?ids->ne[2]:-1),(long)(ids?ids->ne[3]:-1)); } }
            {   // COMPARE: current L routing vs. the prediction for L stored at L-1
                std::lock_guard<std::mutex> _lk(g_procyon_mtx);
                auto _it = g_procyon_pred.find(_L);
                { static int _fc=0; if(_fc<12){_fc++; std::fprintf(stderr,"[procyon-cmp] find(%d)=%d map_size=%zu\n", _L, _it!=g_procyon_pred.end(), g_procyon_pred.size()); } }
                if (_it != g_procyon_pred.end()) {
                    int _hit = 0, _need = 0;
                    for (int e = 0; e < (int) n_expert; ++e) if (rt.uniq_map[e] >= 0) { _need++; if (_it->second.count(e)) _hit++; }
                    static long _ah = 0, _an = 0, _ac = 0;
                    _ah += _hit; _an += _need; _ac++;
                    if (_ac <= 6 || _ac % 64 == 0)
                        std::fprintf(stderr, "[procyon-acc] _ac=%ld need=%d hit=%d | cum %.1f%% (%ld/%ld)\n",
                            _ac, _need, _hit, _an ? 100.0*(double)_ah/(double)_an : 0.0, _ah, _an);
                }
            }
            // PREDICT L+1: GEMV(router[L+1].w, gate_input[L]=src1) -> top-k
            std::unordered_map<int, ProcyonRouter>::iterator _rit;
            size_t _gpn; { std::lock_guard<std::mutex> _lk(g_procyon_mtx); _rit = g_procyon.find(_L + 1); _gpn = g_procyon.size(); }
            // Step-ii prerequisite (2026-07-18): dequantize the router LAZILY to F32. At register time gate_inp->data
            // is still NULL, and the dsv4 router is quantized (wtype!=0) -> procyon_gemv_kernel (const float*) was
            // skipped -> prediction inert. Here (mid-inference) gate_inp_t->data is bound.
            int _curdev = -1; cudaGetDevice(&_curdev);
            if (_rit != g_procyon.end() && _rit->second.gate_inp_t
                && (_rit->second.wtype != 0 || _rit->second.wdev != _curdev)) {
                std::lock_guard<std::mutex> _lk(g_procyon_mtx);
                ProcyonRouter & _rr = _rit->second;
                if (_rr.gate_inp_t && _rr.gate_inp_t->data && (_rr.wtype != 0 || _rr.wdev != _curdev)) {
                    // A buffer on the wrong device that WE created is freed.
                    // wdev == -1 means: `w` is still the original tensor, which does not belong to us.
                    if (_rr.wdev >= 0 && _rr.wdev != _curdev && _rr.w) {
                        cudaFree((void*) _rr.w); _rr.w = _rr.gate_inp_t->data; _rr.wtype = _rr.qtype; _rr.wdev = -1;
                    }
                    const enum ggml_type _qt = (enum ggml_type) _rr.qtype;
                    const size_t _nelem  = (size_t) _rr.n_embd * (size_t) _rr.n_expert;
                    const size_t _qbytes = ggml_row_size(_qt, _rr.n_embd) * (size_t) _rr.n_expert;
                    const struct ggml_type_traits * _tt = ggml_get_type_traits(_qt);
                    if (_tt && _tt->to_float && _nelem > 0 && _qbytes > 0) {
                        std::vector<uint8_t> _qh(_qbytes);
                        std::vector<float>   _fh(_nelem);
                        cudaGetLastError();
                        bool _cp = (cudaMemcpy(_qh.data(), _rr.gate_inp_t->data, _qbytes, cudaMemcpyDeviceToHost) == cudaSuccess);
                        cudaGetLastError();
                        if (_cp) {
                            _tt->to_float(_qh.data(), _fh.data(), (int64_t) _nelem);
                            float * _dw = nullptr;
                            if (cudaMalloc(&_dw, _nelem * sizeof(float)) == cudaSuccess
                                && cudaMemcpy(_dw, _fh.data(), _nelem * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess) {
                                _rr.w = _dw; _rr.wtype = 0; _rr.wdev = _curdev;
                                if (std::getenv("FLE_PAGER_PROCYON_DEBUG"))
                                    std::fprintf(stderr, "[procyon] lazy-dequant L%d -> F32 ok on dev%d (nelem=%zu qbytes=%zu)\n", _L + 1, _curdev, _nelem, _qbytes), std::fflush(stderr);
                            } else if (_dw) cudaFree(_dw);
                        }
                        cudaGetLastError();
                    }
                    // The bias belongs to the router of L+1 and thus lives on ITS card.
                    // The kernel reads it during layer L — so a device-local copy.
                    if (_rr.bias && _rr.bias_type == 0 && _rr.bias_devnum != _curdev && _rr.n_expert > 0) {
                        if (_rr.bias_dev) { cudaFree((void*) _rr.bias_dev); _rr.bias_dev = nullptr; }
                        const size_t _bb = (size_t) _rr.n_expert * sizeof(float);
                        std::vector<float> _bh(_rr.n_expert);
                        void * _bd = nullptr;
                        cudaGetLastError();
                        if (cudaMemcpy(_bh.data(), _rr.bias, _bb, cudaMemcpyDeviceToHost) == cudaSuccess
                            && cudaMalloc(&_bd, _bb) == cudaSuccess
                            && cudaMemcpy(_bd, _bh.data(), _bb, cudaMemcpyHostToDevice) == cudaSuccess) {
                            _rr.bias_dev = _bd; _rr.bias_devnum = _curdev;
                        } else if (_bd) { cudaFree(_bd); }
                        cudaGetLastError();
                    }
                }
            }
            static int _diagc = 0;
            if (_diagc < 6) { _diagc++; std::fprintf(stderr, "[procyon-diag] L%d src1=%p ne0=%ld g_procyon.size=%zu found_L+1=%d\n", _L, (void*)(src1?src1->data:nullptr), (long)(src1?src1->ne[0]:-1), _gpn, _rit != g_procyon.end()); }
            if (_rit != g_procyon.end() && _rit->second.w && _rit->second.wtype == 0) {
                const ProcyonRouter _r = _rit->second;
                // Precaution — not the cause. The suspicion first fell here: ONE
                // static pointer for all layers and both cards. It was then measured that
                // the kernel ALWAYS runs on dev0 (the pager buffer lives there), so
                // _dlog never changes device and was never broken. The condition stays
                // anyway, because it is right and costs nothing: it kicks in as soon as
                // pager tensors live on both cards.
                static float* _dlog = nullptr; static long _dn = 0; static int _ddev = -1;
                if (_dn < _r.n_expert || _ddev != _curdev) {
                    if (_dlog) cudaFree(_dlog);
                    _dlog = nullptr;
                    if (cudaMalloc(&_dlog, (size_t) _r.n_expert * sizeof(float)) != cudaSuccess) _dlog = nullptr;
                    _dn = _dlog ? _r.n_expert : 0; _ddev = _dlog ? _curdev : -1;
                    if (std::getenv("FLE_PAGER_PROCYON_DEBUG"))
                        std::fprintf(stderr, "[procyon] dlog reallocated for dev%d (L%d, n_expert=%ld)\n", _curdev, _L, (long) _r.n_expert), std::fflush(stderr);
                }
                if (_dlog) {
                    // Evidence: which card does each of the three pointers the kernel touches
                    // belong to? `bias` belongs to the router of L+1 and is NOT matched by -ot
                    // (exp_probs_b is not an _exps tensor), so it lives on the card of ITS layer —
                    // beyond the split point that is a different one from the one the kernel runs on.
                    if (std::getenv("FLE_PAGER_PROCYON_DEBUG")) {
                        static int _pc = 0;
                        if (_pc < 40) { _pc++;
                            auto _dev = [](const void* p) -> int {
                                if (!p) return -2;
                                cudaPointerAttributes _a{};
                                if (cudaPointerGetAttributes(&_a, p) != cudaSuccess) { cudaGetLastError(); return -3; }
                                return _a.device;
                            };
                            std::fprintf(stderr, "[procyon-dev] L%d kernel-on-dev%d | w=dev%d bias_raw=dev%d bias_local=dev%d src1=dev%d dlog=dev%d\n",
                                _L, _curdev, _dev(_r.w), _dev(_r.bias), _dev(_r.bias_dev), _dev(src1->data), _dev(_dlog));
                            std::fflush(stderr);
                        }
                    }
                    // With FLE_PROCYON_RAW_BIAS=1 the original pointer is used again.
                    // If the crash comes back with it, the bias is the cause — and only then.
                    static const bool _bias_raw = getenv_alias("FLE_PROCYON_RAW_BIAS", "FLE_PROCYON_BIAS_ROH") != nullptr;   // deprecated alias
                    procyon_gemv_kernel<<<((int) _r.n_expert + 127) / 128, 128, 0, stream>>>(
                        (const float*) _r.w, (const float*) src1->data,
                        _bias_raw ? ((_r.bias && _r.bias_type == 0) ? (const float*) _r.bias : nullptr)
                                  : ((_r.bias_dev && _r.bias_devnum == _curdev) ? (const float*) _r.bias_dev : nullptr),
                        _dlog, (int) _r.n_embd, (int) _r.n_expert);
                    std::vector<float> _hl((size_t) _r.n_expert);
                    if (cudaMemcpyAsync(_hl.data(), _dlog, (size_t) _r.n_expert * sizeof(float), cudaMemcpyDeviceToHost, stream) == cudaSuccess
                        && cudaStreamSynchronize(stream) == cudaSuccess) {
                        std::vector<int> _idx((size_t) _r.n_expert);
                        for (int i = 0; i < (int) _r.n_expert; ++i) _idx[i] = i;
                        const int _k = _r.n_used > 0 ? _r.n_used : 8;
                        std::partial_sort(_idx.begin(), _idx.begin() + _k, _idx.end(), [&](int a, int b){ return _hl[a] > _hl[b]; });
                        std::unordered_set<int> _pred;
                        for (int i = 0; i < _k; ++i) _pred.insert(_idx[i]);
                        if (std::getenv("FLE_PAGER_PROCYON")) {   // part 2b: prediction -> L+1's prefetch signal (+ confidence gate)
                            int _topn = _k;   // "fate trick": prefetch only the top-N most certain (wrong predictions do not evict)
                            const char* _tn = std::getenv("FLE_PAGER_PROCYON_TOPN");
                            if (_tn) { _topn = std::atoi(_tn); if (_topn < 1) _topn = 1; if (_topn > _k) _topn = _k; }
                            auto _pit = rt.pools.find(_r.gate_exps);
                            if (_pit != rt.pools.end()) _pit->second.last_uniq.assign(_idx.begin(), _idx.begin() + _topn);
                        }
                        std::lock_guard<std::mutex> _lk(g_procyon_mtx);
                        size_t _psz=_pred.size(); g_procyon_pred[_L + 1] = std::move(_pred); { static int _sc=0; if(_sc<12){_sc++; std::fprintf(stderr,"[procyon-store] pred[%d] size=%zu map_size=%zu\n", _L+1, _psz, g_procyon_pred.size()); } }
                    } else { static int _ec=0; if(_ec<4){_ec++; std::fprintf(stderr, "[procyon-diag] GEMV/copy FAIL L%d cuda=%s\n", _L, cudaGetErrorString(cudaGetLastError())); } }
                } else { static int _mc=0; if(_mc<4){_mc++; std::fprintf(stderr, "[procyon-diag] dlog-alloc FAIL L%d\n", _L); } }
            }
        }
    }

    const auto _ph_c = PH_NOW();   // host evaluation done, staging begins
    if (n_unique > tp.n_slots) {
        st.prefill++;
        g_walkthrough_active = (g_walkthrough_runs < pager_walkthrough_n());
        if (g_walkthrough_active) {
            g_walkthrough_runs++;
            std::fprintf(stderr,
              "\n[WALKTHROUGH] ======== prefill call #%d, tensor %s ========\n"
              "[WALKTHROUGH] VARIABLES\n"
              "[WALKTHROUGH]   n_expert      %lld      experts in the tensor (src0->ne[2])\n"
              "[WALKTHROUGH]   n_unique      %d      distinct ones in THIS call\n"
              "[WALKTHROUGH]   n_slots       %d      slots in this tensor's region\n"
              "[WALKTHROUGH]   slice         %zu B   src0->nb[2], bytes per expert slice\n"
              "[WALKTHROUGH]   scratch_bytes %zu B   located at vram_pool + 0\n"
              "[WALKTHROUGH]   region_bytes  %zu B   located at vram_pool + base_off\n"
              "[WALKTHROUGH]   base_off      %zu B   offset of THIS tensor\n"
              "[WALKTHROUGH]   src0->ne      [%lld %lld %lld %lld]\n"
              "[WALKTHROUGH]   src0->nb      [%zu %zu %zu %zu]\n"
              "[WALKTHROUGH]   src0->data    %p     (host_backing + tensor offset)\n"
              "[WALKTHROUGH]   vram_pool     %p\n"
              "[WALKTHROUGH]   ids->ne       [%lld %lld %lld %lld]  type=%s\n"
              "[WALKTHROUGH] STEPS\n",
              g_walkthrough_runs, src0->name ? src0->name : "(unnamed)",
              (long long) n_expert, n_unique, tp.n_slots, slice,
              rt.scratch_bytes, rt.region_bytes, tp.base_off,
              (long long) src0->ne[0], (long long) src0->ne[1], (long long) src0->ne[2], (long long) src0->ne[3],
              src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
              src0->data, ctx->vram_pool,
              (long long) ids->ne[0], (long long) ids->ne[1], (long long) ids->ne[2], (long long) ids->ne[3],
              ggml_type_name(ids->type));
        }
        SHOW(1096, "if (n_unique > n_slots)          %d > %d  -> prefill path", n_unique, tp.n_slots);

        const size_t need = (size_t) n_unique * slice;
        SHOW(1099, "need = n_unique * slice          %d * %zu = %zu B (%.1f MiB)",
              n_unique, slice, need, need / 1048576.0);
        SHOW(1100, "if (need > scratch_bytes)        %zu %s %zu  -> %s",
              need, need > rt.scratch_bytes ? ">" : "<=", rt.scratch_bytes,
              need > rt.scratch_bytes ? "FALLBACK (UVA)" : "continue in prefill");
        if (need > rt.scratch_bytes) PAGER_FALLBACK("prefill unique > scratch (increase FLE_PAGER_SCRATCH_MB)");

        uint8_t * scratch = (uint8_t *) ctx->vram_pool;  // scratch is at the pool start
        SHOW(1101, "scratch = vram_pool              %p   (regions only from +%zu)", (void*) scratch, tp.base_off);
        SHOW(1102, "for (e = 0; e < %lld; ++e)        copy loop", (long long) n_expert);

        int copied = 0, skipped = 0, sl_max = -1;
        size_t dst_max = 0, src_max = 0;
        for (int e = 0; e < (int) n_expert; ++e) {
            const int sl = rt.uniq_map[e];
            if (sl < 0) { skipped++; continue; }
            const size_t dst_off   = (size_t) sl * slice;
            const size_t src_off = (size_t) e  * slice;
            if (copied < 3 || sl == n_unique - 1) {
                SHOW(1105, "  e=%-4d sl=%-4d dst=scratch+%-12zu src=src0->data+%zu",
                      e, sl, dst_off, src_off);
            } else if (copied == 3) {
                SHOW(1105, "  ... (further copies not shown individually)");
            }
            if (sl > sl_max) { sl_max = sl; dst_max = dst_off; }
            if (src_off > src_max) src_max = src_off;
            copied++;
            if (cudaMemcpyAsync(scratch + dst_off,
                                (const uint8_t*) src0->data + src_off,
                                slice, cudaMemcpyHostToDevice, stream) != cudaSuccess)
                PAGER_FALLBACK("prefill H2D failed");
        }
        SHOW(1110, "loop done                      %d copied, %d skipped (sum %d = n_expert %lld)",
              copied, skipped, copied + skipped, (long long) n_expert);
        // CAUSE CHECK: does every staged expert sit where its remapped ID points?
        // Synchronize first — the copies above are async.
        if (std::getenv("FLE_PAGER_VERIFY")) {
            cudaStreamSynchronize(stream);
            for (int e = 0; e < (int) n_expert; ++e) {
                const int sl = rt.uniq_map[e];
                if (sl < 0) continue;
                pager_verify_mapping(scratch, (size_t) sl * slice, e, slice, src0, "PREFILL", -1);
            }
        }
        SHOW(1110, "  highest dst edge               scratch+%zu+%zu = %zu  vs scratch_bytes %zu  -> %s",
              dst_max, slice, dst_max + slice, rt.scratch_bytes,
              (dst_max + slice) <= rt.scratch_bytes ? "fits" : "OVERFLOW");
        SHOW(1110, "  highest src edge               src0->data+%zu+%zu = %zu  vs ggml_nbytes(src0) %zu  -> %s",
              src_max, slice, src_max + slice, ggml_nbytes(src0),
              (src_max + slice) <= ggml_nbytes(src0) ? "fits" : "OVERFLOW");

        int id_min = 1 << 30, id_max = -1, id_negative = 0, id_count = 0;
        PAGER_FOR_EACH_ID({
            const int32_t e = *pe;
            if (e >= 0 && e < n_expert) *pe = rt.uniq_map[e];
            const int32_t after_remap = *pe;
            id_count++;
            if (after_remap < 0) id_negative++;
            else { if (after_remap < id_min) id_min = after_remap; if (after_remap > id_max) id_max = after_remap; }
        })
        SHOW(1111, "IDs remapped                    %d items, range %d..%d, of which negative %d",
              id_count, id_min == (1<<30) ? -1 : id_min, id_max, id_negative);
        SHOW(1111, "  check: id_max must be < n_unique         %d < %d  -> %s",
              id_max, n_unique, id_max < n_unique ? "ok" : "ERROR");
        SHOW(1111, "  check: negative must be 0                %d      -> %s",
              id_negative, id_negative == 0 ? "ok" : "ERROR (ID points to no slot)");

        if (cudaMemcpyAsync(rt.d_buf, rt.h_buf, nbytes, cudaMemcpyHostToDevice, stream) != cudaSuccess)
            PAGER_FALLBACK("remap H2D failed");
        SHOW(1114, "remap H2D                        %zu B to d_buf", nbytes);
        *ids_remapped_dev = (const int32_t *) rt.d_buf;
        if (std::getenv("FLE_PAGER_IDSYNC")) cudaStreamSynchronize(stream);
        SHOW(1116, "ids_remapped_dev = d_buf         %p", (void*) rt.d_buf);
        {
            const size_t claimed   = ggml_nbytes(src0);
            const size_t available = rt.scratch_bytes;
            const size_t used      = (size_t) n_unique * slice;
            SHOW(1118, "return scratch                   caller sets src0.data=scratch, ne[2] STAYS %lld",
                  (long long) src0->ne[2]);
            SHOW(1118, "  CHECK tensor claim             ggml_nbytes(src0) = %zu B (%.1f MiB)",
                  claimed, claimed / 1048576.0);
            SHOW(1118, "  CHECK scratch available        scratch_bytes     = %zu B (%.1f MiB)",
                  available, available / 1048576.0);
            SHOW(1118, "  CHECK actually used            n_unique * slice  = %zu B (%.1f MiB)",
                  used, used / 1048576.0);
            SHOW(1118, "  ==> %s",
                  claimed > available
                    ? "OVERFLOW: the tensor claims more than the scratch holds"
                    : "the scratch holds the tensor's claim");
            if (claimed > available) {
                static int _w = 0;
                if (_w < 8) { _w++;
                    std::fprintf(stderr,
                      "[fle-pager] SCRATCH-OVERFLOW %s: ne[2]=%lld claims %zu B, "
                      "scratch holds %zu B (only %zu B are used for %d unique). "
                      "A kernel that iterates over the tensor instead of the remapped ids "
                      "reads %zu B into foreign regions.\n",
                      src0->name ? src0->name : "(unnamed)", (long long) src0->ne[2],
                      claimed, available, used, n_unique, claimed - available);
                }
            }
        }
        if (g_walkthrough_active) {
            std::fprintf(stderr, "[WALKTHROUGH] ======== end of call #%d ========\n\n", g_walkthrough_runs);
            g_walkthrough_active = false;
        }
        // The scratch holds n_unique experts, not n_expert. Without this information the tensor
        // keeps claiming ne[2]=n_expert, and ggml_nbytes() returns a size that reaches beyond the
        // scratch (measured: 891289600 B claimed against 536870912 B available = 338 MiB into
        // foreign regions).
        if (n_experts_out) *n_experts_out = n_unique;
        stat_log();
        return scratch;
    }

    // ── Decode path: persistent region, LRU per UNIQUE (not per element) ──
    for (int e = 0; e < (int) n_expert; ++e) {   // pass 1: pin residents
        if (rt.uniq_map[e] >= 0 && tp.e2s[e] >= 0) {
            tp.slots[tp.e2s[e]].stamp = rt.stamp;
            tp.slots[tp.e2s[e]].last  = rt.clock;
        }
    }
    st.decode++;

    // Measurement point at decode entry. Answers: which tensor, which slice size,
    // how many slots in the same byte budget, how many distinct experts — and
    // which mul_mat_id kernel ggml picks from that. The boundary ne2<=8 (mmvq) versus ne2>8
    // (mmq) is exactly where the fault flips; so far it only lived in a note, not in the run.
    g_walkthrough_dec = (g_walkthrough_dec_runs < pager_walkthrough_dec_n());
    if (g_walkthrough_dec) {
        g_walkthrough_dec_runs++;
        const long long ne2_src1 = src1 ? (long long) src1->ne[2] : -1;
        std::fprintf(stderr,
            "[DECODE] ---- %d ----  %s\n"
            "[DECODE]   slice        %zu B   n_slots %d   base_off %zu   region_bytes %zu\n"
            "[DECODE]   n_expert     %lld   n_unique %d\n"
            "[DECODE]   src1->ne[2]  %lld  -> kernel %s   (limit MMVQ_MAX_BATCH_SIZE=8)\n",
            g_walkthrough_dec_runs, src0->name ? src0->name : "(unnamed)",
            slice, tp.n_slots, tp.base_off, rt.region_bytes,
            (long long) n_expert, n_unique,
            ne2_src1, (ne2_src1 >= 0 && ne2_src1 <= 8) ? "mmvq" : "mmq");
    }
    const bool cpu_miss_mode = pager_cpu_miss();
    const bool keep_stage    = cpu_miss_mode && std::getenv("FLE_PAGER_CPU_MISS_KEEP_STAGE");  // debug: stage AND override
    const int  cpu_miss_T    = cpu_miss_mode ? pager_cpu_miss_freq_t() : 0;                    // hybrid threshold: freq>=T -> GPU, <T -> CPU
    const bool hybrid        = cpu_miss_mode && !keep_stage && cpu_miss_T < (1 << 30);         // stage hot experts, cold ones on CPU
    if (cpu_miss_mode) {                          // capture was_miss BEFORE staging (positions arrive in the value replace)
        if ((int64_t) rt.cpu_wasmiss.size() < n_expert) rt.cpu_wasmiss.assign((size_t) n_expert, 0);
        else std::fill(rt.cpu_wasmiss.begin(), rt.cpu_wasmiss.begin() + n_expert, 0);
    }
    tp.calls++;                                   // working-set measurement: decode calls of this tensor
    for (int e = 0; e < (int) n_expert; ++e) {   // S1 telemetry: COUNT hit/miss before staging
        if (rt.uniq_map[e] < 0) continue;
        st.routed++;
        tp.freq[e]++;                             // usage intensity (reuse-aware eviction)
        const uint64_t rd = tp.seen_call[e] ? (tp.calls - tp.seen_call[e]) : 0;  // reuse distance (tensor calls)
        tp.seen_call[e] = tp.calls;
        if (tp.e2s[e] >= 0) { st.hits++; continue; }
        st.misses++;                              // miss nature: evicted before (reuse=thrash) vs never had (cold)
        if (tp.evicted_at[e] > 0) {
            st.reuse_miss++;
            if (rd > 0) { st.rd_sum += rd; st.rd_cnt++; if ((int64_t) rd <= tp.n_slots) st.rd_fit++; }  // <=slots -> policy, >slots -> capacity
        } else st.cold_miss++;
        if (cpu_miss_mode) rt.cpu_wasmiss[e] = 1; // this expert was a miss -> CPU compute candidate
    }
    // S3: CPU miss mode -> do NOT stage misses (compute them on the CPU), otherwise LRU H2D as before.
    // keep_stage (debug) = stage anyway + override, to separate override correctness from skip-staging.
    rt.cpu_misses.clear();

    if (cpu_miss_mode) {
        static bool _s3_once = false;
        if (!_s3_once && std::getenv("FLE_PAGER_S3_DEBUG")) {
            _s3_once = true;
            std::fprintf(stderr,
                "[S3-PRODUCER] cpu_miss=1 keep_stage=%d hybrid=%d cpu_miss_T=%d"
                "  -> staging block %s\n",
                (int) keep_stage, (int) hybrid, cpu_miss_T,
                (!cpu_miss_mode || keep_stage || hybrid) ? "RUNS" : "skipped");
        }
    }
    if (!cpu_miss_mode || keep_stage || hybrid) {
        // S2 substrate: miss staging optionally on the copy stream. WAR guard: the copy waits for the previous compute
        // (region is reused across tokens) before it writes into it. RAW guard after the loop.
        const bool async_stage = pager_async_stage() && rt.ensure_copy();
        cudaStream_t sstream = async_stage ? rt.copy_stream : stream;
        if (async_stage) {
            cudaEventRecord(rt.compute_ev, stream);              // captured: all compute so far (incl. previous matmul)
            cudaStreamWaitEvent(rt.copy_stream, rt.compute_ev, 0);
        }
        bool staged_any = false;
        unsigned long long _bw_h2d = 0, _bw_d2d = 0;
        if (g_bw_on) {
            // Collect the result of the PREVIOUS call — non-blocking, so the overlap
            // stays untouched. Not finished yet? Discard the sample and count it.
            if (bw_cur().pending) {
                if (cudaEventQuery(bw_cur().ev_b) == cudaSuccess) {
                    float _ms = 0.0f;
                    if (cudaEventElapsedTime(&_ms, bw_cur().ev_a, bw_cur().ev_b) == cudaSuccess) {
                        bw_cur().pcie_ms += _ms; bw_cur().h2d_bytes += bw_cur().pending_bytes; bw_cur().pcie_samples++;
                    }
                } else { bw_cur().pcie_skipped++; }
                bw_cur().pending = false;
            }
            if (!bw_cur().ev_a) {   // WITH timing, unlike rt.stage_ev
                if (cudaEventCreate(&bw_cur().ev_a) != cudaSuccess) bw_cur().ev_a = nullptr;
                if (cudaEventCreate(&bw_cur().ev_b) != cudaSuccess) bw_cur().ev_b = nullptr;
            }
            if (bw_cur().ev_a) cudaEventRecord(bw_cur().ev_a, sstream);
        }
        // ── q*-split: share instead of threshold ────────────────────────────────────
        // Empty = switch off OR too few measurements -> the old threshold logic applies.
        std::vector<char> _split_stage;
        if (g_split_on && hybrid) {
            const double _bp = bw_cur().pcie_ms > 0.0 ? (double) bw_cur().h2d_bytes / bw_cur().pcie_ms : 0.0;
            const double _bc = bw_cur().cpu_ms  > 0.0 ? (double) bw_cur().cpu_bytes / bw_cur().cpu_ms  : 0.0;
            if (bw_cur().pcie_samples >= g_split_min_n && bw_cur().cpu_calls >= g_split_min_n && _bp > 0.0 && _bc > 0.0) {
                // Collect misses together with their frequency (the order is preserved:
                // hot -> bus -> stays resident; cold -> CPU -> occupies nothing).
                std::vector<std::pair<uint32_t,int>> _miss;
                for (int e = 0; e < (int) n_expert; ++e)
                    if (rt.uniq_map[e] >= 0 && tp.e2s[e] < 0) _miss.push_back({tp.freq[e], e});
                if (!_miss.empty()) {
                    const double _q = _bp / (_bp + _bc);          // q* = B_P / (B_P + B_H)
                    size_t _n = (size_t) ((double) _miss.size() * _q + 0.5);
                    if (_n > _miss.size()) _n = _miss.size();
                    std::partial_sort(_miss.begin(), _miss.begin() + _n, _miss.end(),
                                      [](const std::pair<uint32_t,int>& a, const std::pair<uint32_t,int>& b){ return a.first > b.first; });
                    _split_stage.assign(n_expert, 0);
                    for (size_t i = 0; i < _n; ++i) _split_stage[_miss[i].second] = 1;
                    if (bw_cur().pcie_samples % 4096 == 0)
                        std::fprintf(stderr, "[fle-pager] SPLIT q*=%.3f -> %zu of %zu misses over the bus "
                                             "(B_P=%.2f B_H=%.2f GB/s, n=%llu/%llu)\\n",
                                     _q, _n, _miss.size(),
                                     _bp * 1e3 / (1024.0*1024.0*1024.0), _bc * 1e3 / (1024.0*1024.0*1024.0),
                                     bw_cur().pcie_samples, bw_cur().cpu_calls);
                }
            }
        }
        static int _ledger_n = [] { const char * v = getenv_alias("FLE_PAGER_LEDGER", "FLE_PAGER_BILANZ");   // deprecated alias
                                  return (v && *v) ? std::atoi(v) : 0; }();
        static int _ledger_runs = 0;
        const bool ledger = (_ledger_runs < _ledger_n);
        int b_needed = 0, b_resident = 0, b_evicted = 0, b_filled = 0, b_free = 0;
        if (ledger) {
            for (int e = 0; e < (int) n_expert; ++e) {
                if (rt.uniq_map[e] >= 0) { b_needed++; if (tp.e2s[e] >= 0) b_resident++; }
            }
            for (int sl = 0; sl < tp.n_slots; ++sl) if (tp.slots[sl].expert < 0) b_free++;
        }
        for (int e = 0; e < (int) n_expert; ++e) {   // pass 2: stage the misses
            if (rt.uniq_map[e] < 0 || tp.e2s[e] >= 0) continue;
            // With a quota the share decides, without a quota the old threshold.
            if (!_split_stage.empty()) { if (!_split_stage[e]) continue; }          // not in the bus quota -> CPU
            else if (hybrid && tp.freq[e] < cpu_miss_T) continue;   // cold (freq<T) -> CPU compute, do NOT stage
            const bool freq_mode = pager_evict_freq();
            int victim = -1; uint64_t best_last = ~0ull; uint32_t best_freq = ~0u;
            for (int sl = 0; sl < tp.n_slots; ++sl) {
                if (tp.slots[sl].stamp == rt.stamp) continue;             // pinned (this call)
                if (tp.slots[sl].expert < 0) { victim = sl; break; }
                if (freq_mode) {                                          // min freq, tiebreak min last (reuse-aware)
                    const uint32_t f = tp.freq[tp.slots[sl].expert];
                    if (f < best_freq || (f == best_freq && tp.slots[sl].last < best_last)) {
                        best_freq = f; best_last = tp.slots[sl].last; victim = sl;
                    }
                } else if (tp.slots[sl].last < best_last) {               // pure LRU
                    best_last = tp.slots[sl].last; victim = sl;
                }
            }
            if (victim < 0) PAGER_FALLBACK("no victim despite decode path");
            SlotInfo& v = tp.slots[victim];
            if (v.expert >= 0) {
                tp.e2s[v.expert] = -1; tp.evicted_at[v.expert] = rt.clock; tp.freq[v.expert] = 0;
                // The outflow. Until 2026-08-26 only a comment "unregister + eviction" stood here —
                // it happened, but was never counted. That made it impossible to tell
                // whether 28 % misses are one-off or whether the same experts are carried in
                // a circle. Both report the same hit rate.
                st.evicts++; st.evict_bytes += (unsigned long long) slice;
                if (ledger) b_evicted++;
            }  // unregister + eviction time + freq reset
            // Swap-arena hit: the prefetch already loaded this expert into the exchange arena ->
            // fast intra-VRAM D2D swap->region instead of host->region H2D (fetch hidden behind compute).
            const uint8_t* csrc = (const uint8_t*) src0->data + (size_t) e * slice;
            cudaMemcpyKind ckind = cudaMemcpyHostToDevice;
            if (rt.swap_bytes) {
                auto sm = rt.swap_map.find(src0->data);
                if (sm != rt.swap_map.end()) {
                    auto se = sm->second.find(e);
                    if (se != sm->second.end()) { csrc = (const uint8_t*) ctx->vram_pool + se->second; ckind = cudaMemcpyDeviceToDevice; }  // swap hit: real miss, just D2D-fast
                }
            }
            // Bounce buffer only for H2D: D2D from the swap arena is already in VRAM
        // and does not need it (and must not use it, the memcpy would miss).
        bool _via_bounce = false;
        if (ckind == cudaMemcpyHostToDevice && pager_bounce() && rt.ensure_bounce(slice)) {
            const int _bi = rt.bounce_i;
            // Reuse the buffer only once its previous copy has finished.
            if (rt.bounce_ev[_bi]) cudaEventSynchronize(rt.bounce_ev[_bi]);
            std::memcpy(rt.bounce[_bi], csrc, slice);
            if (cudaMemcpyAsync(region + (size_t) victim * tp.slot_bytes, rt.bounce[_bi],
                                slice, cudaMemcpyHostToDevice, sstream) != cudaSuccess)
                PAGER_FALLBACK("bounce H2D failed");
            cudaEventRecord(rt.bounce_ev[_bi], sstream);
            rt.bounce_i = (_bi + 1) % PagerRuntime::BOUNCE_N;
            _via_bounce = true;
        }
        if (!_via_bounce)
        if (cudaMemcpyAsync(region + (size_t) victim * tp.slot_bytes, csrc, slice, ckind, sstream) != cudaSuccess) PAGER_FALLBACK("decode stage failed");
            if (g_bw_on) { if (ckind == cudaMemcpyHostToDevice) _bw_h2d += slice; else _bw_d2d += slice; }
            v.expert = e; tp.e2s[e] = victim; v.stamp = rt.stamp; v.last = rt.clock;
            st.fills++; st.fill_bytes += (unsigned long long) slice;   // counterpart of st.evicts
            if (ledger) b_filled++;
            staged_any = true;
        }
        if (ledger) {
            _ledger_runs++;
            // Recompute: is the bookkeeping consistent after pass 2?
            int n_e2s = 0, n_slots_used = 0, n_duplicate = 0, n_orphan = 0;
            std::vector<int> back((size_t) tp.n_slots, -1);
            for (int e = 0; e < (int) n_expert; ++e) {
                const int sl = tp.e2s[e];
                if (sl < 0) continue;
                n_e2s++;
                if (sl >= tp.n_slots) { n_orphan++; continue; }
                if (back[sl] >= 0) n_duplicate++;      // two experts on the SAME slot
                back[sl] = e;
            }
            for (int sl = 0; sl < tp.n_slots; ++sl) if (tp.slots[sl].expert >= 0) n_slots_used++;
            // Cross-check: does slots[sl].expert point to the same expert as e2s?
            int n_inconsistent = 0;
            for (int sl = 0; sl < tp.n_slots; ++sl) {
                const int e_slot = tp.slots[sl].expert;
                if (e_slot < 0) continue;
                if (e_slot >= (int) n_expert || tp.e2s[e_slot] != sl) n_inconsistent++;
            }
            int missing = 0;
            for (int e = 0; e < (int) n_expert; ++e)
                if (rt.uniq_map[e] >= 0 && tp.e2s[e] < 0) missing++;
            std::fprintf(stderr,
              "[LEDGER %d] %s  n_slots=%d\n"
              "  needed %d · of which resident %d · slots free %d\n"
              "  evicted %d · filled %d · balance %+d\n"
              "  e2s points to %d slots · slots occupied %d%s\n"
              "  DUPLICATE %d%s · orphaned %d%s · inconsistent %d%s · still missing %d%s\n",
              _ledger_runs, src0->name ? src0->name : "?", tp.n_slots,
              b_needed, b_resident, b_free,
              b_evicted, b_filled, b_filled - b_evicted,
              n_e2s, n_slots_used, n_e2s == n_slots_used ? "" : "   <== MISMATCH",
              n_duplicate,  n_duplicate  ? "  <== TWO EXPERTS ON ONE SLOT" : "",
              n_orphan, n_orphan ? "  <== e2s points outside" : "",
              n_inconsistent,  n_inconsistent  ? "  <== slots[] and e2s contradict each other" : "",
              missing,      missing      ? "  <== needed but not staged" : "");
        }
        if (g_bw_on) {
            bw_cur().d2d_bytes += _bw_d2d;                  // immediately: no bus, no time question
            if (bw_cur().ev_b && _bw_h2d > 0) { cudaEventRecord(bw_cur().ev_b, sstream); bw_cur().pending = true; bw_cur().pending_bytes = _bw_h2d; }
        }
        if (async_stage && staged_any) {
            cudaEventRecord(rt.stage_ev, rt.copy_stream);        // staging done ...
            cudaStreamWaitEvent(stream, rt.stage_ev, 0);         // ... compute stream waits on it before mul_mat_id
        }
    }

    if (cpu_miss_mode) {
        // mmq.cu:207 crash fix (2026-07-20): MMQ invariant = ids per token UNIQUE. Collapsing all misses onto
        // slot 0 overflows the compact sort (fresh server: NOTHING admitted -> all 6 ids identical). Miss dummies
        // rotate over UNPINNED slots (stamp != rt.stamp): never collide with really staged ones of this call,
        // unique per token as long as n_slots > n_expert_used. The result of the dummy slots is garbage ->
        // finalize overwrites the dst column (proven with KEEP_STAGE).
        int32_t dummy_rot = -1;
        PAGER_FOR_EACH_ID({                      // cold/not staged -> CPU override; hot (staged, e2s>=0) -> GPU slot
            const int32_t e = *pe;
            if (e >= 0 && e < n_expert) {
                if (rt.cpu_wasmiss[e] && (keep_stage || tp.e2s[e] < 0)) rt.cpu_misses.push_back({(int) i0, (int) i1, e});
                if (tp.e2s[e] >= 0) { *pe = tp.e2s[e]; }
                else {
                    for (int _t = 0; _t < tp.n_slots; ++_t) {
                        dummy_rot = (dummy_rot + 1) % tp.n_slots;
                        if (tp.slots[dummy_rot].stamp != rt.stamp) break;
                    }
                    *pe = (dummy_rot >= 0) ? dummy_rot : 0;
                }
            }
        })
        if (std::getenv("FLE_PAGER_S3_DEBUG")) {
            static long _n = 0, _with_entries = 0, _total = 0;
            _n++; if (!rt.cpu_misses.empty()) { _with_entries++; _total += (long) rt.cpu_misses.size(); }
            if (_n == 1 || _n == 500 || _n == 5000)
                std::fprintf(stderr, "[S3-PRODUCER] after %ld calls: %ld with entries, %ld misses total\n",
                             _n, _with_entries, _total);
        }
    } else {
        int not_staged = 0;
        PAGER_FOR_EACH_ID({                      // replace values in the clone (standard LRU path)
            const int32_t e = *pe;
            if (e >= 0 && e < n_expert) {
                // e2s[e] == -1 means "not staged". Written unchecked, the kernel would read
                // at base + (-1)*slice — outside every region. The cpu_miss path above
                // catches exactly that; here it was missing.
                if (tp.e2s[e] < 0) { not_staged++; *pe = 0; }
                else *pe = tp.e2s[e];
            }
        })
        if (not_staged > 0) {
            static int _w = 0;
            if (_w < 12) { _w++;
                std::fprintf(stderr,
                  "[fle-pager] NOT STAGED: %d of %lld ids pointed to e2s=-1 (%s, "
                  "n_slots=%d, slice=%zu). Unchecked, the kernel would have read at base-%zu.\n",
                  not_staged, (long long) ggml_nelements(ids),
                  src0->name ? src0->name : "?", tp.n_slots, slice, slice);
            }
        }
    }
    #undef PAGER_FOR_EACH_ID

    // S3 OVERLAP preparation: D2H + quantize the activations of the cold misses NOW (before mmvq, stream synced),
    // so finalize computes without sync and the CPU vec_dot overlaps the GPU mmvq kernel. Dedup of identical (iex,token).
    rt.cpu_pre_stride = 0;
    if (cpu_miss_mode && !rt.cpu_misses.empty() && src1) {
        const int64_t Kc = src0->ne[0];
        const struct ggml_type_traits_cpu * trc = pager_type_traits_cpu(src0->type);
        const enum ggml_type vdtc = trc ? trc->vec_dot_type : GGML_TYPE_F32;
        const struct ggml_type_traits_cpu * vtrc = pager_type_traits_cpu(vdtc);
        const size_t nm = rt.cpu_misses.size();
        if (trc && vtrc && vtrc->from_float && Kc > 0 && rt.ensure_xin(nm * (size_t) Kc)) {
            const size_t xqb = ggml_row_size(vdtc, Kc);
            rt.cpu_pre_stride = xqb; rt.cpu_pre_K = Kc; rt.cpu_pre_vdt = (int) vdtc;
            rt.cpu_xq_pre.resize(nm * xqb);
            bool ok = true;                                       // 1. all activations async D2H into the pinned batch (no per-miss sync)
            for (size_t i = 0; i < nm && ok; ++i) {
                const auto & m = rt.cpu_misses[i];
                const int64_t s1e = (src1->ne[1] > 1) ? (int64_t) m.iex : 0;
                const uint8_t * xdev = (const uint8_t *) src1->data + (size_t) s1e * src1->nb[1] + (size_t) m.token * src1->nb[2];
        if (std::getenv("FLE_PAGER_VALIDATE")) {
            const uint8_t* _s1b = (const uint8_t*) src1->data; const size_t _s1n = ggml_nbytes(src1);
            if (xdev < _s1b || xdev + (size_t) Kc * sizeof(float) > _s1b + _s1n) {
                std::fprintf(stderr, "[fle-pager] SRC1-OOB i=%zu iex=%d token=%d Kc=%ld off=%zu s1bytes=%zu ne1=%ld ne2=%ld nb1=%zu nb2=%zu\n",
                    i, m.iex, m.token, (long) Kc, (size_t)(xdev - _s1b), _s1n, (long) src1->ne[1], (long) src1->ne[2], src1->nb[1], src1->nb[2]);
                ok = false; break;
            }
        }
                if (cudaMemcpyAsync(rt.cpu_x_pin + i * (size_t) Kc, xdev, (size_t) Kc * sizeof(float), cudaMemcpyDeviceToHost, stream) != cudaSuccess) ok = false;
            }
            if (!ok || cudaStreamSynchronize(stream) != cudaSuccess) { rt.cpu_pre_stride = 0; }  // 2. ONE sync (before mmvq)
            else {
                #pragma omp parallel for schedule(static) if(nm >= 4)                             // 3. quantize in parallel
                for (int i = 0; i < (int) nm; ++i)
                    vtrc->from_float(rt.cpu_x_pin + (size_t) i * (size_t) Kc, rt.cpu_xq_pre.data() + (size_t) i * xqb, (int) Kc);
            }
        }
    }

    // Control number where the prefill and decode paths converge again: highest
    // slot actually occupied against the region edge. With ONE slice size nothing can show
    // up here — n_slots is computed from the same size. With mixed sizes two tensors have
    // different n_slots in the same byte budget. Without this line "the slot was inside the edge"
    // is an assumption, not a measurement.
    if (g_walkthrough_dec) {
        int highest = -1, used = 0;
        for (int e = 0; e < (int) n_expert; ++e)
            if (rt.uniq_map[e] >= 0 && tp.e2s[e] >= 0) { used++; if (tp.e2s[e] > highest) highest = tp.e2s[e]; }
        const size_t edge = (highest >= 0) ? ((size_t) highest + 1) * slice : 0;
        std::fprintf(stderr,
            "[DECODE]   slots        %d of %d unique occupied   highest slot %d of %d\n"
            "[DECODE]   CHECK        (highest+1)*slice = %zu B  vs region_bytes %zu B  -> %s\n",
            used, n_unique, highest, tp.n_slots - 1,
            edge, rt.region_bytes, edge <= rt.region_bytes ? "fits" : "OVER THE EDGE");
        g_walkthrough_dec = false;
    }

    const auto _ph_d = PH_NOW();   // staging done, remap begins
    if (pager_device_remap() && !cpu_miss_mode) {
        // Step i: e2s->slot_map_dev + device gather instead of host value replace + H2D. No host sync -> graph-legal.
        // (The D2H + sync above is kept in step i for staging; step ii replaces it via the early-routing pre-step.)
        if (!rt.ensure_slotmap((size_t) n_expert)) PAGER_FALLBACK("slot_map alloc failed");
        if (cudaMemcpyAsync(rt.slot_map_dev, tp.e2s.data(), (size_t) n_expert * sizeof(int32_t), cudaMemcpyHostToDevice, stream) != cudaSuccess) PAGER_FALLBACK("slot_map H2D failed");
        if (cudaMemcpyAsync(rt.d_buf, ids->data, nbytes, cudaMemcpyDeviceToDevice, stream) != cudaSuccess) PAGER_FALLBACK("ids D2D failed"); // layout + gaps preserved, kernel overwrites valid elements
        const int64_t total = ids->ne[0]*ids->ne[1]*ids->ne[2]*ids->ne[3];
        const int threads = 256;
        int64_t nblk = (total + threads - 1) / threads; if (nblk > 65535) nblk = 65535; if (nblk < 1) nblk = 1;
        fle_pager_remap_kernel<<<(int) nblk, threads, 0, stream>>>(
            (const uint8_t*) ids->data, (uint8_t*) rt.d_buf, rt.slot_map_dev, (int) n_expert,
            ids->ne[0], ids->ne[1], ids->ne[2], ids->ne[3], ids->nb[0], ids->nb[1], ids->nb[2], ids->nb[3]);
    } else if (cudaMemcpyAsync(rt.d_buf, rt.h_buf, nbytes, cudaMemcpyHostToDevice, stream) != cudaSuccess) PAGER_FALLBACK("remap H2D failed");

    // ── CLGP predictive prefetch: prefetch the next tensor's last unique experts
    //    async on copy_stream -> overlaps the mmvq that follows; the next stage waits
    //    on prefetch_ev. Attacks the reuse misses (recurring experts resident BEFORE they are needed). ──
    if (prefetch_on) {
        tp.last_uniq.clear();                                   // (a) prediction for this tensor's next token
        for (int e = 0; e < (int) n_expert; ++e) if (rt.uniq_map[e] >= 0) tp.last_uniq.push_back(e);
        auto nit = rt.next_ptr.find(src0->data);               // (b) next tensor from the learned chain
        if (nit != rt.next_ptr.end()) {
            auto pit = rt.pools.find(nit->second);
            if (pit != rt.pools.end() && pit->second.self && !pit->second.last_uniq.empty()) {
                TensorPool& np = pit->second;
                cudaEventRecord(rt.compute_ev, stream);         // WAR: target may still be read by the previous compute
                cudaStreamWaitEvent(rt.copy_stream, rt.compute_ev, 0);
                bool any = false;
                if (rt.swap_bytes) {
                    // ── Prefetch into the SWAP ARENA (host->swap, async) — resident regions stay clean
                    //    (speculative prefetch evicts no hot residents). Decode promotes swap->region D2D. ──
                    rt.swap_next = 0; rt.swap_map.clear();       // new round: only the current predictions count
                    auto& smap = rt.swap_map[np.self];
                    for (int32_t e : np.last_uniq) {
                        if (e < 0 || e >= (int) np.e2s.size() || np.e2s[e] >= 0) continue;  // already resident -> no prefetch
                        if (rt.swap_next + np.slot_bytes > rt.swap_bytes) break;            // arena full
                        const size_t soff = rt.swap_off + rt.swap_next;
                        if (cudaMemcpyAsync((uint8_t*) ctx->vram_pool + soff,
                                            (const uint8_t*) np.self + (size_t) e * np.slot_bytes,
                                            np.slot_bytes, cudaMemcpyHostToDevice, rt.copy_stream) != cudaSuccess) break;
                        smap[e] = soff; rt.swap_next += np.slot_bytes; any = true;
                    }
                } else {
                    // ── Old path: in-place into the target region (may evict residents) ──
                    uint8_t* nregion = (uint8_t*) ctx->vram_pool + np.base_off;
                    for (int32_t e : np.last_uniq) {
                        if (e < 0 || e >= (int) np.e2s.size() || np.e2s[e] >= 0) continue;  // already resident
                        int victim = -1; uint64_t best = ~0ull;
                        for (int sl = 0; sl < np.n_slots; ++sl) {
                            if (np.slots[sl].expert < 0) { victim = sl; break; }
                            if (np.slots[sl].last < best) { best = np.slots[sl].last; victim = sl; }
                        }
                        if (victim < 0) continue;
                        SlotInfo& v = np.slots[victim];
                        if (v.expert >= 0) { np.e2s[v.expert] = -1; np.evicted_at[v.expert] = rt.clock; }
                        if (cudaMemcpyAsync(nregion + (size_t) victim * np.slot_bytes,
                                            (const uint8_t*) np.self + (size_t) e * np.slot_bytes,
                                            np.slot_bytes, cudaMemcpyHostToDevice, rt.copy_stream) != cudaSuccess) break;
                        v.expert = e; np.e2s[e] = victim; v.last = rt.clock; any = true;
                    }
                }
                if (any) { cudaEventRecord(rt.prefetch_ev, rt.copy_stream); rt.prefetch_pending = true; }
            }
        }
    }

    *ids_remapped_dev = (const int32_t *) rt.d_buf;
    stat_log();
    // The same gap as in the prefill path, only on the MAIN path (measured:
    // decode=2043 against prefill=5). The region holds tp.n_slots experts, the tensor
    // keeps claiming ne[2]=n_expert. Measured at blk.2.ffn_down_exps:
    //     ggml_nbytes(src0) = 891289600 B    region_bytes = 285212672 B
    //     -> 606 MiB that no kernel may read, but any could.
    if (std::getenv("FLE_PAGER_IDSYNC")) cudaStreamSynchronize(stream);
    if (pager_phases()) {
        const auto _ph_e = PH_NOW();
        g_phases.ids_d2h_sync += PH_MS(_ph_a, _ph_b);
        g_phases.host_scan    += PH_MS(_ph_b, _ph_c);
        g_phases.staging      += PH_MS(_ph_c, _ph_d);
        g_phases.remap_rest   += PH_MS(_ph_d, _ph_e);
        g_phases.n++;
        if (g_phases.n % st_win == 0) {   // st_win = FLE_PAGER_STAT_EVERY, defined earlier in this function
            const double tot_ms = g_phases.ids_d2h_sync + g_phases.host_scan
                             + g_phases.staging + g_phases.remap_rest;
            const double n = (double) g_phases.n;
            std::fprintf(stderr,
                "[fle-phases] %llu calls, %.1f ms total = %.1f us per call\n"
                "[fle-phases]   ids D2H + sync   %8.1f ms  %6.1f us/call  %5.1f %%\n"
                "[fle-phases]   host evaluation  %8.1f ms  %6.1f us/call  %5.1f %%\n"
                "[fle-phases]   staging (H2D)    %8.1f ms  %6.1f us/call  %5.1f %%\n"
                "[fle-phases]   remap + rest     %8.1f ms  %6.1f us/call  %5.1f %%\n",
                g_phases.n, tot_ms, tot_ms * 1000.0 / n,
                g_phases.ids_d2h_sync, g_phases.ids_d2h_sync * 1000.0 / n, 100.0 * g_phases.ids_d2h_sync / tot_ms,
                g_phases.host_scan,    g_phases.host_scan    * 1000.0 / n, 100.0 * g_phases.host_scan    / tot_ms,
                g_phases.staging,      g_phases.staging      * 1000.0 / n, 100.0 * g_phases.staging      / tot_ms,
                g_phases.remap_rest,   g_phases.remap_rest   * 1000.0 / n, 100.0 * g_phases.remap_rest   / tot_ms);
        }
    }
    if (n_experts_out) *n_experts_out = tp.n_slots;
    // CAUSE CHECK on the MAIN path (decode=2043 against prefill=5).
    if (std::getenv("FLE_PAGER_VERIFY")) {
        cudaStreamSynchronize(stream);
        for (int e = 0; e < (int) n_expert; ++e) {
            if (rt.uniq_map[e] < 0 || tp.e2s[e] < 0) continue;
            pager_verify_mapping(region, (size_t) tp.e2s[e] * tp.slot_bytes, e, slice, src0, "DECODE", -1);
        }
    }
    {
        const size_t claimed = ggml_nbytes(src0);
        if (claimed > rt.region_bytes) {
            static int _w = 0;
            if (_w < 4) { _w++;
                std::fprintf(stderr,
                  "[fle-pager] REGION-OVERFLOW %s: ne[2]=%lld claims %zu B, region "
                  "holds %zu B (n_slots=%d) -> %zu B over the edge.\n",
                  src0->name ? src0->name : "(unnamed)", (long long) src0->ne[2],
                  claimed, rt.region_bytes, tp.n_slots, claimed - rt.region_bytes);
            }
        }
    }
    return region;  // region base: slot*nb02 stays inside the region
}

// ── S3: CPU miss finalize ────────────────────────────────────────────────────────────────────
// Called AFTER the GPU mul_mat (wrapper): for every miss position (iex,token,expert) recorded in
// stage_and_remap, compute the expert projection on the CPU (ggml fused vec_dot against the
// packed quant codes in host_backing) and overwrite the dst column (the GPU computed garbage
// there from slot 0). Correctness first: synchronous; overlap is the later optimization.

// Evidence line per exit from pager_cpu_miss_finalize (2026-08-29).
// A diagnostic script wanted to find out which of the six silent exits triggers —
// and could not: FLE_PAGER_S3_DEBUG sat BEHIND all six. A check that
// is silent exactly when the case it should check occurs.
static void s3_exit(const char * wo) {
    static bool enabled = [] { const char * e = std::getenv("FLE_PAGER_S3_DEBUG"); return e && *e && *e != '0'; }();
    if (!enabled) return;
    static std::map<std::string, long> counters;
    static std::mutex m;
    std::lock_guard<std::mutex> g(m);
    long & n = counters[wo];
    if (++n == 1 || n == 1000 || n == 100000)
        std::fprintf(stderr, "[S3-EXIT] %s  (%ld. time)\n", wo, n);
}

FLE_EXPORT void pager_cpu_miss_finalize(ggml_backend_buffer_t buffer,
                                        const struct ggml_tensor * src0,
                                        const struct ggml_tensor * src1,
                                        struct ggml_tensor * dst,
                                        cudaStream_t stream) {
    if (!pager_cpu_miss() || !src0 || !src1 || !dst) { s3_exit("1: switch off or tensor missing"); return; }
    std::lock_guard<std::mutex> lock(g_pager_mtx);
    auto* ctx = (PagerBufferContext*) buffer->context;
    if (!ctx || !ctx->rt) { s3_exit("2: no context/runtime"); return; }
    PagerRuntime& rt = *ctx->rt;
    if (rt.cpu_misses.empty()) { s3_exit("3: no misses recorded"); return; }

    const int64_t K = src0->ne[0];    // contraction dim (n_embd for gate/up, n_ff for down)
    const int64_t N = dst->ne[0];     // output dim (n_ff for gate/up, n_embd for down)
    if (K <= 0 || N <= 0) { s3_exit("4: K or N <= 0"); rt.cpu_misses.clear(); return; }

    const struct ggml_type_traits_cpu * tr = pager_type_traits_cpu(src0->type);
    if (!tr || !tr->vec_dot) { s3_exit("5: no vec_dot for this type"); rt.cpu_misses.clear(); return; }
    const uint8_t * w_base = (const uint8_t *) src0->data;   // host_backing = CPU-readable

    // OVERLAP: the activations were pre-quantized in stage_and_remap (before mmvq) (rt.cpu_xq_pre) -> NO D2H/sync
    // here any more; the CPU vec_dot runs while the GPU mmvq kernel computes. Result goes into PINNED
    // slices (rt.cpu_out_pin) -> true async H2D (no reuse race: the next stage_and_remap ids sync drains).
    const bool overlap = (rt.cpu_pre_stride > 0 && rt.cpu_pre_K == K);
    if (!overlap || !rt.ensure_out(rt.cpu_misses.size() * (size_t) N)) {
        s3_exit(overlap ? "6b: ensure_out failed" : "6a: no overlap (cpu_pre_stride/cpu_pre_K do not match)");
        rt.cpu_misses.clear(); return; }

    // MULTI-THREAD: vec_dot per miss is independent (own expert/pinned slice/dst column) -> OpenMP parallel.
    // This is the core speedup: ggml -ot exps=CPU uses all threads; single-threaded capped S3 at 10.66.
    const int nmiss = (int) rt.cpu_misses.size();
    const int64_t nb1_w = src0->nb[1], nb2_w = src0->nb[2];
    if (std::getenv("FLE_PAGER_S3_DEBUG")) { static int _dbg=0; if (_dbg<3) { _dbg++;
        std::fprintf(stderr, "[S3DBG] K=%ld N=%ld | src0.ne=[%ld,%ld,%ld] nb=[%zu,%zu,%zu] type=%d | src1.ne=[%ld,%ld,%ld] nb=[%zu,%zu,%zu] | dst.ne=[%ld,%ld,%ld] nb=[%zu,%zu,%zu] | nmiss=%zu\n",
          (long)K,(long)N, (long)src0->ne[0],(long)src0->ne[1],(long)src0->ne[2], src0->nb[0],src0->nb[1],src0->nb[2],(int)src0->type,
          (long)src1->ne[0],(long)src1->ne[1],(long)src1->ne[2], src1->nb[0],src1->nb[1],src1->nb[2],
          (long)dst->ne[0],(long)dst->ne[1],(long)dst->ne[2], dst->nb[0],dst->nb[1],dst->nb[2], rt.cpu_misses.size());
        for (size_t i=0;i<rt.cpu_misses.size()&&i<4;i++){auto&m=rt.cpu_misses[i]; std::fprintf(stderr,"[S3DBG]   miss%zu iex=%d token=%d expert=%d\n",i,m.iex,m.token,m.expert);}
    }}
    // PARALLEL over (miss x row): flat enough for all threads even with 1-2 misses (N=2048-4096 rows).
    // The OpenMP team persists in the process -> only a team wake per call, amortized over >=2048 items.
    // Goal: CPU cold compute < GPU mmvq time -> hidden (the GPU is relieved of miss H2D anyway).
    const int64_t total_rc = (int64_t) nmiss * N;
    const auto _cpu_t0 = std::chrono::steady_clock::now();   // CPU work is synchronous -> wall clock is correct
    // Core limit for the CPU pool.
    //
    // There used to be no `num_threads` here, and OMP_NUM_THREADS was not set.
    // The block therefore took ALL cores — 32 on the test host, on top of llama's `-t 24`.
    // 56 threads on 32 cores, which defeated the requirement "do not fully occupy the CPU"
    // without anyone noticing.
    //
    // This is the actual control of the third pool. It needs almost no memory: the weights
    // come straight from the mmap, only `cpu_out_pin` (nmiss*N floats) and `cpu_xq_pre`
    // (nmiss*row_size(K)) are buffered — about 100 KB together. What the CPU contributes
    // is cores, and these must be allocated like sets on a card.
    //
    // Default 0 = unchanged. A silent behavior change in the middle of a running measurement
    // series would be worse than the missing limit; the policy belongs in the measurement
    // script, which knows THREADS anyway.
    static const int cpu_threads = [] {
        const char * e = std::getenv("FLE_PAGER_CPU_THREADS");
        const int v = (e && *e) ? std::atoi(e) : 0;
        return v > 0 ? v : 0;
    }();
#pragma omp parallel for collapse(2) schedule(static) if(total_rc >= 2048) num_threads(cpu_threads > 0 ? cpu_threads : omp_get_max_threads())
    for (int i = 0; i < nmiss; ++i) {
        for (int64_t r = 0; r < N; ++r) {
            const PagerRuntime::CpuMiss & m = rt.cpu_misses[(size_t) i];
            const uint8_t * xq       = rt.cpu_xq_pre.data() + (size_t) i * rt.cpu_pre_stride;
            const uint8_t * w_expert = w_base + (size_t) m.expert * nb2_w;
            tr->vec_dot((int) K, rt.cpu_out_pin + (size_t) i * (size_t) N + r, 0,
                        w_expert + (size_t) r * nb1_w, 0, xq, 0, 1);
        }
    }
    if (g_bw_on) {
        // Weight bytes read: per miss N rows of nb1_w (quant codes from host_backing).
        bw_cur().cpu_bytes += (unsigned long long) nmiss * (unsigned long long) N * (unsigned long long) nb1_w;
        bw_cur().cpu_ms   += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - _cpu_t0).count();
        bw_cur().cpu_calls++;
    }
    for (int i = 0; i < nmiss; ++i) {   // H2D override serially (stream not thread-safe for enqueue)
        const PagerRuntime::CpuMiss & m = rt.cpu_misses[(size_t) i];
        uint8_t * dst_col = (uint8_t *) dst->data + (size_t) m.token * dst->nb[2] + (size_t) m.iex * dst->nb[1];
        cudaMemcpyAsync(dst_col, rt.cpu_out_pin + (size_t) i * (size_t) N, (size_t) N * sizeof(float), cudaMemcpyHostToDevice, stream);
    }
    // NO sync: the override H2Ds are on `stream` after the mmvq -> the next op reads dst correctly afterwards.
    rt.cpu_misses.clear();
}

} // namespace pager
} // namespace fle
