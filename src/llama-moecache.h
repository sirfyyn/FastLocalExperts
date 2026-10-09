#pragma once

// GPU-resident LRU cache for MoE expert weights that -ot pinned to host memory.
//
// Motivation (measured on Qwen3.8-Flash-Next, 512 experts / 10 routed): expert
// routing has strong temporal locality (LRU-64 hit rate ~67% over a mixed
// workload) even though the long-run distribution is near-uniform. Decode on a
// host-offloaded MoE layer is bound by host RAM bandwidth, so serving the hot
// experts from VRAM removes most of the per-token DIMM traffic.
//
// Mechanism (no custom kernels):
//  - per cached layer, companion tensors up_c/gate_c/down_c of shape
//    [ne0, ne1, n_slots+1] live in the device buffer of that layer's router;
//    slot n_slots is permanently zero (the "dummy" slot).
//  - an I32 table[512] maps expert id -> slot, or n_slots when uncached.
//    One copy on device (read by get_rows to remap ids for the cache-side
//    mul_mat_id chain) and one on host (read by the CPU mul_mat_id via
//    src[3] to SKIP cached ids, zeroing their dst rows).
//  - the two down-projection outputs are summed; uncached ids contribute 0
//    through the cache chain (zero slot) and cached ids contribute 0 through
//    the CPU chain (skip), so the result is exact.
//  - llama_moe_cache_step(), called at the end of llama_context::decode(),
//    performs throttled LRU updates: at most LLAMA_MOE_CACHE_INSERTS expert
//    uploads per layer per step via ggml_backend_tensor_set.
//
// Enabled via llama_context_params.n_moe_cache_slots (CLI: --moe-expert-cache).

#include <cstdint>
#include <vector>

struct llama_model;
struct ggml_tensor;

struct llama_moe_cache_layer {
    int il = -1;

    int32_t n_slots = 0;

    // host-resident source weights (the authoritative experts)
    ggml_tensor * up_src   = nullptr;
    ggml_tensor * gate_src = nullptr;
    ggml_tensor * down_src = nullptr;

    // device-resident cache slots, ne[2] == n_slots + 1 (slot n_slots all zeros), plus the PCIe ring slots after it (Patch 47)
    ggml_tensor * up_c   = nullptr;
    ggml_tensor * gate_c = nullptr;
    ggml_tensor * down_c = nullptr;

    // expert id -> slot (or n_slots when uncached); I32 [1, n_expert]
    ggml_tensor * dev_table  = nullptr;
    ggml_tensor * host_table = nullptr;

    // FLE: routing bonus per expert (bias when cached, else 0); F32 [n_expert], only with a cache bias > 0.
    // Mirrored on the host and uploaded once per layer and step (see llama_moe_cache_step).
    ggml_tensor *      dev_bias   = nullptr;
    std::vector<float> bias_host;
    bool               bias_dirty = false;
    float              bias_val   = 0.0f;   // bonus of this layer (FLE_MOE_CACHE_BIAS_LAYERS, otherwise the global value)
};

// build the cache for every host-resident expert layer of the model.
// Safe to call more than once; only the first call does work.
// cache_bias > 0: every cached layer also gets a dev_bias tensor holding cache_bias for resident experts
void llama_moe_cache_init(const llama_model & model, int32_t n_slots, int32_t max_inserts, float cache_bias = 0.0f);

// largest ubatch that runs the cache graph (FLE_MOE_CACHE_BATCH / FLE_MOE_CACHE_MAXTOK, default 8)
int llama_moe_cache_max_tokens();

// nullptr when the cache is disabled or this tensor has no cached layer
const llama_moe_cache_layer * llama_moe_cache_lookup(const ggml_tensor * up_exps);

// apply throttled LRU updates; call between graph executions only
void llama_moe_cache_step();

// Patch 47 (FLE_MOE_PCIE_FRAC > 0): part of the misses of each step is copied over PCIe into a small per-layer ring of
// extra slots and computed by the GPU hit chain instead of the CPU. Read once from the environment; usable before init.
bool llama_moe_cache_pcie_on();

// Patch 48 (FLE_UBATCH_LEND=1): a large prefill ubatch borrows device memory from the expert cache, the way Strata's
// prefill places its compute buffers in the top KV slots. FLE_UBATCH_LEND is read once from the environment.
bool llama_moe_cache_lend_enabled();
// whole cache layers are freed from the last layer down until at least `bytes` are free; returns the bytes freed
// (all lent bytes when a lend is already active). A lent layer has no cache: lookup returns nullptr, the CPU computes.
// Call between graph executions only.
size_t llama_moe_cache_lend(size_t bytes);
// reallocate every lent layer and refill its slots, ring and tables from host memory; no-op without an active lend.
// Free the large compute buffer first. Call between graph executions only.
void llama_moe_cache_return();
// changes on every lend and return: a graph built under another epoch may reference freed cache tensors (or miss the
// refilled ones), so it must not be reused
uint64_t llama_moe_cache_epoch();

// Patch 44a: write the expert-usage profile (FLE_MOE_PROFILE) if it changed since the last save. Cheap no-op
// without the switch. Called from the context destructor; safe to call more than once.
void llama_moe_cache_profile_save();
