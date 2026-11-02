#pragma once

// FLE expert store (FLE_MOE_STORE_GB): a RAM tier for the CPU-side routed experts of an offloaded MoE model, after Strata's
// ds4_store. Whole experts (gate | up | down) are read from the model files with unbuffered reads (O_DIRECT, the page cache
// neither used nor disturbed) into fixed slots of an anonymous arena; the CPU mul_mat_id reads them there through
// ggml_set_moe_data_callback. The slots are filled at load from the usage profile (hottest first) and on demand when a layer
// routes to an expert that is neither in the GPU cache nor in the store; the slot to give up is the least frequently used one.
//
// Thread model: fill() and lookup changes happen on the compute thread 0 inside the op's setup (the other compute threads wait
// at the setup barrier), data() is read by all compute threads after that barrier. The readers are a private thread pool.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

struct ggml_tensor;

struct llama_expert_store_layer {
    int                 il = -1;
    const ggml_tensor * src[3] = { nullptr, nullptr, nullptr };   // gate, up, down (host expert tensors, mmap)
};

class llama_expert_store {
public:
    // budget_bytes: arena size; threads: unbuffered readers. Returns nullptr when the files cannot be mapped (then nothing changes).
    static std::unique_ptr<llama_expert_store> create(const std::vector<llama_expert_store_layer> & layers, size_t budget_bytes, int threads);
    ~llama_expert_store();

    // bytes of one expert (all three matrices) of layer index li
    size_t expert_bytes(size_t li) const;
    size_t n_slots() const;

    // read the experts (li, e) into free slots (no eviction): the warm start from the profile. Returns how many were read.
    size_t preload(const std::vector<std::pair<int32_t, int32_t>> & li_expert);

    // make the given experts of layer li resident (read the missing ones, evicting the least used), count the use of all.
    // Called on compute thread 0 before the layer's CPU experts run. `keep` are experts of this call that must not be evicted.
    // evicted (optional): the (layer index, expert) pairs this call gave up
    void ensure(size_t li, const int32_t * experts, size_t n, std::vector<std::pair<int32_t, int32_t>> * evicted = nullptr);

    bool resident(size_t li, int32_t e) const;

    // start reading expert e of layer li into a cold slot without waiting (a predicted miss); ensure() publishes it when the
    // layer needs it and waits for the reads still running. false: no cold slot or already resident/known.
    bool prefetch(size_t li, int32_t e, std::vector<std::pair<int32_t, int32_t>> * evicted = nullptr);
    size_t queue_len() const;

    // the CPU mul_mat_id's question (ggml_moe_data_cb_t)
    static const void * data_cb(const ggml_tensor * src0, int32_t expert, void * ud);

    struct stats_t {
        uint64_t hits = 0, misses = 0, evictions = 0, read_bytes = 0, preloaded = 0, pre_issued = 0, pre_used = 0, pre_wasted = 0;
        double   wait_ms = 0.0, preload_ms = 0.0;
    };
    const stats_t & stats() const { return st; }

    struct impl;
private:
    explicit llama_expert_store(impl * p) : d(p) {}
    impl *  d;
    stats_t st;
};
