#pragma once

// FLE split experts (v1.2.6, docs/fle/EXPERT-SPLIT.md).
//
// A split GGUF stores the routed experts of a MoE layer in two halves per role: ffn_*_exps_hot (resident on the GPU, the
// role of a static GPU expert cache) and ffn_*_exps_cold (host memory, computed by the FLE CPU path). Each half carries one
// trailing all-zero row. ffn_exps_slot maps original id -> slot (slot < n_hot: hot row, else cold row slot - n_hot).
//
// The router keeps the original numbering. Per split layer this module owns three small I32 tables, built once at load:
//  - hot_of  [1, n_expert] (device of the hot half): original id -> hot row, or n_hot (zero row) for a cold expert
//  - cold_of [1, n_expert] (same device):            original id -> cold row, or n_cold (zero row) for a hot expert
//  - cold_skip [1, n_cold + 1] (host): the src[3] skip table of the CPU chain in cold-local numbering. Every real cold row
//    holds LLAMA_EXPERT_SPLIT_RUN (the "not cached" dummy value: computed), the zero row holds 0 (skipped, never read).
// The GPU chain over the hot half and the CPU chain over the cold half each yield zero for the other half's experts, so
// their sum is the exact MoE output, as with the expert cache.

#include <cstdint>
#include <vector>

struct llama_model;
struct ggml_tensor;

// op_params[0] of the CPU chain (the value of cold_skip for rows the CPU computes)
constexpr int32_t LLAMA_EXPERT_SPLIT_RUN = 1;

struct llama_expert_split_layer {
    int     il     = -1;
    int32_t n_hot  = 0;
    int32_t n_cold = 0;

    ggml_tensor * up_hot    = nullptr;
    ggml_tensor * gate_hot  = nullptr;
    ggml_tensor * down_hot  = nullptr;
    ggml_tensor * up_cold   = nullptr;
    ggml_tensor * gate_cold = nullptr;
    ggml_tensor * down_cold = nullptr;

    ggml_tensor * hot_of    = nullptr;
    ggml_tensor * cold_of   = nullptr;
    ggml_tensor * cold_skip = nullptr;
};

// build the tables of every split layer of the model; once per model, a no-op for unsplit models
void llama_expert_split_init(const llama_model & model);

// true once any loaded model has a split layer (a cheap guard for the graph builder)
bool llama_expert_split_any();

// the split layer whose cold up half is `up_cold`, nullptr otherwise
const llama_expert_split_layer * llama_expert_split_lookup(const ggml_tensor * up_cold);

// the split layers of `model` (layer order), empty for an unsplit model; used to register the cold halves in the expert store
std::vector<const llama_expert_split_layer *> llama_expert_split_layers(const llama_model & model);
