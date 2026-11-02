#pragma once

// FLE fused CPU MoE (FLE_MOE_FUSED_CPU=1): the CPU part of a layer's routed experts (gate, up, clamped SwiGLU, down) as ONE ggml
// custom op with two internal barriers instead of four ops with their own barriers, thread wake-ups and activation copies.
// Same dot kernels, same activation quantization, same arithmetic as the mul_mat_id chain, so the result is the same.

#include <cstdint>

struct ggml_context;
struct ggml_tensor;

// cur [n_embd, 1, n_tokens] F32, ids [n_used, n_tokens] I32, *_exps host expert tensors [n_in, n_out, n_expert],
// table I32 [n_expert] (GPU cache slot or `dummy` when the CPU computes the expert), pred optional (next layer's predicted ids).
// Returns [n_embd, n_used, n_tokens] F32: the CPU experts' down outputs, 0 for the experts the GPU cache computes.
ggml_tensor * llama_moe_fused_cpu(ggml_context * ctx, ggml_tensor * cur, ggml_tensor * ids,
        ggml_tensor * gate_exps, ggml_tensor * up_exps, ggml_tensor * down_exps,
        ggml_tensor * table, int32_t dummy, ggml_tensor * pred, float clamp_limit, int il);
