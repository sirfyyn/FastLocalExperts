// pager_wire.h — narrow P4 interface between ggml-cuda.cu and the FLE expert pager.
// Adapted from FreeToken (Apache-2.0), https://github.com/FlashML-org/FreeToken
// Deliberately minimal: ggml-cuda.cu sees ONLY these symbols (implemented in pager_buffer.cu).
// Precondition: ggml-backend.h + cuda_runtime.h are already included by the includer
// (the case in ggml-cuda.cu); they are included here anyway, idempotently, for standalone checks.
#pragma once

#include "ggml-backend.h"
#include <cuda_runtime.h>
#include <cstdint>

namespace fle {
namespace pager {

// Is this buffer / buffer type our pager buffer? (scheduler gate + injection guard)
bool is_pager_buffer(ggml_backend_buffer_t buffer);
bool is_pager_buffer_type(ggml_backend_buffer_type_t buft);

// P4 core: stages the expert slices selected by `ids` from the host backing into the
// compact VRAM pool and returns remapped ids (device pointer via out-param).
// Return value: pool base (replaces src0->data) — or nullptr = fallback (zero-copy host via UVA,
// ids stay unchanged). The fallback is ALWAYS correct, just slower.
// src1 (activation) + dst (output) are passed through for S3 (CPU miss compute) — unused when
// FLE_PAGER_CPU_MISS is off (pure plumbing, behavior-neutral). May be nullptr.
void * pager_stage_and_remap(ggml_backend_buffer_t buffer,
                             const struct ggml_tensor * src0,
                             const struct ggml_tensor * ids,
                             const int32_t ** ids_remapped_dev,
                             cudaStream_t stream,
                             const struct ggml_tensor * src1 = nullptr,
                             struct ggml_tensor * dst = nullptr,
                             int * n_experts_out = nullptr);

// S3 CPU miss finalize: call AFTER the GPU mul_mat — computes the miss experts recorded in
// stage_and_remap on the CPU (ggml vec_dot against the host_backing quant codes) and overwrites
// the dst columns. No-op if FLE_PAGER_CPU_MISS is off or no misses were recorded.
void pager_cpu_miss_finalize(ggml_backend_buffer_t buffer,
                             const struct ggml_tensor * src0,
                             const struct ggml_tensor * src1,
                             struct ggml_tensor * dst,
                             cudaStream_t stream);

// Early-routing predictor ("Procyon"): registers the router (gate_inp) + bias per layer, keyed by the
// layer index (from gate_exps->name), so that the prefetch can predict layer L+1's experts via early
// routing (router @ gate_input). gate_cosine(L,L+1)~0.92 on M2.7 makes gate_input[L] a valid proxy for L+1.
void pager_register_router(const struct ggml_tensor * gate_exps,
                           const struct ggml_tensor * gate_inp,
                           const struct ggml_tensor * exp_probs_b,
                           int n_expert_used);

// Factory (loader binding, step B): pager buffer type per device.
ggml_backend_buffer_type_t pager_buffer_type(int device, size_t vram_budget_bytes);

} // namespace pager
} // namespace fle
