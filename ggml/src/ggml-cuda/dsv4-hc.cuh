#include "common.cuh"
#include "ggml.h"

void ggml_cuda_op_dsv4_hc_comb(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_dsv4_hc_pre(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_dsv4_hc_post(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_fle_hc_gate_mean(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_fle_hc_combine(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_fle_hc_lr_a(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_fle_hc_lr_b(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
