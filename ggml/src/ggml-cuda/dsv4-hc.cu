#include "common.cuh"
#include "dsv4-hc.cuh"


static constexpr int DSV4_HC = 4;


static __device__ void dsv4_hc_comb_norm_cols(float * comb, float eps) {
    for (int idst = 0; idst < DSV4_HC; ++idst) {
        float sum = eps;
        for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
            sum += comb[idst + DSV4_HC*isrc];
        }

        const float inv_sum = 1.0f / sum;
        for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
            comb[idst + DSV4_HC*isrc] *= inv_sum;
        }
    }
}

static __device__ void dsv4_hc_comb_norm_rows(float * comb, float eps) {
    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
        float sum = eps;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            sum += comb[idst + DSV4_HC*isrc];
        }

        const float inv_sum = 1.0f / sum;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            comb[idst + DSV4_HC*isrc] *= inv_sum;
        }
    }
}

static __global__ void dsv4_hc_comb_f32(
        const float * mixes,
        const float * scale,
        const float * base,
        float * dst,
        int64_t n_tokens,
        int64_t sm0,
        int64_t sm1,
        int64_t ss0,
        int64_t sb0,
        int64_t sd0,
        int64_t sd1,
        int64_t sd2,
        float eps,
        int32_t n_iter) {
    constexpr int comb_offset = 2*DSV4_HC;

    ggml_cuda_pdl_lc();
    const int64_t it = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;

    if (it >= n_tokens) {
        return;
    }

    ggml_cuda_pdl_sync();

    const float scale_comb = scale[2*ss0];
    float comb[DSV4_HC*DSV4_HC];

    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
        float max = -INFINITY;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            const float v = mixes[(comb_offset + idx)*sm0 + it*sm1] * scale_comb + base[(comb_offset + idx)*sb0];
            comb[idx] = v;
            max = fmaxf(max, v);
        }

        float sum = 0.0f;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            const float v = expf(comb[idx] - max);
            comb[idx] = v;
            sum += v;
        }

        const float inv_sum = 1.0f / sum;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            comb[idx] = comb[idx] * inv_sum + eps;
        }
    }

    dsv4_hc_comb_norm_cols(comb, eps);
    for (int32_t i = 1; i < n_iter; ++i) {
        dsv4_hc_comb_norm_rows(comb, eps);
        dsv4_hc_comb_norm_cols(comb, eps);
    }

    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            dst[idst*sd0 + isrc*sd1 + it*sd2] = comb[idx];
        }
    }
}

static __global__ void dsv4_hc_pre_f32(
        const float * x,
        const float * weights,
        float * dst,
        int64_t n_embd,
        int64_t hc,
        int64_t n_tokens,
        int64_t sx0,
        int64_t sx1,
        int64_t sx2,
        int64_t sw0,
        int64_t sw1,
        int64_t sd0,
        int64_t sd1) {
    ggml_cuda_pdl_lc();
    const int64_t ir = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t nr = n_embd * n_tokens;

    if (ir >= nr) {
        return;
    }

    ggml_cuda_pdl_sync();

    const int64_t i0 = ir % n_embd;
    const int64_t it = ir / n_embd;

    float sum = x[i0*sx0 + it*sx2] * weights[it*sw1];
    for (int64_t ih = 1; ih < hc; ++ih) {
        const float xv = x[i0*sx0 + ih*sx1 + it*sx2];
        const float wv = weights[ih*sw0 + it*sw1];
        sum += xv * wv;
    }

    dst[i0*sd0 + it*sd1] = sum;
}

static __global__ void dsv4_hc_post_f32(
        const float * x,
        const float * residual,
        const float * post,
        const float * comb,
        float * dst,
        int64_t n_embd,
        int64_t hc,
        int64_t n_tokens,
        int64_t sx0,
        int64_t sx1,
        int64_t sr0,
        int64_t sr1,
        int64_t sr2,
        int64_t sp0,
        int64_t sp1,
        int64_t sc0,
        int64_t sc1,
        int64_t sc2,
        int64_t sd0,
        int64_t sd1,
        int64_t sd2) {
    ggml_cuda_pdl_lc();
    const int64_t ir = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t nr = n_embd * hc * n_tokens;

    if (ir >= nr) {
        return;
    }

    ggml_cuda_pdl_sync();

    const int64_t i0   = ir % n_embd;
    const int64_t idst = (ir / n_embd) % hc;
    const int64_t it   = ir / (n_embd * hc);

    float sum = x[i0*sx0 + it*sx1] * post[idst*sp0 + it*sp1];
    for (int64_t isrc = 0; isrc < hc; ++isrc) {
        sum += residual[i0*sr0 + isrc*sr1 + it*sr2] * comb[idst*sc0 + isrc*sc1 + it*sc2];
    }

    dst[i0*sd0 + idst*sd1 + it*sd2] = sum;
}

void ggml_cuda_op_dsv4_hc_comb(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * mixes = dst->src[0];
    const ggml_tensor * scale = dst->src[1];
    const ggml_tensor * base  = dst->src[2];

    GGML_ASSERT(mixes->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32);
    GGML_ASSERT(base->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    constexpr int64_t hc_mix_dim = (2 + DSV4_HC)*DSV4_HC;

    GGML_ASSERT(mixes->ne[0] == hc_mix_dim);
    GGML_ASSERT(dst->ne[0] == DSV4_HC);
    GGML_ASSERT(dst->ne[1] == DSV4_HC);
    GGML_ASSERT(dst->ne[2] == mixes->ne[1]);
    GGML_ASSERT(scale->ne[0] >= 3);
    GGML_ASSERT(base->ne[0] == hc_mix_dim);

    GGML_TENSOR_LOCALS(size_t, nbm, mixes, nb);
    GGML_TENSOR_LOCALS(size_t, nbs, scale, nb);
    GGML_TENSOR_LOCALS(size_t, nbb, base,  nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,   nb);

    const int64_t n_tokens = mixes->ne[1];
    const float eps = ggml_get_op_params_f32(dst, 0);
    const int32_t n_iter = ggml_get_op_params_i32(dst, 1);

    const int block_size = 256;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((n_tokens + block_size - 1) / block_size, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    ggml_cuda_kernel_launch(dsv4_hc_comb_f32, launch_params,
            (const float *) mixes->data, (const float *) scale->data, (const float *) base->data, (float *) dst->data,
            n_tokens,
            nbm0 / sizeof(float), nbm1 / sizeof(float),
            nbs0 / sizeof(float),
            nbb0 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float), nbd2 / sizeof(float),
            eps, n_iter);
}

void ggml_cuda_op_dsv4_hc_pre(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x       = dst->src[0];
    const ggml_tensor * weights = dst->src[1];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_LOCALS(size_t, nbx, x,       nb);
    GGML_TENSOR_LOCALS(size_t, nbw, weights, nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,     nb);

    const int64_t n_embd   = x->ne[0];
    const int64_t hc       = x->ne[1];
    const int64_t n_tokens = x->ne[2];

    const int block_size = 256;
    const int64_t nr = n_embd * n_tokens;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((nr + block_size - 1) / block_size, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    ggml_cuda_kernel_launch(dsv4_hc_pre_f32, launch_params,
            (const float *) x->data, (const float *) weights->data, (float *) dst->data,
            n_embd, hc, n_tokens,
            nbx0 / sizeof(float), nbx1 / sizeof(float), nbx2 / sizeof(float),
            nbw0 / sizeof(float), nbw1 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float));
}

void ggml_cuda_op_dsv4_hc_post(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x        = dst->src[0];
    const ggml_tensor * residual = dst->src[1];
    const ggml_tensor * post     = dst->src[2];
    const ggml_tensor * comb     = dst->src[3];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(residual->type == GGML_TYPE_F32);
    GGML_ASSERT(post->type == GGML_TYPE_F32);
    GGML_ASSERT(comb->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_LOCALS(size_t, nbx, x,        nb);
    GGML_TENSOR_LOCALS(size_t, nbr, residual, nb);
    GGML_TENSOR_LOCALS(size_t, nbp, post,     nb);
    GGML_TENSOR_LOCALS(size_t, nbc, comb,     nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,      nb);

    const int64_t n_embd   = x->ne[0];
    const int64_t n_tokens = x->ne[1];
    const int64_t hc       = residual->ne[1];

    const int block_size = 256;
    const int64_t nr = n_embd * hc * n_tokens;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((nr + block_size - 1) / block_size, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    ggml_cuda_kernel_launch(dsv4_hc_post_f32, launch_params,
            (const float *) x->data, (const float *) residual->data,
            (const float *) post->data, (const float *) comb->data, (float *) dst->data,
            n_embd, hc, n_tokens,
            nbx0 / sizeof(float), nbx1 / sizeof(float),
            nbr0 / sizeof(float), nbr1 / sizeof(float), nbr2 / sizeof(float),
            nbp0 / sizeof(float), nbp1 / sizeof(float),
            nbc0 / sizeof(float), nbc1 / sizeof(float), nbc2 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float), nbd2 / sizeof(float));
}


// FLE: hyper-connection kernels of the qwen4exp architecture (ggml_fle_hc_gate_mean, ggml_fle_hc_combine, see ggml.h).
// Same floating-point steps in the same order as the chain of primitive operations; __fmul_rn/__fadd_rn prevent
// the compiler from fusing product and sum into an FMA (the separate kernels round after the product).
static __device__ __forceinline__ float fle_hc_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));   // wie op_sigmoid in unary.cu
}

static __global__ void fle_hc_gate_mean_f32(
        const float * xn,
        const float * g,
        float * dst,
        int64_t n_embd,
        int hc,
        int64_t n_tokens,
        int64_t sxn1,
        int64_t sg1,
        int64_t sd1,
        float scale) {
    ggml_cuda_pdl_lc();
    const int64_t ir = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t nr = n_embd * n_tokens;

    if (ir >= nr) {
        return;
    }

    ggml_cuda_pdl_sync();

    const int64_t i0 = ir % n_embd;
    const int64_t it = ir / n_embd;

    const float * xr = xn + it*sxn1;
    const float * gr = g  + it*sg1;

    float sum = __fmul_rn(xr[i0], fle_hc_sigmoid(gr[i0]));
    for (int c = 1; c < hc; ++c) {
        sum = __fadd_rn(sum, __fmul_rn(xr[c*n_embd + i0], fle_hc_sigmoid(gr[c*n_embd + i0])));
    }

    dst[it*sd1 + i0] = __fmul_rn(sum, scale);
}

static __global__ void fle_hc_combine_f32(
        const float * residual,
        const float * x,
        const float * inject,
        float * dst,
        int64_t n_embd,
        int64_t hc,
        int64_t n_tokens,
        int64_t sr1,
        int64_t sr2,
        int64_t sx1,
        int64_t si1,
        int64_t sd1,
        int64_t sd2,
        float scale_inject,
        float scale_w) {
    ggml_cuda_pdl_lc();
    const int64_t ir = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t nr = n_embd * hc * n_tokens;

    if (ir >= nr) {
        return;
    }

    ggml_cuda_pdl_sync();

    const int64_t i0 = ir % n_embd;
    const int64_t ic = (ir / n_embd) % hc;
    const int64_t it = ir / (n_embd * hc);

    const float t = __fmul_rn(inject[it*si1 + ic], scale_inject);
    const float w = __fmul_rn(fle_hc_sigmoid(t), scale_w);
    const float p = __fmul_rn(x[it*sx1 + i0], w);

    dst[it*sd2 + ic*sd1 + i0] = __fadd_rn(residual[it*sr2 + ic*sr1 + i0], p);
}

void ggml_cuda_op_fle_hc_gate_mean(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * xn = dst->src[0];
    const ggml_tensor * g  = dst->src[1];

    GGML_ASSERT(xn->type == GGML_TYPE_F32);
    GGML_ASSERT(g->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(xn) && ggml_is_contiguous(g));

    int   hc;
    float scale;
    memcpy(&hc,    (const int32_t *) dst->op_params + 0, sizeof(int));
    memcpy(&scale, (const float *)   dst->op_params + 1, sizeof(float));

    const int64_t n_embd   = dst->ne[0];
    const int64_t n_tokens = dst->ne[1];

    GGML_ASSERT(xn->ne[0] == n_embd*hc);
    GGML_ASSERT(xn->ne[1] == n_tokens);

    const int block_size = 256;
    const int64_t nr = n_embd * n_tokens;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((nr + block_size - 1) / block_size, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    ggml_cuda_kernel_launch(fle_hc_gate_mean_f32, launch_params,
            (const float *) xn->data, (const float *) g->data, (float *) dst->data,
            n_embd, hc, n_tokens,
            xn->nb[1] / sizeof(float), g->nb[1] / sizeof(float), dst->nb[1] / sizeof(float),
            scale);
}

void ggml_cuda_op_fle_hc_combine(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * residual = dst->src[0];
    const ggml_tensor * x        = dst->src[1];
    const ggml_tensor * inject   = dst->src[2];

    GGML_ASSERT(residual->type == GGML_TYPE_F32);
    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(inject->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(residual) && ggml_is_contiguous(x) && inject->nb[0] == sizeof(float));

    float scale_inject, scale_w;
    memcpy(&scale_inject, (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&scale_w,      (const float *) dst->op_params + 1, sizeof(float));

    const int64_t n_embd   = dst->ne[0];
    const int64_t hc       = dst->ne[1];
    const int64_t n_tokens = dst->ne[2];

    const int block_size = 256;
    const int64_t nr = n_embd * hc * n_tokens;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((nr + block_size - 1) / block_size, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    ggml_cuda_kernel_launch(fle_hc_combine_f32, launch_params,
            (const float *) residual->data, (const float *) x->data, (const float *) inject->data, (float *) dst->data,
            n_embd, hc, n_tokens,
            residual->nb[1] / sizeof(float), residual->nb[2] / sizeof(float),
            x->nb[1] / sizeof(float), inject->nb[1] / sizeof(float),
            dst->nb[1] / sizeof(float), dst->nb[2] / sizeof(float),
            scale_inject, scale_w);
}


// ---- Low-rank projection of the hyper-connection (ggml_fle_hc_lr_a / ggml_fle_hc_lr_b, see ggml.h) ----
// Small batches (n_tokens <= 8). Q8_0 weights are dequantized block-wise and multiplied with F32 activations (no Q8_1 quantization of the activation).
#define FLE_LR_MAXTOK 8

static __device__ __forceinline__ float fle_lr_warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        v += __shfl_xor_sync(0xffffffff, v, o);
    }
    return v;
}

// Kernel A: FLE_LR_R rows per block (r rows of w_down, then the hc inject rows in a block of their own). Each warp takes every eighth Q8_0 block of the rows, each lane one element
// per block; the activation is loaded once per block and used for all rows of the block (less L2 traffic), the weight loads are independent (many in flight at once).
#define FLE_LR_R 4

static __global__ void __launch_bounds__(256) fle_hc_lr_a_f32(
        const float * xn,
        const block_q8_0 * w_down,
        const float * w_inj,
        float * dst,
        int64_t k,
        int r,
        int hc,
        int nt,
        int64_t sxn1,
        int64_t swi1,
        int64_t sd1,
        float scale) {
    ggml_cuda_pdl_lc();
    ggml_cuda_pdl_sync();

    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int row0 = blockIdx.x * FLE_LR_R;

    float acc[FLE_LR_R][FLE_LR_MAXTOK];
#pragma unroll
    for (int rr = 0; rr < FLE_LR_R; ++rr) {
#pragma unroll
        for (int t = 0; t < FLE_LR_MAXTOK; ++t) {
            acc[rr][t] = 0.0f;
        }
    }

    if (row0 < r) {
        const int nb = (int) (k / QK8_0);
        const block_q8_0 * w0 = w_down + (int64_t) row0 * nb;
#pragma unroll 4
        for (int b = warp; b < nb; b += 8) {
            float xv[FLE_LR_MAXTOK];
#pragma unroll
            for (int t = 0; t < FLE_LR_MAXTOK; ++t) {
                xv[t] = t < nt ? xn[t * sxn1 + b * QK8_0 + lane] : 0.0f;
            }
#pragma unroll
            for (int rr = 0; rr < FLE_LR_R; ++rr) {
                const block_q8_0 * wb = w0 + (int64_t) rr * nb + b;
                const float wq = __half2float(wb->d) * (float) wb->qs[lane];
#pragma unroll
                for (int t = 0; t < FLE_LR_MAXTOK; ++t) {
                    acc[rr][t] = fmaf(wq, xv[t], acc[rr][t]);
                }
            }
        }
    } else {
        for (int i = tid; i < k; i += 256) {
            float xv[FLE_LR_MAXTOK];
#pragma unroll
            for (int t = 0; t < FLE_LR_MAXTOK; ++t) {
                xv[t] = t < nt ? xn[t * sxn1 + i] : 0.0f;
            }
#pragma unroll
            for (int rr = 0; rr < FLE_LR_R; ++rr) {
                if (rr < hc) {
                    const float w = w_inj[(int64_t) rr * swi1 + i];
#pragma unroll
                    for (int t = 0; t < FLE_LR_MAXTOK; ++t) {
                        acc[rr][t] = fmaf(w, xv[t], acc[rr][t]);
                    }
                }
            }
        }
    }

    __shared__ float red[8][FLE_LR_R][FLE_LR_MAXTOK];
#pragma unroll
    for (int rr = 0; rr < FLE_LR_R; ++rr) {
#pragma unroll
        for (int t = 0; t < FLE_LR_MAXTOK; ++t) {
            const float v = fle_lr_warp_sum(acc[rr][t]);
            if (lane == 0) {
                red[warp][rr][t] = v;
            }
        }
    }
    __syncthreads();

    if (tid < FLE_LR_R * FLE_LR_MAXTOK) {
        const int rr = tid % FLE_LR_R;
        const int t  = tid / FLE_LR_R;
        if (t < nt && (row0 < r || rr < hc)) {
            float sum = 0.0f;
#pragma unroll
            for (int w = 0; w < 8; ++w) {
                sum += red[w][rr][t];
            }
            if (row0 < r) {
                const float x = sum * scale;
                sum = x / (1.0f + expf(-x));     // wie ggml_cuda_op_silu_single
                dst[(int64_t) t * sd1 + row0 + rr] = sum;
            } else {
                dst[(int64_t) t * sd1 + r + rr] = sum;
            }
        }
    }
}

// Kernel B: one warp per output column j (8 columns per block): all Q8_0 blocks of the hc rows (up to 4 x 16) are loaded first (independent loads), then follow the dot products with lo from shared memory,
// then sigmoid, product with xn, sum over the streams (fixed order like the chain) and scaling.
static __global__ void __launch_bounds__(256) fle_hc_lr_b_f32(
        const float * xn,
        const float * a,
        const block_q8_0 * w_up,
        float * dst,
        int n_embd,
        int hc,
        int r,
        int nt,
        int64_t sxn1,
        int64_t sa1,
        int64_t sd1,
        float scale) {
    ggml_cuda_pdl_lc();
    __shared__ float slo[FLE_LR_MAXTOK][512];
    ggml_cuda_pdl_sync();

    for (int idx = threadIdx.x; idx < nt * r; idx += 256) {
        const int t = idx / r;
        const int i = idx - t * r;
        slo[t][i] = a[(int64_t) t * sa1 + i];
    }
    __syncthreads();

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int j    = blockIdx.x * 8 + warp;
    if (j >= n_embd) {
        return;
    }

    const int nb = r / QK8_0;
    float wq[4][16];
#pragma unroll
    for (int c = 0; c < 4; ++c) {
        const block_q8_0 * wr = w_up + ((int64_t) c * n_embd + j) * nb;
#pragma unroll
        for (int b = 0; b < 16; ++b) {
            wq[c][b] = (c < hc && b < nb) ? __half2float(wr[b].d) * (float) wr[b].qs[lane] : 0.0f;
        }
    }

    float g[4][FLE_LR_MAXTOK];
#pragma unroll
    for (int c = 0; c < 4; ++c) {
#pragma unroll
        for (int t = 0; t < FLE_LR_MAXTOK; ++t) {
            g[c][t] = 0.0f;
        }
    }
#pragma unroll
    for (int c = 0; c < 4; ++c) {
#pragma unroll
        for (int b = 0; b < 16; ++b) {
            if (c < hc && b < nb) {
#pragma unroll
                for (int t = 0; t < FLE_LR_MAXTOK; ++t) {
                    if (t < nt) {
                        g[c][t] = fmaf(wq[c][b], slo[t][b * QK8_0 + lane], g[c][t]);
                    }
                }
            }
        }
    }

#pragma unroll
    for (int c = 0; c < 4; ++c) {
#pragma unroll
        for (int t = 0; t < FLE_LR_MAXTOK; ++t) {
            if (c < hc && t < nt) {
                g[c][t] = fle_lr_warp_sum(g[c][t]);
            }
        }
    }

    if (lane == 0) {
#pragma unroll
        for (int t = 0; t < FLE_LR_MAXTOK; ++t) {
            if (t < nt) {
                const float * x = xn + (int64_t) t * sxn1;
                float s = __fmul_rn(x[j], fle_hc_sigmoid(g[0][t]));
#pragma unroll
                for (int c = 1; c < 4; ++c) {
                    if (c < hc) {
                        s = __fadd_rn(s, __fmul_rn(x[c * n_embd + j], fle_hc_sigmoid(g[c][t])));
                    }
                }
                dst[(int64_t) t * sd1 + j] = __fmul_rn(s, scale);
            }
        }
    }
}

void ggml_cuda_op_fle_hc_lr_a(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * xn     = dst->src[0];
    const ggml_tensor * w_down = dst->src[1];
    const ggml_tensor * w_inj  = dst->src[2];

    GGML_ASSERT(xn->type == GGML_TYPE_F32);
    GGML_ASSERT(w_down->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(w_inj->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    float scale;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    const int64_t k  = xn->ne[0];
    const int     nt = (int) xn->ne[1];
    const int     r  = (int) w_down->ne[1];
    const int64_t hc = w_inj->ne[1];

    GGML_ASSERT(nt <= FLE_LR_MAXTOK);
    GGML_ASSERT(k % QK8_0 == 0);

    GGML_ASSERT(r % FLE_LR_R == 0 && hc <= FLE_LR_R);

    const dim3 block_dims(256, 1, 1);
    const dim3 grid_dims((unsigned) (r / FLE_LR_R + 1), 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    ggml_cuda_kernel_launch(fle_hc_lr_a_f32, launch_params,
            (const float *) xn->data, (const block_q8_0 *) w_down->data, (const float *) w_inj->data, (float *) dst->data,
            k, r, (int) hc, nt,
            (int64_t) (xn->nb[1] / sizeof(float)), (int64_t) (w_inj->nb[1] / sizeof(float)), (int64_t) (dst->nb[1] / sizeof(float)),
            scale);
}

void ggml_cuda_op_fle_hc_lr_b(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * xn   = dst->src[0];
    const ggml_tensor * a    = dst->src[1];
    const ggml_tensor * w_up = dst->src[2];

    GGML_ASSERT(xn->type == GGML_TYPE_F32);
    GGML_ASSERT(a->type == GGML_TYPE_F32);
    GGML_ASSERT(w_up->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    int   hc;
    float scale;
    memcpy(&hc,    (const int32_t *) dst->op_params + 0, sizeof(int));
    memcpy(&scale, (const float *)   dst->op_params + 1, sizeof(float));

    const int n_embd = (int) dst->ne[0];
    const int nt     = (int) dst->ne[1];
    const int r      = (int) w_up->ne[0];

    GGML_ASSERT(nt <= FLE_LR_MAXTOK);
    GGML_ASSERT(hc <= 4);
    GGML_ASSERT(r <= 512 && r % QK8_0 == 0);

    const dim3 block_dims(256, 1, 1);
    const dim3 grid_dims((unsigned) ((n_embd + 7) / 8), 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    ggml_cuda_kernel_launch(fle_hc_lr_b_f32, launch_params,
            (const float *) xn->data, (const float *) a->data, (const block_q8_0 *) w_up->data, (float *) dst->data,
            n_embd, hc, r, nt,
            (int64_t) (xn->nb[1] / sizeof(float)), (int64_t) (a->nb[1] / sizeof(float)), (int64_t) (dst->nb[1] / sizeof(float)),
            scale);
}
