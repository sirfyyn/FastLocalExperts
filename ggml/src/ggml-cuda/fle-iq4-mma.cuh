#pragma once
// FLE_IQ4_MMA (Infernix-Assimilation, Nr. 1103, Mechanismus 6): Prüfkern für IQ4_XS bei 9..24 Spalten, siehe fle-iq4-mma.cu.
#include "common.cuh"

#define FLE_IQ4_MMA_MIN   9
#define FLE_IQ4_MMA_MAX   16  // 17..24 waren in Fassung 6 langsamer als mul_mat_q (GPU1)
#define FLE_IQ4_MMA_WARPS 4
#define FLE_IQ4_MMA_MT    2
#define FLE_IQ4_MMA_ROWS  (16 * FLE_IQ4_MMA_MT * FLE_IQ4_MMA_WARPS)
#define FLE_IQ4_MMA_ROWS  (16 * FLE_IQ4_MMA_MT * FLE_IQ4_MMA_WARPS)

bool fle_iq4_mma_passt(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc);
void fle_iq4_mma(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
