#include "convert.cuh"
#include "dequantize.cuh"

#include <cstdint>

#define CUDA_Q8_0_NE_ALIGN 2048

template <int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static __global__ void dequantize_block(const void * __restrict__ vx, dst_t * __restrict__ y,
        const int64_t ne00, const int64_t ne01,
        const int64_t ne0203, const uint3 ne02,
        const int64_t s01, const int64_t s02, const int64_t s03) {
    const int64_t i00 = 2 * (int64_t(blockDim.x)*blockIdx.x + threadIdx.x);

    if (i00 >= ne00) {
        return;
    }

    for (int64_t i01 = blockIdx.y; i01 < ne01; i01 += gridDim.y) {
        for (int64_t i0203 = blockIdx.z; i0203 < ne0203; i0203 += gridDim.z) {
            const uint2 dm = fast_div_modulo((uint32_t)i0203, ne02);
            const int64_t i02 = dm.y;
            const int64_t i03 = dm.x;

            const int64_t ibx0 = i03*s03 + i02*s02 + i01*s01;

            const int64_t ib = ibx0 + i00/qk; // block index
            const int64_t iqs = (i00%qk)/qr; // quant index
            const int64_t iybs = i00 - i00%qk; // y block start index
            const int64_t y_offset = qr == 1 ? 1 : qk/2;

            // dequantize
            float2 v;
            dequantize_kernel(vx, ib, iqs, v);

            const int64_t iy0 = (i0203*ne01 + i01)*ne00 + iybs + iqs;
            y[iy0 + 0]        = ggml_cuda_cast<dst_t>(v.x);
            y[iy0 + y_offset] = ggml_cuda_cast<dst_t>(v.y);
        }
    }
}

template <bool need_check>
static __global__ void dequantize_block_q8_0_f16(const void * __restrict__ vx, half * __restrict__ y, const int64_t k) {
#if __CUDA_ARCH__ >= GGML_CUDA_CC_PASCAL
    constexpr int nint = CUDA_Q8_0_NE_ALIGN/sizeof(int) + WARP_SIZE;

    const int64_t   i0 = CUDA_Q8_0_NE_ALIGN*blockIdx.x;
    const int * x0 = ((int *) vx) + blockIdx.x * nint;
    half2 * y2 = (half2 *) (y + i0);

    __shared__ int vals[nint];

#pragma unroll
    for (int ix0 = 0; ix0 < nint; ix0 += WARP_SIZE) {
        if (need_check && i0*sizeof(block_q8_0)/QK8_0 + sizeof(int)*(ix0 + threadIdx.x) >= k*sizeof(block_q8_0)/QK8_0) {
            break;
        }

        const int ix = ix0 + threadIdx.x;
        vals[ix] = x0[ix];
    }

    __syncthreads();

#pragma unroll
    for (int iy = 0; iy < CUDA_Q8_0_NE_ALIGN; iy += 2*WARP_SIZE) {
        if (need_check && i0 + iy + 2*threadIdx.x >= k) {
            return;
        }

        const half * b0 = ((const half  *) vals) + (sizeof(block_q8_0)/sizeof(half)) * ((iy + 2*threadIdx.x)/QK8_0);
        const half    d = *b0;
        const char2  qs = ((const char2 *) (b0 + 1))[threadIdx.x % (QK8_0/2)];

        y2[iy/2 + threadIdx.x] = __hmul2(make_half2(qs.x, qs.y), __half2half2(d));
    }
#else
    GGML_UNUSED_VARS(vx, y, k);
    NO_DEVICE_CODE;
#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_PASCAL
}

template<typename dst_t>
static __global__ void dequantize_block_q4_0(const void * __restrict__ vx, dst_t * __restrict__ yy, int nb32) {

    const int64_t i = blockIdx.x;

    // assume 32 threads
    const int64_t tid = threadIdx.x;
    const int64_t il  = tid/8;
    const int64_t ir  = tid%8;
    const int64_t ib = 8*i + ir;
    if (ib >= nb32) {
        return;
    }

    dst_t * y = yy + 256*i + 32*ir + 4*il;

    const block_q4_0 * x = (const block_q4_0 *)vx + ib;
    const float d = __half2float(x->d);
    const float dm = -8*d;

    const uint8_t * q = x->qs + 4*il;

    for (int l = 0; l < 4; ++l) {
        y[l+ 0] = ggml_cuda_cast<dst_t>(d * (q[l] & 0xF) + dm);
        y[l+16] = ggml_cuda_cast<dst_t>(d * (q[l] >>  4) + dm);
    }
}

template<typename dst_t>
static __global__ void dequantize_block_q4_1(const void * __restrict__ vx, dst_t * __restrict__ yy, int nb32) {

    const int64_t i = blockIdx.x;

    // assume 32 threads
    const int64_t tid = threadIdx.x;
    const int64_t il  = tid/8;
    const int64_t ir  = tid%8;
    const int64_t ib = 8*i + ir;
    if (ib >= nb32) {
        return;
    }

    dst_t * y = yy + 256*i + 32*ir + 4*il;

    const block_q4_1 * x = (const block_q4_1 *)vx + ib;
    const float2 d = __half22float2(x->dm);

    const uint8_t * q = x->qs + 4*il;

    for (int l = 0; l < 4; ++l) {
        y[l+ 0] = ggml_cuda_cast<dst_t>(d.x * (q[l] & 0xF) + d.y);
        y[l+16] = ggml_cuda_cast<dst_t>(d.x * (q[l] >>  4) + d.y);
    }
}

//================================== k-quants

template<typename dst_t>
static __global__ void dequantize_block_q2_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;

    dequantize_q2_K(vx, i, yy + i*QK_K, threadIdx.x);
}

template<typename dst_t>
static __global__ void dequantize_block_q3_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;

    dequantize_q3_K(vx, i, yy + i*QK_K, threadIdx.x);
}

template<typename dst_t>
static __global__ void dequantize_block_q4_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;

    dequantize_q4_K(vx, i, yy + i*QK_K, threadIdx.x);
}

template<typename dst_t>
static __global__ void dequantize_block_q5_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;

    dequantize_q5_K(vx, i, yy + i*QK_K, threadIdx.x);
}

template<typename dst_t>
static __global__ void dequantize_block_q6_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;

    dequantize_q6_K(vx, i, yy + i*QK_K, threadIdx.x);
}

template<typename dst_t>
static __global__ void dequantize_block_iq2_xxs(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;

    dequantize_iq2_xxs(vx, i, yy + i*QK_K, threadIdx.x);
}

template<typename dst_t>
static __global__ void dequantize_block_iq2_xs(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;

    dequantize_iq2_xs(vx, i, yy + i*QK_K, threadIdx.x);
}

template<typename dst_t>
static __global__ void dequantize_block_iq2_s(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;

    dequantize_iq2_s(vx, i, yy + i*QK_K, threadIdx.x);
}

template<typename dst_t>
static __global__ void dequantize_block_iq3_xxs(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;

    dequantize_iq3_xxs(vx, i, yy + i*QK_K, threadIdx.x);
}

template<typename dst_t>
static __global__ void dequantize_block_iq3_s(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;

    dequantize_iq3_s(vx, i, yy + i*QK_K, threadIdx.x);
}

template<typename dst_t>
static __global__ void dequantize_block_iq1_s(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;

    dequantize_iq1_s(vx, i, yy + i*QK_K, threadIdx.x);
}

template<typename dst_t>
static __global__ void dequantize_block_iq1_m(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;

    dequantize_iq1_m(vx, i, yy + i*QK_K, threadIdx.x);
}

template<typename dst_t>
static __global__ void dequantize_block_iq4_nl(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;

    dequantize_iq4_nl(vx, i, yy + i*QK_K, threadIdx.x);
}

// ==== IQ_KT CUDA dequant (adapted verbatim from ik_llama.cpp (MIT), https://github.com/ikawrakow/ik_llama.cpp; with fudge factor = GPU serving ground truth) ====
static const __device__ int8_t fle_iq4k_values[16] = {-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
static __device__ __forceinline__ int fle_trellis_next_int(uint32_t & val) {
    val = 0xCBAC1FEDu * val;                       // QTIP 3INST
    uint32_t s = val & 0x3f3f3f3fu;                // bytes 0..63 -> signed==unsigned
    const int8_t * p = (const int8_t *)&s;
    return p[0] + p[1] + p[2] + p[3] - 126;
}
template<typename dst_t>
static __global__ void fle_dequantize_block_iq2_kt(const void * __restrict__ vx, dst_t * __restrict__ yy, int64_t n_per_row, int64_t row_size) {
    int64_t ii  = blockIdx.x;
    int64_t row = (QK_K * ii) / n_per_row;
    const char * cx = (const char *)vx + row * row_size;
    float scale = *(const float *)cx;
    const block_iq2_kt * x = (const block_iq2_kt *)(cx + sizeof(float));
    const int64_t i = ii - (row*n_per_row)/QK_K;
    const int64_t tid = threadIdx.x;
    const int64_t ib = tid; // 0...31
    dst_t * y = yy + ii*QK_K + 8*ib;
    const uint16_t * ql = (const uint16_t *)x[i].ql;
    uint32_t idx = ql[ib] + 4096;
    const float dl = scale * fle_iq4k_values[((x[i].scales[(ib/4)%4] >> 4*(ib/16)) & 0xf)] * 1.05f;
    for (int j = 0; j < 8; ++j) y[j] = dl * fle_trellis_next_int(idx);
}
template<typename dst_t>
static __global__ void fle_dequantize_block_iq3_kt(const void * __restrict__ vx, dst_t * __restrict__ yy, int64_t n_per_row, int64_t row_size) {
    int64_t ii  = blockIdx.x;
    int64_t row = (QK_K * ii) / n_per_row;
    const char * cx = (const char *)vx + row * row_size;
    float scale = *(const float *)cx;
    const block_iq3_kt * x = (const block_iq3_kt *)(cx + sizeof(float));
    const int64_t i = ii - (row*n_per_row)/QK_K;
    const int64_t tid = threadIdx.x;
    const int64_t ib = tid; // 0...31
    dst_t * y = yy + ii*QK_K + 8*ib;
    const uint16_t * ql = (const uint16_t *)x[i].ql;
    uint32_t idx = ql[ib] + 4096;
    const float dl = scale * ((x[i].scales[(ib/4)%4] >> 4*(ib/16)) & 0xf) * 1.01f;
    uint8_t mask = 1 << (ib/4);
    for (int j = 0; j < 8; ++j)
        y[j] = dl * abs(fle_trellis_next_int(idx)) * (x[i].qh[(8*ib+j)%32] & mask ? -1.f : 1.f);
}
template<typename dst_t>
static __global__ void fle_dequantize_block_iq4_kt(const void * __restrict__ vx, dst_t * __restrict__ yy, int64_t n_per_row, int64_t row_size) {
    int64_t ii  = blockIdx.x;
    int64_t row = (QK_K * ii) / n_per_row;
    const float * dptr = (const float *)((const char *)vx + row * row_size);
    float scale = dptr[0] * 1.00f;
    const block_iq4_kt * x = (const block_iq4_kt *)(dptr + 1);
    const int64_t i = ii - (row*n_per_row)/QK_K;
    constexpr int kNumGroups = 64;
    const int64_t tid = threadIdx.x;
    const int64_t ib = tid; // 0...31
    dst_t * y = yy + ii*QK_K + 8*ib;
    const uint32_t * shb = x[i].qs;
    const uint8_t * ql = (const uint8_t *)(shb + 8);
    const uint8_t * qh = ql + kNumGroups;
    const int ib32 = ib/4;
    const int ig = ib%4;
    const int jj = ib32*8 + 2*ig;
    uint32_t offset = shb[ib32] & 1 ? 4096 + 32768 : 4096;
    uint32_t idx1 = ql[jj+0] + ((qh[(jj+0)%(kNumGroups/2)] << (8 - 4*((jj+0)/(kNumGroups/2)))) & 0xf00) + (((shb[ib32] >> (8 + 6*ig+0)) & 7) << 12) + offset;
    uint32_t idx2 = ql[jj+1] + ((qh[(jj+1)%(kNumGroups/2)] << (8 - 4*((jj+1)/(kNumGroups/2)))) & 0xf00) + (((shb[ib32] >> (8 + 6*ig+3)) & 7) << 12) + offset;
    int ls = ((shb[ib32] & 0xff) >> 1) - 64;
    const float dl = scale * ls;
    for (int j = 0; j < 4; ++j) { y[j+0] = dl * fle_trellis_next_int(idx1); y[j+4] = dl * fle_trellis_next_int(idx2); }
}
// Exported test wrapper: host-in -> device -> launch -> host-out. type: 2/3/4.
extern "C" int fle_dequant_iqkt_test(int type, const void * h_vx, float * h_y, int64_t n_per_row, int64_t nrows) {
    const int64_t nbrow = n_per_row / QK_K;
    int64_t blkbytes = type==2 ? 68 : (type==3 ? 100 : 128);
    int64_t row_size = 4 + nbrow * blkbytes;
    int64_t nblocks  = nbrow * nrows;
    int64_t vxbytes  = row_size * nrows;
    int64_t ybytes   = (int64_t)n_per_row * nrows * sizeof(float);
    void * d_vx = nullptr; float * d_y = nullptr;
    if (cudaMalloc(&d_vx, vxbytes) != cudaSuccess) return 1;
    if (cudaMalloc(&d_y, ybytes)  != cudaSuccess) { cudaFree(d_vx); return 2; }
    cudaMemcpy(d_vx, h_vx, vxbytes, cudaMemcpyHostToDevice);
    if      (type==2) fle_dequantize_block_iq2_kt<float><<<nblocks,32>>>(d_vx, d_y, n_per_row, row_size);
    else if (type==3) fle_dequantize_block_iq3_kt<float><<<nblocks,32>>>(d_vx, d_y, n_per_row, row_size);
    else              fle_dequantize_block_iq4_kt<float><<<nblocks,32>>>(d_vx, d_y, n_per_row, row_size);
    cudaError_t e = cudaDeviceSynchronize();
    cudaMemcpy(h_y, d_y, ybytes, cudaMemcpyDeviceToHost);
    cudaFree(d_vx); cudaFree(d_y);
    return e==cudaSuccess ? 0 : 10;
}

// ==== IQ2_KT custom matmul kernel (dmmv style, row-meta capable) ====
// dst[row] = sum_col dequant(W[row])[col] * y[col]. Mirrors fle_dequantize_block_iq2_kt (fudge 1.05f).
static __global__ void fle_mmv_iq2_kt_kernel(const void * __restrict__ vx, const float * __restrict__ y,
        float * __restrict__ dst, int ncols, int64_t row_size) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x; // 0..31, each thread = 8 elements per super-block
    const char * rowp = (const char *)vx + (int64_t)row * row_size;
    const float rscale = *(const float *)rowp;
    const block_iq2_kt * x = (const block_iq2_kt *)(rowp + sizeof(float));
    const int nblocks = ncols / QK_K;
    float sum = 0.f;
    for (int ibl = 0; ibl < nblocks; ++ibl) {
        const int ib = tid; // 0..31
        const uint16_t * ql = (const uint16_t *)x[ibl].ql;
        uint32_t idx = ql[ib] + 4096;
        const float dl = rscale * fle_iq4k_values[((x[ibl].scales[(ib/4)%4] >> 4*(ib/16)) & 0xf)] * 1.05f;
        const float * yb = y + ibl*QK_K + 8*ib;
        for (int j = 0; j < 8; ++j) sum += dl * fle_trellis_next_int(idx) * yb[j];
    }
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_down_sync(0xffffffff, sum, off);
    if (tid == 0) dst[row] = sum;
}
// Test-Launcher: host-in -> device -> launch -> host-out. y = ncols floats, W = nrows*row_size, dst = nrows.
extern "C" int fle_mmv_iq2_kt_test(const void * h_vx, const float * h_y, float * h_dst, int ncols, int nrows) {
    const int64_t nbrow = ncols / QK_K;
    const int64_t row_size = 4 + nbrow * 68;
    void * d_vx = nullptr; float * d_y = nullptr, * d_dst = nullptr;
    if (cudaMalloc(&d_vx, row_size*nrows) != cudaSuccess) return 1;
    if (cudaMalloc(&d_y, (int64_t)ncols*sizeof(float)) != cudaSuccess) { cudaFree(d_vx); return 2; }
    if (cudaMalloc(&d_dst, (int64_t)nrows*sizeof(float)) != cudaSuccess) { cudaFree(d_vx); cudaFree(d_y); return 3; }
    cudaMemcpy(d_vx, h_vx, row_size*nrows, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, (int64_t)ncols*sizeof(float), cudaMemcpyHostToDevice);
    fle_mmv_iq2_kt_kernel<<<nrows, 32>>>(d_vx, d_y, d_dst, ncols, row_size);
    cudaError_t e = cudaDeviceSynchronize();
    cudaMemcpy(h_dst, d_dst, (int64_t)nrows*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_vx); cudaFree(d_y); cudaFree(d_dst);
    return e==cudaSuccess ? 0 : 10;
}

// ==== IQ3_KT custom matmul (mirrors fle_dequantize_block_iq3_kt, fudge 1.01f) ====
static __global__ void fle_mmv_iq3_kt_kernel(const void * __restrict__ vx, const float * __restrict__ y,
        float * __restrict__ dst, int ncols, int64_t row_size) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const char * rowp = (const char *)vx + (int64_t)row * row_size;
    const float rscale = *(const float *)rowp;
    const block_iq3_kt * x = (const block_iq3_kt *)(rowp + sizeof(float));
    const int nblocks = ncols / QK_K;
    float sum = 0.f;
    for (int ibl = 0; ibl < nblocks; ++ibl) {
        const int ib = tid;
        const uint16_t * ql = (const uint16_t *)x[ibl].ql;
        uint32_t idx = ql[ib] + 4096;
        const float dl = rscale * ((x[ibl].scales[(ib/4)%4] >> 4*(ib/16)) & 0xf) * 1.01f;
        const uint8_t mask = 1 << (ib/4);
        const float * yb = y + ibl*QK_K + 8*ib;
        for (int j = 0; j < 8; ++j)
            sum += dl * abs(fle_trellis_next_int(idx)) * (x[ibl].qh[(8*ib+j)%32] & mask ? -1.f : 1.f) * yb[j];
    }
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_down_sync(0xffffffff, sum, off);
    if (tid == 0) dst[row] = sum;
}
extern "C" int fle_mmv_iq3_kt_test(const void * h_vx, const float * h_y, float * h_dst, int ncols, int nrows) {
    const int64_t nbrow = ncols / QK_K;
    const int64_t row_size = 4 + nbrow * 100;
    void * d_vx=nullptr; float * d_y=nullptr, * d_dst=nullptr;
    if (cudaMalloc(&d_vx, row_size*nrows)!=cudaSuccess) return 1;
    if (cudaMalloc(&d_y, (int64_t)ncols*sizeof(float))!=cudaSuccess){cudaFree(d_vx);return 2;}
    if (cudaMalloc(&d_dst,(int64_t)nrows*sizeof(float))!=cudaSuccess){cudaFree(d_vx);cudaFree(d_y);return 3;}
    cudaMemcpy(d_vx,h_vx,row_size*nrows,cudaMemcpyHostToDevice);
    cudaMemcpy(d_y,h_y,(int64_t)ncols*sizeof(float),cudaMemcpyHostToDevice);
    fle_mmv_iq3_kt_kernel<<<nrows,32>>>(d_vx,d_y,d_dst,ncols,row_size);
    cudaError_t e=cudaDeviceSynchronize();
    cudaMemcpy(h_dst,d_dst,(int64_t)nrows*sizeof(float),cudaMemcpyDeviceToHost);
    cudaFree(d_vx);cudaFree(d_y);cudaFree(d_dst);
    return e==cudaSuccess?0:10;
}

// ==== IQ4_KT custom matmul (mirrors fle_dequantize_block_iq4_kt, fudge 1.00f) ====
static __global__ void fle_mmv_iq4_kt_kernel(const void * __restrict__ vx, const float * __restrict__ y,
        float * __restrict__ dst, int ncols, int64_t row_size) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const char * rowp = (const char *)vx + (int64_t)row * row_size;
    const float rscale = *(const float *)rowp;
    const block_iq4_kt * x = (const block_iq4_kt *)(rowp + sizeof(float));
    const int nblocks = ncols / QK_K;
    constexpr int kNumGroups = 64;
    float sum = 0.f;
    for (int ibl = 0; ibl < nblocks; ++ibl) {
        const int ib = tid;
        const uint32_t * shb = x[ibl].qs;
        const uint8_t * ql = (const uint8_t *)(shb + 8);
        const uint8_t * qh = ql + kNumGroups;
        const int ib32 = ib/4, ig = ib%4, jj = ib32*8 + 2*ig;
        uint32_t offset = shb[ib32] & 1 ? 4096 + 32768 : 4096;
        uint32_t idx1 = ql[jj+0] + ((qh[(jj+0)%(kNumGroups/2)] << (8 - 4*((jj+0)/(kNumGroups/2)))) & 0xf00) + (((shb[ib32] >> (8 + 6*ig+0)) & 7) << 12) + offset;
        uint32_t idx2 = ql[jj+1] + ((qh[(jj+1)%(kNumGroups/2)] << (8 - 4*((jj+1)/(kNumGroups/2)))) & 0xf00) + (((shb[ib32] >> (8 + 6*ig+3)) & 7) << 12) + offset;
        const int ls = ((shb[ib32] & 0xff) >> 1) - 64;
        const float dl = rscale * ls;
        const float * yb = y + ibl*QK_K + 8*ib;
        for (int j = 0; j < 4; ++j) {
            sum += dl * fle_trellis_next_int(idx1) * yb[j+0];
            sum += dl * fle_trellis_next_int(idx2) * yb[j+4];
        }
    }
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_down_sync(0xffffffff, sum, off);
    if (tid == 0) dst[row] = sum;
}
extern "C" int fle_mmv_iq4_kt_test(const void * h_vx, const float * h_y, float * h_dst, int ncols, int nrows) {
    const int64_t nbrow = ncols / QK_K;
    const int64_t row_size = 4 + nbrow * 128;
    void * d_vx=nullptr; float * d_y=nullptr, * d_dst=nullptr;
    if (cudaMalloc(&d_vx, row_size*nrows)!=cudaSuccess) return 1;
    if (cudaMalloc(&d_y, (int64_t)ncols*sizeof(float))!=cudaSuccess){cudaFree(d_vx);return 2;}
    if (cudaMalloc(&d_dst,(int64_t)nrows*sizeof(float))!=cudaSuccess){cudaFree(d_vx);cudaFree(d_y);return 3;}
    cudaMemcpy(d_vx,h_vx,row_size*nrows,cudaMemcpyHostToDevice);
    cudaMemcpy(d_y,h_y,(int64_t)ncols*sizeof(float),cudaMemcpyHostToDevice);
    fle_mmv_iq4_kt_kernel<<<nrows,32>>>(d_vx,d_y,d_dst,ncols,row_size);
    cudaError_t e=cudaDeviceSynchronize();
    cudaMemcpy(h_dst,d_dst,(int64_t)nrows*sizeof(float),cudaMemcpyDeviceToHost);
    cudaFree(d_vx);cudaFree(d_y);cudaFree(d_dst);
    return e==cudaSuccess?0:10;
}

// ==== ggml-integrated launcher: column loop over ncols_y, reuses the kernels ====
extern "C" float fle_mmv_iq2_kt_bench(int nrows, int ncols, int iters) {
    const int64_t row_size = 4 + (int64_t)(ncols/256)*68;
    void * d_vx=nullptr; float * d_y=nullptr, * d_dst=nullptr;
    if (cudaMalloc(&d_vx, row_size*nrows)!=cudaSuccess) return -1.f;
    if (cudaMalloc(&d_y, (int64_t)ncols*4)!=cudaSuccess) { cudaFree(d_vx); return -2.f; }
    if (cudaMalloc(&d_dst,(int64_t)nrows*4)!=cudaSuccess){ cudaFree(d_vx);cudaFree(d_y); return -3.f; }
    fle_mmv_iq2_kt_kernel<<<nrows,32>>>(d_vx, d_y, d_dst, ncols, row_size);
    cudaDeviceSynchronize();
    cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    for (int i=0;i<iters;i++) fle_mmv_iq2_kt_kernel<<<nrows,32>>>(d_vx, d_y, d_dst, ncols, row_size);
    cudaEventRecord(b); cudaEventSynchronize(b);
    float ms=0; cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a); cudaEventDestroy(b);
    cudaFree(d_vx); cudaFree(d_y); cudaFree(d_dst);
    return ms/iters;
}
extern "C" void fle_mmv_kt_launch(int type, const void * vx, const float * vy, float * dst,
        int ncols, int ncols_y, int nrows, int64_t row_size, cudaStream_t stream) {
    for (int col = 0; col < ncols_y; ++col) {
        const float * y = vy + (int64_t)col*ncols;
        float * d = dst + (int64_t)col*nrows;
        if      (type==2) fle_mmv_iq2_kt_kernel<<<nrows,32,0,stream>>>(vx, y, d, ncols, row_size);
        else if (type==3) fle_mmv_iq3_kt_kernel<<<nrows,32,0,stream>>>(vx, y, d, ncols, row_size);
        else              fle_mmv_iq4_kt_kernel<<<nrows,32,0,stream>>>(vx, y, d, ncols, row_size);
    }
}

template<typename dst_t>
static __global__ void dequantize_block_iq4_xs(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;

    dequantize_iq4_xs(vx, i, yy + i*QK_K, threadIdx.x);
}

template<typename dst_t>
static __global__ void dequantize_block_mxfp4(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i = blockIdx.x;

    dequantize_mxfp4(vx, i, yy + i*QK_K, threadIdx.x);
}

template <int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static void dequantize_block_cuda(const void * vx, dst_t * y,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t s01, const int64_t s02, const int64_t s03, cudaStream_t stream) {
    const int64_t ne0203 = ne02*ne03;
    const uint3 ne02_fdv = init_fastdiv_values(ne02);
    const dim3 num_blocks((ne00 + 2*CUDA_DEQUANTIZE_BLOCK_SIZE - 1) / (2*CUDA_DEQUANTIZE_BLOCK_SIZE), (int)std::min(ne01, (int64_t)65535), (int)std::min(ne0203, (int64_t)65535));
    dequantize_block<qk, qr, dequantize_kernel><<<num_blocks, CUDA_DEQUANTIZE_BLOCK_SIZE, 0, stream>>>
        (vx, y, ne00, ne01, ne0203, ne02_fdv, s01, s02, s03);
}

template <int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static void dequantize_block_cont_cuda(const void * __restrict__ vx, dst_t * __restrict__ y, const int64_t k, cudaStream_t stream) {
    dequantize_block_cuda<qk, qr, dequantize_kernel, dst_t>(vx, y, k, 1, 1, 1, k/qk, k/qk, k/qk, stream);
}

static void dequantize_block_q8_0_f16_cuda(const void * __restrict__ vx, half * __restrict__ y, const int64_t k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_Q8_0_NE_ALIGN - 1) / CUDA_Q8_0_NE_ALIGN;
    if (k % CUDA_Q8_0_NE_ALIGN == 0) {
        const bool need_check = false;
        dequantize_block_q8_0_f16<need_check><<<num_blocks, WARP_SIZE, 0, stream>>>(vx, y, k);
    } else {
        const bool need_check = true;
        dequantize_block_q8_0_f16<need_check><<<num_blocks, WARP_SIZE, 0, stream>>>(vx, y, k);
    }
}

template<typename dst_t>
static void dequantize_row_q2_K_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_q2_K<<<nb, 64, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_q3_K_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_q3_K<<<nb, 64, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_q4_0_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb32 = k / 32;
    const int nb = (k + 255) / 256;
    dequantize_block_q4_0<<<nb, 32, 0, stream>>>(vx, y, nb32);
}

template<typename dst_t>
static void dequantize_row_q4_1_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb32 = k / 32;
    const int nb = (k + 255) / 256;
    dequantize_block_q4_1<<<nb, 32, 0, stream>>>(vx, y, nb32);
}

template<typename dst_t>
static void dequantize_row_q4_K_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_q4_K<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_q5_K_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_q5_K<<<nb, 64, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_q6_K_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_q6_K<<<nb, 64, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq2_xxs_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq2_xxs<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq2_xs_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq2_xs<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq2_s_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq2_s<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq3_xxs_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq3_xxs<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq3_s_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq3_s<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq1_s_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq1_s<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq4_nl_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = (k + QK_K - 1) / QK_K;
    dequantize_block_iq4_nl<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq1_m_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq1_m<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq4_xs_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = (k + QK_K - 1) / QK_K;
    dequantize_block_iq4_xs<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_mxfp4_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = (k + QK_K - 1) / QK_K;
    dequantize_block_mxfp4<<<nb, 32, 0, stream>>>(vx, y);
}

template <typename dst_t>
static __global__ void dequantize_block_nvfp4(
        const void * __restrict__ vx,
        dst_t * __restrict__ yy,
        const int64_t ne) {
    const int64_t i = blockIdx.x;
    const int     tid = threadIdx.x;

    const int64_t base = i * QK_NVFP4;
    if (base >= ne) {
        return;
    }

    const block_nvfp4 * x = (const block_nvfp4 *) vx;
    const block_nvfp4 & xb = x[i];

    const int sub = tid / (QK_NVFP4_SUB / 2);
    const int j = tid % (QK_NVFP4_SUB / 2);

    const float d = ggml_cuda_ue4m3_to_fp32(xb.d[sub]);
    const uint8_t q = xb.qs[sub * (QK_NVFP4_SUB / 2) + j];

    const int64_t y0 = base + sub * QK_NVFP4_SUB + j;
    const int64_t y1 = y0 + QK_NVFP4_SUB / 2;

    yy[y0] = ggml_cuda_cast<dst_t>(d * kvalues_mxfp4[q & 0x0F]);
    yy[y1] = ggml_cuda_cast<dst_t>(d * kvalues_mxfp4[q >> 4]);
}

template <typename dst_t>
static void dequantize_row_nvfp4_cuda(
        const void * vx,
        dst_t * y,
        const int64_t k,
        cudaStream_t stream) {
    GGML_ASSERT(k % QK_NVFP4 == 0);
    const int nb = k / QK_NVFP4;
    dequantize_block_nvfp4<<<nb, 32, 0, stream>>>(vx, y, k);
}
template <typename src_t, typename dst_t>
static __global__ void convert_unary(
        const void * __restrict__ vx, dst_t * __restrict__ y, const int64_t ne00, const int64_t ne01,
        const int64_t ne0203, const uint3 ne02,
        const int64_t s01, const int64_t s02, const int64_t s03) {
    const int64_t i00 = (int64_t)blockDim.x*blockIdx.x + threadIdx.x;

    if (i00 >= ne00) {
        return;
    }

    const src_t * x = (const src_t *) vx;

    for (int64_t i01 = blockIdx.y; i01 < ne01; i01 += gridDim.y) {
        for (int64_t i0203 = blockIdx.z; i0203 < ne0203; i0203 += gridDim.z) {
            const uint2 dm = fast_div_modulo((uint32_t)i0203, ne02);
            const int64_t i02 = dm.y;
            const int64_t i03 = dm.x;

            const int64_t ix = i03*s03 + i02*s02 + i01*s01 + i00;
            const int64_t iy = (i0203*ne01 + i01)*ne00 + i00;
            y[iy] = ggml_cuda_cast<dst_t>(x[ix]);
        }
    }
}

template <typename src_t, typename dst_t>
static void convert_unary_cuda(const void * vx, dst_t * y,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t s01, const int64_t s02, const int64_t s03, cudaStream_t stream) {
    const int64_t ne0203 = ne02*ne03;
    const uint3 ne02_fdv = init_fastdiv_values(ne02);
    const dim3 num_blocks((ne00 + CUDA_DEQUANTIZE_BLOCK_SIZE - 1) / CUDA_DEQUANTIZE_BLOCK_SIZE, (int)std::min(ne01, (int64_t)65535), (int)std::min(ne0203, (int64_t)65535));
    convert_unary<src_t><<<num_blocks, CUDA_DEQUANTIZE_BLOCK_SIZE, 0, stream>>>
        (vx, y, ne00, ne01, ne0203, ne02_fdv, s01, s02, s03);
}

template <typename src_t, typename dst_t>
static void convert_unary_cont_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    convert_unary_cuda<src_t>(vx, y, k, 1, 1, 1, k, k, k, stream);
}

to_bf16_cuda_t ggml_get_to_bf16_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:
            return dequantize_block_cont_cuda<QK1_0, QR1_0, dequantize_q1_0>;
        case GGML_TYPE_Q2_0:
            return dequantize_block_cont_cuda<QK2_0, QR2_0, dequantize_q2_0>;
        case GGML_TYPE_Q4_0:
            return dequantize_row_q4_0_cuda;
        case GGML_TYPE_Q4_1:
            return dequantize_row_q4_1_cuda;
        case GGML_TYPE_Q5_0:
            return dequantize_block_cont_cuda<QK5_0, QR5_0, dequantize_q5_0>;
        case GGML_TYPE_Q5_1:
            return dequantize_block_cont_cuda<QK5_1, QR5_1, dequantize_q5_1>;
        case GGML_TYPE_Q8_0:
            return dequantize_block_cont_cuda<QK8_0, QR8_0, dequantize_q8_0>;
        case GGML_TYPE_Q2_K:
            return dequantize_row_q2_K_cuda;
        case GGML_TYPE_Q3_K:
            return dequantize_row_q3_K_cuda;
        case GGML_TYPE_Q4_K:
            return dequantize_row_q4_K_cuda;
        case GGML_TYPE_Q5_K:
            return dequantize_row_q5_K_cuda;
        case GGML_TYPE_Q6_K:
            return dequantize_row_q6_K_cuda;
        case GGML_TYPE_IQ2_XXS:
            return dequantize_row_iq2_xxs_cuda;
        case GGML_TYPE_IQ2_XS:
            return dequantize_row_iq2_xs_cuda;
        case GGML_TYPE_IQ2_S:
            return dequantize_row_iq2_s_cuda;
        case GGML_TYPE_IQ3_XXS:
            return dequantize_row_iq3_xxs_cuda;
        case GGML_TYPE_IQ1_S:
            return dequantize_row_iq1_s_cuda;
        case GGML_TYPE_IQ1_M:
            return dequantize_row_iq1_m_cuda;
        case GGML_TYPE_IQ4_NL:
            return dequantize_row_iq4_nl_cuda;
        case GGML_TYPE_IQ4_XS:
            return dequantize_row_iq4_xs_cuda;
        case GGML_TYPE_IQ3_S:
            return dequantize_row_iq3_s_cuda;
        case GGML_TYPE_MXFP4:
            return dequantize_row_mxfp4_cuda;
        case GGML_TYPE_NVFP4:
            return dequantize_row_nvfp4_cuda;
        case GGML_TYPE_F32:
            return convert_unary_cont_cuda<float>;
        case GGML_TYPE_F16:
            return convert_unary_cont_cuda<half>;
        default:
            return nullptr;
    }
}

to_fp16_cuda_t ggml_get_to_fp16_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:
            return dequantize_block_cont_cuda<QK1_0, QR1_0, dequantize_q1_0>;
        case GGML_TYPE_Q2_0:
            return dequantize_block_cont_cuda<QK2_0, QR2_0, dequantize_q2_0>;
        case GGML_TYPE_Q4_0:
            return dequantize_row_q4_0_cuda;
        case GGML_TYPE_Q4_1:
            return dequantize_row_q4_1_cuda;
        case GGML_TYPE_Q5_0:
            return dequantize_block_cont_cuda<QK5_0, QR5_0, dequantize_q5_0>;
        case GGML_TYPE_Q5_1:
            return dequantize_block_cont_cuda<QK5_1, QR5_1, dequantize_q5_1>;
        case GGML_TYPE_Q8_0:
            if (fp16_available(ggml_cuda_info().devices[ggml_cuda_get_device()].cc)) {
                return dequantize_block_q8_0_f16_cuda;
            }
            return dequantize_block_cont_cuda<QK8_0, QR8_0, dequantize_q8_0>;
        case GGML_TYPE_Q2_K:
            return dequantize_row_q2_K_cuda;
        case GGML_TYPE_Q3_K:
            return dequantize_row_q3_K_cuda;
        case GGML_TYPE_Q4_K:
            return dequantize_row_q4_K_cuda;
        case GGML_TYPE_Q5_K:
            return dequantize_row_q5_K_cuda;
        case GGML_TYPE_Q6_K:
            return dequantize_row_q6_K_cuda;
        case GGML_TYPE_IQ2_XXS:
            return dequantize_row_iq2_xxs_cuda;
        case GGML_TYPE_IQ2_XS:
            return dequantize_row_iq2_xs_cuda;
        case GGML_TYPE_IQ2_S:
            return dequantize_row_iq2_s_cuda;
        case GGML_TYPE_IQ3_XXS:
            return dequantize_row_iq3_xxs_cuda;
        case GGML_TYPE_IQ1_S:
            return dequantize_row_iq1_s_cuda;
        case GGML_TYPE_IQ1_M:
            return dequantize_row_iq1_m_cuda;
        case GGML_TYPE_IQ4_NL:
            return dequantize_row_iq4_nl_cuda;
        case GGML_TYPE_IQ4_XS:
            return dequantize_row_iq4_xs_cuda;
        case GGML_TYPE_IQ3_S:
            return dequantize_row_iq3_s_cuda;
        case GGML_TYPE_MXFP4:
            return dequantize_row_mxfp4_cuda;
        case GGML_TYPE_NVFP4:
            return dequantize_row_nvfp4_cuda;
        case GGML_TYPE_F32:
            return convert_unary_cont_cuda<float>;
        case GGML_TYPE_BF16:
            return convert_unary_cont_cuda<nv_bfloat16>;
        default:
            return nullptr;
    }
}

to_fp32_cuda_t ggml_get_to_fp32_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:
            return dequantize_block_cont_cuda<QK1_0, QR1_0, dequantize_q1_0>;
        case GGML_TYPE_Q2_0:
            return dequantize_block_cont_cuda<QK2_0, QR2_0, dequantize_q2_0>;
        case GGML_TYPE_Q4_0:
            return dequantize_row_q4_0_cuda;
        case GGML_TYPE_Q4_1:
            return dequantize_row_q4_1_cuda;
        case GGML_TYPE_Q5_0:
            return dequantize_block_cont_cuda<QK5_0, QR5_0, dequantize_q5_0>;
        case GGML_TYPE_Q5_1:
            return dequantize_block_cont_cuda<QK5_1, QR5_1, dequantize_q5_1>;
        case GGML_TYPE_Q8_0:
            return dequantize_block_cont_cuda<QK8_0, QR8_0, dequantize_q8_0>;
        case GGML_TYPE_Q2_K:
            return dequantize_row_q2_K_cuda;
        case GGML_TYPE_Q3_K:
            return dequantize_row_q3_K_cuda;
        case GGML_TYPE_Q4_K:
            return dequantize_row_q4_K_cuda;
        case GGML_TYPE_Q5_K:
            return dequantize_row_q5_K_cuda;
        case GGML_TYPE_Q6_K:
            return dequantize_row_q6_K_cuda;
        case GGML_TYPE_IQ2_XXS:
            return dequantize_row_iq2_xxs_cuda;
        case GGML_TYPE_IQ2_XS:
            return dequantize_row_iq2_xs_cuda;
        case GGML_TYPE_IQ2_S:
            return dequantize_row_iq2_s_cuda;
        case GGML_TYPE_IQ3_XXS:
            return dequantize_row_iq3_xxs_cuda;
        case GGML_TYPE_IQ1_S:
            return dequantize_row_iq1_s_cuda;
        case GGML_TYPE_IQ1_M:
            return dequantize_row_iq1_m_cuda;
        case GGML_TYPE_IQ4_NL:
            return dequantize_row_iq4_nl_cuda;
        case GGML_TYPE_IQ4_XS:
            return dequantize_row_iq4_xs_cuda;
        case GGML_TYPE_IQ3_S:
            return dequantize_row_iq3_s_cuda;
        case GGML_TYPE_MXFP4:
            return dequantize_row_mxfp4_cuda;
        case GGML_TYPE_NVFP4:
            return dequantize_row_nvfp4_cuda;
        case GGML_TYPE_F16:
            return convert_unary_cont_cuda<half>;
        case GGML_TYPE_BF16:
            return convert_unary_cont_cuda<nv_bfloat16>;
        default:
            return nullptr;
    }
}

to_fp16_nc_cuda_t ggml_get_to_fp16_nc_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
            return convert_unary_cuda<float>;
        case GGML_TYPE_Q1_0:
            return dequantize_block_cuda<QK1_0, QR1_0, dequantize_q1_0>;
        case GGML_TYPE_Q2_0:
            return dequantize_block_cuda<QK2_0, QR2_0, dequantize_q2_0>;
        case GGML_TYPE_Q4_0:
            return dequantize_block_cuda<QK4_0, QR4_0, dequantize_q4_0>;
        case GGML_TYPE_Q4_1:
            return dequantize_block_cuda<QK4_1, QR4_1, dequantize_q4_1>;
        case GGML_TYPE_Q5_0:
            return dequantize_block_cuda<QK5_0, QR5_0, dequantize_q5_0>;
        case GGML_TYPE_Q5_1:
            return dequantize_block_cuda<QK5_1, QR5_1, dequantize_q5_1>;
        case GGML_TYPE_Q8_0:
            return dequantize_block_cuda<QK8_0, QR8_0, dequantize_q8_0>;
        case GGML_TYPE_BF16:
            return convert_unary_cuda<nv_bfloat16>;
        default:
            return nullptr;
    }
}

to_bf16_nc_cuda_t ggml_get_to_bf16_nc_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
            return convert_unary_cuda<float, nv_bfloat16>;
        case GGML_TYPE_Q1_0:
            return dequantize_block_cuda<QK1_0, QR1_0, dequantize_q1_0>;
        case GGML_TYPE_Q2_0:
            return dequantize_block_cuda<QK2_0, QR2_0, dequantize_q2_0>;
        case GGML_TYPE_Q4_0:
            return dequantize_block_cuda<QK4_0, QR4_0, dequantize_q4_0>;
        case GGML_TYPE_Q4_1:
            return dequantize_block_cuda<QK4_1, QR4_1, dequantize_q4_1>;
        case GGML_TYPE_Q5_0:
            return dequantize_block_cuda<QK5_0, QR5_0, dequantize_q5_0>;
        case GGML_TYPE_Q5_1:
            return dequantize_block_cuda<QK5_1, QR5_1, dequantize_q5_1>;
        case GGML_TYPE_Q8_0:
            return dequantize_block_cuda<QK8_0, QR8_0, dequantize_q8_0>;
        case GGML_TYPE_F16:
            return convert_unary_cuda<half, nv_bfloat16>;
        default:
            return nullptr;
    }
}

to_fp32_nc_cuda_t ggml_get_to_fp32_nc_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F16:
            return convert_unary_cuda<half, float>;
        case GGML_TYPE_Q1_0:
            return dequantize_block_cuda<QK1_0, QR1_0, dequantize_q1_0>;
        case GGML_TYPE_Q2_0:
            return dequantize_block_cuda<QK2_0, QR2_0, dequantize_q2_0>;
        case GGML_TYPE_Q4_0:
            return dequantize_block_cuda<QK4_0, QR4_0, dequantize_q4_0>;
        case GGML_TYPE_Q4_1:
            return dequantize_block_cuda<QK4_1, QR4_1, dequantize_q4_1>;
        case GGML_TYPE_Q5_0:
            return dequantize_block_cuda<QK5_0, QR5_0, dequantize_q5_0>;
        case GGML_TYPE_Q5_1:
            return dequantize_block_cuda<QK5_1, QR5_1, dequantize_q5_1>;
        case GGML_TYPE_Q8_0:
            return dequantize_block_cuda<QK8_0, QR8_0, dequantize_q8_0>;
        case GGML_TYPE_BF16:
            return convert_unary_cuda<nv_bfloat16, float>;
        default:
            return nullptr;
    }
}
