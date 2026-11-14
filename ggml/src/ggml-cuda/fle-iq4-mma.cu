// FLE_IQ4_MMA=1 (FLE assimiliert Infernix, Nr. 1103, Mechanismus 6): eigener Prüfkern für IQ4_XS, Q5_K, Q4_K und Q6_K bei 9..16 Spalten.
//
// Warum: Katy prüft die Entwürfe des DSpine-Kopfs mit 16 Spalten. Dafür nimmt FLE heute mul_mat_q (Aktivierungen erst nach
// q8_1 quantisiert, int8-mma, danach stream-K-Fixup); das ist 72,5 % der Kernelzeit und liest IQ4_XS mit ~610 GB/s gegen
// ~775 GB/s bei F16 (gemv_bench, RTX PRO 4500). Infernix rechnet diese Breiten mit mma auf 16-Bit-Aktivierungen
// (src/ops/linear/q4/q4_a16_*.cuh, Apache-2.0): die 4-Bit-Codes werden in Registern zu 16-Bit-Werten, gerechnet wird mit
// mma.sync, K ist geteilt, Gewichte und Aktivierungen laufen über eine Bühne im Shared Memory.
//
// Werte als F16 mit Versatz (Fassung 6): jedes Gewicht liegt als vorzeichenloses Byte b vor (IQ4_XS: Tabellenwert + 128,
// Q5_K: q in 0..31); zwei Bytes werden mit einem prmt zu zwei F16 1024 + b (exakt, b < 1024). Den Versatz zieht der Kern je
// Teilblock über die Summe der Aktivierungen ab:  Σ (V + b)·x = Σ b·x + V·Σx.  Ergebnis je Zeile und Teilblock s:
//   IQ4_XS   d·(ls−32) · (G − 1152·Σx)                          (V = 1024 + 128)
//   Q5_K     d·sc · (G − 1024·Σx) − dmin·m · Σx                 (V = 1024; die Minima fallen über dieselbe Summe heraus)
// zusammengefasst  so·G − of·Σx  mit je Zeile und Teilblock vorgerechnetem so und of.
// Die Reihenfolge über k ist für A und B gleich vertauscht, damit jeder Thread je Zeile, Teilblock und mma-Schritt ein
// 32-Bit-Wort mit vier Werten und je Token ein 8-Byte-Wort der Aktivierungen braucht:
//   mma-k  2·lid+e      ↔ Element 4·lid+e      (e = 0, 1)
//   mma-k  2·lid+8+e    ↔ Element 4·lid+2+e
//
// Aufbau: ein Block rechnet FLE_IQ4_MMA_ROWS = 16·MT·W Zeilen; alle Warps gehen über dieselben 256er-Blöcke von K. Die
// Aktivierungen (F16) und ihre Summen je 32 liegen einmal im Shared Memory, die Gewichte je Warp in seiner Bühne, alles per
// cp.async doppelt gepuffert. Bei wenigen Zeilen wird K auf bis zu 8 Blöcke geteilt; die Teile werden in fester
// Reihenfolge summiert (reproduzierbar).
// Ohne Schalter läuft alles wie vorher. Ergebnis nicht bitgleich zu mul_mat_q (F16 statt q8_1 auf der Aktivierung).

#include "fle-iq4-mma.cuh"
#include "vecdotq.cuh"

#include <cstdlib>

static bool fle_iq4_mma_an() {
    static const bool an = [] {
        const char * v = getenv("FLE_IQ4_MMA");
        return v != nullptr && atoi(v) != 0;
    }();
    return an;
}

// Q6_K eigens: FLE_IQ4_MMA_Q6=1 (zusätzlich zu FLE_IQ4_MMA), damit er getrennt gemessen werden kann
static bool fle_iq4_mma_q6_an() {
    static const bool an = [] {
        const char * v = getenv("FLE_IQ4_MMA_Q6");
        return v != nullptr && atoi(v) != 0;
    }();
    return an;
}

bool fle_iq4_mma_passt(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc) {
    if (!fle_iq4_mma_an()) {
        return false;
    }
    if (src0->type == GGML_TYPE_Q6_K && !fle_iq4_mma_q6_an()) {
        return false;
    }
    if (!(GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_AMPERE)) {
        return false;
    }
    if ((src0->type != GGML_TYPE_IQ4_XS && src0->type != GGML_TYPE_Q5_K && src0->type != GGML_TYPE_Q4_K && src0->type != GGML_TYPE_Q6_K) || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    const int64_t ncols = src1->ne[1];
    if (ncols < FLE_IQ4_MMA_MIN || ncols > FLE_IQ4_MMA_MAX) {
        return false;
    }
    // nur der einfache Fall: eine Matrix, eine Ebene, alles zusammenhängend, Zeilen ein Vielfaches der Blockhöhe
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) {
        return false;
    }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1) || !ggml_is_contiguous(dst)) {
        return false;
    }
    if (src0->ne[0] % QK_K != 0 || src0->ne[1] % FLE_IQ4_MMA_ROWS != 0) {
        return false;
    }
    return true;
}

// ── Aktivierungen F32 → F16, [T][K], dazu je 32 Werte die Summe (aus den gerundeten Werten, für den Versatz) ─────────────
static __global__ void fle_iq4_f32_f16(const float * __restrict__ x, half * __restrict__ y, float * __restrict__ summe, const int64_t nteil) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;  // ein Teilblock aus 32 je Thread
    if (i < nteil) {
        const float4 * q = (const float4 *) (x + i * 32);
        half2 *        z = (half2 *) (y + i * 32);
        float s = 0.0f;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float4 v = q[j];
            const half2 h0 = __floats2half2_rn(v.x, v.y);
            const half2 h1 = __floats2half2_rn(v.z, v.w);
            z[2 * j]     = h0;
            z[2 * j + 1] = h1;
            s += __low2float(h0) + __high2float(h0) + __low2float(h1) + __high2float(h1);
        }
        summe[i] = s;
    }
}

static __device__ __forceinline__ void fle_mma_f16(float & d0, float & d1, float & d2, float & d3,
                                                   const unsigned a0, const unsigned a1, const unsigned a2, const unsigned a3,
                                                   const unsigned b0, const unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                 : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// ── je Gewichtstyp: Blockform, Skalen je Teilblock, vier Werte je Thread als Bytes ─────────────────────────────────────────
template <ggml_type TYP> struct fle_art;

template <> struct fle_art<GGML_TYPE_IQ4_XS> {
    using block = block_iq4_xs;
    static constexpr bool  affin = false;    // Versatz ist eine Konstante: so·(G − V·Σx)
    static constexpr float V     = 1152.0f;
    // so und of für die 8 Teilblöcke einer Zeile
    static __device__ __forceinline__ void skalen(const block * b, float * so, float * of) {
        const float    d  = __half2float(b->d);
        const uint32_t sh = b->scales_h;
        const uint32_t sl = *(const uint32_t *) b->scales_l;
#pragma unroll
        for (int s = 0; s < QK_K / 32; ++s) {
            const float v = d * (float) ((int) (((sl >> (4 * s)) & 0xF) | (((sh >> (2 * s)) & 3) << 4)) - 32);
            so[s] = v;
            of[s] = 1152.0f * v;
        }
    }
    // vier Bytes (Wert + 128) je mma-Schritt des Teilblocks s, Elemente 16h + 4lid .. +3; .x = h 0, .y = h 1
    static __device__ __forceinline__ uint2 werte(const block * b, const int s, const int lid) {
        const int2 w = get_int_from_table_16(*(const int *) (b->qs + s * 16 + 4 * lid), kvalues_iq4nl);
        return make_uint2((unsigned) w.x ^ 0x80808080u, (unsigned) w.y ^ 0x80808080u);
    }
};

template <> struct fle_art<GGML_TYPE_Q5_K> {
    using block = block_q5_K;
    static constexpr bool  affin = true;     // Minima je Teilblock: so·G − of·Σx
    static constexpr float V     = 1024.0f;
    static __device__ __forceinline__ void skalen(const block * b, float * so, float * of) {
        const float d    = __low2float(b->dm);
        const float dmin = __high2float(b->dm);
#pragma unroll
        for (int s = 0; s < QK_K / 32; ++s) {
            uint8_t sc, m;
            if (s < 4) {
                sc = b->scales[s] & 63;
                m  = b->scales[s + 4] & 63;
            } else {
                sc = (b->scales[s + 4] & 0xF) | ((b->scales[s - 4] >> 6) << 4);
                m  = (b->scales[s + 4] >> 4) | ((b->scales[s] >> 6) << 4);
            }
            so[s] = d * (float) sc;
            of[s] = 1024.0f * so[s] + dmin * (float) m;
        }
    }
    // Teilblock s = 2j + hh: unteres (hh 0) oder oberes Nibble von qs[32j + l], Bit s von qh[l]; l = 16h + 4lid .. +3
    static __device__ __forceinline__ uint2 werte(const block * b, const int s, const int lid) {
        const int j  = s / 2;
        const int hh = s % 2;
        const uint32_t q0 = *(const uint32_t *) (b->qs + 32 * j + 4 * lid);
        const uint32_t q1 = *(const uint32_t *) (b->qs + 32 * j + 16 + 4 * lid);
        const uint32_t h0 = *(const uint32_t *) (b->qh + 4 * lid);
        const uint32_t h1 = *(const uint32_t *) (b->qh + 16 + 4 * lid);
        const uint32_t v0 = ((q0 >> (4 * hh)) & 0x0F0F0F0Fu) | (((h0 >> s) & 0x01010101u) << 4);
        const uint32_t v1 = ((q1 >> (4 * hh)) & 0x0F0F0F0Fu) | (((h1 >> s) & 0x01010101u) << 4);
        return make_uint2(v0, v1);
    }
};

// Q4_K wie Q5_K ohne das fünfte Bit (u. a. der DSpine-Kopf von Katy, Q4_K_M)
template <> struct fle_art<GGML_TYPE_Q4_K> {
    using block = block_q4_K;
    static constexpr bool  affin = true;
    static constexpr float V     = 1024.0f;
    static __device__ __forceinline__ void skalen(const block * b, float * so, float * of) {
        const float d    = __low2float(b->dm);
        const float dmin = __high2float(b->dm);
#pragma unroll
        for (int s = 0; s < QK_K / 32; ++s) {
            uint8_t sc, m;
            if (s < 4) {
                sc = b->scales[s] & 63;
                m  = b->scales[s + 4] & 63;
            } else {
                sc = (b->scales[s + 4] & 0xF) | ((b->scales[s - 4] >> 6) << 4);
                m  = (b->scales[s + 4] >> 4) | ((b->scales[s] >> 6) << 4);
            }
            so[s] = d * (float) sc;
            of[s] = 1024.0f * so[s] + dmin * (float) m;
        }
    }
    static __device__ __forceinline__ uint2 werte(const block * b, const int s, const int lid) {
        const int j  = s / 2;
        const int hh = s % 2;
        const uint32_t q0 = *(const uint32_t *) (b->qs + 32 * j + 4 * lid);
        const uint32_t q1 = *(const uint32_t *) (b->qs + 32 * j + 16 + 4 * lid);
        return make_uint2((q0 >> (4 * hh)) & 0x0F0F0F0Fu, (q1 >> (4 * hh)) & 0x0F0F0F0Fu);
    }
};

// NT Tokenkacheln zu 8. Block = W Warps × 16·MT Zeilen; blockIdx.y = Teil von K (ks Teile).
template <int NT, ggml_type TYP>
static __global__ void __launch_bounds__(32 * FLE_IQ4_MMA_WARPS) fle_iq4_mma_kern(
        const typename fle_art<TYP>::block * __restrict__ w, const half * __restrict__ x, const float * __restrict__ xsum,
        float * __restrict__ dst, const int nrows, const int k, const int ncols, const int ks) {
    using art   = fle_art<TYP>;
    using block = typename art::block;
    constexpr int W  = FLE_IQ4_MMA_WARPS;
    constexpr int MT = FLE_IQ4_MMA_MT;                     // 16·MT Zeilen je Warp
    constexpr int T  = NT * 8;                             // Tokens im Block
    constexpr int ZW = (int) (sizeof(block) / 8);          // 8-Byte-Wörter je Zeile und Block (IQ4_XS 17, Q5_K 22)
    static_assert(sizeof(block) % 8 == 0, "Block muss in 8-Byte-Wörter aufgehen");
    constexpr int NW = 16 * MT * ZW;                       // Wörter je Warp und Block
    constexpr int JE = (NW + 31) / 32;                     // je Lane
    constexpr int XW = T * QK_K / 8;                       // Aktivierungen eines Blocks in 16-Byte-Wörtern (8 F16)
    constexpr int XJ = (XW + 32 * W - 1) / (32 * W);       // je Thread
    constexpr int SW = T * 2;                              // Summen eines Blocks (T × 8 float) in 16-Byte-Wörtern

    __shared__ float skal[W][16 * MT][QK_K / 32];
    __shared__ float offs[W][16 * MT][QK_K / 32];
    extern __shared__ __align__(16) unsigned char dyn[];
    uint2 (*buehne)[W][NW] = (uint2 (*)[W][NW]) dyn;
    uint4 (*xs)[XW]        = (uint4 (*)[XW]) (dyn + 2 * sizeof(uint2) * W * NW);
    float (*su8)[T][8]     = (float (*)[T][8]) (dyn + 2 * sizeof(uint2) * W * NW + 2 * sizeof(uint4) * XW);

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int gid  = lane >> 2;
    const int lid  = lane & 3;
    const int nsb  = k / QK_K;                                          // Blöcke je Zeile
    const int wrow = blockIdx.x * (16 * MT * W) + warp * (16 * MT);    // erste Zeile dieses Warps

    const int sb0 = (int) ((int64_t) nsb * blockIdx.y / ks);
    const int sb1 = (int) ((int64_t) nsb * (blockIdx.y + 1) / ks);

    float acc[MT][NT][4];
#pragma unroll
    for (int mi = 0; mi < MT; ++mi)
#pragma unroll
        for (int ni = 0; ni < NT; ++ni)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[mi][ni][c] = 0.0f;

    const auto lade = [&](const int sb, const int st) {
#pragma unroll
        for (int j = 0; j < JE; ++j) {
            const int idx = lane + 32 * j;
            if (idx < NW) {
                const int r = idx / ZW;
                const int o = idx % ZW;
                const unsigned ziel = (unsigned) __cvta_generic_to_shared(&buehne[st][warp][idx]);
                const void *   quel = (const uint2 *) (w + (int64_t) (wrow + r) * nsb + sb) + o;
                asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" :: "r"(ziel), "l"(quel));
            }
        }
#pragma unroll
        for (int j = 0; j < XJ; ++j) {
            const int idx = threadIdx.x + 32 * W * j;
            if (idx < XW) {
                const int tok = idx / (QK_K / 8);
                const int o   = idx % (QK_K / 8);
                const bool gilt = tok < ncols;
                const unsigned ziel = (unsigned) __cvta_generic_to_shared(&xs[st][idx]);
                const void *   quel = gilt ? (const void *) ((const uint4 *) (x + (int64_t) tok * k + (int64_t) sb * QK_K) + o) : (const void *) x;
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(ziel), "l"(quel), "r"(gilt ? 16 : 0));
            }
        }
        for (int idx = threadIdx.x; idx < SW; idx += 32 * W) {
            const int tok = idx / 2;
            const int o   = idx % 2;
            const bool gilt = tok < ncols;
            const unsigned ziel = (unsigned) __cvta_generic_to_shared(&su8[st][tok][4 * o]);
            const void *   quel = gilt ? (const void *) (xsum + (int64_t) tok * (k / 32) + sb * 8 + 4 * o) : (const void *) xsum;
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(ziel), "l"(quel), "r"(gilt ? 16 : 0));
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };

    if (sb0 < sb1) {
        lade(sb0, 0);
    }
    for (int sb = sb0; sb < sb1; ++sb) {
        const int st = (sb - sb0) & 1;
        if (sb + 1 < sb1) {
            lade(sb + 1, st ^ 1);  // nächster Block läuft, während dieser rechnet
            asm volatile("cp.async.wait_group 1;\n" ::);
        } else {
            asm volatile("cp.async.wait_group 0;\n" ::);
        }
        __syncthreads();  // Bühne st ist da
        for (int zr = lane; zr < 16 * MT; zr += 32) {
            // Skalen und Versätze dieses Blocks: jede Lane rechnet die ihrer Zeile(n) einmal vor
            art::skalen((const block *) (&buehne[st][warp][zr * ZW]), skal[warp][zr], offs[warp][zr]);
        }
        __syncwarp();
        const half * xb = (const half *) xs[st];

#pragma unroll
        for (int ib = 0; ib < QK_K / 32; ++ib) {
            // Aktivierungen dieses Teilblocks: je Tokenkachel ni der Token ni*8+gid, Elemente 4*lid .. 4*lid+3 (+16 für die zweite Hälfte)
            unsigned b[NT][2][2];
#pragma unroll
            for (int ni = 0; ni < NT; ++ni) {
                const int tok = ni * 8 + gid;
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const uint2 v = *(const uint2 *) (xb + tok * QK_K + ib * 32 + h * 16 + 4 * lid);
                    b[ni][h][0] = v.x;
                    b[ni][h][1] = v.y;
                }
            }
#pragma unroll
            for (int mi = 0; mi < MT; ++mi) {
                const block * bo = (const block *) (&buehne[st][warp][(mi * 16 + gid) * ZW]);
                const block * bu = (const block *) (&buehne[st][warp][(mi * 16 + gid + 8) * ZW]);
                const uint2 wo = art::werte(bo, ib, lid);
                const uint2 wu = art::werte(bu, ib, lid);
                float g[NT][4];
#pragma unroll
                for (int ni = 0; ni < NT; ++ni)
#pragma unroll
                    for (int c = 0; c < 4; ++c) g[ni][c] = 0.0f;
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    // je zwei Bytes mit 0x64 davor: F16 1024 + Byte (exakt)
                    const unsigned vo = h ? wo.y : wo.x;
                    const unsigned vu = h ? wu.y : wu.x;
                    const unsigned a0 = __byte_perm(vo, 0x6464u, 0x5140);  // Zeile gid,   Elemente 4lid, 4lid+1
                    const unsigned a1 = __byte_perm(vu, 0x6464u, 0x5140);  // Zeile gid+8
                    const unsigned a2 = __byte_perm(vo, 0x6464u, 0x5342);  // Zeile gid,   Elemente 4lid+2, 4lid+3
                    const unsigned a3 = __byte_perm(vu, 0x6464u, 0x5342);  // Zeile gid+8
#pragma unroll
                    for (int ni = 0; ni < NT; ++ni) {
                        fle_mma_f16(g[ni][0], g[ni][1], g[ni][2], g[ni][3], a0, a1, a2, a3, b[ni][h][0], b[ni][h][1]);
                    }
                }
                const float so = skal[warp][mi * 16 + gid][ib];
                const float su = skal[warp][mi * 16 + gid + 8][ib];
                if constexpr (!art::affin) {
#pragma unroll
                    for (int ni = 0; ni < NT; ++ni) {
                        // so·(G − V·Σx) je Token (fester Versatz)
                        const float x0 = art::V * su8[st][ni * 8 + 2 * lid][ib];
                        const float x1 = art::V * su8[st][ni * 8 + 2 * lid + 1][ib];
                        acc[mi][ni][0] = fmaf(g[ni][0] - x0, so, acc[mi][ni][0]);
                        acc[mi][ni][1] = fmaf(g[ni][1] - x1, so, acc[mi][ni][1]);
                        acc[mi][ni][2] = fmaf(g[ni][2] - x0, su, acc[mi][ni][2]);
                        acc[mi][ni][3] = fmaf(g[ni][3] - x1, su, acc[mi][ni][3]);
                    }
                } else {
                    const float oo = offs[warp][mi * 16 + gid][ib];
                    const float ou = offs[warp][mi * 16 + gid + 8][ib];
#pragma unroll
                    for (int ni = 0; ni < NT; ++ni) {
                        // so·G − of·Σx je Token (Versatz und Minima)
                        const float x0 = su8[st][ni * 8 + 2 * lid][ib];
                        const float x1 = su8[st][ni * 8 + 2 * lid + 1][ib];
                        acc[mi][ni][0] = fmaf(g[ni][0], so, fmaf(-oo, x0, acc[mi][ni][0]));
                        acc[mi][ni][1] = fmaf(g[ni][1], so, fmaf(-oo, x1, acc[mi][ni][1]));
                        acc[mi][ni][2] = fmaf(g[ni][2], su, fmaf(-ou, x0, acc[mi][ni][2]));
                        acc[mi][ni][3] = fmaf(g[ni][3], su, fmaf(-ou, x1, acc[mi][ni][3]));
                    }
                }
            }
        }
        __syncthreads();  // alle sind fertig, bevor die nächste Runde in diese Bühne lädt
    }

    // Schreiben: dst ist [ncols][nrows] (ne0 = Zeilen); bei ks Teilen von K je Teil eine Ebene
#pragma unroll
    for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
        for (int ni = 0; ni < NT; ++ni) {
            const int r  = wrow + mi * 16 + gid;
            const int t0 = ni * 8 + 2 * lid;
            const float v[4] = {acc[mi][ni][0], acc[mi][ni][1], acc[mi][ni][2], acc[mi][ni][3]};
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                const int t  = t0 + (c & 1);
                const int rr = r + 8 * (c >> 1);
                if (t < ncols) {
                    dst[(int64_t) blockIdx.y * ncols * nrows + (int64_t) t * nrows + rr] = v[c];
                }
            }
        }
    }
}

// ── Q6_K (Fassung 9): Katys output.weight (6,8 % der Bytes) und Teile des DSpine-Kopfs ─────────────────────────────────────
// Eigener Kern, damit der abgestimmte Pfad oben unberührt bleibt. Unterschiede:
//   * Blöcke zu 210 Bytes liegen nur auf 2-Byte-Grenzen: je Zeile und Block wird ein auf 16 Bytes ausgerichtetes Fenster aus
//     15 Wörtern (240 Bytes) per cp.async geladen (über das Tensorende hinaus mit Nullen), gelesen wird ab dem Versatz darin.
//   * eine Skala je 16 Werte: jeder mma-Schritt (16 Werte) wird einzeln skaliert; Summen der Aktivierungen je 16.
//   * Werte q (0..63) als F16 1024 + q, Gewicht d·sc·(q − 32): Versatz 1024 + 32 = 1056.
// Zuordnung (dequantize_row_q6_K): Teilblock ib = 4n + j aus 32 Werten, Hälfte h; Elemente 128n + 32j + 16h + l (l 0..15):
//   ql[64n + 32(j&1) + 16h + l], Nibble j>>1;  qh[32n + 16h + l], Bits 2j..2j+1;  Skala scales[8n + 2j + h] = scales[2ib + h].

// Summen der Aktivierungen je 16 (Q6_K)
static __global__ void fle_q6k_f32_f16(const float * __restrict__ x, half * __restrict__ y, float * __restrict__ summe, const int64_t nteil) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;  // 32 Werte je Thread, zwei Summen
    if (i < nteil) {
        const float4 * q = (const float4 *) (x + i * 32);
        half2 *        z = (half2 *) (y + i * 32);
        float s[2] = {0.0f, 0.0f};
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float4 v = q[j];
            const half2 h0 = __floats2half2_rn(v.x, v.y);
            const half2 h1 = __floats2half2_rn(v.z, v.w);
            z[2 * j]     = h0;
            z[2 * j + 1] = h1;
            s[j / 4] += __low2float(h0) + __high2float(h0) + __low2float(h1) + __high2float(h1);
        }
        summe[2 * i]     = s[0];
        summe[2 * i + 1] = s[1];
    }
}

// vier Bytes ab einer beliebigen (geraden) Stelle im Shared Memory
static __device__ __forceinline__ uint32_t fle_lade32(const uint8_t * p) {
    const uintptr_t a  = (uintptr_t) p;
    const uint32_t * w = (const uint32_t *) (a & ~(uintptr_t) 3);
    return __funnelshift_r(w[0], w[1], (unsigned) ((a & 3) * 8));
}

static __global__ void __launch_bounds__(32 * FLE_IQ4_MMA_WARPS) fle_q6k_mma_kern(
        const uint8_t * __restrict__ w, const int64_t w_bytes, const half * __restrict__ x, const float * __restrict__ xsum,
        float * __restrict__ dst, const int nrows, const int k, const int ncols, const int ks) {
    constexpr int NT = 2;
    constexpr int W  = FLE_IQ4_MMA_WARPS;
    constexpr int MT = FLE_IQ4_MMA_MT;
    constexpr int T  = NT * 8;
    constexpr int BL = (int) sizeof(block_q6_K);           // 210
    constexpr int ZW = 15;                                 // 16-Byte-Wörter je Zeile (Fenster 240 Bytes)
    constexpr int NW = 16 * MT * ZW;
    constexpr int JE = (NW + 31) / 32;
    constexpr int XW = T * QK_K / 8;
    constexpr int XJ = (XW + 32 * W - 1) / (32 * W);
    constexpr int SW = T * 4;                              // Summen je 16: T × 16 float = T × 4 Wörter

    __shared__ float skal[W][16 * MT][QK_K / 16];
    extern __shared__ __align__(16) unsigned char dyn[];
    uint4 (*buehne)[W][NW] = (uint4 (*)[W][NW]) dyn;
    uint4 (*xs)[XW]        = (uint4 (*)[XW]) (dyn + 2 * sizeof(uint4) * W * NW);
    float (*su16)[T][16]   = (float (*)[T][16]) (dyn + 2 * sizeof(uint4) * W * NW + 2 * sizeof(uint4) * XW);

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int gid  = lane >> 2;
    const int lid  = lane & 3;
    const int nsb  = k / QK_K;
    const int wrow = blockIdx.x * (16 * MT * W) + warp * (16 * MT);
    const int sb0  = (int) ((int64_t) nsb * blockIdx.y / ks);
    const int sb1  = (int) ((int64_t) nsb * (blockIdx.y + 1) / ks);
    const uintptr_t wbasis = (uintptr_t) w;

    // Versatz des Blocks (Zeile r dieses Warps, Block sb) in seinem 16-Byte-Fenster
    // 210 ≡ 2 (mod 16): der Versatz hängt nur von der Blocknummer mod 8 und dem Anfang des Tensors ab
    const int wlo = (int) (wbasis & 15);
    const auto versatz = [&](const int r, const int sb) -> int {
        return (wlo + 2 * (((wrow + r) * nsb + sb) & 7)) & 15;
    };

    float acc[MT][NT][4];
#pragma unroll
    for (int mi = 0; mi < MT; ++mi)
#pragma unroll
        for (int ni = 0; ni < NT; ++ni)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[mi][ni][c] = 0.0f;

    const auto lade = [&](const int sb, const int st) {
#pragma unroll
        for (int j = 0; j < JE; ++j) {
            const int idx = lane + 32 * j;
            if (idx < NW) {
                const int r = idx / ZW;
                const int o = idx % ZW;
                const int64_t anf = ((int64_t) (wrow + r) * nsb + sb) * BL;      // Byte des Blocks im Tensor
                const int64_t a16 = (int64_t) (((wbasis + (uint64_t) anf) & ~(uint64_t) 15) - wbasis) + 16 * o;
                const int64_t rest = w_bytes - a16;
                const int     n    = rest >= 16 ? 16 : (rest > 0 ? (int) rest : 0);
                const unsigned ziel = (unsigned) __cvta_generic_to_shared(&buehne[st][warp][idx]);
                const void *   quel = n > 0 ? (const void *) (w + a16) : (const void *) w;
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(ziel), "l"(quel), "r"(n));
            }
        }
#pragma unroll
        for (int j = 0; j < XJ; ++j) {
            const int idx = threadIdx.x + 32 * W * j;
            if (idx < XW) {
                const int tok = idx / (QK_K / 8);
                const int o   = idx % (QK_K / 8);
                const bool gilt = tok < ncols;
                const unsigned ziel = (unsigned) __cvta_generic_to_shared(&xs[st][idx]);
                const void *   quel = gilt ? (const void *) ((const uint4 *) (x + (int64_t) tok * k + (int64_t) sb * QK_K) + o) : (const void *) x;
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(ziel), "l"(quel), "r"(gilt ? 16 : 0));
            }
        }
        for (int idx = threadIdx.x; idx < SW; idx += 32 * W) {
            const int tok = idx / 4;
            const int o   = idx % 4;
            const bool gilt = tok < ncols;
            const unsigned ziel = (unsigned) __cvta_generic_to_shared(&su16[st][tok][4 * o]);
            const void *   quel = gilt ? (const void *) (xsum + (int64_t) tok * (k / 16) + sb * 16 + 4 * o) : (const void *) xsum;
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(ziel), "l"(quel), "r"(gilt ? 16 : 0));
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };

    if (sb0 < sb1) {
        lade(sb0, 0);
    }
    for (int sb = sb0; sb < sb1; ++sb) {
        const int st = (sb - sb0) & 1;
        if (sb + 1 < sb1) {
            lade(sb + 1, st ^ 1);
            asm volatile("cp.async.wait_group 1;\n" ::);
        } else {
            asm volatile("cp.async.wait_group 0;\n" ::);
        }
        __syncthreads();
        for (int zr = lane; zr < 16 * MT; zr += 32) {
            const uint8_t * bz = (const uint8_t *) &buehne[st][warp][zr * ZW] + versatz(zr, sb);
            const float d = __half2float(__ushort_as_half((unsigned short) (bz[208] | (bz[209] << 8))));
#pragma unroll
            for (int s = 0; s < QK_K / 16; ++s) {
                skal[warp][zr][s] = d * (float) (int8_t) bz[192 + s];
            }
        }
        __syncwarp();
        const half * xb = (const half *) xs[st];
        // Zeiger auf die Blöcke der eigenen Zeilen, einmal je Block
        const uint8_t * zo[MT];
        const uint8_t * zu[MT];
#pragma unroll
        for (int mi = 0; mi < MT; ++mi) {
            zo[mi] = (const uint8_t *) &buehne[st][warp][(mi * 16 + gid) * ZW] + versatz(mi * 16 + gid, sb);
            zu[mi] = (const uint8_t *) &buehne[st][warp][(mi * 16 + gid + 8) * ZW] + versatz(mi * 16 + gid + 8, sb);
        }

#pragma unroll
        for (int ib = 0; ib < QK_K / 32; ++ib) {
            const int n = ib >> 2;
            const int j = ib & 3;
            unsigned b[NT][2][2];
#pragma unroll
            for (int ni = 0; ni < NT; ++ni) {
                const int tok = ni * 8 + gid;
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const uint2 v = *(const uint2 *) (xb + tok * QK_K + ib * 32 + h * 16 + 4 * lid);
                    b[ni][h][0] = v.x;
                    b[ni][h][1] = v.y;
                }
            }
#pragma unroll
            for (int mi = 0; mi < MT; ++mi) {
                const int ro = mi * 16 + gid;
                const int ru = ro + 8;
                const uint8_t * bo = zo[mi];
                const uint8_t * bu = zu[mi];
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int pl = 64 * n + 32 * (j & 1) + 16 * h + 4 * lid;
                    const int ph = 128 + 32 * n + 16 * h + 4 * lid;
                    const uint32_t vo = ((fle_lade32(bo + pl) >> (4 * (j >> 1))) & 0x0F0F0F0Fu) | (((fle_lade32(bo + ph) >> (2 * j)) & 0x03030303u) << 4);
                    const uint32_t vu = ((fle_lade32(bu + pl) >> (4 * (j >> 1))) & 0x0F0F0F0Fu) | (((fle_lade32(bu + ph) >> (2 * j)) & 0x03030303u) << 4);
                    const unsigned a0 = __byte_perm(vo, 0x6464u, 0x5140);
                    const unsigned a1 = __byte_perm(vu, 0x6464u, 0x5140);
                    const unsigned a2 = __byte_perm(vo, 0x6464u, 0x5342);
                    const unsigned a3 = __byte_perm(vu, 0x6464u, 0x5342);
                    float g[NT][4];
#pragma unroll
                    for (int ni = 0; ni < NT; ++ni) {
#pragma unroll
                        for (int c = 0; c < 4; ++c) g[ni][c] = 0.0f;
                        fle_mma_f16(g[ni][0], g[ni][1], g[ni][2], g[ni][3], a0, a1, a2, a3, b[ni][h][0], b[ni][h][1]);
                    }
                    const int   s16 = 2 * ib + h;
                    const float so  = skal[warp][ro][s16];
                    const float su  = skal[warp][ru][s16];
#pragma unroll
                    for (int ni = 0; ni < NT; ++ni) {
                        const float x0 = 1056.0f * su16[st][ni * 8 + 2 * lid][s16];
                        const float x1 = 1056.0f * su16[st][ni * 8 + 2 * lid + 1][s16];
                        acc[mi][ni][0] = fmaf(g[ni][0] - x0, so, acc[mi][ni][0]);
                        acc[mi][ni][1] = fmaf(g[ni][1] - x1, so, acc[mi][ni][1]);
                        acc[mi][ni][2] = fmaf(g[ni][2] - x0, su, acc[mi][ni][2]);
                        acc[mi][ni][3] = fmaf(g[ni][3] - x1, su, acc[mi][ni][3]);
                    }
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
        for (int ni = 0; ni < NT; ++ni) {
            const int r  = wrow + mi * 16 + gid;
            const int t0 = ni * 8 + 2 * lid;
            const float v[4] = {acc[mi][ni][0], acc[mi][ni][1], acc[mi][ni][2], acc[mi][ni][3]};
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                const int t  = t0 + (c & 1);
                const int rr = r + 8 * (c >> 1);
                if (t < ncols) {
                    dst[(int64_t) blockIdx.y * ncols * nrows + (int64_t) t * nrows + rr] = v[c];
                }
            }
        }
    }
}

static void fle_q6k_start(const dim3 grid, const dim3 block, cudaStream_t stream, const void * w, const int64_t w_bytes,
                          const half * x, const float * xsum, float * d, const int nrows, const int k, const int ncols, const int ks) {
    constexpr int W  = FLE_IQ4_MMA_WARPS;
    constexpr int NW = 16 * FLE_IQ4_MMA_MT * 15;
    constexpr int XW = 16 * QK_K / 8;
    constexpr size_t bytes = 2 * sizeof(uint4) * W * NW + 2 * sizeof(uint4) * XW + 2 * sizeof(float) * 16 * 16;
    static bool frei = false;
    if (!frei) {
        CUDA_CHECK(cudaFuncSetAttribute(fle_q6k_mma_kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) bytes));
        frei = true;
    }
    fle_q6k_mma_kern<<<grid, block, bytes, stream>>>((const uint8_t *) w, w_bytes, x, xsum, d, nrows, k, ncols, ks);
}

// Summe der ks Teile von K in fester Reihenfolge
static __global__ void fle_iq4_summe(const float * __restrict__ teile, float * __restrict__ dst, const int64_t n, const int ks) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float s = teile[i];
        for (int j = 1; j < ks; ++j) {
            s += teile[(int64_t) j * n + i];
        }
        dst[i] = s;
    }
}

// Zwei Bühnen brauchen mehr als 48 KB Shared Memory: einmal je Fassung freischalten
template <int NT, ggml_type TYP>
static void fle_iq4_start(const dim3 grid, const dim3 block, cudaStream_t stream, const void * w, const half * x,
                          const float * xsum, float * d, const int nrows, const int k, const int ncols, const int ks) {
    using blk = typename fle_art<TYP>::block;
    constexpr int W  = FLE_IQ4_MMA_WARPS;
    constexpr int NW = 16 * FLE_IQ4_MMA_MT * (int) (sizeof(blk) / 8);
    constexpr int XW = NT * 8 * QK_K / 8;
    constexpr size_t bytes = 2 * sizeof(uint2) * W * NW + 2 * sizeof(uint4) * XW + 2 * sizeof(float) * NT * 8 * 8;
    static bool frei = false;
    if (!frei) {
        CUDA_CHECK(cudaFuncSetAttribute(fle_iq4_mma_kern<NT, TYP>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) bytes));
        frei = true;
    }
    fle_iq4_mma_kern<NT, TYP><<<grid, block, bytes, stream>>>((const blk *) w, x, xsum, d, nrows, k, ncols, ks);
}

void fle_iq4_mma(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    static bool gemeldet = false;
    if (!gemeldet) {
        gemeldet = true;
        GGML_LOG_WARN("FLE_IQ4_MMA aktiv: erster Pruefkern %s (%s) %lldx%lld, %lld Spalten\n", src0->name, ggml_type_name(src0->type),
                      (long long) src0->ne[1], (long long) src0->ne[0], (long long) src1->ne[1]);
    }
    const int nrows = (int) src0->ne[1];
    const int k     = (int) src0->ne[0];
    const int ncols = (int) src1->ne[1];
    cudaStream_t stream = ctx.stream();

    const bool q6 = src0->type == GGML_TYPE_Q6_K;  // Q6_K: Summen je 16 statt je 32
    ggml_cuda_pool_alloc<half>  x_f16(ctx.pool(), (size_t) k * ncols);
    ggml_cuda_pool_alloc<float> x_sum(ctx.pool(), (size_t) (k / (q6 ? 16 : 32)) * ncols);
    {
        const int64_t nteil = (int64_t) (k / 32) * ncols;
        const int     bs    = 128;
        if (q6) {
            fle_q6k_f32_f16<<<(unsigned) ((nteil + bs - 1) / bs), bs, 0, stream>>>((const float *) src1->data, x_f16.get(), x_sum.get(), nteil);
        } else {
            fle_iq4_f32_f16<<<(unsigned) ((nteil + bs - 1) / bs), bs, 0, stream>>>((const float *) src1->data, x_f16.get(), x_sum.get(), nteil);
        }
    }

    // Füllen die Zeilen allein die Karte nicht (etwa 5120 oder 6144 Zeilen), wird K auf bis zu 8 Teile verteilt, je Teil
    // mindestens zwei 256er-Blöcke. Die Teile landen in eigenen Ebenen und werden danach in fester Reihenfolge summiert.
    const int nblk = nrows / FLE_IQ4_MMA_ROWS;
    const int nsm  = ggml_cuda_info().devices[ctx.device].nsm;
    const int nsb  = k / QK_K;
    int ks = 1;
    while (ks < 8 && nblk * ks < 2 * nsm && nsb / (ks * 2) >= 2) {
        ks *= 2;
    }
    const dim3 grid(nblk, ks, 1);
    const dim3 block(32 * FLE_IQ4_MMA_WARPS, 1, 1);
    ggml_cuda_pool_alloc<float> teile(ctx.pool(), ks > 1 ? (size_t) ks * ncols * nrows : 1);
    float * d = ks > 1 ? teile.get() : (float *) dst->data;
    switch (src0->type) {
        case GGML_TYPE_IQ4_XS: fle_iq4_start<2, GGML_TYPE_IQ4_XS>(grid, block, stream, src0->data, x_f16.get(), x_sum.get(), d, nrows, k, ncols, ks); break;
        case GGML_TYPE_Q5_K:   fle_iq4_start<2, GGML_TYPE_Q5_K>(grid, block, stream, src0->data, x_f16.get(), x_sum.get(), d, nrows, k, ncols, ks); break;
        case GGML_TYPE_Q6_K:   fle_q6k_start(grid, block, stream, src0->data, (int64_t) ggml_nbytes(src0), x_f16.get(), x_sum.get(), d, nrows, k, ncols, ks); break;
        default:               fle_iq4_start<2, GGML_TYPE_Q4_K>(grid, block, stream, src0->data, x_f16.get(), x_sum.get(), d, nrows, k, ncols, ks); break;
    }
    if (ks > 1) {
        const int64_t n  = (int64_t) ncols * nrows;
        const int     bs = 256;
        fle_iq4_summe<<<(unsigned) ((n + bs - 1) / bs), bs, 0, stream>>>(teile.get(), (float *) dst->data, n, ks);
    }
    CUDA_CHECK(cudaGetLastError());
}
