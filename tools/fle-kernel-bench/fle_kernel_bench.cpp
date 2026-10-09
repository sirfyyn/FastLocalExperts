// fle-kernel-bench: compute time of the CPU expert path per quantization type.
//
// Why: with the expert cache, the CPU computes every miss (expert not in the GPU cache) itself. Measurements with 160 versus
// 128 cache slots give roughly 0.095 ms per miss, which corresponds to only about 25 GB/s of read load: the path is compute-bound,
// not memory-bound. The kernel speed of the type (IQ4_XS versus IQ3_S, IQ3_XXS, IQ4_NL ...) then matters, not only
// the byte count. This tool measures it with the expert shape of the reference model (2560 x 640, 512 experts per layer) and the
// production thread pool (8 threads, cores 0-7, strict).
//
// Per type and shape (gate/up: k = 2560, m = 640; down: k = 640, m = 2560), ggml_mul_mat_id runs over n_used = 10 randomly
// chosen experts for n_tokens = 1 or 2 tokens (environment FLE_KB_TOKENS, FLE_KB_ITERS, FLE_KB_TYPES "IQ4_XS,IQ3_S").
// The expert data is one quantized expert copied 512 times (real memory addresses, no cache hits
// from repetition because the ids change on every call).
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <sstream>
#include <string>
#include <vector>

struct TypeSpec { const char * name; ggml_type type; };

static const TypeSpec ALL_TYPES[] = {
    {"Q8_0", GGML_TYPE_Q8_0},   {"Q6_K", GGML_TYPE_Q6_K},     {"Q5_K", GGML_TYPE_Q5_K},   {"Q4_K", GGML_TYPE_Q4_K},
    {"IQ4_XS", GGML_TYPE_IQ4_XS}, {"IQ4_NL", GGML_TYPE_IQ4_NL}, {"IQ3_S", GGML_TYPE_IQ3_S}, {"IQ3_XXS", GGML_TYPE_IQ3_XXS},
    {"Q3_K", GGML_TYPE_Q3_K},   {"Q2_K", GGML_TYPE_Q2_K},     {"Q4_0", GGML_TYPE_Q4_0},   {"Q2_0", GGML_TYPE_Q2_0},
    {"IQ2_S", GGML_TYPE_IQ2_S}, {"IQ2_XS", GGML_TYPE_IQ2_XS}, {"IQ2_XXS", GGML_TYPE_IQ2_XXS},
};

static double bench(ggml_backend_t backend, ggml_type type, int64_t k, int64_t m, int n_expert, int n_used, int n_tokens,
                    bool broadcast_b, int iters, size_t * expert_bytes_out) {
    const size_t overhead = ggml_tensor_overhead() * 16 + ggml_graph_overhead() + 4096;
    ggml_init_params ip = { overhead, nullptr, /*no_alloc=*/true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * as  = ggml_new_tensor_3d(ctx, type, k, m, n_expert);
    ggml_tensor * b   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, broadcast_b ? 1 : n_used, n_tokens);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n_tokens);
    ggml_tensor * out = ggml_mul_mat_id(ctx, as, b, ids);
    ggml_cgraph * gf  = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) { std::fprintf(stderr, "could not allocate memory for %s\n", ggml_type_name(type)); ggml_free(ctx); return -1; }

    // one expert: random weights, quantized, then copied 512 times
    std::mt19937 rng(12345);
    std::normal_distribution<float> nd(0.f, 1.f);
    const int64_t n_el = k * m;
    std::vector<float> src(n_el);
    for (auto & v : src) v = nd(rng);
    const size_t row_bytes = ggml_row_size(type, k);
    const size_t expert_bytes = row_bytes * m;
    std::vector<uint8_t> q(expert_bytes);
    std::vector<float> ones(k, 1.0f);   // importance matrix of ones for types that require one (IQ2_XS/IQ2_XXS)
    ggml_quantize_chunk(type, src.data(), q.data(), 0, m, k, ggml_quantize_requires_imatrix(type) ? ones.data() : nullptr);
    for (int e = 0; e < n_expert; ++e) ggml_backend_tensor_set(as, q.data(), (size_t) e * expert_bytes, expert_bytes);

    std::vector<float> act(ggml_nelements(b));
    for (auto & v : act) v = nd(rng);
    ggml_backend_tensor_set(b, act.data(), 0, ggml_nbytes(b));

    std::vector<int32_t> idv(n_used * n_tokens), perm(n_expert);
    std::vector<double> ms;
    for (int it = 0; it < iters + 20; ++it) {
        for (int t = 0; t < n_tokens; ++t) {
            for (int e = 0; e < n_expert; ++e) perm[e] = e;
            std::shuffle(perm.begin(), perm.end(), rng);
            for (int u = 0; u < n_used; ++u) idv[t * n_used + u] = perm[u];
        }
        ggml_backend_tensor_set(ids, idv.data(), 0, ggml_nbytes(ids));
        const auto t0 = std::chrono::steady_clock::now();
        ggml_backend_graph_compute(backend, gf);
        const double dt = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (it >= 20) ms.push_back(dt);
    }
    std::sort(ms.begin(), ms.end());
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    *expert_bytes_out = expert_bytes;
    return ms[ms.size() / 2];
}

int main() {
    const int iters    = std::getenv("FLE_KB_ITERS") ? std::atoi(std::getenv("FLE_KB_ITERS")) : 300;
    const int n_tokens = std::getenv("FLE_KB_TOKENS") ? std::atoi(std::getenv("FLE_KB_TOKENS")) : 1;
    const int n_used = 10, n_expert = 512;
    std::vector<std::string> want;
    if (std::getenv("FLE_KB_TYPES")) {
        std::stringstream ss(std::getenv("FLE_KB_TYPES")); std::string it;
        while (std::getline(ss, it, ',')) want.push_back(it);
    }
    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_threadpool_params tp = ggml_threadpool_params_default(8);
    for (int i = 0; i < 8; ++i) tp.cpumask[i] = true;   // -C 0x000000FF
    tp.strict_cpu = true;                                // --cpu-strict 1
    ggml_threadpool_t pool = ggml_threadpool_new(&tp);
    ggml_backend_cpu_set_threadpool(backend, pool);
    ggml_backend_cpu_set_n_threads(backend, 8);

    for (const auto & ts : ALL_TYPES) {
        if (!want.empty() && std::find(want.begin(), want.end(), ts.name) == want.end()) continue;
        for (int shape = 0; shape < 2; ++shape) {
            const int64_t k = shape == 0 ? 2560 : 640, m = shape == 0 ? 640 : 2560;
            if (k % ggml_blck_size(ts.type) != 0) continue;   // down (k = 640) only works with blocks of 32/64, not with QK_K = 256
            size_t eb = 0;
            const double ms = bench(backend, ts.type, k, m, n_expert, n_used, n_tokens, shape == 0, iters, &eb);
            if (ms < 0) { std::printf("KB type=%s shape=%s skipped (%.0f)\n", ts.name, shape == 0 ? "gateup" : "down", ms); continue; }
            const double per_expert_us = 1000.0 * ms / n_used;
            const double gbps = (double) eb * n_used / (ms * 1e-3) / 1e9;   // weights read per second at 1 token
            std::printf("KB type=%s shape=%s tokens=%d expert_mb=%.3f call_ms=%.4f us_per_expert=%.1f gbps=%.1f\n",
                        ts.name, shape == 0 ? "gateup" : "down", n_tokens, eb / 1e6, ms, per_expert_us, gbps);
            std::fflush(stdout);
        }
    }
    ggml_threadpool_free(pool);
    ggml_backend_free(backend);
    return 0;
}
