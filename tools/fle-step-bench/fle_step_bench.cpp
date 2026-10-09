// fle-step-bench: cost of a decode step as a function of the number of tokens m verified per step
// (cost curve t(m)), with the expert cache as in production. Question: how expensive does verifying
// several tokens become for a MoE with an expert cache in host RAM, and from which acceptance length does a drafter pay off?
//
// Environment: FLE_BENCH_CORPUS (text file), FLE_BENCH_M (tokens per step, list "1,2,3,4,6,8"), FLE_BENCH_STEPS (measured
//              steps per m, 300), FLE_BENCH_WARM (warm-up steps per m, 150), FLE_BENCH_PREFILL (tokens before measuring, 512),
//              FLE_BENCH_STRIDE (text offset between two m values in tokens, 6000: each m value sees different text).
// The text is fed teacher-forced (real tokens, real routing); logits only for the last token (in production the server
// samples on the backend). Invoked like llama-perplexity (model, -ngl 99 -cmoe -fit off, --moe-expert-cache 160 ...).
// Output: one line "BENCH m=.. median_ms=.. mean_ms=.. p10=.. p90=.. tok_per_s=.." on stdout.
#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_PERPLEXITY)) return 1;
    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

    const char * corpus = std::getenv("FLE_BENCH_CORPUS");
    if (!corpus) { LOG_ERR("fle-step-bench: FLE_BENCH_CORPUS is required\n"); return 1; }
    std::vector<int> ms_list;
    {
        std::stringstream ls(std::getenv("FLE_BENCH_M") ? std::getenv("FLE_BENCH_M") : "1");
        std::string item;
        while (std::getline(ls, item, ',')) if (!item.empty()) ms_list.push_back(std::atoi(item.c_str()));
    }
    const int steps   = std::getenv("FLE_BENCH_STEPS") ? std::atoi(std::getenv("FLE_BENCH_STEPS")) : 300;
    const int warm    = std::getenv("FLE_BENCH_WARM") ? std::atoi(std::getenv("FLE_BENCH_WARM")) : 150;
    const int prefill = std::getenv("FLE_BENCH_PREFILL") ? std::atoi(std::getenv("FLE_BENCH_PREFILL")) : 512;
    const int stride  = std::getenv("FLE_BENCH_STRIDE") ? std::atoi(std::getenv("FLE_BENCH_STRIDE")) : 6000;

    auto init = common_init_from_params(params);
    auto * model = init->model();
    auto * ctx   = init->context();
    if (!model || !ctx) { LOG_ERR("fle-step-bench: model or context not loaded\n"); return 1; }

    std::ifstream fin(corpus, std::ios::binary);
    std::stringstream ss; ss << fin.rdbuf();
    const std::vector<llama_token> tokens = common_tokenize(ctx, ss.str(), true, true);
    int max_m = 1;
    for (int m : ms_list) max_m = std::max(max_m, m);
    const int per_m = prefill + (warm + steps) * max_m + 8;
    if (per_m > stride) { LOG_ERR("fle-step-bench: FLE_BENCH_STRIDE %d is smaller than the required %d\n", stride, per_m); return 1; }
    if ((int) tokens.size() < stride * (int) ms_list.size()) {
        LOG_ERR("fle-step-bench: corpus too short (%zu tokens, %d needed)\n", tokens.size(), stride * (int) ms_list.size());
        return 1;
    }

    llama_batch batch = llama_batch_init(std::max<int>(prefill, max_m), 0, 1);
    int mi = 0;
    for (int m : ms_list) {
        const int base = mi++ * stride;
        llama_memory_clear(llama_get_memory(ctx), true);
        // prefill in chunks of n_ubatch (large batch: the cache graph does not run there, as in production)
        int pos = 0;
        while (pos < prefill) {
            const int n = std::min<int>(prefill - pos, (int) params.n_ubatch);
            common_batch_clear(batch);
            for (int i = 0; i < n; ++i) common_batch_add(batch, tokens[base + pos + i], pos + i, { 0 }, pos + i == prefill - 1);
            if (llama_decode(ctx, batch)) { LOG_ERR("fle-step-bench: prefill failed\n"); return 1; }
            pos += n;
        }
        llama_synchronize(ctx);

        std::vector<double> ms;
        ms.reserve(steps);
        int ti = base + prefill;
        for (int s = 0; s < warm + steps; ++s) {
            common_batch_clear(batch);
            for (int i = 0; i < m; ++i) common_batch_add(batch, tokens[ti + i], pos + i, { 0 }, i == m - 1);
            const auto t0 = std::chrono::steady_clock::now();
            if (llama_decode(ctx, batch)) { LOG_ERR("fle-step-bench: step %d (m=%d) failed\n", s, m); return 1; }
            llama_synchronize(ctx);
            const double dt = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            if (s >= warm) ms.push_back(dt);
            ti += m; pos += m;
        }
        std::sort(ms.begin(), ms.end());
        double mean = 0; for (double v : ms) mean += v; mean /= ms.size();
        const double med = ms[ms.size() / 2];
        std::printf("BENCH m=%d steps=%zu median_ms=%.3f mean_ms=%.3f p10=%.3f p90=%.3f tok_per_s=%.2f\n",
                    m, ms.size(), med, mean, ms[ms.size() / 10], ms[ms.size() * 9 / 10], 1000.0 * m / med);
        std::fflush(stdout);
    }
    llama_batch_free(batch);
    llama_backend_free();
    return 0;
}
