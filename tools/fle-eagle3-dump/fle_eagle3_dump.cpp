// fle-eagle3-dump: EAGLE3 hidden-state dump using the CAUSAL convention.
// An earlier version ran in embeddings/pooling-none mode with layers [1,29,56], which builds the wrong graph and
// uses the wrong layers (deployed acceptance was only 3.7%).
// This version: CAUSAL context (embedding=false, as in inference, see speculative.cpp), aux via set_embeddings_layer_inp,
//     final via eval callback on "result_norm". Default layers [2,30,57] (conversion/llama.py: [2,n//2,n-3]).
// .bin (assemble_eagle3 format): header int32[nt,n_embd,n_aux] + token_ids[nt] + aux[n_aux*nt*ne] + final[nt*ne].
#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

// The C API llama_set/get_embeddings_layer_inp is declared in llama.h (LLAMA_API).

// --- eval callback: captures the final "result_norm" hidden state per token (causal, without embeddings mode) ---
static std::vector<float> g_final;   // [n_embd * n_tokens] last capture
static int g_final_ne = 0, g_final_nt = 0;
static bool capture_final_cb(struct ggml_tensor * t, bool ask, void * ud) {
    (void) ud;
    const bool want = t->name[0] && std::strcmp(t->name, "result_norm") == 0 && t->type == GGML_TYPE_F32;
    if (ask) return want;
    if (!want) return true;
    const int ne = (int) t->ne[0];
    const int nt = (int) t->ne[1];
    if (ne <= 0 || nt <= 0) return true;
    g_final_ne = ne; g_final_nt = nt;
    g_final.resize((size_t) ne * nt);
    ggml_backend_tensor_get(t, g_final.data(), 0, (size_t) ne * nt * sizeof(float));
    return true;
}

static std::vector<int> parse_csv_ints(const char * s) {
    std::vector<int> out; if (!s) return out;
    std::stringstream ss(s); std::string tok;
    while (std::getline(ss, tok, ',')) if (!tok.empty()) out.push_back(std::atoi(tok.c_str()));
    return out;
}

int main(int argc, char ** argv) {
    common_params params;
    params.n_batch = 4096;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_EMBEDDING)) return 1;
    params.embedding = false;                 // CAUSAL (no embeddings/pooling-none graph)
    params.cb_eval = capture_final_cb;        // grab the final result_norm hidden state
    params.cb_eval_user_data = nullptr;

    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();
    if (!model || !ctx) { LOG_ERR("fle-eagle3-dump: model/ctx load fail\n"); return 1; }
    const int n_embd = llama_model_n_embd(model);

    std::vector<int> aux = parse_csv_ints(std::getenv("FLE_DUMP_AUX"));
    if (aux.empty()) aux = {2, 30, 57};       // consistent with the engine (conversion/llama.py)
    const int n_aux = (int) aux.size();
    for (int lid : aux) llama_set_embeddings_layer_inp(ctx, (uint32_t) lid, true);

    const char * corpus = std::getenv("FLE_DUMP_CORPUS");
    const char * outdir = std::getenv("FLE_DUMP_OUTDIR");
    if (!corpus || !outdir) { LOG_ERR("fle-eagle3-dump: FLE_DUMP_CORPUS + FLE_DUMP_OUTDIR required\n"); return 1; }
    const int start = std::getenv("FLE_DUMP_START") ? std::atoi(std::getenv("FLE_DUMP_START")) : 0;

    std::ifstream fin(corpus);
    if (!fin) { LOG_ERR("fle-eagle3-dump: corpus %s is not readable\n", corpus); return 1; }
    LOG_INF("[dump] CAUSAL n_embd=%d n_aux=%d aux=[", n_embd, n_aux);
    for (int i = 0; i < n_aux; i++) LOG_INF("%d%s", aux[i], i+1<n_aux?",":"");
    LOG_INF("] outdir=%s start=%d\n", outdir, start);

    std::string line; int idx = 0, written = 0;
    llama_batch batch = llama_batch_init(params.n_batch, 0, 1);
    while (std::getline(fin, line)) {
        if (line.empty()) { idx++; continue; }
        if (idx < start) { idx++; continue; }
        std::vector<llama_token> inp = common_tokenize(ctx, line, true, true);
        const int nt = (int) inp.size();
        if (nt < 2 || nt > (int) params.n_batch) { idx++; continue; }

        common_batch_clear(batch);
        for (int i = 0; i < nt; i++) common_batch_add(batch, inp[i], i, {0}, true);   // logits for all -> result_norm for all
        llama_memory_clear(llama_get_memory(ctx), true);
        g_final_nt = 0;
        if (llama_decode(ctx, batch) < 0) { LOG_ERR("[dump] decode fail line %d\n", idx); idx++; continue; }
        if (g_final_nt != nt || g_final_ne != n_embd) { LOG_ERR("[dump] final-capture mismatch line %d (nt=%d/%d ne=%d/%d)\n", idx, g_final_nt, nt, g_final_ne, n_embd); idx++; continue; }

        char path[4096]; std::snprintf(path, sizeof(path), "%s/s%06d.bin", outdir, idx);
        FILE * f = std::fopen(path, "wb");
        if (!f) { LOG_ERR("[dump] open %s fail\n", path); idx++; continue; }
        int32_t hdr[3] = { nt, n_embd, n_aux };
        std::fwrite(hdr, sizeof(int32_t), 3, f);
        std::fwrite(inp.data(), sizeof(int32_t), nt, f);
        for (int a = 0; a < n_aux; a++) {
            const float * h = llama_get_embeddings_layer_inp(ctx, (uint32_t) aux[a]);   // [nt x n_embd] causal
            std::vector<float> z; if (!h) { z.assign((size_t) nt * n_embd, 0.0f); h = z.data(); LOG_ERR("[dump] aux %d null line %d\n", aux[a], idx); }
            std::fwrite(h, sizeof(float), (size_t) nt * n_embd, f);
        }
        std::fwrite(g_final.data(), sizeof(float), (size_t) nt * n_embd, f);   // final from callback
        std::fclose(f); written++;
        if ((written % 100) == 0) LOG_INF("[dump] %d written (idx=%d)\n", written, idx);
        idx++;
    }
    llama_batch_free(batch);
    LOG_INF("[dump] DONE: %d .bin to %s\n", written, outdir);
    llama_backend_free();
    return 0;
}
