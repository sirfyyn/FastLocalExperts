// fle-capture-layers — DFlash multi-layer feature capture
// Loads a model, enables set_embeddings_layer_inp for N target layers, batches texts through it,
// and dumps per sample: [int32 n_tok][int32 n_layers][int32 n_embd][n_tok*int32 input_ids][n_layers*n_tok*n_embd f32].
// Config via ENV: FLE_CAP_LAYERS=csv, FLE_CAP_CORPUS=texts.txt (one text per line), FLE_CAP_OUT=dir, FLE_CAP_N, FLE_CAP_MAXTOK.
// Model/-ot/-ngl/-c via the standard common_params arguments.
#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cstdlib>
#include <cstdio>

// from src/llama-ext.h (not on the examples include path) — C++ linkage (mangled, no extern "C")
void    llama_set_embeddings_layer_inp(struct llama_context * ctx, uint32_t lid, bool value);
float * llama_get_embeddings_layer_inp(struct llama_context * ctx, uint32_t lid);

static std::vector<uint32_t> parse_layers(const char * s) {
    std::vector<uint32_t> v;
    if (!s) return v;
    std::stringstream ss(s);
    std::string t;
    while (std::getline(ss, t, ',')) if (!t.empty()) v.push_back((uint32_t) std::stoi(t));
    return v;
}

int main(int argc, char ** argv) {
    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_EMBEDDING)) return 1;

    // NO embeddings mode: layer-inp extraction (set_embeddings_layer_inp) runs on a normal causal context
    // (as in DFlash speculative.cpp). embeddings/pooling-none builds a different graph -> t_layer_inp is null.
    params.n_parallel   = 1;
    params.kv_unified   = true;

    llama_backend_init();
    llama_numa_init(params.numa);

    common_init_result_ptr llama_init = common_init_from_params(params);
    llama_model   * model = llama_init->model();
    llama_context * ctx   = llama_init->context();
    if (!model || !ctx) { fprintf(stderr, "[cap] model/ctx load failed\n"); return 1; }

    auto layers = parse_layers(std::getenv("FLE_CAP_LAYERS"));
    const char * corpus = std::getenv("FLE_CAP_CORPUS");
    const char * outdir = std::getenv("FLE_CAP_OUT");
    int n_target = std::getenv("FLE_CAP_N")      ? std::stoi(std::getenv("FLE_CAP_N"))      : 500;
    int max_tok  = std::getenv("FLE_CAP_MAXTOK") ? std::stoi(std::getenv("FLE_CAP_MAXTOK")) : 64;
    if (layers.empty() || !corpus || !outdir) {
        fprintf(stderr, "[cap] need FLE_CAP_LAYERS (csv), FLE_CAP_CORPUS, FLE_CAP_OUT\n");
        return 1;
    }

    for (auto lid : layers) llama_set_embeddings_layer_inp(ctx, lid, true);
    const int n_embd = llama_model_n_embd(model);
    fprintf(stderr, "[cap] n_embd=%d layers=%zu n_target=%d max_tok=%d\n", n_embd, layers.size(), n_target, max_tok);

    std::ifstream in(corpus);
    if (!in) { fprintf(stderr, "[cap] cannot open corpus %s\n", corpus); return 1; }
    std::string line;
    int done = 0;
    while (std::getline(in, line) && done < n_target) {
        if (line.empty()) continue;
        std::vector<llama_token> toks = common_tokenize(ctx, line, true, true);
        if ((int) toks.size() < 4) continue;
        if ((int) toks.size() > max_tok) toks.resize(max_tok);
        const int n = (int) toks.size();

        llama_memory_clear(llama_get_memory(ctx), true);
        llama_batch batch = llama_batch_init(n, 0, 1);
        for (int i = 0; i < n; i++) common_batch_add(batch, toks[i], i, { 0 }, true);
        if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "[cap] decode fail @%d\n", done); llama_batch_free(batch); continue; }

        char path[512];
        snprintf(path, sizeof(path), "%s/s%05d.bin", outdir, done);
        std::ofstream out(path, std::ios::binary);
        int32_t hn = n, hl = (int32_t) layers.size(), he = n_embd;
        out.write((char *) &hn, 4); out.write((char *) &hl, 4); out.write((char *) &he, 4);
        for (int i = 0; i < n; i++) { int32_t id = toks[i]; out.write((char *) &id, 4); }
        bool ok = true;
        for (auto lid : layers) {
            float * p = llama_get_embeddings_layer_inp(ctx, lid);
            if (!p) { fprintf(stderr, "[cap] layer %u not extracted\n", lid); ok = false; break; }
            out.write((char *) p, (size_t) n * n_embd * sizeof(float));
        }
        out.close();
        llama_batch_free(batch);
        if (!ok) return 2;
        done++;
        if (done % 50 == 0) fprintf(stderr, "[cap] %d\n", done);
    }
    fprintf(stderr, "[cap] DONE %d samples -> %s\n", done, outdir);
    return 0;
}
