// fle-moe-capture: writes, per MoE layer, the router input and the output of the routed experts
// (ffn_moe_out) for a text corpus. This is the dataset for layer-local router repair: input x,
// teacher output y = sum of the top-k experts of the unmodified model.
//
// Environment: FLE_CAP_CORPUS (text file), FLE_CAP_OUTDIR, FLE_CAP_CHUNKS (chunks of n_ctx tokens, default 12),
//              FLE_CAP_SKIP (skip the first chunks, default 0),
//              FLE_CAP_HNEXTN=1 (additionally the state before hc_head, 4 streams, and the top-32 logits per position:
//              dataset for a block drafter; hnextn.f16, top32.ids.i32, top32.logit.f32).
// Invoked like llama-perplexity (model, -c 2048 -b 2048 -ub 2048 -ngl 99 -cmoe -fit off ...).
// Output per layer L (two digits): L<L>.x.f16 and L<L>.y.f16, rows of n_embd values (fp16) in token order,
// plus tokens.i32 and meta.json.
#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "../../src/llama-ext.h"   // interim interface: llama_set_embeddings_nextn / llama_get_embeddings_nextn

#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

struct capture {
    std::string dir;
    std::map<int, FILE *> fx, fy;
    std::vector<uint8_t> raw;
    std::vector<ggml_fp16_t> half;
    long rows_x = 0, rows_y = 0;
    int n_embd = 0;
};
static capture g;

static FILE * open_layer(std::map<int, FILE *> & files, int il, const char * suffix) {
    auto it = files.find(il);
    if (it != files.end()) return it->second;
    char path[4096];
    std::snprintf(path, sizeof(path), "%s/L%02d.%s.f16", g.dir.c_str(), il, suffix);
    FILE * f = std::fopen(path, "ab");
    if (!f) { LOG_ERR("fle-moe-capture: %s is not writable\n", path); std::exit(2); }
    files[il] = f;
    return f;
}

static void write_rows(FILE * f, const ggml_tensor * src, long & rows) {
    const int64_t ne = src->ne[0], nt = src->ne[1];
    if (ne != g.n_embd) { LOG_ERR("fle-moe-capture: unexpected width %ld\n", (long) ne); std::exit(3); }
    const size_t nbytes = ggml_nbytes(src);
    g.raw.resize(nbytes);
    ggml_backend_tensor_get(src, g.raw.data(), 0, nbytes);
    g.half.resize((size_t) ne * nt);
    if (src->type == GGML_TYPE_F32) {
        ggml_fp32_to_fp16_row((const float *) g.raw.data(), g.half.data(), ne * nt);
    } else if (src->type == GGML_TYPE_F16) {
        std::memcpy(g.half.data(), g.raw.data(), (size_t) ne * nt * sizeof(ggml_fp16_t));
    } else {
        LOG_ERR("fle-moe-capture: type %s is not supported\n", ggml_type_name(src->type));
        std::exit(4);
    }
    std::fwrite(g.half.data(), sizeof(ggml_fp16_t), (size_t) ne * nt, f);
    rows += nt;
}

// "ffn_moe_logits-<il>" is the router matrix multiplication, src[1] its input; "ffn_moe_out-<il>" is the sum of the experts.
static bool capture_cb(struct ggml_tensor * t, bool ask, void *) {
    const char * n = t->name;
    int il = -1;
    bool is_in = false;
    if (std::strncmp(n, "ffn_moe_logits-", 15) == 0) { il = std::atoi(n + 15); is_in = true; }
    else if (std::strncmp(n, "ffn_moe_out-", 12) == 0) { il = std::atoi(n + 12); }
    if (il < 0) return ask ? false : true;
    if (ask) return true;
    if (is_in) {
        if (!t->src[1]) return true;
        write_rows(open_layer(g.fx, il, "x"), t->src[1], g.rows_x);
    } else {
        write_rows(open_layer(g.fy, il, "y"), t, g.rows_y);
    }
    return true;
}

int main(int argc, char ** argv) {
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_PERPLEXITY)) return 1;
    const bool cap_layers = !(std::getenv("FLE_CAP_LAYERS") && std::atoi(std::getenv("FLE_CAP_LAYERS")) == 0);   // FLE_CAP_LAYERS=0: no layer capture
    params.cb_eval = cap_layers ? capture_cb : nullptr;
    params.cb_eval_user_data = nullptr;
    params.warmup = false;

    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

    const char * corpus = std::getenv("FLE_CAP_CORPUS");
    const char * outdir = std::getenv("FLE_CAP_OUTDIR");
    if (!corpus || !outdir) { LOG_ERR("fle-moe-capture: FLE_CAP_CORPUS and FLE_CAP_OUTDIR are required\n"); return 1; }
    g.dir = outdir;
    const int n_chunks = std::getenv("FLE_CAP_CHUNKS") ? std::atoi(std::getenv("FLE_CAP_CHUNKS")) : 12;
    const int skip     = std::getenv("FLE_CAP_SKIP") ? std::atoi(std::getenv("FLE_CAP_SKIP")) : 0;

    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();
    if (!model || !ctx) { LOG_ERR("fle-moe-capture: model or context not loaded\n"); return 1; }
    g.n_embd = llama_model_n_embd(model);
    const bool want_h = std::getenv("FLE_CAP_HNEXTN") && std::atoi(std::getenv("FLE_CAP_HNEXTN")) != 0;
    if (want_h) { llama_set_embeddings_nextn(ctx, true, false); }
    FILE * fh = want_h ? std::fopen((g.dir + "/hnextn.f16").c_str(), "ab") : nullptr;
    FILE * fti = want_h ? std::fopen((g.dir + "/top32.ids.i32").c_str(), "ab") : nullptr;
    FILE * ftl = want_h ? std::fopen((g.dir + "/top32.logit.f32").c_str(), "ab") : nullptr;
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const int n_ctx = params.n_ctx;
    if ((int) params.n_batch < n_ctx) { LOG_ERR("fle-moe-capture: -b must be >= -c (one chunk per call)\n"); return 1; }

    std::ifstream fin(corpus, std::ios::binary);
    if (!fin) { LOG_ERR("fle-moe-capture: corpus %s is not readable\n", corpus); return 1; }
    std::stringstream ss; ss << fin.rdbuf();
    const std::vector<llama_token> tokens = common_tokenize(ctx, ss.str(), true, true);
    const int n_avail = (int) (tokens.size() / n_ctx);
    LOG_INF("[capture] %zu tokens, %d chunks of %d, of which %d from %d\n", tokens.size(), n_avail, n_ctx, n_chunks, skip);
    const bool add_bos = llama_vocab_get_add_bos(llama_model_get_vocab(model));

    FILE * ftok = std::fopen((g.dir + "/tokens.i32").c_str(), "ab");
    llama_batch batch = llama_batch_init(n_ctx, 0, 1);
    int done = 0;
    for (int c = skip; c < n_avail && done < n_chunks; ++c, ++done) {
        common_batch_clear(batch);
        for (int k = 0; k < n_ctx; ++k) {
            llama_token tok = tokens[(size_t) c * n_ctx + k];
            if (k == 0 && add_bos) tok = llama_vocab_bos(llama_model_get_vocab(model));
            common_batch_add(batch, tok, k, { 0 }, true);   // output for all tokens: otherwise the last layer is trimmed to the output rows
            if (ftok) std::fwrite(&tok, sizeof(int32_t), 1, ftok);
        }
        llama_memory_clear(llama_get_memory(ctx), true);
        if (llama_decode(ctx, batch)) { LOG_ERR("fle-moe-capture: chunk %d failed\n", c); return 1; }
        if (want_h) {
            const float * hn = llama_get_embeddings_nextn(ctx);
            const int n_hc = (int) llama_model_n_embd_out(model);
            if (!hn) { LOG_ERR("fle-moe-capture: no h_nextn (n_hc=%d)\n", n_hc); return 1; }
            std::vector<ggml_fp16_t> hh((size_t) n_ctx * n_hc);
            ggml_fp32_to_fp16_row(hn, hh.data(), (int64_t) n_ctx * n_hc);
            std::fwrite(hh.data(), sizeof(ggml_fp16_t), hh.size(), fh);
            std::vector<int32_t> ids(32);
            std::vector<float> lg(32);
            for (int k = 0; k < n_ctx; ++k) {
                const float * l = llama_get_logits_ith(ctx, k);
                std::vector<std::pair<float, int32_t>> best;
                best.reserve(n_vocab);
                for (int v = 0; v < n_vocab; ++v) { best.emplace_back(l[v], v); }
                std::partial_sort(best.begin(), best.begin() + 32, best.end(), [](const auto & a, const auto & b) { return a.first > b.first; });
                for (int j = 0; j < 32; ++j) { lg[j] = best[j].first; ids[j] = best[j].second; }
                std::fwrite(ids.data(), sizeof(int32_t), 32, fti);
                std::fwrite(lg.data(), sizeof(float), 32, ftl);
            }
        }
        LOG_INF("[capture] chunk %d done (%d of %d), rows x=%ld y=%ld\n", c, done + 1, n_chunks, g.rows_x, g.rows_y);
    }
    if (ftok) std::fclose(ftok);
    if (fh) std::fclose(fh);
    if (fti) std::fclose(fti);
    if (ftl) std::fclose(ftl);
    for (auto & kv : g.fx) std::fclose(kv.second);
    for (auto & kv : g.fy) std::fclose(kv.second);
    llama_batch_free(batch);
    {
        std::ofstream meta(g.dir + "/meta.json");
        meta << "{\"n_embd\": " << g.n_embd << ", \"n_ctx\": " << n_ctx << ", \"chunks\": " << done
             << ", \"layers\": " << g.fx.size() << ", \"rows_x\": " << g.rows_x << ", \"rows_y\": " << g.rows_y << "}\n";
    }
    LOG_INF("[capture] DONE: %d chunks, %zu layers to %s\n", done, g.fx.size(), outdir);
    llama_backend_free();
    return 0;
}
