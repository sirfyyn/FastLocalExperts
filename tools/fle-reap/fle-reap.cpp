// llama-fle-reap: per-expert REAP saliency of an MoE model over a calibration text.
//
// REAP (Router-weighted Expert Activation Pruning, Lasby et al. 2025) ranks the routed experts of a layer by
//   S_j = mean over the tokens routed to expert j of  g_j(x) * ||f_j(x)||_2
// where g_j is the router weight and f_j the expert output. Both are read from the graph node "ffn_moe_weighted-<il>"
// (expert output times its router weight; g >= 0, so the row norm is g * ||f||); the expert ids come from the ids
// operand of "ffn_moe_down-<il>" (the down projection's MUL_MAT_ID) of the same layer.
//
// Output (-o, default reap.tsv), one line per layer and expert:
//   layer  expert  count  sum_gate  sum_saliency  sum_norm
// count = routed tokens, sum_gate = summed router weight (routing mass), sum_saliency = summed g * ||f||,
// sum_norm = summed ||f|| without the gate. REAP score = sum_saliency / count; the gate-free mean activation norm
// (MAN, arXiv 2606.15716) = sum_norm / count. The router weights are those after the top-k renormalization.
//
// Usage: llama-fle-reap -m MODEL.gguf -f calib.txt -c 2048 --chunks 256 -o reap.tsv [-ngl 99 -cmoe ...]
// The text is tokenized with special tokens parsed (chat markup counts as tokens). Each chunk of -c tokens is
// evaluated from an empty context. The table is written every 16 chunks and at the end.

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml-backend.h"

#include <chrono>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

struct layer_stats {
    std::vector<int64_t> count;
    std::vector<double>  gate;
    std::vector<double>  sal;
    std::vector<double>  norm;
};

static std::map<int, layer_stats> g_stats;
static std::vector<uint8_t> g_ids;
static int64_t g_ids_ne0   = 0;
static int64_t g_ids_ne1   = 0;
static size_t  g_ids_nb0   = 0;
static size_t  g_ids_nb1   = 0;
static int     g_ids_layer = -1;
static int64_t g_n_expert  = 0;
static int64_t g_skipped   = 0;
static std::vector<uint8_t> g_buf;
static std::vector<uint8_t> g_wbuf;

static int layer_of(const char * name, const char * prefix) {
    const size_t n = strlen(prefix);
    if (strncmp(name, prefix, n) != 0) {
        return -1;
    }
    return atoi(name + n);
}

static const uint8_t * host_data(const ggml_tensor * t, std::vector<uint8_t> & buf) {
    if (t->buffer && ggml_backend_buffer_is_host(t->buffer)) {
        return (const uint8_t *) t->data;
    }
    buf.resize(ggml_nbytes(t));
    ggml_backend_tensor_get(t, buf.data(), 0, ggml_nbytes(t));
    return buf.data();
}

static bool reap_collect(ggml_tensor * t, bool ask, void * /*user_data*/) {
    const int l_down = t->op == GGML_OP_MUL_MAT_ID ? layer_of(t->name, "ffn_moe_down-") : -1;
    const int l_w    = t->op == GGML_OP_MUL        ? layer_of(t->name, "ffn_moe_weighted-") : -1;
    if (ask) {
        return l_down >= 0 || l_w >= 0;
    }
    if (l_down >= 0) {
        const ggml_tensor * ids = t->src[2];
        GGML_ASSERT(ids && ids->type == GGML_TYPE_I32);
        g_n_expert = t->src[0]->ne[2];
        g_ids.resize(ggml_nbytes(ids));
        ggml_backend_tensor_get(ids, g_ids.data(), 0, ggml_nbytes(ids));
        g_ids_ne0   = ids->ne[0];
        g_ids_ne1   = ids->ne[1];
        g_ids_nb0   = ids->nb[0];
        g_ids_nb1   = ids->nb[1];
        g_ids_layer = l_down;
        return true;
    }
    if (l_w < 0) {
        return true;
    }
    const ggml_tensor * w = t->src[1];   // router weights [1, n_used, n_tokens]
    if (l_w != g_ids_layer || !w || t->type != GGML_TYPE_F32 || w->type != GGML_TYPE_F32 ||
            t->ne[1] != g_ids_ne0 || t->ne[2] != g_ids_ne1 || w->ne[1] != t->ne[1] || w->ne[2] != t->ne[2]) {
        if (g_skipped++ < 5) {
            LOG_WRN("%s: skipping %s (ids of layer %d, shapes %lld/%lld vs %lld/%lld)\n", __func__, t->name, g_ids_layer,
                    (long long) t->ne[1], (long long) t->ne[2], (long long) g_ids_ne0, (long long) g_ids_ne1);
        }
        return true;
    }
    GGML_ASSERT(t->nb[0] == sizeof(float));
    const int64_t n_embd = t->ne[0];
    const int64_t n_used = t->ne[1];
    const int64_t n_tok  = t->ne[2];
    const uint8_t * d  = host_data(t, g_buf);
    const uint8_t * wd = host_data(w, g_wbuf);

    layer_stats & st = g_stats[l_w];
    if (st.count.empty()) {
        st.count.assign(g_n_expert, 0);
        st.gate.assign(g_n_expert, 0.0);
        st.sal.assign(g_n_expert, 0.0);
        st.norm.assign(g_n_expert, 0.0);
    }
    for (int64_t tok = 0; tok < n_tok; ++tok) {
        for (int64_t s = 0; s < n_used; ++s) {
            const int32_t e = *(const int32_t *) (g_ids.data() + tok * g_ids_nb1 + s * g_ids_nb0);
            if (e < 0 || e >= g_n_expert) {
                continue;
            }
            const float * row = (const float *) (d + s * t->nb[1] + tok * t->nb[2]);
            double ss = 0.0;
            for (int64_t i = 0; i < n_embd; ++i) {
                ss += (double) row[i] * row[i];
            }
            const float g = *(const float *) (wd + s * w->nb[1] + tok * w->nb[2]);
            const double nrm = std::sqrt(ss);   // = g * ||f||
            st.count[e] += 1;
            st.gate[e]  += g;
            st.sal[e]   += nrm;
            st.norm[e]  += g > 0.0f ? nrm / g : 0.0;
        }
    }
    return true;
}

static void write_table(const std::string & path) {
    const std::string tmp = path + ".tmp";
    FILE * f = fopen(tmp.c_str(), "w");
    if (!f) {
        LOG_ERR("%s: cannot write %s\n", __func__, tmp.c_str());
        return;
    }
    fprintf(f, "layer\texpert\tcount\tsum_gate\tsum_saliency\tsum_norm\n");
    for (const auto & [il, st] : g_stats) {
        for (size_t e = 0; e < st.count.size(); ++e) {
            fprintf(f, "%d\t%zu\t%lld\t%.9g\t%.9g\t%.9g\n", il, e, (long long) st.count[e], st.gate[e], st.sal[e],
                    st.norm[e]);
        }
    }
    fclose(f);
    rename(tmp.c_str(), path.c_str());
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.out_file = "reap.tsv";
    params.n_ctx    = 2048;
    params.escape   = false;

    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_IMATRIX)) {
        return 1;
    }
    if (params.prompt.empty()) {
        LOG_ERR("%s: no calibration text (-f FILE)\n", __func__);
        return 1;
    }
    const int n_ctx = params.n_ctx;
    params.n_batch    = std::min(params.n_batch, n_ctx);
    params.n_ubatch   = std::min(params.n_ubatch, params.n_batch);
    params.n_parallel = 1;
    params.warmup     = false;

    llama_backend_init();
    llama_numa_init(params.numa);

    params.cb_eval           = reap_collect;
    params.cb_eval_user_data = nullptr;

    auto llama_init = common_init_from_params(params);
    llama_model   * model = llama_init->model();
    llama_context * ctx   = llama_init->context();
    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s: failed to init\n", __func__);
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, true, true);
    const int n_chunk_max = (int) (tokens.size() / n_ctx);
    const int n_chunk     = params.n_chunks < 0 ? n_chunk_max : std::min(params.n_chunks, n_chunk_max);
    LOG_INF("%s: %zu tokens, %d chunks of %d (batch %d)\n", __func__, tokens.size(), n_chunk, n_ctx, params.n_batch);
    if (n_chunk < 1) {
        LOG_ERR("%s: text shorter than one chunk\n", __func__);
        return 1;
    }

    const bool add_bos = llama_vocab_get_add_bos(vocab);
    llama_batch batch = llama_batch_init(params.n_batch, 0, 1);
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < n_chunk; ++i) {
        llama_memory_clear(llama_get_memory(ctx), true);
        const int start = i * n_ctx;
        for (int j = 0; j < n_ctx; j += params.n_batch) {
            const int n = std::min(params.n_batch, n_ctx - j);
            common_batch_clear(batch);
            for (int k = 0; k < n; ++k) {
                llama_token tok = tokens[start + j + k];
                if (add_bos && j == 0 && k == 0) {
                    tok = llama_vocab_bos(vocab);
                }
                common_batch_add(batch, tok, j + k, { 0 }, j + k == n_ctx - 1);
            }
            if (llama_decode(ctx, batch)) {
                LOG_ERR("%s: decode failed in chunk %d\n", __func__, i);
                llama_batch_free(batch);
                write_table(params.out_file);
                return 1;
            }
        }
        llama_synchronize(ctx);
        const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        LOG_INF("chunk %d/%d, %.0f tok/s, %.1f min left\n", i + 1, n_chunk, (i + 1) * n_ctx / secs,
                secs / (i + 1) * (n_chunk - i - 1) / 60.0);
        if ((i + 1) % 16 == 0) {
            write_table(params.out_file);
        }
    }
    llama_batch_free(batch);
    write_table(params.out_file);
    LOG_INF("%s: %zu layers written to %s (%lld skipped nodes)\n", __func__, g_stats.size(), params.out_file.c_str(),
            (long long) g_skipped);

    llama_perf_context_print(ctx);
    llama_backend_free();
    return 0;
}
