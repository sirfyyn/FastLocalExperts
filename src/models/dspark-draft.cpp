#include "models.h"

// ─────────────────────────────────────────────────────────────────────────────────
// dspark-draft — DeepSeek-V4 DSpark drafter as its own architecture (FLE)
//
// Why its own arch instead of masquerading as deepseek4 (four failed attempts):
//   deepseek4 derives from the LAYER INDEX which tensors must exist
//   (compressors via compress_ratios, tid2eid via hash_layer_count). The drafter
//   is a 3-layer stack without compression -- every masquerade ends in tensor conflicts.
//
// Reuses the VERIFIED building blocks of the parent class (mHC Sinkhorn checked against
// a reference implementation, MLA/inverse-RoPE/sink checked against a RoPE reference).
// ─────────────────────────────────────────────────────────────────────────────────

void llama_model_dspark_draft::load_arch_hparams(llama_model_loader & ml) {
    // The base loads ALL V4 hparams (including derived values such as n_lora_kv, mscale, rope-yarn).
    // Re-implementing them was error-prone: every forgotten value is 0 and leads to SIGFPE in the graph.
    llama_model_deepseek4::load_arch_hparams(ml);

    // ── Draft-specific corrections ──────────────────────────────────────────────
    // 1) NO attention compression: the drafter has neither attn_compressor_* nor indexer_*.
    //    (The cache tolerates ratio == 0 since the guard in dsv4_comp_size.)
    hparams.dsv4_compress_ratios.fill(0);

    // 2) NO hash routing: all draft layers route via ffn_gate_inp + exp_probs_b.
    hparams.dsv4_hash_layer_count = 0;
}

void llama_model_dspark_draft::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    const int64_t q_lora_rank     = hparams.n_lora_q;
    const int64_t n_ff_exp        = hparams.n_ff_exp();
    const int64_t n_expert_shared = hparams.n_expert_shared;

    const int64_t n_embd_head = hparams.n_embd_head_k();
    const int64_t o_groups    = hparams.dsv4_o_group_count;
    const int64_t o_lora_rank = hparams.dsv4_o_lora_rank;
    const int64_t hc_mult     = hparams.dsv4_hc_mult;
    const int64_t hc_dim      = hc_mult * n_embd;
    const int64_t hc_mix_dim  = (2 + hc_mult) * hc_mult;

    // The embedding and the output head are OPTIONAL -- a drafter shares them with the
    // target model. No published DSpark artifact contains them: both released
    // versions have 81 tensors, none of them `token_embd` or `output`. The same
    // holds for the converted XS drafter.
    //
    // They used to be required, so the loader demanded something that does not exist.
    // The fallback to the target is the same as in dflash.cpp and is proven;
    // the graph in deepseek4.cpp picks it up with this change.
    tok_embd    = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,  "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);

    hc_head_fn    = create_tensor(tn(LLM_TENSOR_HC_HEAD_FN,    "weight"), {hc_dim, hc_mult}, 0);
    // Context combiner (concat aux L40|L41|L42 -> 12288 -> 4096) + Markov head.
    // Loaded optionally, so that older artifacts without these tensors keep working.
    dspark_main_proj = create_tensor(tn(LLM_TENSOR_DSPARK_MAIN_PROJ, "weight"), {3*n_embd, n_embd}, TENSOR_NOT_REQUIRED);
    dspark_main_norm = create_tensor(tn(LLM_TENSOR_DSPARK_MAIN_NORM, "weight"), {n_embd},           TENSOR_NOT_REQUIRED);
    dspark_markov_w1 = create_tensor(tn(LLM_TENSOR_DSPARK_MARKOV_W1, "weight"), {256, n_vocab},     TENSOR_NOT_REQUIRED);
    dspark_markov_w2 = create_tensor(tn(LLM_TENSOR_DSPARK_MARKOV_W2, "weight"), {256, n_vocab},     TENSOR_NOT_REQUIRED);
    if (dspark_main_proj) { LLAMA_LOG_INFO("%s: dspark context combiner loaded\n", __func__); }
    if (dspark_markov_w1) { LLAMA_LOG_INFO("%s: dspark Markov head loaded\n", __func__); }


    hc_head_base  = create_tensor(tn(LLM_TENSOR_HC_HEAD_BASE,  "weight"), {hc_mult}, 0);
    hc_head_scale = create_tensor(tn(LLM_TENSOR_HC_HEAD_SCALE, "weight"), {1}, 0);

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm     = create_tensor(tn(LLM_TENSOR_ATTN_NORM,     "weight", i), {n_embd}, 0);
        layer.attn_sinks    = create_tensor(tn(LLM_TENSOR_ATTN_SINKS,    "weight", i), {n_head}, 0);
        layer.wq_a          = create_tensor(tn(LLM_TENSOR_ATTN_Q_A,      "weight", i), {n_embd, q_lora_rank}, 0);
        layer.attn_q_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_A_NORM, "weight", i), {q_lora_rank}, 0);
        layer.wq_b          = create_tensor(tn(LLM_TENSOR_ATTN_Q_B,      "weight", i), {q_lora_rank, n_head * n_embd_head}, 0);
        layer.wkv           = create_tensor(tn(LLM_TENSOR_ATTN_KV,       "weight", i), {n_embd, n_embd_head}, 0);
        layer.attn_kv_norm  = create_tensor(tn(LLM_TENSOR_ATTN_KV_NORM,  "weight", i), {n_embd_head}, 0);
        // Upstream changed the shape of wo_a in deepseek4.cpp to three dimensions.
        // build_attention_impl is used by BOTH models and has relied on that ever since --
        // the two-dimensional shape yields eight times the size there and dies in ggml_cont_2d.
        // The stored weights stay [4096, 8192]; TENSOR_ALLOW_RESHAPE turns them into the
        // view [4096, 1024, 8].
        layer.wo_a          = create_tensor(tn(LLM_TENSOR_ATTN_OUT_A,    "weight", i), {n_head * n_embd_head / o_groups, o_lora_rank, o_groups}, TENSOR_ALLOW_RESHAPE);
        layer.wo_b          = create_tensor(tn(LLM_TENSOR_ATTN_OUT_B,    "weight", i), {o_groups * o_lora_rank, n_embd}, 0);

        layer.hc_attn_fn    = create_tensor(tn(LLM_TENSOR_HC_ATTN_FN,    "weight", i), {hc_dim, hc_mix_dim}, 0);
        layer.hc_attn_base  = create_tensor(tn(LLM_TENSOR_HC_ATTN_BASE,  "weight", i), {hc_mix_dim}, 0);
        layer.hc_attn_scale = create_tensor(tn(LLM_TENSOR_HC_ATTN_SCALE, "weight", i), {3}, 0);
        layer.hc_ffn_fn     = create_tensor(tn(LLM_TENSOR_HC_FFN_FN,     "weight", i), {hc_dim, hc_mix_dim}, 0);
        layer.hc_ffn_base   = create_tensor(tn(LLM_TENSOR_HC_FFN_BASE,   "weight", i), {hc_mix_dim}, 0);
        layer.hc_ffn_scale  = create_tensor(tn(LLM_TENSOR_HC_FFN_SCALE,  "weight", i), {3}, 0);

        // NO compressor block (ratio == 0 for all draft layers).

        layer.ffn_gate_inp    = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,   "weight", i), {n_embd, n_expert}, 0);
        layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias",  i), {n_expert}, 0);
        // NO tid2eid: the drafter routes via ffn_gate_inp + exp_probs_b.
        // The inherited cache is omitted when the GGUF is built (tensor count!).

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

        layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd,   n_ff_exp, n_expert}, 0);
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp, n_embd,   n_expert}, 0);
        layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd,   n_ff_exp, n_expert}, 0);

        layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd,                     n_ff_exp * n_expert_shared}, 0);
        layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {n_ff_exp * n_expert_shared, n_embd                    }, 0);
        layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd,                     n_ff_exp * n_expert_shared}, 0);
    }
}
