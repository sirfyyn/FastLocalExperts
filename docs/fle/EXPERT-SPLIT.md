# Split experts (FLE v1.2.6)

A split GGUF stores the routed experts of a MoE layer in two halves: a **hot** half that lives on the GPU for good and a
**cold** half in host memory that the FLE CPU path computes. The hot half takes the role of a static GPU expert cache
(`--moe-expert-cache-inserts 0`), but it is fixed in the file, can use its own quant type and costs no uploads, no
tables to maintain and no cache bookkeeping at run time. There is no switch: the engine uses the split path whenever the
split tensors are present. Unsplit models and unsplit layers build exactly the graph they built before.

## File format (v1, written by the Forge tool)

Per split layer `N` (DeepSeek-V4.1, `src/models/deepseek41.cpp`):

| Tensor | Shape | Content |
|--------|-------|---------|
| `blk.N.ffn_{gate,up,down}_exps_hot.weight` | `[ne0, ne1, n_hot + 1]` | the hot experts in ascending original id, then one all-zero row (index `n_hot`) |
| `blk.N.ffn_{gate,up,down}_exps_cold.weight` | `[ne0, ne1, n_cold + 1]` | the other experts in ascending original id, then one all-zero row (index `n_cold`) |
| `blk.N.ffn_exps_slot.weight` | I32 `[n_expert]` | original id -> slot; `slot < n_hot`: row `slot` of the hot half, else row `slot - n_hot` of the cold half |
| `blk.N.ffn_exps_id.weight` | I32 `[n_expert]` | slot -> original id (the inverse; checked at load) |

KV: `fle.expert_split.version` = 1 (u32), `fle.expert_split.zero_row` = 1 (u32, required; the hot zero rows are also
checked byte by byte at load), `fle.expert_split.hot` (u32, the planned maximum hot count), `fle.expert_split.source` (string). `n_hot` may differ per layer (`ne2` of the hot half minus 1). A split layer has no
`ffn_*_exps`; a layer may also stay unsplit and keep them. The hot and cold halves may have different quant types (gate
and up of one half must share a `vec_dot_type`, as for any CPU MoE layer). Router, `exp_probs_b` and the shared expert keep
the original numbering.

## Design

- **Loader** (`deepseek41.cpp`, `llama-arch.*`, `llama-model.h`): with the KV present, a layer whose `ffn_up_exps_hot`
  exists loads all eight split tensors (required), otherwise the normal experts (required). Both in one layer is an error.
  The hot half follows the layer's device even if a CPU override (`-cmoe`, `-ot exps=CPU`) matches it
  (`llama-model-loader.cpp`); `LLM_FFN_EXPS_REGEX` and the `-fit` patterns exclude `_exps_hot`. The cold half goes where
  `-cmoe` puts experts today (host, mmap).
- **Tables** (`src/llama-expertsplit.{h,cpp}`, built once per model in the context constructor, owned by the model
  (`llama_model::fle_expert_split`, freed with it) and reached through `llama_layer::fle_split`): `hot_of[id]` (hot row or
  `n_hot`) and `cold_of[id]` (cold row or `n_cold`) on the hot half's device (plain CPU buffer when the hot half is in host
  memory), and the host skip table `cold_skip[n_cold + 1]` (`1` for every real cold row, `0` for the zero row). The model
  builder passes the layer's tables to `build_moe_ffn` through `llm_graph_context::fle_split`.
- **Cold half not in host memory** (no `-cmoe`, or `-ngmoe`): no skip table, no fused CPU op and no expert store for that
  layer; its backend computes the zero row, which yields zeros.
- **Graph** (`build_moe_ffn`, `src/llama-graph.cpp`): routing and weights stay in original ids. The GPU chain is the expert
  cache's hit chain over the hot half with `ids = get_rows(hot_of, selected)`. The CPU chain is the usual one (fused CPU op
  or `mul_mat_id`, the expert store, the CPU pool) over the cold half with `ids = get_rows(cold_of, selected)` and
  `src[3] = cold_skip`, `op_params[0] = 1`: the CPU computes every real cold row and skips the zero row (its dst rows are
  zeroed, the row is never read). The two `[n_embd, n_used, n_tok]` outputs are summed, then weighted as before. The split
  path runs for every token count; overlap (`FLE_MC_OVERLAP`) and deferral (`FLE_MOE_DEFER`) apply where the cache graph
  would run (up to `FLE_MOE_CACHE_MAXTOK` tokens).
- **CUDA**: the zero row is named once per expert of the other half, so one token can repeat it. MMVQ (decode) handles
  that; past the MMVQ batch sizes `fle_is_moe_cache_w` sends `_exps_hot` and `_exps_cold` (op-offload copy) to the
  per-expert path, which tolerates repeats (MMQ/MMF would leave the repeated rows unwritten).
- **Expert cache / store** (`src/llama-moecache.cpp`): split layers have no `ffn_*_exps`, so the GPU cache never selects
  them. Their cold halves (host memory, weights loaded) are registered in the RAM expert store (`FLE_MOE_STORE_GB`) after
  the cache layers and served in cold-local numbering (`ensure` from the observer, `data_cb`, `ready_cb`, pinning). Without
  an expert cache the store is still created for the cold halves; -fit's estimation model (no weights) does not end the
  one-time init, so the real model still gets the store.

## Not in v1

- The observer does no LRU, profile (`FLE_MOE_PROFILE`), `FLE_MOE_WILLNEED_NOW` or `FLE_MOE_SHARE_EVERY` work for split
  layers; `FLE_MOE_HOT_MLOCK_GB` and the profile warm start of the store do not cover them.
- `FLE_MOE_PREFETCH` is not computed into a split layer (its prediction is in original ids); `FLE_MOE_CPU_PRED` and the
  CUDA pager (`FLE_PAGER_VRAM_MB`) do not register split layers. Predictions out of a split layer into an unsplit one still work.
- No cache routing bonus (`FLE_MOE_CACHE_BIAS`) for the hot experts.
- Prefill helpers skip split layers: `FLE_UBATCH_LEND` lends only cache layers, `FLE_UBATCH_PREFETCH` sizes its slots from
  unsplit experts. Large prefill batches compute the hot half with the per-expert CUDA path and the cold half by op offload
  (also per-expert) or on the CPU; this is correct but slower than MMQ.
- `GGML_MOE_LOG` logs a split layer's cold-local ids.

## How to verify

1. Load log: `FLE split experts v1: up to H hot experts ...` (loader) and `FLE split experts: L layers, X hot experts on
   CUDA0, Y cold experts on CPU` (tables). With `FLE_MOE_STORE_GB` and no expert cache: `FLE split experts: expert store
   for the cold halves of L layers (no expert cache)`.
2. Correctness: same prompt, greedy, against the unsplit source GGUF with the GPU cache off; the logits should match to
   rounding (different quant types per half change them only as much as that requantization does). A split file whose hot
   half is the profile's top experts at the same quant type must give the same tokens as the unsplit file with a static
   cache of the same experts (`--moe-expert-cache-inserts 0` and that profile).
3. Graph: `GGML_SCHED_DEBUG` or the eval callback shows `ffn_moe_split_hot_ids`, `ffn_moe_split_cold_ids` and
   `ffn_moe_split_merged` per split layer, and no `moe_cache_*` tensors for them.
4. Prefill: a long prompt (more than `FLE_MOE_CACHE_MAXTOK` tokens) must give the same result as decode-by-decode
   (checks the per-expert CUDA path with repeated zero rows).
