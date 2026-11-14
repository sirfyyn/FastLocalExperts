# FLE environment variables (v1.3.0, dense line)

v1.3.0 is FLE's dense line: upstream llama.cpp `344846b4f` (2026-10-30) plus the changes that run Katy and Kate
(Qwen3.8-27B fine-tunes with Gated DeltaNet layers) with the DSpine draft head. Every change listed here is off unless
its variable is set; with none set, the engine behaves like upstream at the same commit, except for the n-gram table
(see `--spec-type ngram-mod` below) and the DSpine model type, which upstream cannot load.

The MoE switches of v1.2.x (`FLE_MOE_*`, expert store, split experts, `--moe-expert-cache*`, `/fle/status`) are **not**
part of v1.3.0. For DeepSeek V4.1 Flash, Rin and Qwen3.8-Flash-Next use
[v1.2.6](https://github.com/sirfyyn/FastLocalExperts/tree/v1.2.6) and its
[ENV.md](https://github.com/sirfyyn/FastLocalExperts/blob/v1.2.6/docs/fle/ENV.md).

The run IDs (k28, k30, ...) are the paired A-B-B-A measurements in the Kate/Katy report (Appendix C); all were measured
on one RTX PRO 4500 Blackwell (32 GB) with Katy (IQ4_XS) and the DSpine b16 head, n-max 15.

## Verify rounds in hybrid models

| Variable | Default | What it does | Measured |
|---|---|---|---|
| `FLE_RS_WINDOW` | off | The short causal convolution in front of each Gated DeltaNet layer keeps its last inputs as a window addressed by an index input, so rolling a rejected draft back moves the index instead of copying buffers. Needs `--parallel 1`. Prints one line at start saying whether it took effect. | +1.2 % (+1.1…+1.3), k28; recurrent state 35.6 instead of 90 MiB; bit-identical on 30 of 30 tasks |
| `FLE_GDN_REPLAY` | off | Needs `FLE_RS_WINDOW`. Instead of one recurrent-state snapshot per drafted position, a verify round keeps one saved level and replays the accepted tokens through the DeltaNet update. Only layers on CUDA; layers on the CPU keep snapshots. | +4.6 % (+4.5…+4.7) over the window, k30; recurrent state 348 instead of 2,340 MiB; bit-identical 30/30. With less VRAM the freed memory matters more: +23.7 % at 20 GB (k32), +16.6 % at 16 GB (k33) |
| `FLE_RS_SEQ` | off | `<n>`: deeper recurrent rollback than the draft model needs, so n-gram drafts of up to n tokens roll back through the recurrent state instead of a checkpoint. Cheap with `FLE_GDN_REPLAY`, expensive with snapshots. | not in the production setup |

## Speculative decoding

| Variable / flag | Default | What it does | Measured |
|---|---|---|---|
| DSpine model type | - | Loads DSpine block-diffusion draft heads (for example `tencent/Qwen3.8-27B-DSpine-b16` as GGUF) with `--spec-type draft-dflash -md <head>.gguf`. Adds the GGUF conversion in `conversion/qwen.py`. | DSpine b16, n-max 15: +34.4 % (+25.9…+43.2) over DFlash2, k6; +19.0 % at temperature 1.0, k7 |
| `--spec-type ngram-mod` | - | FLE's n-gram table v2: tagged buckets (4 ways), age-based replacement instead of the global reset at 25 % occupancy, same flags as upstream. **Always active with `ngram-mod`**, there is no switch back to upstream's table. | +19.8 % with a pre-warmed table at temperature 1.0, k10 |
| `FLE_NGRAM_MOD_FILE` | off | `<file>`: the n-gram table is loaded from this file at start (warm start) and saved to it in the background (written to `<file>.tmp`, then renamed). A file for another `n_match` or size is ignored. A warm table makes benchmark runs depend on what ran before: measure with a cold or frozen table. | - |
| `FLE_DRAFT_TEMP` | 1 | Factor on the target's temperature for the draft head's own sampling with `--spec-draft-sampling probabilistic`. | 0.7: +3.5 % (−3.3…+11.4), unclear, k44/k66 |
| `FLE_DFT_ROLLBACK` | off | `1`: after every verify round, remove the rows of rejected positions from the draft head's KV cache (after upstream PR #32045). | −0.5 %, noise, k65 |
| `LLAMA_DFLASH_DEV` | off | Upstream #31506 (cherry-picked): copy the hidden features for DFlash/DSpine with device copies instead of through the host. Stays on the host path while `FLE_MITSCHNITT` records. | +0.8 %, noise, k64 |
| `FLE_MITSCHNITT` | off | `<file>`: records the target's hidden states and verify outcomes of accepted rows for training a draft head on your own traffic (format in `common/speculative.cpp`). A path that cannot be opened is reported and the server runs without recording. **Records your prompts' hidden states: only use it on traffic you may store.** | −1.7 % (−1.8…−1.6) while recording, k39 |
| `FLE_MITSCHNITT_ALLES` | off | With `FLE_MITSCHNITT`: also write rows of chunks without a verify step (prefill). | - |
| `FLE_MITSCHNITT_MAX_GB` | off | With `FLE_MITSCHNITT`: stop recording when the file reaches this size. | - |

## Sampler

| Variable | Default | What it does | Measured |
|---|---|---|---|
| `FLE_SAMPLE_TOPK` | off | `1`: when no sampler before top-k needs the full vocabulary (no penalties, no DRY, no logit bias, no mirostat, top-k between 1 and 128), build only the top-k candidates in one pass over the logits, skipping blocks of 64 whose maximum is below the current k-th best. A grammar still checks the drawn token and falls back to the full list on rejection; while a reasoning budget forces tokens, the full list is used. | Decode +2.6 % (+1.9…+3.1) agent, +2.2 % (+1.6…+3.2) code, k77; host time per 16-wide verify round 0.82 → 0.29 ms; outputs bit-identical, tool calls included |
| `FLE_SAMPLE_ZEIT` | off | `<n>`: log where the sampler's host time goes, every n calls (diagnostics). | - |
| `FLE_RUNDEN_ZEIT` | off | `1`: log the host time per verify round, averaged over 200 rounds (`<n>` > 1: every n rounds). Diagnostics. | - |

## CUDA

| Variable | Default | What it does | Measured |
|---|---|---|---|
| `FLE_GRAPH_FORMEN` | off | `1`: one CUDA graph per batch shape instead of one graph that is rebuilt when the shape changes (draft, verify and prefill alternate). Technique from Infernix. | +0.5 %, noise, k72 |
| `FLE_IQ4_MMA` | off | `1`: Tensor Core verify kernel for IQ4_XS, Q4_K and Q5_K at 9 to 16 columns (weights stay packed, F16 activations, `mma.sync m16n8k16`, cp.async double buffering, deterministic split-K). Technique from Infernix. KLD is slightly better because activations stay in F16. | **slower in decode**: −7.1 % (−15.1…−0.9) agent, k73c; KLD −0.0015 / −0.0004 / −0.0003, k76. Leave off |
| `FLE_IQ4_MMA_Q6` | off | With `FLE_IQ4_MMA`: the same kernel for Q6_K. Slower than the stock path on our card. | - |

## What we run

Katy in production (and Kate, same build): `FLE_RS_WINDOW=1 FLE_GDN_REPLAY=1 LLAMA_DFLASH_DEV=1 FLE_SAMPLE_TOPK=1`,
optionally `FLE_NGRAM_MOD_FILE`. The command line is in the [README](../../README.md#kate-and-katy).
