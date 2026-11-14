# FLE v1.3.0 (draft release notes)

> Draft. Not tagged, not released, not signed. `{{PENDING}}` marks what is filled in before the tag.

Portable binary packages of the FLE engine, built from source by GitHub Actions from commit `{{PENDING: sha8}}`, the
tagged commit. The workflow is [`fle-binaries.yml`](https://github.com/sirfyyn/FastLocalExperts/blob/v1.3.0/.github/workflows/fle-binaries.yml),
run `{{PENDING}}`.

The easiest way to run this release is [**FLE Link**](https://flelink.run), the desktop app for FLE: it installs the
engine, picks the recipe and sets the flags ([source](https://github.com/sirfyyn/flelink)). The recipe for Kate is
`kate-27b-dspine`.

## A new line: dense models with DSpine, about 155 tok/s on one 32 GB card

v1.3.0 is a different engine line from v1.2.x. It is upstream llama.cpp `344846b4f` (2026-10-30) plus the engine
changes that run **Katy** and **Kate**, two fine-tunes of Qwen3.8-27B (48 Gated DeltaNet layers and 16 attention layers),
with the DSpine b16 draft head. It is exactly the source of the build Katy has been served with since 13 November 2026.

**The MoE features of v1.2.x are not in v1.3.0**: GPU expert cache (`--moe-expert-cache*`, `FLE_MOE_*`), expert store,
split experts, the MTP vocabulary, the trellis types `IQ*_KT` and `GET /fle/status`. For DeepSeek V4.1 Flash, Rin and
Qwen3.8-Flash-Next keep using [v1.2.6](https://github.com/sirfyyn/FastLocalExperts/releases/tag/v1.2.6).
{{PENDING: plan for bringing the two lines together.}}

Measured on one RTX PRO 4500 Blackwell (32 GB), Ryzen 9 9950X3D, Linux, with Katy (IQ4_XS) and the public DSpine b16
head (Q4_K_M), n-max 15, temperature 1.0 unless noted. Every number is a paired A-B-B-A run with a 95 % bootstrap
interval; the run IDs refer to Appendix C of the Kate report ({{PENDING: link}}).

| Change | Switch | Effect | 95 % interval | Run |
|---|---|---|---|---|
| DSpine b16 draft head instead of DFlash2, n-max 15 | `--spec-type draft-dflash -md <DSpine>` | **+34.4 %** (107.7 → 144.8 tok/s) | +25.9…+43.2 | k6 |
| same at temperature 1.0 | | +19.0 % (104.7 → 124.5) | +12.5…+25.9 | k7 |
| Convolution window instead of buffer copies | `FLE_RS_WINDOW=1` | +1.2 %, recurrent state 35.6 instead of 90 MiB, bit-identical 30/30 | +1.1…+1.3 | k28 |
| Gated DeltaNet replay instead of state snapshots | `FLE_GDN_REPLAY=1` | **+4.6 %**, recurrent state 348 instead of 2,340 MiB, bit-identical 30/30 | +4.5…+4.7 | k30 |
| Replay on smaller cards (VRAM capped) | `FLE_GDN_REPLAY=1` | 20 GB: +23.7 % (DSpine 15 + replay vs DSpine 7 + window); 16 GB: +16.6 % | | k32, k33 |
| Sampler fast path, top-k only | `FLE_SAMPLE_TOPK=1` | **+2.6 %** agent, **+2.2 %** code, outputs bit-identical incl. tool calls; host time per round 0.82 → 0.29 ms | +1.9…+3.1 / +1.6…+3.2 | k77 |
| n-gram table v2, pre-warmed, temperature 1.0 | `--spec-type ngram-mod`, `FLE_NGRAM_MOD_FILE` | +19.8 % | | k10 |
| f16 instead of q8_0 KV cache at ≈ 150k context (flag, not an engine change) | `-ctk f16 -ctv f16` | +27.7 % (78.0 → 99.6) | +15.1…+45.5 | k62 |
| CUDA graph per batch shape | `FLE_GRAPH_FORMEN=1` | +0.5 %, noise | | k72 |
| Upstream commits incl. device-side DFlash hidden-state copy | `LLAMA_DFLASH_DEV=1` | +0.4 % / +0.8 %, noise | | k63, k64 |

What we run: `FLE_RS_WINDOW=1 FLE_GDN_REPLAY=1 LLAMA_DFLASH_DEV=1 FLE_SAMPLE_TOPK=1`. In production on 12 November
Katy decoded a median of 154.6 tok/s over 1,729 real agent responses (P10 106.5, P90 214.4); paired with a cold n-gram
table 161.8 tok/s on agent tasks and 151.4 on code. Kate: {{PENDING}}.

## New

- **DSpine draft heads** (`--spec-type draft-dflash`): block-diffusion heads that propose up to 16 tokens per pass, after
  AjHomelab's DSpine patch for llama.cpp ([NOTICE-dspine.md](NOTICE-dspine.md)). GGUF conversion in `conversion/qwen.py`.
- **Verify rounds for Gated DeltaNet models**: `FLE_RS_WINDOW`, `FLE_GDN_REPLAY`, `FLE_RS_SEQ`.
- **n-gram table v2** for `--spec-type ngram-mod`: tagged buckets, age-based replacement, optional persistence
  (`FLE_NGRAM_MOD_FILE`). This replaces upstream's table whenever `ngram-mod` is used.
- **Sampler fast path** `FLE_SAMPLE_TOPK`, diagnostics `FLE_SAMPLE_ZEIT`, `FLE_RUNDEN_ZEIT`.
- **Draft-head training data** `FLE_MITSCHNITT`: records hidden states and verify outcomes of accepted rows (−1.7 %
  while on, k39). Only use it on traffic you may store.
- Optional and off: `FLE_GRAPH_FORMEN`, `FLE_IQ4_MMA` (Tensor Core verify kernel after Infernix, see
  [NOTICE-infernix.md](NOTICE-infernix.md)), `FLE_DRAFT_TEMP`, `FLE_DFT_ROLLBACK`.
- Cherry-picked upstream fixes on top of `344846b4f`: #31506 (DFlash hidden features via device copies,
  `LLAMA_DFLASH_DEV`), CUDA graph reuse and eviction fixes, reference-counted server checkpoints, on-device state copy
  sizes for quantized KV, halved VMM pool reservation.

Every variable is documented in [docs/fle/ENV.md](docs/fle/ENV.md).

## Known issues

- `FLE_IQ4_MMA` is slower in decode (−7.1 %, CI −15.1…−0.9, k73c) although its KLD is slightly better (k76). Leave it off.
- `FLE_RS_WINDOW` and `FLE_GDN_REPLAY` work with `--parallel 1` only; layers on the CPU keep snapshots.
- A persistent n-gram table (`FLE_NGRAM_MOD_FILE`) makes runs depend on earlier runs. Benchmark with a cold or frozen
  table; our own +17.3 % for a re-trained head came entirely from this (report §8.1).
- Two slots with a shared KV cache: one stream −6.5 %, two streams together +27 % throughput (k37).
- {{PENDING: Windows package tested with Kate?}}

## Requirements

- **CUDA 13.4**, so an NVIDIA driver of release 580 or newer.
- **Windows:** the package needs the upstream cudart (`cudart-llama-bin-win-cuda-13.4-x64.zip` from llama.cpp's
  releases) next to the binaries.

## Links

- [FLE Link](https://flelink.run): one-click FLE, recipe `kate-27b-dspine` ([source](https://github.com/sirfyyn/flelink))
- [Kate on Hugging Face](https://huggingface.co/sirfyyn/Kate-Qwen-27B-GGUF-SpedUp-Abliterated)
- [Katy on Hugging Face](https://huggingface.co/sirfyyn/Katy-27B-GGUF)

## Checksums (SHA-256)

```
{{PENDING}}  fle-{{PENDING}}-bin-ubuntu-cuda-13.4-x64.tar.gz
{{PENDING}}  fle-{{PENDING}}-bin-win-cuda-13.4-x64.zip
```

The same values are in the attached `SHA256SUMS`.
