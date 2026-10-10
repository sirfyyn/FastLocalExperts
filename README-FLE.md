# FLE - Fast Local Experts

FLE is a [llama.cpp](https://github.com/ggml-org/llama.cpp) fork for running large Mixture-of-Experts (MoE) models on
**one 32 GB GPU plus CPU experts**. The attention layers, shared weights and a cache of hot experts live on the GPU;
the bulk of the routed experts stays in host RAM and is computed on the CPU when they are not cached.

Everything not described here is llama.cpp; see [README.md](README.md) and the upstream documentation.

> Benchmark numbers, ablations and the evaluation setup are not part of this repository; they are in the technical
> report: <https://sirfyyn.github.io/FastLocalExperts/> (also as [PDF](https://sirfyyn.github.io/FastLocalExperts/fle-paper.pdf)).
> The release with the report is archived on Zenodo: [doi:10.5281/zenodo.23272400](https://doi.org/10.5281/zenodo.23272400).
> The model files for Qwen3.8-Flash-Next are at <https://huggingface.co/sirfyyn/Qwen3.8-Flash-Next-FLE-GGUF>; the pruned
> FLE-IQ3_M-REAP448 (448 of 512 experts per layer) has its own repo:
> <https://huggingface.co/sirfyyn/Qwen3.8-Flash-Next-FLE-REAP448-GGUF>.

## Project status

This is a private project: I build it for my own machine and publish it as it is. Hints, findings and bug reports are always welcome; acting on them, reviewing and merging can take a while, and there is no support promise. Issues and pull requests are the best way to send them.

## Key features

- **GPU expert cache with a global byte budget and warm-start profile.** Hot experts are cached in VRAM. The slots per
  layer are derived from one global byte budget (`FLE_MOE_CACHE_BUDGET_MIB`) instead of a fixed slot count, and the cache
  can be pre-populated from a usage profile recorded in an earlier run (`FLE_MOE_PROFILE`), so a restarted server starts
  warm. The budget can be coupled to `--fit-target` (`FLE_MOE_CACHE_BUDGET_FIT`).
- **Overlap of GPU hits and CPU misses.** Experts that hit the cache are computed on the GPU while the missed experts are
  computed on the CPU at the same time (`FLE_MC_OVERLAP`).
- **MTP speculative decoding** with the model's own multi-token-prediction head loaded from a separate file and an optional restricted vocabulary list
  for the draft (`FLE_MTP_VOCAB`, `FLE_MTP_VOCAB_N`), with draft-time sampler penalties and an adaptive draft length.
- **Lean prefill** (`FLE_PREFILL_LEAN`): caps the physical micro-batch of the MTP context and computes the sparse
  attention indexer in token blocks, which lowers peak memory during long prefills.
- **Pinned expert loading** with `--load-mode none` (no mmap): expert weights are read into host buffers that can be
  pinned for fast host-to-device transfers.
- **AVX2 `Q2_0` kernel** (`ggml_vec_dot_q2_0_q8_0`): on x86 the CPU path previously had only the scalar fallback.
- Early slot save for the server prompt cache (`FLE_SLOT_EARLY_SAVE`), `IQ2_KT`/`IQ3_KT`/`IQ4_KT` trellis quant types and
  a CUDA hybrid pager path (see the notices below).

All FLE features are opt-in via environment variables or the flags below and are off by default.
The full list is in [docs/fle/ENV.md](docs/fle/ENV.md).

## Making Rhea fast: eight short films

Eight short films (English, subtitles burned in, 33 to 63 s each, 7 min in total) show how FLE was measured on one machine
and what did and did not work. Every number in them comes from the measurement database behind the technical report
(6,130 measurements, state of 2026-10-08); where a film and the report differ, the report is the reference. GitHub does
not play videos from repository files, so the posters below link to the films on Hugging Face. All eight are also on
YouTube as one playlist: <https://www.youtube.com/playlist?list=PLMYhpsBSEhIQ>.

<table>
<tr>
<td width="25%"><a href="https://huggingface.co/sirfyyn/Qwen3.8-Flash-Next-FLE-GGUF/resolve/main/media/making-rhea-fast-1.mp4"><img src="https://huggingface.co/sirfyyn/Qwen3.8-Flash-Next-FLE-GGUF/resolve/main/media/making-rhea-fast-1.png" alt="Film 1: The race"></a><br><b>1 · The race</b> (0:33)</td>
<td width="25%"><a href="https://huggingface.co/sirfyyn/Qwen3.8-Flash-Next-FLE-GGUF/resolve/main/media/making-rhea-fast-2.mp4"><img src="https://huggingface.co/sirfyyn/Qwen3.8-Flash-Next-FLE-GGUF/resolve/main/media/making-rhea-fast-2.png" alt="Film 2: The bracket"></a><br><b>2 · The bracket</b> (0:34)</td>
<td width="25%"><a href="https://huggingface.co/sirfyyn/Qwen3.8-Flash-Next-FLE-GGUF/resolve/main/media/making-rhea-fast-3.mp4"><img src="https://huggingface.co/sirfyyn/Qwen3.8-Flash-Next-FLE-GGUF/resolve/main/media/making-rhea-fast-3.png" alt="Film 3: The staircase"></a><br><b>3 · The staircase</b> (1:03)</td>
<td width="25%"><a href="https://huggingface.co/sirfyyn/Qwen3.8-Flash-Next-FLE-GGUF/resolve/main/media/making-rhea-fast-4.mp4"><img src="https://huggingface.co/sirfyyn/Qwen3.8-Flash-Next-FLE-GGUF/resolve/main/media/making-rhea-fast-4.png" alt="Film 4: The expert cache"></a><br><b>4 · The expert cache</b> (0:59)</td>
</tr>
<tr>
<td width="25%"><a href="https://huggingface.co/sirfyyn/Qwen3.8-Flash-Next-FLE-GGUF/resolve/main/media/making-rhea-fast-5.mp4"><img src="https://huggingface.co/sirfyyn/Qwen3.8-Flash-Next-FLE-GGUF/resolve/main/media/making-rhea-fast-5.png" alt="Film 5: Draft and verify"></a><br><b>5 · Draft and verify</b> (1:00)</td>
<td width="25%"><a href="https://huggingface.co/sirfyyn/Qwen3.8-Flash-Next-FLE-GGUF/resolve/main/media/making-rhea-fast-6.mp4"><img src="https://huggingface.co/sirfyyn/Qwen3.8-Flash-Next-FLE-GGUF/resolve/main/media/making-rhea-fast-6.png" alt="Film 6: What did not work"></a><br><b>6 · What did not work</b> (0:56)</td>
<td width="25%"><a href="https://huggingface.co/sirfyyn/Qwen3.8-Flash-Next-FLE-GGUF/resolve/main/media/making-rhea-fast-7.mp4"><img src="https://huggingface.co/sirfyyn/Qwen3.8-Flash-Next-FLE-GGUF/resolve/main/media/making-rhea-fast-7.png" alt="Film 7: The rebench"></a><br><b>7 · The rebench</b> (0:58)</td>
<td width="25%"><a href="https://huggingface.co/sirfyyn/Qwen3.8-Flash-Next-FLE-GGUF/resolve/main/media/making-rhea-fast-8.mp4"><img src="https://huggingface.co/sirfyyn/Qwen3.8-Flash-Next-FLE-GGUF/resolve/main/media/making-rhea-fast-8.png" alt="Film 8: Why we do this"></a><br><b>8 · Why we do this</b> (0:58)</td>
</tr>
</table>

All eight in one file: [making-rhea-fast-complete.mp4](https://huggingface.co/sirfyyn/Qwen3.8-Flash-Next-FLE-GGUF/resolve/main/media/making-rhea-fast-complete.mp4) (6:59).
The models, the MTP head and the films live in
[sirfyyn/Qwen3.8-Flash-Next-FLE-GGUF](https://huggingface.co/sirfyyn/Qwen3.8-Flash-Next-FLE-GGUF), the pruned
FLE-IQ3_M-REAP448 in [sirfyyn/Qwen3.8-Flash-Next-FLE-REAP448-GGUF](https://huggingface.co/sirfyyn/Qwen3.8-Flash-Next-FLE-REAP448-GGUF).

## Inspirations and third-party code

| Project | What was adopted | Notice |
|---------|------------------|--------|
| [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) (Iwan Kawrakow, MIT) | The trellis-quantized `IQ2_KT`/`IQ3_KT`/`IQ4_KT` tensor types and their decoding kernels, taken essentially verbatim. | [NOTICE-ik_llama.md](NOTICE-ik_llama.md) |
| [FreeToken](https://github.com/FlashML-org/FreeToken) (Apache-2.0, arXiv:2608.16157) | The hybrid pager path and the bandwidth rule for staging experts, re-implemented in the CUDA backend. | [NOTICE-freetoken.md](NOTICE-freetoken.md) |
| [KTransformers](https://github.com/kvcache-ai/ktransformers) (Apache-2.0) | The idea of running MoE experts on the CPU while attention and shared weights stay on the GPU. No KTransformers code is part of this tree. | - |
| [Strata](https://github.com/Niko1221/Strata) (MIT) | Selected ideas and code from the serving stack (for example the early slot save). | [NOTICE-strata.md](NOTICE-strata.md) |

## Build

FLE builds like llama.cpp; see [docs/build.md](docs/build.md). For the CUDA path:

```bash
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
```

Use `-DCMAKE_CUDA_ARCHITECTURES=<arch>` to build only for your GPU (for example `89` for Ada, `120` for Blackwell).
The CPU kernels use AVX2 where available; build natively (`-DGGML_NATIVE=ON`, the default) for best results.

## Example server command

The expert tensors are kept on the CPU with `-cmoe` (or `-ncmoe N` / `--override-tensor`), the rest goes to the GPU.
The cache budget is set in MiB through the environment; `--fit-target` is the free VRAM to leave besides the cache.

```bash
FLE_MOE_CACHE_BUDGET_MIB=17035 \
FLE_MOE_CACHE_BUDGET_FIT=1 \
FLE_MOE_PROFILE=/path/to/expert-profile.bin \
FLE_MOE_PROFILE_SAVE_SEC=600 \
FLE_MC_OVERLAP=2 \
FLE_MTP_VOCAB=/path/to/mtp-vocab.txt \
FLE_MTP_VOCAB_N=32768 \
FLE_PREFILL_LEAN=1 \
FLE_UBATCH_LEND=1 \
FLE_UBATCH_PREFETCH=1 \
./build/bin/llama-server \
    -m model.gguf \
    --spec-type draft-mtp \
    -md mtp.gguf \
    -b 4096 -ub 4096 \
    --load-mode none \
    -cmoe \
    -ngl 99 \
    -c 65536 \
    --fit-target 1024 \
    --host 127.0.0.1 --port 8080
```

Notes:

- `-md mtp.gguf` is the MTP head (the model's own MTP module, experts requantized to IQ4_NL); omit it (and `--spec-type`) to run without speculative decoding.
- `FLE_UBATCH_LEND=1` with `-ub 4096`: decode runs with a 1024-token ubatch and the full expert cache; prompts of
  2048 tokens or more borrow cache layers for the large ubatch and return them afterwards. `FLE_UBATCH_PREFETCH=1`
  copies the next expert weights on a second stream during such a prefill. Both only affect prefill speed.
- The byte budget alone enables the expert cache; `--moe-expert-cache N` (fixed slots per layer) is not needed when
  `FLE_MOE_CACHE_BUDGET_MIB` is set. See [docs/fle/ENV.md](docs/fle/ENV.md) for how the budget, `--moe-expert-cache`
  and `--fit-target` interact.

## Licenses

FLE is released under the MIT License (see [LICENSE](LICENSE)). Third-party notices are in the `NOTICE-*.md` files;
the Apache-2.0 license text is in [licenses/LICENSE-Apache-2.0](licenses/LICENSE-Apache-2.0).
