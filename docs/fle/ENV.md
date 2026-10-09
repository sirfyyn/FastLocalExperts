# FLE environment variables and flags

This page lists every `FLE_*` environment variable read by the code (including `ggml/`) and the command line flags the
fork adds. FLE features are opt-in: with none of these set the fork behaves like the llama.cpp it is based on.

How to read the tables:

- **Default** is the behaviour when the variable is unset. "off" means the feature is disabled.
- **Ref.** is the value used in the reference configuration (the setup used for the measurements in the technical
  report, <https://sirfyyn.github.io/FastLocalExperts/>). "-" means it is not set there.
- Flag-like variables ("on") are usually enabled by any non-empty value that does not start with `0`; where the code
  checks existence only, the table says "set". Numeric variables are parsed with `atoi`/`atoll`/`atof`.
- Variables marked **experimental** are research switches that may change or disappear. **Obsolete** variables were
  superseded or measured as not useful and are kept only so old scripts keep working. **Debug** variables print or dump
  diagnostics and have no effect on results.
- Deprecated aliases: the pre-release German names of some variables are still read as a fallback (the new name is read
  first). They are listed in the last section and will be removed in a future release.

## GPU expert cache (`src/llama-moecache.cpp`)

The cache keeps hot MoE experts of host-resident layers in VRAM slots next to the router. It is enabled by
`--moe-expert-cache N` (fixed slots per layer) **or** by a byte budget (`FLE_MOE_CACHE_BUDGET_MIB`); the budget alone is
enough.

| Variable | Default | Meaning | Ref. |
|----------|---------|---------|------|
| `FLE_MOE_CACHE_BUDGET_MIB` | off (0) | Global byte budget (MiB) for the cache slot tensors (up + gate + down) of all layers. Slots per layer are planned from the budget instead of using a fixed count: every layer pays for its fixed part first, gets a floor of `FLE_MOE_CACHE_FLOOR_PCT` percent of the uniform slot count, and the rest goes to the experts with the most routed hits per byte (from the profile, otherwise spread evenly). Counts the slot tensors only; the I32 tables and bias vector come on top. | 17035 |
| `FLE_MOE_CACHE_BUDGET_FIT` | off | `1`: add the budget to the margin of device 0 in the `-fit` step (see [`--fit-target`](#--fit-target-and-the-cache-budget)), so the fit leaves the budget free. | 1 |
| `FLE_MOE_CACHE_FLOOR_PCT` | 10 | Per-layer floor of the budget plan, in percent of the uniform slot count (at least one slot while the budget allows). | - |
| `FLE_MOE_CACHE_SLOTS_MAX` | 0 (= number of experts) | Cap on the usable slots of one layer. | - |
| `FLE_MOE_PROFILE` | off (unset) | Path of the expert-usage profile. If the file exists and matches the model it is loaded before allocation: the budget plan uses its counts, the cache is pre-filled with the most frequent experts (warm start) and the admission counters are seeded. The file is updated while running and at shutdown. A missing file means cold start (created at the first save); a file that does not match the model is rejected and replaced. | `<file>` |
| `FLE_MOE_PROFILE_SAVE_SEC` | 600 | Seconds between profile saves; `0` = only at shutdown. | 600 |
| `FLE_MOE_PROFILE_MAX_OBS` | 134217728 (2^27) | Per-layer observation total at which the counts are halved (aging, so old runs do not dominate forever). | - |
| `FLE_MOE_ADMIT_FREQ` | 0 (off) | `> 0`: admission by aged request frequency; the counters are halved every this many observations per layer. Decides which miss is admitted and which slot is evicted. | 2000 |
| `FLE_MOE_EVICT_OLDEST` | 0 (off) | `>= 2`: evict the least frequent of the N oldest slots instead of the oldest one. | 32 |
| `FLE_MOE_OBS_MAXTOK` | 4 | Batches with more tokens than this are not observed by the LRU/frequency bookkeeping (prefill must not pollute it). | 8 |
| `FLE_MOE_CACHE_MAXTOK` | 8 | The cache graph is built for ubatches of up to this many tokens (decode and draft verification). | - |
| `FLE_MOE_CACHE_BATCH` | unset | `off`: restore the old guard (cache only for single-token batches). | - |
| `FLE_MOE_CACHE_BIAS` | value of `--moe-cache-bias` (0) | Additive routing bonus (probability units) for experts resident in the cache; steers selection only, the weights stay unbiased. Environment overrides the flag; the MTP context always uses 0. | 0.001 |
| `FLE_MOE_CACHE_BIAS_LAYERS` | unset | Per-layer override of the bias, `"from-to:value,layer:value"` (for example `0-3:0.002,4-45:0.001`). | - |
| `FLE_MC_OVERLAP` | 0 (off) | Overlap of GPU cache hits and CPU misses. `1`: the hit chain is placed in the graph before the CPU chain and its first node is marked; the scheduler then fetches the inputs of the CPU part before queueing the hit chain, so the GPU computes the hits while the CPU computes the misses. `2` (qwen4exp): additionally the shared expert is moved into the early chain. Read in `src/llama-graph.cpp`, `src/models/qwen4exp.cpp` and `ggml/src/ggml-backend.cpp`. | 2 |
| `FLE_MOE_PCIE_FRAC` | 0 (off) | Experimental (Patch 47, needs `FLE_MC_OVERLAP`). Share (0-1) of the misses of every layer call that is copied over PCIe into a small ring of extra slots and computed by the GPU hit chain instead of the CPU. The copies are enqueued on the hit chain's stream after the routed ids are known and before the hit chain starts; ring entries stay valid until their slot is reused. Evidence line `[fle-pcie]` every 512 steps. | - |
| `FLE_MOE_PCIE_RING` | 2 | Ring slots per layer for `FLE_MOE_PCIE_FRAC` (1-16), placed after the zero slot and paid out of the cache budget like the shadow slots. | - |
| `FLE_UBATCH_LEND` | 0 (off) | Experimental (Patch 48, after Strata's prefill). `-ub` becomes the large prefill ubatch: compute buffers are sized for `FLE_UBATCH_SMALL`, and a decode call with at least `FLE_UBATCH_LEND_MIN` tokens frees whole expert-cache layers (last layer first) for the larger buffer, runs with `-ub`, and the next short call gives the memory back and refills those layers from host memory. A lent layer has no cache (the CPU computes all of its experts). Needs one cache device (not with `FLE_MOECACHE_DEV1_SHARE`). Evidence lines `[fle-lend]`. | - |
| `FLE_UBATCH_SMALL` | 1024 | Ubatch outside a lend (`FLE_UBATCH_LEND`). | - |
| `FLE_UBATCH_LEND_MIN` | 2048 | Tokens per decode call from which the large ubatch is used (`FLE_UBATCH_LEND`). | - |
| `FLE_UBATCH_LEND_RESERVE_MIB` | 512 | Extra MiB lent on top of the measured compute-buffer difference (CUDA pool growth at the large ubatch). | - |
| `FLE_UBATCH_PREFETCH` | 0 (off) | Patch 49, needs `FLE_UBATCH_LEND`. During a lend the context also lends a ring of staging slots, each the size of the largest expert tensor in host memory; while one expert weight split computes, a second stream on the same device copies the whole weights of the next splits into the free slots, and each split only copies slot -> compute buffer on the device (no wait for its routed ids). Evidence: `[fle-prefetch] ON` at start, `prefetch on` in the `[fle-lend]` line, prefetched weights in the `FLE_TRANSFER_TIMING` output. | 1 |
| `FLE_UBATCH_PREFETCH_SLOTS` | 3 | Number of staging slots in the ring (1-8). One slot reaches only one split ahead; three cover one layer (up, gate, down). Each slot costs the size of the largest host expert tensor (FLE-IQ3_M: 450 MiB), lent from the expert cache together with the large ubatch. | 3 |
| `FLE_MOECACHE_SHADOW` | off (set to enable) | Experimental. Adds `2 * max_inserts` shadow slots per layer, so that an upload never goes into an occupied slot. Costs one slot of VRAM per shadow slot and layer. | - |
| `FLE_ADMIT_FREQUENT` | off (set to enable) | Experimental. Admit the most requested miss of a step instead of the latest. | - |
| `FLE_BUDGET_GLOBAL` | off (set to enable) | Experimental. The unused insert budget of a layer carries over to the next layer instead of expiring. | - |
| `FLE_MOECACHE_DEV1_SHARE` | 0 | Experimental. Percentage (0-100) of the layers whose cache lives on the second GPU; the distribution is per layer, never per expert. | - |
| `FLE_MOECACHE_DEV1_BLOCK` | off (set to enable) | Experimental. Give the second GPU a contiguous block of layers instead of distributing round-robin (only with `FLE_MOECACHE_DEV1_SHARE`). Measured as irrelevant without peer access. | - |
| `FLE_MOECACHE_PEER` | off | Experimental (`ggml/src/ggml-cuda/ggml-cuda.cu`). Request peer access for the second GPU's cache. | - |
| `FLE_UPLOAD_ASYNC` | off | Experimental. Bundle the slot copies of a step. | - |
| `FLE_UPLOAD_COUNTERS` | 0 (off) | Debug. Print upload counters every N calls. | - |
| `FLE_SLOWDOWN_UPLOAD_US` | 0 | Experiment. Artificially lengthen every slot copy by this many microseconds to find out whether the copy time lies in series with decoding. | - |
| `FLE_SLOWDOWN_TABLE_US` | 0 | Experiment. The same question for the table synchronization. | - |

## MoE routing at decode time (qwen4exp)

These are environment overrides of the `--moe-active-*` flags below; the environment wins over the flag.

| Variable | Default | Meaning | Ref. |
|----------|---------|---------|------|
| `FLE_MOE_K1` | flag value (0 = trained k) | Routed experts per token while decoding. | - |
| `FLE_MOE_K2` | flag value | Reference set of the renormalisation denominator (default: the trained k). | - |
| `FLE_MOE_SCALE` | flag value (0 = off) | Fixed factor on the renormalised expert weights. | - |
| `FLE_MOE_K_MAXTOK` | flag value (8) | Ubatches of at most this many tokens use the reduced routing; prefill keeps the trained routing. | - |
| `FLE_MOE_KL` | flag value | Per-layer k, `"from-to:k,layer:k"`. | - |

## MTP speculative decoding and prefill

| Variable | Default | Meaning | Ref. |
|----------|---------|---------|------|
| `FLE_MTP_VOCAB` | off (unset) | File with one token id per line (`#` comments allowed): the vocabulary list for the MTP head. The head is evaluated only for these tokens; EOG, control, user-defined and unknown tokens are always added. A missing or unreadable file switches the list off. Also enables `FLE_MTP_ARGMAX` unless that is set explicitly. | `<file>` |
| `FLE_MTP_VOCAB_N` | whole list | Use only the first N entries of the list file. | 32768 |
| `FLE_MTP_ARGMAX` | on if `FLE_MTP_VOCAB` is set, else off | Greedy draft by a single block-maximum pass instead of the sampler chain (same token; for exactly equal logits the smallest id). Needs `--no-spec-draft-backend-sampling`. | - |
| `FLE_MTP_PENALTIES` | on | Apply the repetition, frequency and presence penalties of the main sampler to the draft logits as well, so that fewer drafted tokens are penalized away by the main path. Changes only which tokens are drafted, not the output. `0` turns it off. Needs `--no-spec-draft-backend-sampling`. | 1 |
| `FLE_MTP_PSTOP` | off (0) | Adaptive draft length: end the draft when the product of the draft probabilities falls below `p` (0 < p < 1). Greedy path only. | - |
| `FLE_PREFILL_LEAN` | off | Lean prefill: caps the physical ubatch of the MTP context (`FLE_MTP_UBATCH`) so that its compute buffer stays small and `-fit` can measure it, and computes the QSA indexer of large ubatches in token blocks (`FLE_QSA_CHUNK`) so that one `[n_kv, n_batch]` mask less is allocated. | 1 |
| `FLE_MTP_UBATCH` | 512 | Ubatch cap of the MTP context with `FLE_PREFILL_LEAN`. | - |
| `FLE_QSA_CHUNK` | 256 | Token block size of the QSA indexer with `FLE_PREFILL_LEAN` (minimum 16). | - |
| `FLE_HC_FUSED` | off | `1`: fused kernels for the hyper-connection gate mean and combine instead of a chain of primitive ops (bit-identical). | 1 |
| `FLE_HC_LR` | off | Low-rank hyper-connection kernels for small batches (needs `FLE_HC_FUSED`; Q8_0 weights, not bit-identical to the mul_mat chain). | - |
| `FLE_GDN_L2` | reference | `old`: use `ggml_l2_norm` in the gated delta net instead of the reference normalisation, to measure both variants in one build. | - |
| `FLE_DSPARK_CTX` | - | Experimental. `zero`: use zeroed instead of captured context for the DSpark drafter. | - |
| `FLE_DSPARK_AUX` | off | Experimental. Capture auxiliary layer features for the DSpark drafter. | - |
| `FLE_PLE_NVME` | off | Comma-separated list of tensor name fragments (matched as substrings) to keep on disk instead of pulling them into RAM with `MAP_POPULATE`. Intended for huge lookup tables with small rows (per-layer token embeddings), for which read-ahead and residency are both wrong. | - |

## Server (`tools/server`)

| Variable | Default | Meaning | Ref. |
|----------|---------|---------|------|
| `FLE_SLOT_EARLY_SAVE` | off | `1`: when a slot is freed its content is saved into the host-RAM prompt cache immediately, instead of when a later request needs the slot. | 1 |
| `FLE_CACHE_GUARD_SLOT` | unset | Slot id whose prompt-cache entries are evicted last. Without the guard, short automated prompts can evict a long interactive context just because it is the oldest entry. | 0 |
| `FLE_CKPT_TAIL` | n_ubatch | Distance (tokens) from the end of the prompt of the second-to-last context checkpoint. By default it equals `n_ubatch`, which with a large `-ub` makes a follow-up request with the same prefix recompute up to `n_ubatch` tokens. | 512 |
| `FLE_SPEC_ADAPTIVE` | off (set to enable) | Experimental. Adaptive speculation gate: skip the draft during cold accept streaks, before the expensive verify. | - |
| `FLE_SPEC_COLD_N` | 4 | With `FLE_SPEC_ADAPTIVE`: length of the cold streak that triggers skipping. | - |
| `FLE_SPEC_SKIP_M` | 8 | With `FLE_SPEC_ADAPTIVE`: number of steps to skip. | - |
| `FLE_SPEC_PREFILL_SKIP` | off | Experimental. Skip the draft-context prefill while no slot is generating. | - |
| `FLE_PHASE_CLOCK` | 0 (off) | Debug. Print a split report (draft / verify / process) every N target decodes. | - |

## Sampling and loading

| Variable | Default | Meaning | Ref. |
|----------|---------|---------|------|
| `FLE_SAMPLER_NO_PREFILTER` | off (set to disable) | Disable the exact top-k prefilter of the sampler (to compare against the old path). | - |
| `FLE_GGUF_LENIENT_OFFSETS` | off (set to enable) | Instead of rejecting a GGUF whose tensor offsets differ from the expected layout, recompute the offsets cumulatively and warn (for files assembled by other tools that omitted the per-row metadata of the `IQ*_KT` types). | - |
| `FLE_PAGER_VRAM_MB`, `FLE_PAGER_VRAM_MB1` | off | Enable the CUDA pager buffer type with the given VRAM pool (MiB) on device 0 / 1; tensors are bound with `-ot <pattern>=PAGER` (`PAGER1` for the second device). See the pager section. | - |

## Diagnostics and debug

None of these changes results unless noted. They are meant for developers.

| Variable | Default | Meaning |
|----------|---------|---------|
| `FLE_TIME_PROFILE` | off | Per-node-name time profile through the eval callback; value = report interval in calls (default 20000 when set but empty). |
| `FLE_NAN_GUARD`, `FLE_NAN_GUARD_CAP` | off, 3 | Report the first tensor that contains NaN together with its inputs; cap on the number of reports. |
| `FLE_TRACE_WRITE=<file>` / `FLE_TRACE_READ=<file>` | off | Two-run tracer: one known-good run writes name and L2 norm of every F32 tensor; a second run compares step by step and stops cleanly at the first deviation. |
| `FLE_TRACE_TOLERANCE` | 0.02 | Relative deviation above which the tracer counts a break. |
| `FLE_TRACE_DISTRIBUTION` | off | Collect all deviations instead of stopping at the first break. |
| `FLE_TRACE_CTX` | 262144 | Context budget used for the deviation check. |
| `FLE_DUMP_MOE`, `FLE_DUMP_MOE_CAP` | off, 4000 | Dump MoE top-k selections; cap on the number of dumps. |
| `FLE_DUMP_ROUTER_W` | off | Print the router weights per layer (`[ROUTER-W] <layer> w1,...,wk`). |
| `FLE_DUMP_FFN_DENSITY` | off | Activation density statistics of the MoE FFN intermediates (cap via `FLE_DUMP_MOE_CAP`). |
| `FLE_DUMP_MTP`, `FLE_DUMP_MTP_CAP` | off, 12 | Dump MTP tensors; cap on the number of dumps. |
| `FLE_DSPARK_AUX_DUMP`, `FLE_DSPARK_AUX_TRACE` | off | Dump / trace the DSpark auxiliary features. |
| `FLE_CAPTURE_DIAG` | off | Always print the layer-capture diagnostic (otherwise only when it fails). |
| `FLE_REUSE_REASON` | off | Count why the compute graph could not be reused. |
| `FLE_SPEC_TRACE` | off | Trace KV-cache slot searches and speculative positions. |
| `FLE_SPEC_DEBUG`, `FLE_SPEC_DEBUG_DUMP_DIR` | off | EAGLE3/MTP draft debug output. With `FLE_SPEC_DEBUG_DUMP_DIR=<dir>` the first engine calls (tokens and aux features) are additionally written to `<dir>/eagle3_engine_call<N>.bin`; without the directory nothing is written. |
| `FLE_SPEC_FEATDIAG` | off | Count non-finite values in the captured features. |
| `FLE_SPEC_TIMINGS` | off | Time shares of the draft step. |
| `FLE_KLD_DUMP=<file>` | off | `llama-perplexity`: write KL divergence and delta-p per token for paired comparisons. |
| `FLE_META_LEAF_DEBUG`, `FLE_META_MM_DEBUG`, `FLE_META_SPLIT_DEBUG` | off | Print split-state decisions of the meta backend (value = maximum number of lines). |
| `FLE_SPLIT_EVENTS` | off | Experimental (`ggml/src/ggml-backend.cpp`). Use stream events as markers for the pre-queued expert copies of the next layer. |
| `FLE_TRANSFER_TIMING` | 0 (off) | Wall-clock time around the synchronization points between backends; value = report interval. |
| `GGML_BACKEND_BUF_TRACE` | off | Log every backend buffer allocation with size and buffer type (`ggml/src/ggml-backend.cpp`). |
| `GGML_CUDA_POOL_TRACE` | off | Log every CUDA pool allocation with pointer, requested and actual size (`ggml/src/ggml-cuda/common.cuh`). |

## CUDA hybrid pager (`ggml/src/ggml-cuda/fle/pager_buffer.cu`)

The pager is a research path: stacked expert tensors live in pinned host memory, a compact VRAM pool serves as staging
target, and misses are either staged over PCIe or computed on the CPU. It is off unless `FLE_PAGER_VRAM_MB` is set. All
pager variables are **experimental** unless marked otherwise; none of them is part of the reference configuration.

| Variable | Default | Meaning |
|----------|---------|---------|
| `FLE_PAGER_VRAM_MB`, `FLE_PAGER_VRAM_MB1` | off | VRAM pool size (MiB) for device 0 / 1; enables the pager. |
| `FLE_PAGER_REGIONS`, `FLE_PAGER_REGIONS1` | 160 | Number of regions of the pool (per device). |
| `FLE_PAGER_SCRATCH_MB`, `FLE_PAGER_SCRATCH_MB1` | 2048 | Scratch area (MiB), capped at half of the pool. |
| `FLE_PAGER_SWAP_MB`, `FLE_PAGER_SWAP_MB1` | 0 | Swap arena (MiB) at the end of the pool, capped at 25 %. |
| `FLE_PAGER_CPU_MISS` | off | Compute missed experts on the CPU instead of staging them over PCIe (only the activation crosses the bus). |
| `FLE_PAGER_CPU_MISS_FREQ_T` | off | Hybrid mode with `FLE_PAGER_CPU_MISS`: experts requested at least this often are staged to the GPU, colder ones run on the CPU. |
| `FLE_PAGER_CPU_MISS_KEEP_STAGE` | off | Debug. Stage and override at the same time. |
| `FLE_PAGER_CPU_THREADS` | 0 (default pool) | Thread count of the CPU miss path. |
| `FLE_PAGER_SPLIT`, `FLE_PAGER_SPLIT_MIN_N` | off, 64 | Split the misses between the PCIe and CPU branches by measured throughput; minimum samples per side. |
| `FLE_PAGER_ASYNC_STAGE` | off | Stage misses on a dedicated copy stream. |
| `FLE_PAGER_BOUNCE` | off | Force a bounce copy through a pinned buffer. |
| `FLE_PAGER_PREFETCH` | off | Prefetch the experts of the last token during the current compute. |
| `FLE_PAGER_EVICT_FREQ` | off | Frequency-aware eviction (protects recurrently hot experts). |
| `FLE_PAGER_DEVICE_REMAP` | off | Remap expert ids with a device kernel instead of a host round trip. |
| `FLE_PAGER_NO_UVA` | off | Abort instead of silently reading host memory over UVA. |
| `FLE_PAGER_NO_PIN`, `FLE_PAGER_MMAP_FILE` | off | Use pageable instead of pinned host memory / back the host tier with a file. |
| `FLE_PAGER_HOST_REGISTER`, `FLE_PAGER_HOST_REGISTER_MB` | off | Register existing host memory with CUDA (optionally only the first N MiB). |
| `FLE_PAGER_BW` | off | Debug. Bandwidth statistics of the PCIe and CPU paths. |
| `FLE_PAGER_STAT_EVERY` | 1024 | Debug. Statistics window in calls. |
| `FLE_PAGER_RESET_FILE` | unset | Debug. If this file exists at a statistics point it is removed and the counters are reset (to drop warm-up). |
| `FLE_PAGER_FREQ_DUMP`, `FLE_PAGER_TRACE_FILE` | off | Debug. Dump the expert frequency distribution / the routing trace for offline simulation. |
| `FLE_PAGER_PHASES`, `FLE_PAGER_LEDGER`, `FLE_PAGER_WALKTHROUGH`, `FLE_PAGER_WALKTHROUGH_DECODE` | off | Debug. Phase timing, transfer ledger, and step-by-step comparison of the pager against the unpaged result. |
| `FLE_PAGER_VERIFY`, `FLE_PAGER_VERIFY_CAP`, `FLE_PAGER_VALIDATE`, `FLE_PAGER_IDSYNC`, `FLE_PAGER_DUMP_SHAPES`, `FLE_PAGER_S3_DEBUG` | off | Debug. Consistency checks and dumps of the staging path. |
| `FLE_PAGER_PROCYON`, `FLE_PAGER_PROCYON_ACC`, `FLE_PAGER_PROCYON_TOPN`, `FLE_PAGER_PROCYON_DEBUG`, `FLE_PROCYON_RAW_BIAS` | off | **Obsolete.** Routing prediction for the next layer; measured prediction accuracy was close to chance, so the path is kept only for reproducibility and has known multi-GPU issues. |

## Tools

| Variable | Used by | Meaning |
|----------|---------|---------|
| `FLE_CAP_CORPUS`, `FLE_CAP_OUTDIR`, `FLE_CAP_CHUNKS` (12), `FLE_CAP_SKIP` (0), `FLE_CAP_HNEXTN`, `FLE_CAP_LAYERS` | `fle-moe-capture` | Text corpus, output directory, number of chunks, chunks to skip, additionally dump the pre-head state, `0` = no layer capture. |
| `FLE_CAP_LAYERS`, `FLE_CAP_CORPUS`, `FLE_CAP_OUT`, `FLE_CAP_N` (500), `FLE_CAP_MAXTOK` (64) | `fle-capture-layers` | Layers to capture (csv), texts file, output directory, sample count, tokens per sample. |
| `FLE_DUMP_AUX` (2,30,57), `FLE_DUMP_CORPUS`, `FLE_DUMP_OUTDIR`, `FLE_DUMP_START` (0) | `fle-eagle3-dump` | Auxiliary layer ids, corpus, output directory, first line. |
| `FLE_BENCH_CORPUS`, `FLE_BENCH_M` (1), `FLE_BENCH_STEPS` (300), `FLE_BENCH_WARM` (150), `FLE_BENCH_PREFILL` (512), `FLE_BENCH_STRIDE` (6000) | `fle-step-bench` | Cost curve of verifying m tokens per step with the expert cache. |
| `FLE_KB_ITERS` (300), `FLE_KB_TOKENS` (1), `FLE_KB_TYPES` | `fle-kernel-bench` | Iterations, tokens per call and quantization types of the expert kernel benchmark. |

## Command line flags added by the fork

| Flag | Meaning |
|------|---------|
| `--moe-expert-cache N` | GPU cache slots per host-resident MoE expert layer; 0 = disabled (default). Not needed when `FLE_MOE_CACHE_BUDGET_MIB` is set. Env: `LLAMA_ARG_MOE_EXPERT_CACHE`. |
| `--moe-expert-cache-inserts N` | Maximum expert uploads per layer and decode step (default 2). Env: `LLAMA_ARG_MOE_EXPERT_CACHE_INSERTS`. |
| `--moe-cache-bias G` | Routing bonus for cached experts, see `FLE_MOE_CACHE_BIAS` (default 0). |
| `--moe-active-experts K1[,K2]`, `--moe-active-scale S`, `--moe-active-max-tokens N`, `--moe-active-layers SPEC` | Reduced routing while decoding (qwen4exp); see the routing table above. |
| `-lm`, `--load-mode none` | `--load-mode` is upstream; the reference configuration uses `none`: tensors are read into allocated host buffers instead of being memory-mapped (no mmap, mlock or direct I/O). Only without mmap can the backend place CPU-resident expert weights in its own (pinned) host buffers, which is what makes cache uploads and CPU/GPU overlap fast. |
| `-cmoed`, `--spec-draft-cpu-moe` (`--cpu-moe-draft`) | Keep all MoE weights of the draft model (for example the MTP head) on the CPU. `-ncmoed N` does the same for the first N layers. |
| `--no-spec-draft-backend-sampling` | Disable offloading of draft sampling to the backend (default: enabled). Required for the host-side draft paths `FLE_MTP_ARGMAX` and `FLE_MTP_PENALTIES`, which are skipped while a backend sampler chain is active. |
| `--spec-type draft-mtp`, `-md FNAME` | Upstream flags used to run MTP speculative decoding with a separate MTP head file. |

### `--fit-target` and the cache budget

`-fit` (on by default) sizes the context and layer placement so that `--fit-target MiB[,MiB...]` of free memory remains
on each device. The expert cache is allocated after the fit, from whatever the fit leaves free. Without help the fit
would therefore fill the memory the cache needs.

With `FLE_MOE_CACHE_BUDGET_FIT=1` and `FLE_MOE_CACHE_BUDGET_MIB=B`, the budget `B` is added to the margin of device 0
(`fit_target[0] += B`; the added amount is logged). `--fit-target` then keeps its meaning of "free memory besides the
cache", and the cache is guaranteed to find its budget free. Without `FLE_MOE_CACHE_BUDGET_FIT` the margin is exactly
`--fit-target`. Devices are not split: the cache of a layer sits next to its router, which is device 0 in the usual
single-GPU setup.

## Deprecated aliases

The following pre-release names are still read when the new name is unset. Use the new names in new scripts.

| Deprecated name | Current name |
|-----------------|--------------|
| `FLE_SLOT_FRUEH` | `FLE_SLOT_EARLY_SAVE` |
| `FLE_PREFILL_SCHLANK` | `FLE_PREFILL_LEAN` |
| `FLE_MTP_STRAFEN` | `FLE_MTP_PENALTIES` |
| `FLE_CACHE_SCHUTZ_SLOT` | `FLE_CACHE_GUARD_SLOT` |
| `FLE_MOECACHE_SCHATTEN` | `FLE_MOECACHE_SHADOW` |
| `FLE_UEBERGANGSZEIT` | `FLE_TRANSFER_TIMING` |
| `FLE_BREMSE` | `FLE_SLOWDOWN_UPLOAD_US` |
| `FLE_BREMSE_TABELLE` | `FLE_SLOWDOWN_TABLE_US` |
| `FLE_UPLOAD_ZAEHLER` | `FLE_UPLOAD_COUNTERS` |
| `FLE_ZULASSUNG_HAEUFIG` | `FLE_ADMIT_FREQUENT` |
| `FLE_MOECACHE_DEV1_ANTEIL` | `FLE_MOECACHE_DEV1_SHARE` |
| `FLE_NAN_WACHE`, `FLE_NAN_WACHE_CAP` | `FLE_NAN_GUARD`, `FLE_NAN_GUARD_CAP` |
| `FLE_SPUR_SCHREIBEN`, `FLE_SPUR_LESEN`, `FLE_SPUR_TOLERANZ`, `FLE_SPUR_VERTEILUNG`, `FLE_SPUR_KONTEXT` | `FLE_TRACE_WRITE`, `FLE_TRACE_READ`, `FLE_TRACE_TOLERANCE`, `FLE_TRACE_DISTRIBUTION`, `FLE_TRACE_CTX` |
| `FLE_ZEITPROFIL` | `FLE_TIME_PROFILE` |
| `FLE_SPEC_ZEITEN` | `FLE_SPEC_TIMINGS` |
| `FLE_REUSE_GRUND` | `FLE_REUSE_REASON` |
| `FLE_PAGER_PHASEN`, `FLE_PAGER_BILANZ`, `FLE_PAGER_DURCHZUG`, `FLE_PAGER_DURCHZUG_DECODE`, `FLE_PROCYON_BIAS_ROH` | `FLE_PAGER_PHASES`, `FLE_PAGER_LEDGER`, `FLE_PAGER_WALKTHROUGH`, `FLE_PAGER_WALKTHROUGH_DECODE`, `FLE_PROCYON_RAW_BIAS` |
| `GGML_BACKEND_BUF_BELEG`, `GGML_CUDA_POOL_BELEG` | `GGML_BACKEND_BUF_TRACE`, `GGML_CUDA_POOL_TRACE` |

Two value aliases are accepted as well: `FLE_GDN_L2=alt` (now `old`) and `FLE_MOE_CACHE_BATCH=aus` (now `off`).
