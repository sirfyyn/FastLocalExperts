# `/fle/status`

`GET /fle/status` returns one JSON object (schema 1) for FLE Link: what the engine decided at start and live counters.
Unknown or unavailable values are `null`, never left out: every field of the schema is always present. The endpoint only reads process state and counters; it does not
touch the compute path.

```json
{
  "schema": 1,
  "engine": {"name": "fle", "version": null, "commit": "<build info>", "model": "<model name>", "uptime_s": 812},
  "cuda": {"runtime": "13.4", "driver": "580.95.05", "driver_cuda": "13.4", "gpus": [{"index": 0, "name": "...", "vram_mib": 32623}]},
  "switches": {"FLE_MOE_CACHE_BUDGET_MIB": "15500"},
  "profile": {"loaded": true, "path": "...", "checksum_ok": true, "readonly": true, "steps": 121752, "rejected_reason": null, "state": "loaded"},
  "gpu_cache": {"requested_mib": 15500, "fitted_mib": 15500, "inserts": 0, "experts": 1885, "hit": 0.633},
  "store": {"gb": 90, "warm_experts": 9120, "hits": 0, "misses": 0, "evictions": 0, "read_gb": 0.0},
  "draft": {"kind": "draft-dspark", "n_max": 5, "block": null},
  "warnings": [{"code": "prefill_window_gt_8", "level": "warn", "text": "..."}],
  "live": {"since_start": {"decode_tps": 50.6, "prefill_tps": 540.0, "draft_accept": 0.94, "draft_mean_len": 5.1, "cache_hit": 0.633,
                           "store_hits": 0, "store_misses": 0, "store_evictions": 0, "store_read_gb": 0.0, "cpu_experts_per_round": null, "requests": 3},
           "last_request": {"...": "the same fields; store and cache counters are since start only, so null here"}}
}
```

- `engine.version` comes from `FLE_VERSION` if set (the packages do not embed the tag yet), `commit` is the build info string.
- `cuda.runtime` is read from the CUDA backend module (`fle_cuda_versions` via the backend's proc address, works with
  `GGML_BACKEND_DL`); `driver` is the kernel module release from `/proc/driver/nvidia/version` (Linux only, else null);
  `driver_cuda` is the CUDA version the driver supports.
- `switches`: every `FLE_*` environment variable of the process.
- `profile.state`: `loaded`, `not_found` (cold start), `rejected` (with `rejected_reason`, e.g. a checksum or model
  mismatch) or `none` (no `FLE_MOE_PROFILE`).
- `gpu_cache.fitted_mib`: device memory the cache actually took; `experts`: usable slots over all layers.
- Warnings (`code` stays fixed, `text` is English): `inserts0_no_profile`, `cuda_lt_13_4`, `prefill_window_gt_8`.
- `live`: decode and prefill tok/s and DSpark acceptance from finished requests; `draft_mean_len` = 1 + accepted per verify step.

Not yet filled (null): `draft.block`, `cpu_experts_per_round`.

## Start errors

A failed start ends the process before the HTTP server answers. The server then writes, once, one line to stderr:

```
[fle-status-error] {"error":{"code":"fit_budget_too_large","max_mib":14200,"text":"..."}}
```

- `fit_budget_too_large`: with `FLE_MOE_CACHE_BUDGET_FIT=1` the requested `FLE_MOE_CACHE_BUDGET_MIB` did not fit next to
  model and context; `max_mib` is the largest budget that would have left the `-fit-target` margin on device 0.
- `model_load_failed`: any other load or context failure (no `max_mib`); the log above has the cause.

Library side: `llama_fle_status_json()` (include/llama.h, implemented in src/llama-moecache.cpp) returns the profile,
GPU cache and store parts; the server (tools/server/server-context.cpp, `get_fle_status`) adds the rest.
