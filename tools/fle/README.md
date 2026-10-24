# `fle` — setup helper for this checkout

`./fle` runs the Python helper in `tools/fle` against the `llama-server` next to this checkout. It reads the model
header, finds the engine, sizes the GPU expert cache, writes a start script and can start the server. Every command
accepts a model file, the first shard, or the folder that contains the shards.

It is a plain Python 3 program with no extra packages and does not change the engine.

```bash
./fle inspect  /models/Maya-S              # architecture, parts, expert bytes, KV cache
./fle verify   /models/Maya-S              # parts, tensor count, offsets, quant types
./fle doctor   /models/Maya-S              # model + engine + RAM/VRAM + the plan
./fle plan     /models/Maya-S --profile auto
./fle serve    /models/Maya-S --profile auto
./fle bench    --url http://127.0.0.1:18425
./fle report   /models/Maya-S --out fle-report.json
```

`serve` runs the same checks as `doctor` first. `--dry-run` prints the command without starting it; `--force` skips the
preflight when you know better.

## Profiles

| Profile | What it does |
|---|---|
| `off` | No `FLE_*` variables and no FLE-only flags. A reproducible upstream baseline. |
| `auto` | Sizes the GPU expert cache and the context from free VRAM and RAM. The default. |
| `balanced` | Leaves more free VRAM for other work; a little slower. |
| `max` | Uses the memory for the expert cache and a larger context; best on an idle GPU. |
| `nvme` | For a model larger than RAM: mmap, `--no-warmup`, a small cache and 16k context. |
| `debug` | `auto` plus FLE timing lines in the engine log. |

The profile only changes the values the helper derives. Flags passed on the command line always win. The profile names
and the config keys match FLE Studio, so a plan can move between the terminal and the app.

## Speculation

`--spec` controls drafting independently of the profile:

| Value | Behaviour |
|---|---|
| `auto` | Uses MTP only when the model has both MTP sidecars and the profile is not `off` or `nvme`. Never turns on ngram by itself. |
| `none` | No MTP and no ngram. |
| `mtp` | Requires the MTP head and vocabulary; fails with a hint when they are missing. |
| `ngram-mod` / `ngram-simple` | Adds `--spec-type` and the mode; needs an engine that supports it. Prompt-dependent, often strong on code and repetitive text. |

`--no-mtp` still works and only disables MTP. The `off` profile is the reproducibility baseline, so it rejects
`--spec ngram-mod`; use `--profile auto` for speculation. `plan` prints the effective speculation, so `auto` shows
`none` when no usable drafter is present.

## Context and slots

`plan` shows total context, per-slot context, slot count and whether unified KV is on. If you pass
`--arg=--parallel N` through to llama-server, also pass `--arg=--kv-unified`; otherwise every slot gets about
`context / N` tokens. The helper warns about this instead of silently changing the command.

## Config and precedence

Settings live in `~/.config/fle/config.toml` (or `$XDG_CONFIG_HOME/fle/config.toml`). The order is:

**CLI flag → environment variable → config file → profile default**

`FLE_CONFIG` points at another file, `FLE_PROFILE` overrides the profile, `FLE_SPEC` overrides speculation,
`FLE_ENGINE` overrides the engine and `FLE_MODEL` supplies a model when none is given. `FLE_PROFILE=off` is the quick
reproducibility switch.

```bash
./fle config init
./fle config show
./fle config set profile balanced
./fle config set spec ngram-mod
./fle config set engine /path/to/build/bin/llama-server
./fle config set env.FLE_MC_OVERLAP 2
./fle config set pinned_load off
./fle config --json show
```

Keys that are not in the file's schema are ignored, so a config from a newer helper cannot break an older one. `fle
config unset KEY` removes an override.

## Commands

- **`inspect`** — header only, no weights; works while a split download is still running.
- **`verify`** — checks the split count, `split.no`, tensor count, offsets, overlaps and quant ids. Missing or incomplete
  parts are reported separately, with the bytes present so far.
- **`doctor`** — one read-only report: model, parts, engine, GPU, RAM, disk, MTP and the plan. `--strict` also exits
  non-zero on warnings.
- **`engines`** — lists the `llama-server` builds the helper can find, with FLE/upstream flavor, backend and whether
  they support `--spec-type`. Useful before `--fallback` or when comparing builds; `--all` also shows missing
  candidate paths.
- **`plan`** — prints the environment, the exact command, the cache calculation and notes. `--cache-mib` sets the
  budget by hand; `0` means automatic.
- **`serve`** — starts `llama-server` with the plan and forwards its output. `--dry-run` only prints. `--fallback`
  retries with an installed upstream llama.cpp and profile `off` if the FLE plan cannot be built; set
  `FLE_UPSTREAM_ENGINE=/path/to/upstream/llama-server` if it is not found automatically. This is plan-time fallback
  only, not a rescue from a model that crashes after loading.
- **`bench`** — measures prompt and decode tokens/s through the OpenAI-compatible `/v1` endpoint. It starts no server.
  `--prompt-file` reads a longer prompt, `--output` writes the JSON result, `--compare URL` runs the same prompt
  against a second running server/model, and the report shows `finish_reason`, draft acceptance and a warning when a
  run hit the token limit.
- **`report`** — a redacted JSON (or `--zip` with `plan.sh`) for a bug report. Secrets are not included.
- **`config`** — read and write the user config (`path`, `init`, `show`, `set`, `unset`, `edit`).

`--json` is available on the read-only commands for scripts. `--debug` prints a traceback for an unexpected failure.

## Notes for large models

- A split GGUF keeps tensor offsets relative to each shard. `verify` checks them per shard; a shard that is still
  downloading shows up as incomplete rather than as a corrupt file.
- If the experts are larger than RAM, use `--profile nvme`; experts stay memory-mapped and the GPU cache stays small.
  Keep the model on fast local NVMe, not on a network mount.
- `FLE_MOE_PROFILE` is written under `~/.config/fle/profiles/`. The first run records which experts are used; later runs
  start warm.
- The helper never edits `rhea.service` or another running service. Stop the service before `serve` if it already holds
  the GPU.

## Layout

```text
fle                     # the launcher
tools/fle/cli.py        # argument parsing and commands
tools/fle/gguf.py       # header and split parser, verification
tools/fle/hardware.py   # GPU/RAM/engine detection
tools/fle/plan.py       # profiles, cache budget, command
tools/fle/doctor.py     # the checks
tools/fle/report.py     # redacted JSON/zip
tools/fle/bench.py      # /v1 throughput
tools/fle/config.py     # TOML config and precedence
```
