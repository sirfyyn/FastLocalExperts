# `fle` — setup helper for this checkout

`./fle` runs the Python helper in `tools/fle` against the `llama-server` next to this checkout. It reads the model
header, finds the engine, sizes the GPU expert cache, writes a start script and can start the server. Every command
accepts a model file, the first shard, or the folder that contains the shards.

It is a plain Python 3 program with no extra packages and does not change the engine.

```bash
./fle inspect  /models/MyModel              # architecture, parts, expert bytes, KV cache
./fle verify   /models/MyModel              # parts, tensor count, offsets, quant types
./fle doctor   /models/MyModel              # model + engine + RAM/VRAM + the plan
./fle plan     /models/MyModel --profile auto
./fle serve    /models/MyModel --profile auto
./fle bench    --url http://127.0.0.1:18425
./fle fetch-sidecars /models/Qwen3.8-Flash-Next-FLE-IQ3_M --dry-run
./fle report   /models/MyModel --out fle-report.json
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
| `mtp` | Requires the MTP head and vocabulary; fails with a hint (`fle fetch-sidecars MODEL`) when they are missing. |
| `ngram-mod` / `ngram-simple` | Adds `--spec-type` and the mode; needs an engine that supports it. Prompt-dependent, often strong on code and repetitive text. |

`--no-mtp` still works and only disables MTP. `--spec mtp` together with a switched-off MTP (`--no-mtp`, `mtp = false` in
the config, or `FLE_MTP=0`) is a contradiction and fails, instead of being half obeyed. The `off` profile is the reproducibility baseline, so it rejects
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
  non-zero on warnings. When another profile would resolve a warning (experts larger than RAM -> `nvme`, little free
  VRAM -> `balanced`, no GPU or no FLE engine -> `off`) it ends with the ready command, for example
  `fle plan /models/MyModel --profile nvme`; `--json` has the same as `suggestions` (a list, empty when nothing helps).
- **`engines`** — lists the `llama-server` builds the helper can find, with FLE/upstream flavor, backend and whether
  they support `--spec-type`. Useful before `--fallback` or when comparing builds; `--all` also shows missing
  candidate paths.
- **`plan`** — prints the environment, the exact command, the cache calculation and notes. `--cache-mib` sets the
  budget by hand; `0` means automatic.
- **`serve`** — starts `llama-server` with the plan and forwards its output. `--dry-run` only prints. `--fallback`
  retries with an installed upstream llama.cpp and profile `off` in two cases: the FLE plan cannot be built, or the
  started server exits with a non-zero code before `/health` reports ready. The second case runs the server as a child
  process for the start-up phase, prints one hint line and restarts once; a crash after "ready" is not restarted.
  `--ready-timeout SEC` (default 3600) is how long the start is watched. An explicit `--spec ngram-*` or `mtp` is
  never dropped silently: with it there is no fallback, only the error. Set
  `FLE_UPSTREAM_ENGINE=/path/to/upstream/llama-server` if no upstream build is found automatically. Without
  `--fallback`, `serve` replaces itself with the server (`exec`) as before.
- **`bench`** — measures prompt and decode tokens/s through the OpenAI-compatible `/v1` endpoint. It starts no server.
  `--prompt-file` reads a longer prompt, `--output` writes the JSON result, `--compare URL` runs the same prompt
  against a second running server/model, and the report shows `finish_reason`, draft acceptance and a warning when a
  run hit the token limit. `--baseline FILE` compares the median tok/s with the JSON from an earlier `--output` and
  prints the change in percent; `--max-regression PCT` (default 10) sets the allowed drop, and a bigger drop exits with
  code 2 (a CI gate). No comparison is made, with a note, when either side has truncated runs, the model differs, or one
  URL is local and the other is not.
- **`fetch-sidecars MODEL`** — downloads the missing MTP head and vocabulary next to the model, from the Hugging Face
  repo that FLE Studio's catalog also uses. Only models with a known source are covered (today the Qwen3.8 Flash Next
  FLE files); for any other model it says `no known source for this model` and exits 1 rather than guessing a URL.
  `--dry-run` lists the files, sizes and targets. Files are written under a temporary name, checked for size and SHA-256
  and then renamed; existing files are never replaced without `--force`. `HF_TOKEN` is sent if set.
- **`report`** — a redacted JSON (or `--zip` with `plan.sh`) for a bug report. Secrets are not included.
- **`config`** — read and write the user config (`path`, `init`, `show`, `set`, `unset`, `edit`).

`--json` is available on the read-only commands for scripts. `--debug` prints a traceback for an unexpected failure.

## Notes for large models

- A split GGUF keeps tensor offsets relative to each shard. `verify` checks them per shard; a shard that is still
  downloading shows up as incomplete rather than as a corrupt file.
- If the experts are larger than RAM, use `--profile nvme`; experts stay memory-mapped and the GPU cache stays small.
  Keep the model on fast local NVMe, not on a network mount.
- `FLE_MOE_PROFILE` is written under `~/.config/fle/profiles/<model>.bin`. The name is the model's stable alias; when the
  GGUF name is only a quant type like `Bf16`, a short hash of the model path is appended so two models never share a
  file. The first run records which experts are used; later runs start warm. Dense models (no experts) get no
  `FLE_MOE_*` settings and no MoE flags.
- The helper never edits a running service. Stop the service before `serve` if it already holds
  the GPU.

## Tests and the shared contract

```bash
python3 -m pytest tools/fle/tests        # needs pytest; the helper itself needs nothing
```

The tests use tiny synthetic GGUF files (`tests/gguf_fixture.py`), a fake machine and fake engines, a local fake `/v1`
server for `bench` and fake processes for `serve`. They start no server and need no GPU or network.

`tests/spec_cases.json` is the contract for the speculation rules and is shared with FLE Studio (the same file lives in
its repo). `tests/test_spec_contract.py` runs every case against `plan.build`. Do not edit the file here alone: when a
rule changes, change both copies and both implementations together. The error cases check that the lower-case message
plus hint contains `sidecar`, `contradict`, `spec-type` or `unknown`.

## Layout

```text
fle                     # the launcher
tools/fle/cli.py        # argument parsing and commands
tools/fle/gguf.py       # header and split parser, verification
tools/fle/hardware.py   # GPU/RAM/engine detection
tools/fle/plan.py       # profiles, cache budget, command
tools/fle/doctor.py     # the checks
tools/fle/report.py     # redacted JSON/zip
tools/fle/bench.py      # /v1 throughput, baseline comparison
tools/fle/serve.py      # start-up watching for serve --fallback
tools/fle/sidecars.py   # known MTP sources and the verified download
tools/fle/tests/        # pytest suite, synthetic GGUF writer, spec_cases.json
tools/fle/config.py     # TOML config and precedence
```
