# `fle` in Rust

The terminal program for FLE: **one file, one command**. `fle` (no arguments) sets everything up on first run: it looks at
the machine, installs the engine that fits (checked against the SHA-256 GitHub publishes), downloads a model that fits
(resumable, checked), chooses the start settings and opens a chat. Nothing to type, no `FLE_*` variable to set.

```bash
fle            # the terminal interface
fle doctor     # what this machine can do, and which engine fits it
fle models     # known models and whether each fits (before anything is downloaded)
fle run MODEL  # load a model and serve it (OpenAI-compatible API) until Ctrl+C
fle ask MODEL "question"
```

Install (one file, no Python, no `uv`, no `git`, no `sudo`):

```bash
curl -fsSL https://raw.githubusercontent.com/sirfyyn/FastLocalExperts/main/rust/install.sh | sh        # macOS, Linux
irm https://raw.githubusercontent.com/sirfyyn/FastLocalExperts/main/rust/install.ps1 | iex              # Windows
```

The installer asks: Terminal, Browser (FLE Studio) or both.

## How this relates to `tools/fle` (the Python helper)

`./fle` in the checkout root (`tools/fle`, Python) is the setup helper for a source checkout: `inspect`, `verify`,
`doctor`, `plan`, `serve`, `bench`, `fetch-sidecars`, `report`, `config`, with the profiles `off/auto/balanced/max/nvme/debug`
and the precedence CLI flag, environment, config file, profile default. The Rust program is its successor, not a second
tool:

- `crates/fle-core` is the library (hardware, engines, catalog, download, plan, server, chat). A Tauri app and FLE Studio's
  `/api` will sit on it.
- The Python helper and its tests (`tools/fle/tests`, 14 files) are the **specification**. Command by command the Rust
  program takes over, and the same cases run against it. Until a command is ported, `./fle` stays the way to use it.

Status of the port: see `docs/fle` plan (workstream A). Ported so far: hardware detection, engine choice, download,
start plan (simplified), server supervision, chat, terminal interface.

## Build and test

```bash
cd rust && cargo test --all && cargo build --release -p fle     # target/release/fle
```

Releases: tag `cli-vX.Y.Z` runs `.github/workflows/fle-cli-release.yml` (macOS arm64/x64, Linux x64, Windows x64).
