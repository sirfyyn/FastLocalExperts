# FLE - Fast Local Experts

This page moved: what FLE is, its results and how to build and run it are now in the main [README.md](README.md).
Every FLE environment variable and flag is in [docs/fle/ENV.md](docs/fle/ENV.md), and the upstream llama.cpp README is
kept in [README-llama.cpp.md](README-llama.cpp.md).

`./fle` is a small setup helper for a checkout: it reads the model header, finds the engine, sizes the GPU expert cache,
explains the plan and can start the server. Commands, profiles and the tests are in [tools/fle/README.md](tools/fle/README.md).
