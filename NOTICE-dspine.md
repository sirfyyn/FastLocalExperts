# DSpine draft model support

The DSpine model type (`src/models/dflash.cpp`, `src/llama-arch.*`, `src/llama-model.h`, `src/llama-quant.cpp`,
`conversion/qwen.py`, `gguf-py/gguf/constants.py`, `gguf-py/gguf/tensor_mapping.py`) comes from the patch
"dflash : add DSpine draft model support" by AjHomelab (commit `d493f84`, 2026-11-08), written for llama.cpp and
contributed under its MIT License. FLE adds its verify-round changes on top.

The DSpine head itself (`tencent/Qwen3.8-27B-DSpine-b16`) is a model by Tencent under the Apache License 2.0 and is not
part of this repository.
