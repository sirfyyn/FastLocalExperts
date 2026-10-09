# Ideas and code adopted from FreeToken

FLE's hybrid pager path (GPU-resident expert pool plus pinned host backing store, with UVA-readable fallback) and the
bandwidth rule that decides when staging an expert to the GPU pays off were re-implemented following the design in
[FreeToken](https://github.com/FlashML-org/FreeToken) (commit `4b94bdc`, Apache License 2.0; see also
arXiv:2608.16157). The implementation lives in `ggml/src/ggml-cuda/fle/pager_buffer.cu` (with `pager_wire.h`).

FreeToken is Copyright 2026 FreeToken Authors and licensed under the Apache License, Version 2.0. A copy of the license
is provided in [`licenses/LICENSE-Apache-2.0`](licenses/LICENSE-Apache-2.0) and at
http://www.apache.org/licenses/LICENSE-2.0.

## Statement of changes (Apache-2.0, section 4(b))

The FLE code is an independent re-implementation inside ggml's CUDA backend: it works on llama.cpp's stacked expert
tensors (custom backend buffer, per-call staging with LRU eviction, id remapping for `mul_mat_id`), whereas FreeToken
targets a different runtime. Where algorithms, constants or structure follow FreeToken, they were modified for this
integration. Files that carry such adaptations state so in their header comments.
