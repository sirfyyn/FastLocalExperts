# Code taken from ik_llama.cpp

FLE contains code from [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) by Iwan Kawrakow
and contributors, released under the MIT License:

    MIT License

    Copyright (c) 2023-2024 The ggml authors (https://github.com/ggml-org/ggml/blob/master/AUTHORS)
    Copyright (c) 2023-2024 The llama.cpp authors

    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
    documentation files (the "Software"), to deal in the Software without restriction, including without limitation
    the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
    permit persons to whom the Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all copies or substantial portions of
    the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
    THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
    CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
    IN THE SOFTWARE.

The copyright lines above are those of the ik_llama.cpp license file; the KT quantization code itself was written by
Iwan Kawrakow. The authoritative license text is https://github.com/ikawrakow/ik_llama.cpp/blob/main/LICENSE.

## What was taken

The trellis-quantized (QTIP-style) `IQ2_KT`, `IQ3_KT` and `IQ4_KT` tensor types, their block layouts and the
decoding kernels were taken from ik_llama.cpp essentially verbatim, and adapted only as far as needed to fit the
surrounding code (type numbers, per-row metadata via `row_meta_size`, FLE kernel wrappers):

| File | Content |
|------|---------|
| `ggml/include/ggml.h` | `GGML_TYPE_IQ2_KT` / `IQ3_KT` / `IQ4_KT`, `row_meta_size` in the type traits |
| `ggml/src/ggml-common.h` | `block_iq2_kt`, `block_iq3_kt`, `block_iq4_kt` (per-row float scale precedes the blocks) |
| `ggml/src/ggml-quants.c`, `ggml/src/ggml-quants.h` | scalar dequantization (`dequantize_row_iq{2,3,4}_kt`); the `quantize_row_*_kt_ref` functions are stubs that abort (dequantization only) |
| `ggml/src/ggml-cpu/quants.c`, `ggml/src/ggml-cpu/quants.h` | `ggml_vec_dot_iq{2,3,4}_kt_q8_K` |
| `ggml/src/ggml-cuda/convert.cu` | CUDA dequantization (`fle_dequantize_block_iq{2,3,4}_kt`) taken from ik_llama.cpp, and the matrix-vector kernels `fle_mmv_iq{2,3,4}_kt_kernel` that mirror them |
| `ggml/src/ggml.c`, `ggml/src/ggml-cpu/ggml-cpu.c` | type-trait table entries |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | dispatch hook `ggml_cuda_mul_mat_fle_kt` and exclusion of KT types from `mul_mat_id` |
| `ggml/src/gguf.cpp` | mapping of ik_llama.cpp type numbers 153/154/155 to FLE's 43/44/45 when reading GGUF files |

The type numbers of the KT types differ from ik_llama.cpp; GGUF files written by ik_llama.cpp are remapped on load.
