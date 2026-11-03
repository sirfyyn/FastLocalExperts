# Adoptions from Strata

Parts of FLE adopt ideas or code from Strata v0.1.39 and v0.2.9 (repository state c62d5e3)
(https://github.com/Niko1221/Strata), released under the MIT License:

    MIT License
    Copyright (c) 2026 Niko1221 and the Strata contributors

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

Every commit that adopts Strata code names the source file and lines in its commit message and in the header of the change.

Adopted from Strata 0.2.9 (c62d5e3):
- `ggml/src/ggml-cpu/arch/x86/quants.c`: the AVX-512 VNNI paths for IQ1_M (after `rows_m`), IQ2_XXS and IQ2_XS (after
  `FmtX<16>`, `FmtX<17>` and `row_dot_x`) in `src/kernels/cpu/iq_avx512.cpp`; the IQ1_S path follows the same scheme.
- `src/llama-expertstore.cpp`, `ggml/src/ggml-cpu/ggml-cpu.c`: the order of the asynchronous expert store (resident experts
  first, each missing one when its read lands) after `cpu_experts` in `src/dsv41/ds4_model.cu`; own code.
