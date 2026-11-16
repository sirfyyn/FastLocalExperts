# Adoptions from Infernix

Two optional CUDA paths in FLE v1.3.0 re-implement techniques from Infernix
(https://github.com/wallawalla47/infernix, state a3edb45), released under the Apache License 2.0
(full text: [licenses/LICENSE-Apache-2.0](licenses/LICENSE-Apache-2.0)):

- `FLE_GRAPH_FORMEN` (`ggml/src/ggml-cuda/ggml-cuda.cu`): one CUDA graph per batch shape.
- `FLE_IQ4_MMA` (`ggml/src/ggml-cuda/fle-iq4-mma.cu`): a Tensor Core kernel for verify rounds over 9 to 16 columns
  that keeps quantized weights packed, expands them to F16 with byte permutes and removes the offsets through
  per-sub-block activation sums (`mma.sync m16n8k16`, cp.async double buffering, deterministic split-K).

No Infernix source code was copied: both paths were written for FLE after reading Infernix's design and source. A
line-by-line comparison against Infernix a3edb45 finds no shared line apart from the standard PTX `mma.sync`
instruction. Infernix's licence file names no copyright holder; the project is by wallawalla47. Infernix itself began as a fork of
NInfer (https://github.com/Neroued/ninfer) by Neroued. Both paths are off by default.
