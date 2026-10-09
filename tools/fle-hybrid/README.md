# fle-hybrid: building layer hybrids from existing quantizations

The FLE GGUF files (G3, FLE-Q3, FLE-Q2) are not new quantizations. They are assembled from whole MoE expert layers of
public quantizations of the same model; tensor bytes are copied, nothing is re-quantized. Three small tools do this:

| Tool | What it does |
|---|---|
| `gguf_splice.py` | Builds a GGUF from a base file and one or more donor files: `--take 'DONOR:REGEX@layers'` selects which tensors come from which donor. Metadata and all other tensors come from the base. Checks type, shape and the first and last 64 KiB of every tensor against its source after writing. |
| `fle_select.py` | Chooses per MoE layer the donor whose expert tensors give the smallest predicted error for a fixed target size (multiple-choice knapsack). Optionally calls `gguf_splice.py` to build the file. |
| `gguf_provenance.py` | Checks a finished hybrid: for every MoE layer, which source file its expert tensors were copied from. Reads only the first 64 KiB of each expert tensor. |

All three use `gguf-py` from this repository (`../../gguf-py`) if it is there, else an installed `gguf` package
(`GGUF_PY` overrides the path). Requirements: Python 3.10+, numpy.

## Per-layer error prediction

`fle_select.py` needs, for each donor, a per-layer error estimate in a JSON file:

```json
{"layers": [{"layer": 0, "err": {"B": {"all": 0.0123}}}, ...]}
```

`err` is the relative output error of the MoE block when only that layer's expert tensors come from the donor,
`|y_Q - y_BF16|^2 / |y_BF16|^2`, on captured layer inputs (4,096 tokens in our runs), with the same capture and the same
BF16 reference for every donor so the values are comparable. The method and the numbers are described in the FLE
report; the prediction is only a selection aid, and every built file was measured against BF16 (KLD) afterwards.

## Example

```sh
# FLE-Q2: target 72 GB, base ISTA IQ2_XS, donors ISTA IQ3_XXS and Unsloth UD-Q2_K_XL
python3 fle_select.py --name Q2 --target-gb 72 --base B \
    --source B=ista-IQ2_XS-00001-of-00002.gguf --source X=ista-IQ3_XXS-00001-of-00002.gguf \
    --source K=unsloth-UD-Q2_K_XL-00001-of-00003.gguf \
    --err B=err-iq2xs.json --err X=err-iq3xxs.json --err K=err-q2kxl.json \
    --out q2-selection.json --build FLE-Q2.gguf

# check where every expert layer came from
python3 gguf_provenance.py FLE-Q2.gguf B=ista-IQ2_XS-00001-of-00002.gguf \
    X=ista-IQ3_XXS-00001-of-00002.gguf K=unsloth-UD-Q2_K_XL-00001-of-00003.gguf
```

The donors' licences apply to the hybrids; see the NOTICE of the model repository.
