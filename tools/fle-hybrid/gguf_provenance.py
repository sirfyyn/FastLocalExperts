#!/usr/bin/env python3
"""gguf_provenance.py - provenance per MoE layer of a hybrid GGUF: compares the first 64 KiB of the expert tensors
(gate/up/down) of every layer with the same tensors in the source files. Reads only a few MB per file.

  gguf_provenance.py HYBRID.gguf NAME=first-shard.gguf [NAME=...]

Further shards (-0000k-of-0000n) of a source are read automatically. Output: per source combination the number of
layers and their indices; '?' marks a layer that matches no source.
"""
import glob
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

_GGUF_PY = os.environ.get("GGUF_PY", str(Path(__file__).resolve().parents[2] / "gguf-py"))
if os.path.isdir(_GGUF_PY):
    sys.path.insert(0, _GGUF_PY)
from gguf import GGUFReader  # noqa: E402

PATTERN = re.compile(r"^blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$")
N = 65536


def tensors(path):
    shards = sorted(glob.glob(re.sub(r"-\d{5}-of-(\d{5})\.gguf$", r"-*-of-\1.gguf", path))) or [path]
    out = {}
    for shard in shards:
        for t in GGUFReader(shard).tensors:
            if PATTERN.match(t.name):
                out[t.name] = bytes(t.data.reshape(-1).view("uint8")[:N])
    return out


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    hybrid = tensors(sys.argv[1])
    sources = {}
    for arg in sys.argv[2:]:
        name, path = arg.split("=", 1)
        sources[name] = tensors(path)
    per_layer = defaultdict(set)
    for tname, head in hybrid.items():
        layer = int(PATTERN.match(tname).group(1))
        hits = [s for s, ts in sources.items() if ts.get(tname) == head]
        per_layer[layer].add("+".join(hits) or "?")
    groups = defaultdict(list)
    for layer in sorted(per_layer):
        groups["/".join(sorted(per_layer[layer]))].append(layer)
    for origin, layers in sorted(groups.items()):
        print(f"{origin:10s} {len(layers):2d} layers: {','.join(map(str, layers))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
