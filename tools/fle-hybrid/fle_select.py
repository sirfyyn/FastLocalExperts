#!/usr/bin/env python3
"""fle_select.py - choose, per MoE layer, which donor quantization supplies the expert tensors so that the predicted
error is smallest for a fixed target file size (used for FLE-Q3 and FLE-Q2).

Whole expert layers (gate, up and down together) come from one of several donor files; all other tensors and the
metadata come from the base. The prediction per layer and donor is the relative output error of the MoE block,
err = |y_Q - y_BF16|^2 / |y_BF16|^2 on captured layer inputs, measured for every donor with the same capture and the
same BF16 reference, so the values are comparable. Selection is a multiple-choice knapsack (dynamic programming over
the expert bytes in 16 MB steps, keeping the Pareto front). The prediction is only a selection aid: measure the built
file against BF16 (KLD) afterwards.

  fle_select.py --name Q2 --target-gb 72 --base B --source B=IQ2_XS.gguf --source K=Q2_K_XL.gguf \\
                --err B=err-iq2xs.json --err K=err-q2kxl.json --out q2.json [--build Q2.gguf]

--err NAME=FILE[:KEY] reads {"layers": [{"layer": L, "err": {KEY: {"all": E}}}, ...]}; KEY defaults to NAME.
--build calls gguf_splice.py from this directory.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gguf_splice as gs  # noqa: E402

STEP = 16_000_000
PATTERN = r"^blk\.\d+\.ffn_(gate|up|down)_exps\.weight$"


def file_size(first: str) -> int:
    if "-00001-of-" in first:
        return sum(os.path.getsize(p) for p in glob.glob(first.replace("-00001-of-", "-*-of-")))
    return os.path.getsize(first)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--name", required=True)
    ap.add_argument("--target-gb", type=float, required=True)
    ap.add_argument("--base", required=True, help="name of the base source")
    ap.add_argument("--source", action="append", required=True, help="NAME=first GGUF file")
    ap.add_argument("--err", action="append", required=True, help="NAME=FILE[:KEY] per-layer error prediction")
    ap.add_argument("--out", required=True)
    ap.add_argument("--build", default="", help="also build the file with gguf_splice.py")
    a = ap.parse_args()

    path = dict(s.split("=", 1) for s in a.source)
    size, n_layer = {}, 0
    for name, p in path.items():
        _, t, order = gs.load(p)
        n_layer = 1 + max(int(n.split(".")[1]) for n in order if n.startswith("blk."))
        size[name] = {l: sum(t[f"blk.{l}.ffn_{g}_exps.weight"].n_bytes for g in ("gate", "up", "down"))
                      for l in range(n_layer)}
    err = {}
    for spec in a.err:
        name, _, rest = spec.partition("=")
        fname, _, key = rest.partition(":")
        with open(fname) as f:
            rows = json.load(f)["layers"]
        err[name] = {r["layer"]: r["err"][key or name]["all"] for r in rows}
    missing = [n for n in path if n not in err]
    if missing:
        sys.exit(f"no prediction for {missing}")
    layers = [l for l in range(n_layer) if all(l in err[n] for n in path)]
    if len(layers) != n_layer:
        sys.exit(f"prediction incomplete: {len(layers)} of {n_layer} layers")

    rest_bytes = file_size(path[a.base]) - sum(size[a.base].values())
    budget = int((a.target_gb * 1e9 - rest_bytes) // STEP)
    # state: used expert bytes (steps) -> (sum of err, choice)
    states = {0: (0.0, {})}
    for l in layers:
        nxt = {}
        for b0, (e0, choice) in states.items():
            for n in path:
                b1 = b0 + -(-size[n][l] // STEP)
                if b1 > budget:
                    continue
                e1 = e0 + err[n][l]
                if b1 not in nxt or e1 < nxt[b1][0]:
                    nxt[b1] = (e1, {**choice, l: n})
        # keep only states that are not worse with fewer bytes (Pareto front)
        front, best = {}, float("inf")
        for b in sorted(nxt):
            if nxt[b][0] < best:
                front[b], best = nxt[b], nxt[b][0]
        states = front
        if not states:
            sys.exit(f"target {a.target_gb} GB not reachable (non-expert rest {rest_bytes / 1e9:.1f} GB)")
    total_err, choice = min(states.values(), key=lambda v: v[0])
    total = rest_bytes + sum(size[choice[l]][l] for l in layers)
    base_err = sum(err[a.base][l] for l in layers)
    by_source = {n: sorted(l for l in layers if choice[l] == n) for n in path}
    result = {"name": a.name, "target_gb": a.target_gb, "size_gb": round(total / 1e9, 2), "base": a.base,
              "sources": path, "layers": by_source, "sum_err": total_err, "sum_err_base": base_err}
    print(f"{a.name}: {total / 1e9:.2f} GB (target {a.target_gb}), sum err {total_err:.4f} "
          f"(base {a.base} alone {base_err:.4f})")
    for n, ls in by_source.items():
        print(f"   {n}: {len(ls)} layers {ls}")
    with open(a.out, "w") as f:
        json.dump(result, f, indent=1)
    if a.build:
        cmd = [sys.executable, str(HERE / "gguf_splice.py"), "--base", path[a.base], "--out", a.build]
        for n, ls in by_source.items():
            if n != a.base and ls:
                cmd += ["--donor", f"{n}={path[n]}", "--take", f"{n}:{PATTERN}@{','.join(map(str, ls))}"]
        if len(cmd) == 7:
            sys.exit("no donor layer chosen: the hybrid would be the base itself")
        print("build:", " ".join(cmd), flush=True)
        return subprocess.run(cmd).returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
