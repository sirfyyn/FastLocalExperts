#!/usr/bin/env python3
"""reap_prune.py - prune the routed experts of a GGUF MoE model to the K best per layer (REAP-style).

The ranking comes from the table of llama-fle-reap (layer, expert, count, sum_gate, sum_saliency):
  --score reap   sum_saliency / count   (REAP: mean router-weighted expert output norm over the routed tokens)
  --score freq   count                  (how often the expert was routed)
  --score mass   sum_gate               (summed router weight)
  --score man    sum_norm / count       (gate-free mean expert output norm, MAN, arXiv 2606.15716)
The routing mass kept, printed and stored in the manifest, describes a selection; it does not rank two selections
(each one wins on the corpus that produced it).
Experts that were never routed score 0 and go first. Per layer the K best experts are kept in their original order;
ffn_gate_exps, ffn_up_exps, ffn_down_exps, ffn_gate_up_exps and their biases/scales lose the other expert slices, the
router ffn_gate_inp (and exp_probs_b) loses the matching rows, and <arch>.expert_count becomes K. Nothing is
re-quantized: every kept expert slice is copied byte for byte. The kept ids go to a JSON manifest.

Examples:
  reap_prune.py --model G3.gguf --table reap.tsv --keep 384 --out G3-REAP384.gguf
  reap_prune.py --model G3.gguf --table reap.tsv --keep 384 --out x.gguf --dry-run   # only the manifest and sizes

gguf-py: as in gguf_splice.py (../../gguf-py, else installed `gguf`; GGUF_PY overrides).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

_GGUF_PY = os.environ.get("GGUF_PY", str(Path(__file__).resolve().parents[2] / "gguf-py"))
if os.path.isdir(_GGUF_PY):
    sys.path.insert(0, _GGUF_PY)
import gguf  # noqa: E402
from gguf import GGUFReader, GGUFWriter  # noqa: E402

LAYER = re.compile(r"^blk\.(\d+)\.")
# tensors whose slowest axis (data axis 0) is the routed expert
PER_EXPERT = re.compile(r"\.(ffn_(gate|up|down|gate_up)_exps(\.(weight|bias|scale))?|ffn_(gate|up|down)_exps_s.*|"
                        r"ffn_gate_inp\.weight|exp_probs_b(\.bias)?)$")


def load_table(path: str) -> dict[int, dict[int, tuple[int, float, float, float]]]:
    table: dict[int, dict[int, tuple[int, float, float, float]]] = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            table.setdefault(int(row["layer"]), {})[int(row["expert"])] = (
                int(row["count"]), float(row["sum_gate"]), float(row["sum_saliency"]),
                float(row.get("sum_norm") or 0.0))
    return table


def score(entry: tuple[int, float, float, float], kind: str) -> float:
    count, gate, sal, norm = entry
    if kind == "freq":
        return float(count)
    if kind == "mass":
        return gate
    if kind == "man":
        return norm / count if count else 0.0
    return sal / count if count else 0.0


def select(table, layers: list[int], n_expert: int, keep: int, kind: str) -> dict[int, list[int]]:
    chosen: dict[int, list[int]] = {}
    for il in layers:
        stats = table.get(il)
        if stats is None:
            raise ValueError(f"layer {il} missing in the table")
        ranked = sorted(range(n_expert), key=lambda e: (score(stats.get(e, (0, 0.0, 0.0, 0.0)), kind), -e), reverse=True)
        chosen[il] = sorted(ranked[:keep])
    return chosen


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--table", required=True, help="TSV from llama-fle-reap")
    ap.add_argument("--keep", type=int, required=True, help="experts kept per layer")
    ap.add_argument("--score", choices=["reap", "freq", "mass", "man"], default="reap")
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", help="default: OUT with .manifest.json")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    reader = GGUFReader(args.model, "r")
    arch = reader.get_field(gguf.Keys.General.ARCHITECTURE).contents()
    key_count = f"{arch}.expert_count"
    n_expert = int(reader.get_field(key_count).contents())
    if not 0 < args.keep < n_expert:
        raise SystemExit(f"--keep must lie between 1 and {n_expert - 1}")
    layers = sorted({int(m.group(1)) for t in reader.tensors if (m := LAYER.match(t.name)) and PER_EXPERT.search(t.name)})
    table = load_table(args.table)
    chosen = select(table, layers, n_expert, args.keep, args.score)

    kept_mass = []
    for il in layers:
        tot = sum(v[1] for v in table[il].values()) or 1.0
        kept_mass.append(sum(table[il].get(e, (0, 0.0, 0.0, 0.0))[1] for e in chosen[il]) / tot)
    manifest = {"model": os.path.basename(args.model), "table": os.path.basename(args.table), "score": args.score,
                "n_expert": n_expert, "keep": args.keep,
                "routing_mass_kept": {"mean": sum(kept_mass) / len(kept_mass), "min": min(kept_mass),
                                      "min_layer": layers[kept_mass.index(min(kept_mass))]},
                "layers": {str(il): chosen[il] for il in layers}}
    man_path = args.manifest or re.sub(r"\.gguf$", "", args.out) + ".manifest.json"
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)

    # (tensor, kept expert ids or None, output shape, output bytes); slices are only materialized one tensor at a time
    # while writing and checking, so RAM stays at about one expert tensor
    plan = []
    total = 0
    for t in reader.tensors:
        m = LAYER.match(t.name)
        if m and PER_EXPERT.search(t.name):
            if t.data.shape[0] != n_expert:
                raise ValueError(f"{t.name}: expert axis {t.data.shape[0]} != {n_expert}")
            ids = chosen[int(m.group(1))]
            shape = (len(ids),) + tuple(t.data.shape[1:])
            nbytes = t.data.nbytes // n_expert * len(ids)
        else:
            ids, shape, nbytes = None, tuple(t.data.shape), t.data.nbytes
        plan.append((t, ids, shape, nbytes))
        total += nbytes

    def data_of(t, ids):
        return t.data if ids is None else t.data[ids]
    print(f"{args.model}: {n_expert} -> {args.keep} experts in {len(layers)} layers, score {args.score}; "
          f"routing mass kept mean {manifest['routing_mass_kept']['mean']:.4f}, "
          f"min {manifest['routing_mass_kept']['min']:.4f} (layer {manifest['routing_mass_kept']['min_layer']}); "
          f"{total / 1e9:.2f} GB of tensors", flush=True)
    if args.dry_run:
        return 0

    writer = GGUFWriter(args.out + ".part", arch)
    for field in reader.fields.values():
        if (field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF.")
                or field.name.startswith("split.")):
            continue
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        value = args.keep if field.name == key_count else field.contents()
        writer.add_key_value(field.name, value, val_type, sub_type=sub_type)
    for t, ids, shape, nbytes in plan:
        writer.add_tensor_info(t.name, shape, t.data.dtype, nbytes, t.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    done, t0, next_report = 0, time.time(), 5 * 10**9
    for t, ids, shape, nbytes in plan:
        writer.write_tensor_data(data_of(t, ids), tensor_endianess=reader.endianess)
        done += nbytes
        if done >= next_report:
            print(f"  written {done / 1e9:.1f} of {total / 1e9:.1f} GB ({done / 1e6 / (time.time() - t0):.0f} MB/s)",
                  flush=True)
            next_report += 5 * 10**9
    writer.close()

    # check: every tensor of the output equals the selected slices of the source, compared in blocks of rows
    def equal(a, b) -> bool:
        if a.shape != b.shape:
            return False
        step = max(1, (256 << 20) // max(1, a[:1].nbytes))
        return all((a[i:i + step] == b[i:i + step]).all() for i in range(0, a.shape[0], step))

    out = GGUFReader(args.out + ".part", "r")
    src = {t.name: (t, ids) for t, ids, _, _ in plan}
    bad = [o.name for o in out.tensors if not equal(o.data, data_of(*src[o.name]))]
    if bad or int(out.get_field(key_count).contents()) != args.keep:
        print(f"CHECK FAILED: {bad[:5]}", flush=True)
        return 1
    os.replace(args.out + ".part", args.out)
    print(f"ok: {args.out} ({os.path.getsize(args.out) / 1e9:.2f} GB), manifest {man_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
