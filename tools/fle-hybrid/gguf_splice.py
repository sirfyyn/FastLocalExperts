#!/usr/bin/env python3
"""gguf_splice.py - build a GGUF from the tensors of existing quantizations of the same model.

Metadata and every tensor that is not replaced come from BASE. Tensors matched by --take (optionally restricted to
layers) come from a DONOR. Split GGUFs (...-00001-of-0000N.gguf) are resolved; the output is a single file. Nothing is
re-quantized: tensor bytes are copied as they are, and the output is checked against the sources afterwards.

Several donors: --donor I=ista-iq3s.gguf --donor X=ista-iq3xxs.gguf --take 'I:REGEX@0,2' --take 'X:REGEX@5-7'
(a tensor may come from one donor only; a pattern that two donors claim is an error).

Examples:
  # expert layers 0-11 from the donor, everything else from the base
  gguf_splice.py --base BASE-00001-of-00003.gguf --donor DONOR-00001-of-00002.gguf \\
      --take 'ffn_(gate|up|down)_exps' --layers 0-11 --out hybrid.gguf
  # only show the plan
  gguf_splice.py --base BASE.gguf --donor DONOR.gguf --take 'ffn_down_exps@3,5' --out h.gguf --dry-run

gguf-py: the copy in this repository (../../gguf-py) is used if present, else an installed `gguf` package; GGUF_PY
overrides the path.
"""
from __future__ import annotations

import argparse
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


def shard_paths(first: str) -> list[str]:
    m = re.match(r"(.*)-(\d{5})-of-(\d{5})\.gguf$", first)
    if not m:
        return [first]
    prefix, _, total = m.groups()
    return [f"{prefix}-{i:05d}-of-{total}.gguf" for i in range(1, int(total) + 1)]


def load(first: str):
    readers = [GGUFReader(p, "r") for p in shard_paths(first)]
    tensors: dict[str, object] = {}
    order: list[str] = []
    for reader in readers:
        for tensor in reader.tensors:
            if tensor.name in tensors:
                raise ValueError(f"tensor twice in {first}: {tensor.name}")
            tensors[tensor.name] = tensor
            order.append(tensor.name)
    return readers, tensors, order


def parse_layers(text: str) -> set[int]:
    layers: set[int] = set()
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        layers.update(range(int(lo), int(hi or lo) + 1))
    return layers


def split_take(spec: str) -> tuple[re.Pattern, set[int] | None]:
    """'REGEX' or 'REGEX@0-7,24-31': the pattern then applies only in these layers."""
    regex, sep, tail = spec.rpartition("@")
    if sep and re.fullmatch(r"[0-9,\-]+", tail):
        return re.compile(regex), parse_layers(tail)
    return re.compile(spec), None


def choose_multi(order: list[str], base, donors: dict, take: list, layers: set[int] | None) -> dict[str, str]:
    """Per tensor name 'base' or the name of a donor. `take` holds (donor, pattern, layer set or None).
    A match without a counterpart in the donor, or with a different shape, is an error rather than a silent fallback
    to BASE; two donors claiming the same tensor is an error too."""
    plan: dict[str, str] = {}
    for name in order:
        m = LAYER.match(name)
        layer = int(m.group(1)) if m else None
        claimed = None
        for donor_name, pattern, own in take:
            if not pattern.search(name):
                continue
            allowed = own if own is not None else layers
            if allowed is not None and (layer is None or layer not in allowed):
                continue
            if claimed is not None and claimed != donor_name:
                raise ValueError(f"{name}: claimed by two donors ({claimed}, {donor_name})")
            claimed = donor_name
        if claimed is None:
            plan[name] = "base"
            continue
        donor = donors[claimed]
        if name not in donor:
            raise ValueError(f"donor {claimed} has no tensor {name}")
        if list(donor[name].shape) != list(base[name].shape):
            raise ValueError(f"shape differs for {name}: {list(base[name].shape)} vs {list(donor[name].shape)}")
        plan[name] = claimed
    return plan


def write_output(out: str, base_readers, base, donors: dict, order, plan) -> tuple[int, int]:
    reader0 = base_readers[0]
    arch = reader0.get_field(gguf.Keys.General.ARCHITECTURE).contents()
    writer = GGUFWriter(out, arch)
    for field in reader0.fields.values():
        if (field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF.")
                or field.name.startswith("split.")):
            continue
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), val_type, sub_type=sub_type)
    sources = {"base": base, **donors}
    total = 0
    for name in order:
        t = sources[plan[name]][name]
        total += t.n_bytes
        writer.add_tensor_info(name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    done = 0
    t0 = time.time()
    next_report = 5 * 10**9
    for name in order:
        t = sources[plan[name]][name]
        writer.write_tensor_data(t.data, tensor_endianess=reader0.endianess)
        done += t.n_bytes
        if done >= next_report:
            print(f"  written {done / 1e9:.1f} of {total / 1e9:.1f} GB "
                  f"({done / 1e6 / (time.time() - t0):.0f} MB/s)", flush=True)
            next_report += 5 * 10**9
    writer.close()
    return total, len(order)


def verify(out: str, order, plan, base, donors: dict) -> list[str]:
    """Type, shape and the first and last 64 KiB of every tensor against its source."""
    problems: list[str] = []
    reader = GGUFReader(out, "r")
    got = {t.name: t for t in reader.tensors}
    if list(got) != list(order):
        problems.append("tensor order or set differs")
    sources = {"base": base, **donors}
    edge = 65536
    for name in order:
        if name not in got:
            problems.append(f"missing: {name}")
            continue
        src, dst = sources[plan[name]][name], got[name]
        if src.tensor_type != dst.tensor_type or list(src.shape) != list(dst.shape):
            problems.append(f"type/shape: {name}")
            continue
        a, b = src.data.reshape(-1).view("uint8"), dst.data.reshape(-1).view("uint8")
        if a.nbytes != b.nbytes or a[:edge].tobytes() != b[:edge].tobytes() \
                or a[-edge:].tobytes() != b[-edge:].tobytes():
            problems.append(f"bytes: {name}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", required=True, help="first file of the base GGUF")
    ap.add_argument("--donor", action="append", required=True,
                    help="first file of a donor GGUF, optionally NAME=PATH (repeatable)")
    ap.add_argument("--take", action="append", default=[],
                    help="[DONOR:]REGEX[@0-7,24-31]; matching tensors (in these layers) come from the donor")
    ap.add_argument("--layers", default="", help="only these layers for patterns without @, e.g. 0-11,20")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true", help="only print the plan")
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()
    if not args.take:
        ap.error("at least one --take is needed")
    if os.path.exists(args.out) and not args.dry_run:
        ap.error(f"output exists already: {args.out}")

    base_readers, base, order = load(args.base)
    donors: dict = {}
    for spec in args.donor:
        name, sep, path = spec.partition("=")
        if not sep or "/" in name:
            name, path = "donor", spec
        donors[name] = load(path)[1]
    layers = parse_layers(args.layers) if args.layers else None
    take = []
    for spec in args.take:
        m = re.match(r"^([A-Za-z][A-Za-z0-9_]*):(.*)$", spec)
        if m and m.group(1) in donors:
            donor_name, spec = m.group(1), m.group(2)
        elif len(donors) == 1:
            donor_name = next(iter(donors))
        else:
            ap.error(f"--take without a donor name, but {len(donors)} donors: {spec}")
        pattern, own = split_take(spec)
        take.append((donor_name, pattern, own))
    plan = choose_multi(order, base, donors, take, layers)
    sources = {"base": base, **donors}
    counts = {k: sum(1 for v in plan.values() if v == k) for k in ("base", *donors)}
    bytes_ = {k: sum(sources[k][n].n_bytes for n in order if plan[n] == k) for k in counts}
    print("tensors: %d, %s; bytes %s = %.2f GB" % (
        len(order), ", ".join(f"{k} {counts[k]}" for k in counts),
        " + ".join(f"{k} {bytes_[k] / 1e9:.2f} GB" for k in bytes_), sum(bytes_.values()) / 1e9))
    if args.dry_run:
        return 0
    total, n = write_output(args.out, base_readers, base, donors, order, plan)
    print(f"written: {n} tensors, {total / 1e9:.2f} GB to {args.out}")
    if not args.no_verify:
        problems = verify(args.out, order, plan, base, donors)
        if problems:
            print(f"VERIFICATION FAILED ({len(problems)}): {problems[:5]}")
            return 1
        print("verification ok: type, shape and edges of every tensor match the sources")
    return 0


if __name__ == "__main__":
    sys.exit(main())
