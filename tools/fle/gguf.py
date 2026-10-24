"""Read GGUF headers without loading weights.

The reader is deliberately independent of the engine build: it only needs the GGUF layout and the quantization type
table. Tensor byte counts are checked against the offsets and the file size, so a wrong type guess is visible.
"""

from __future__ import annotations

import math
import os
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

from .util import FleError, GiB, MiB, human_bytes

SPLIT_RE = re.compile(r"^(?P<base>.+)-(?P<no>\d{5})-of-(?P<count>\d{5})\.gguf$", re.IGNORECASE)
SIDECAR_RE = re.compile(r"(?:^|[-_.])(?:mmproj|mtp|draft)(?:[-_.]|$)", re.IGNORECASE)
EXPERT_RE = re.compile(r"\.ffn_(?:gate_up|gate|up|down)_exps\.")
INPUT_RE = re.compile(r"(?:^|\.)(?:token_embd|per_layer_token_embd|engram|engram_embd|ple)(?:\.|$)")

# name, block size, bytes per block, extra bytes at the start of every row
TYPE_TABLE: dict[int, tuple[str, int, int, int]] = {
    0: ("F32", 1, 4, 0),
    1: ("F16", 1, 2, 0),
    2: ("Q4_0", 32, 18, 0),
    3: ("Q4_1", 32, 20, 0),
    6: ("Q5_0", 32, 22, 0),
    7: ("Q5_1", 32, 24, 0),
    8: ("Q8_0", 32, 34, 0),
    9: ("Q8_1", 32, 40, 0),
    10: ("Q2_K", 256, 84, 0),
    11: ("Q3_K", 256, 110, 0),
    12: ("Q4_K", 256, 144, 0),
    13: ("Q5_K", 256, 176, 0),
    14: ("Q6_K", 256, 210, 0),
    15: ("Q8_K", 256, 292, 0),
    16: ("IQ2_XXS", 256, 66, 0),
    17: ("IQ2_XS", 256, 74, 0),
    18: ("IQ3_XXS", 256, 98, 0),
    19: ("IQ1_S", 256, 50, 0),
    20: ("IQ4_NL", 32, 18, 0),
    21: ("IQ3_S", 256, 110, 0),
    22: ("IQ2_S", 256, 82, 0),
    23: ("IQ4_XS", 256, 136, 0),
    24: ("I8", 1, 1, 0),
    25: ("I16", 1, 2, 0),
    26: ("I32", 1, 4, 0),
    27: ("I64", 1, 8, 0),
    28: ("F64", 1, 8, 0),
    29: ("IQ1_M", 256, 56, 0),
    30: ("BF16", 1, 2, 0),
    34: ("TQ1_0", 256, 54, 0),
    35: ("TQ2_0", 256, 66, 0),
    39: ("MXFP4", 32, 17, 0),
    40: ("NVFP4", 64, 36, 0),
    41: ("Q1_0", 128, 18, 0),
    42: ("Q2_0", 64, 18, 0),
}

# The FLE release and the DeepSeek41 port disagree on the last four ids. The offsets decide which table is meant.
TYPE_TABLE_RELEASE = dict(TYPE_TABLE)
TYPE_TABLE_RELEASE.update(
    {
        43: ("IQ2_KT", 256, 68, 4),
        44: ("IQ3_KT", 256, 96, 4),
        45: ("IQ4_KT", 256, 128, 0),
    }
)
TYPE_TABLE_V41 = dict(TYPE_TABLE)
TYPE_TABLE_V41.update(
    {
        43: ("MXFP8", 32, 33, 0),
        44: ("IQ2_KT", 256, 68, 4),
        45: ("IQ3_KT", 256, 96, 4),
        46: ("IQ4_KT", 256, 128, 0),
    }
)
LAYOUTS = {"release": TYPE_TABLE_RELEASE, "deepseek41": TYPE_TABLE_V41}

_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
_STRING, _ARRAY = 8, 9
_MAX_STRING = 32 * 1024 * 1024
_MAX_ARRAY_KEEP = 32


class GGUFError(FleError):
    pass


class _Reader:
    def __init__(self, f: BinaryIO) -> None:
        self.f = f

    def read(self, n: int) -> bytes:
        data = self.f.read(n)
        if len(data) != n:
            raise GGUFError("the file ends inside the header (incomplete download?)")
        return data

    def u32(self) -> int:
        return struct.unpack("<I", self.read(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.read(8))[0]

    def string(self) -> str:
        n = self.u64()
        if n > _MAX_STRING:
            raise GGUFError("implausible string length in header")
        return self.read(n).decode("utf-8", errors="replace")

    def skip(self, vtype: int) -> None:
        if vtype in _SCALAR:
            self.f.seek(struct.calcsize(_SCALAR[vtype]), os.SEEK_CUR)
            return
        if vtype == _STRING:
            n = self.u64()
            if n > _MAX_STRING:
                raise GGUFError("implausible string length in header")
            self.f.seek(n, os.SEEK_CUR)
            return
        if vtype == _ARRAY:
            etype, count = self.u32(), self.u64()
            if count > 100_000_000:
                raise GGUFError("implausible array length in header")
            if etype == _STRING:
                for _ in range(count):
                    n = self.u64()
                    if n > _MAX_STRING:
                        raise GGUFError("implausible string length in header")
                    self.f.seek(n, os.SEEK_CUR)
            elif etype in _SCALAR:
                self.f.seek(struct.calcsize(_SCALAR[etype]) * count, os.SEEK_CUR)
            elif etype == _ARRAY:
                for _ in range(count):
                    self.skip(_ARRAY)
            else:
                raise GGUFError(f"unknown value type {etype} in header")
            return
        raise GGUFError(f"unknown value type {vtype} in header")

    def value(self, vtype: int, keep: int = _MAX_ARRAY_KEEP) -> Any:
        if vtype in _SCALAR:
            fmt = _SCALAR[vtype]
            return struct.unpack(fmt, self.read(struct.calcsize(fmt)))[0]
        if vtype == _STRING:
            return self.string()
        if vtype == _ARRAY:
            etype, count = self.u32(), self.u64()
            if count > 100_000_000:
                raise GGUFError("implausible array length in header")
            if etype == _STRING:
                if count <= keep:
                    return [self.string() for _ in range(count)]
                for _ in range(count):
                    n = self.u64()
                    if n > _MAX_STRING:
                        raise GGUFError("implausible string length in header")
                    self.f.seek(n, os.SEEK_CUR)
                return {"array_of": etype, "count": count}
            if etype in _SCALAR:
                size = struct.calcsize(_SCALAR[etype])
                if count <= keep:
                    return list(struct.unpack(f"<{count}{_SCALAR[etype][1]}", self.read(size * count)))
                self.f.seek(size * count, os.SEEK_CUR)
                return {"array_of": etype, "count": count}
            if etype == _ARRAY:
                if count <= keep:
                    return [self.value(_ARRAY, keep) for _ in range(count)]
                for _ in range(count):
                    self.skip(_ARRAY)
                return {"array_of": etype, "count": count}
            raise GGUFError(f"unknown value type {etype} in header")
        raise GGUFError(f"unknown value type {vtype} in header")


@dataclass
class Tensor:
    name: str
    shape: tuple[int, ...]
    type_id: int
    offset: int
    nbytes: int = 0
    type_name: str = ""

    @property
    def rows(self) -> int:
        return math.prod(self.shape[1:]) if len(self.shape) > 1 else 1

    @property
    def elements(self) -> int:
        return math.prod(self.shape) if self.shape else 0


@dataclass
class Shard:
    path: str
    version: int
    file_size: int
    data_start: int
    alignment: int
    tensor_count: int
    meta: dict[str, Any]
    tensors: list[Tensor]
    layout: str = "release"
    expected_end: int = 0

    @property
    def pay_load_size(self) -> int:
        return max(0, self.file_size - self.data_start)

    @property
    def complete(self) -> bool:
        return self.expected_end <= self.pay_load_size


@dataclass
class Model:
    first: Path
    parts: list[Path]
    missing: list[Path]
    shards: list[Shard]
    meta: dict[str, Any]
    layout: str
    arch: str
    name: str
    size: int
    tensor_count: int
    expert_bytes: int
    input_bytes: int
    gpu_bytes: int
    context_length: int | None
    layers: int | None
    experts: int | None
    experts_used: int | None
    file_type: int | None
    type_histogram: dict[str, dict[str, int]] = field(default_factory=dict)
    unknown_types: list[int] = field(default_factory=list)
    incomplete_shards: list[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return not self.missing and not self.incomplete_shards

    @property
    def is_moe(self) -> bool:
        return bool(self.experts and self.expert_bytes)

    def kv_bytes_per_token(self, type_k: str = "q8_0", type_v: str = "q8_0") -> float:
        m, a = self.meta, self.arch
        layers = int(m.get(f"{a}.block_count") or 0)
        heads = m.get(f"{a}.attention.head_count")
        heads_kv = m.get(f"{a}.attention.head_count_kv", heads)
        emb = m.get(f"{a}.embedding_length")
        head_dim = (emb // heads) if isinstance(emb, int) and isinstance(heads, int) and heads else 128
        k_len = int(m.get(f"{a}.attention.key_length") or head_dim)
        v_len = int(m.get(f"{a}.attention.value_length") or k_len)
        per_layer = heads_kv if isinstance(heads_kv, list) else [heads_kv or 0] * layers
        interval = m.get(f"{a}.full_attention_interval")
        if isinstance(interval, int) and interval > 1:
            per_layer = [h for i, h in enumerate(per_layer) if (i + 1) % interval == 0]
        bk = _KV_TYPE_BYTES.get(type_k, 2.0)
        bv = _KV_TYPE_BYTES.get(type_v, 2.0)
        return float(sum(int(h or 0) * (k_len * bk + v_len * bv) for h in per_layer))

    def summary(self) -> dict[str, Any]:
        return {
            "path": str(self.first),
            "name": self.name,
            "arch": self.arch,
            "layout": self.layout,
            "complete": self.complete,
            "missing": [p.name for p in self.missing],
            "incomplete": list(self.incomplete_shards),
            "size": self.size,
            "size_human": human_bytes(self.size),
            "parts": [p.name for p in self.parts],
            "shards_present": len(self.shards),
            "tensor_count": self.tensor_count,
            "context_length": self.context_length,
            "layers": self.layers,
            "experts": self.experts,
            "experts_used": self.experts_used,
            "expert_bytes": self.expert_bytes,
            "expert_bytes_human": human_bytes(self.expert_bytes),
            "input_bytes": self.input_bytes,
            "gpu_bytes": self.gpu_bytes,
            "gpu_bytes_human": human_bytes(self.gpu_bytes),
            "file_type": self.file_type,
            "kv_bytes_per_token_q8": round(self.kv_bytes_per_token("q8_0", "q8_0")),
            "kv_bytes_per_token_f16": round(self.kv_bytes_per_token("f16", "f16")),
            "type_histogram": self.type_histogram,
            "unknown_types": self.unknown_types,
            "meta": {k: v for k, v in self.meta.items() if not isinstance(v, (list, dict))},
        }


_KV_TYPE_BYTES = {
    "f32": 4.0,
    "f16": 2.0,
    "bf16": 2.0,
    "q8_0": 34 / 32,
    "q5_1": 24 / 32,
    "q5_0": 22 / 32,
    "q4_1": 20 / 32,
    "q4_0": 18 / 32,
    "iq4_nl": 18 / 32,
}


def type_info(type_id: int, layout: str = "release") -> tuple[str, int, int, int] | None:
    return LAYOUTS.get(layout, TYPE_TABLE_RELEASE).get(type_id)


def type_name(type_id: int, layout: str = "release") -> str:
    info = type_info(type_id, layout)
    return info[0] if info else f"type-{type_id}"


def tensor_nbytes(tensor: Tensor, layout: str = "release") -> int:
    info = type_info(tensor.type_id, layout)
    if not info or not tensor.shape or any(d <= 0 for d in tensor.shape):
        return 0
    _, block, size, row_meta = info
    if block <= 0:
        return 0
    if block == 1:
        return int(size * math.prod(tensor.shape))
    if tensor.shape[0] % block:
        return 0
    row_size = row_meta + size * (tensor.shape[0] // block)
    return int(row_size * (math.prod(tensor.shape[1:]) if len(tensor.shape) > 1 else 1))


def read_shard(path: str | os.PathLike[str]) -> Shard:
    p = Path(path)
    with p.open("rb") as f:
        r = _Reader(f)
        if r.read(4) != b"GGUF":
            raise GGUFError(f"{p.name}: not a GGUF file")
        version = r.u32()
        if version not in (2, 3):
            raise GGUFError(f"{p.name}: GGUF version {version} is not supported")
        n_tensors, n_kv = r.u64(), r.u64()
        if n_tensors > 2_000_000 or n_kv > 1_000_000:
            raise GGUFError(f"{p.name}: implausible header")
        meta: dict[str, Any] = {}
        for _ in range(n_kv):
            key = r.string()
            meta[key] = r.value(r.u32())
        raw: list[tuple[str, tuple[int, ...], int, int]] = []
        for _ in range(n_tensors):
            name = r.string()
            n_dims = r.u32()
            if n_dims > 8:
                raise GGUFError(f"{p.name}: implausible tensor rank")
            dims = struct.unpack(f"<{n_dims}Q", r.read(8 * n_dims))
            raw.append((name, tuple(int(d) for d in dims), r.u32(), r.u64()))
        align = meta.get("general.alignment", 32)
        align = align if isinstance(align, int) and align > 0 else 32
        data_start = -(-f.tell() // align) * align
    file_size = p.stat().st_size
    if data_start > file_size:
        raise GGUFError(f"{p.name}: header ends behind the end of the file")
    tensors = [
        Tensor(name=name, shape=shape, type_id=type_id, offset=offset, nbytes=tensor_nbytes(
            Tensor(name, shape, type_id, offset), "release"))
        for name, shape, type_id, offset in raw
    ]
    shard = Shard(
        path=str(p),
        version=version,
        file_size=file_size,
        data_start=data_start,
        alignment=align,
        tensor_count=n_tensors,
        meta=meta,
        tensors=tensors,
    )
    shard.layout = detect_layout([shard])
    for t in shard.tensors:
        t.type_name = type_name(t.type_id, shard.layout)
        t.nbytes = tensor_nbytes(t, shard.layout)
    shard.expected_end = max((t.offset + t.nbytes for t in shard.tensors), default=0)
    return shard


def _layout_score(shards: list[Shard], layout: str) -> float:
    score = 0.0
    for shard in shards:
        if any(t.type_id not in LAYOUTS[layout] for t in shard.tensors):
            return float("inf")
        intervals: list[tuple[int, int]] = []
        end = 0
        for t in shard.tensors:
            n = tensor_nbytes(t, layout)
            if not n:
                return float("inf")
            if t.offset < 0 or t.offset + n > shard.pay_load_size:
                return float("inf")
            intervals.append((t.offset, t.offset + n))
            end = max(end, t.offset + n)
        intervals.sort()
        for (a_start, a_end), (b_start, _) in zip(intervals, intervals[1:]):
            if b_start < a_end:
                return float("inf")
        # The last tensor normally ends at the end of the file. A small gap is padding, not an error.
        score += abs(shard.pay_load_size - end)
    return score


def detect_layout(shards: list[Shard]) -> str:
    high = {t.type_id for shard in shards for t in shard.tensors if t.type_id >= 43}
    if not high:
        arch = str((shards[0].meta.get("general.architecture") if shards else "") or "")
        return "deepseek41" if arch == "deepseek41" else "release"
    scores = {name: _layout_score(shards, name) for name in LAYOUTS}
    valid = {name: score for name, score in scores.items() if math.isfinite(score)}
    if not valid:
        return "release"
    return min(valid, key=lambda name: valid[name])


def split_parts(first: Path) -> list[Path]:
    first = Path(first)
    m = SPLIT_RE.match(first.name)
    if not m:
        return [first]
    count = int(m.group("count"))
    if count > 1000:
        raise GGUFError(f"{first.name}: implausible split count {count}")
    base = m.group("base")
    return [first.with_name(f"{base}-{i:05d}-of-{count:05d}.gguf") for i in range(1, count + 1)]


def resolve_model(path: str | os.PathLike[str]) -> Path:
    p = Path(path).expanduser()
    if p.is_dir():
        candidates = [q for q in p.glob("*.gguf") if not SIDECAR_RE.search(q.name)]
        firsts = [q for q in candidates if not SPLIT_RE.match(q.name) or SPLIT_RE.match(q.name).group("no") == "00001"]
        if not firsts:
            firsts = candidates
        if not firsts:
            raise GGUFError(f"No GGUF model in {p}", "Pass the model file or a folder that contains it.")
        if len(firsts) > 1:
            firsts.sort(key=lambda q: q.stat().st_size, reverse=True)
        p = firsts[0]
    m = SPLIT_RE.match(p.name)
    if m and m.group("no") != "00001":
        p = p.with_name(f"{m.group('base')}-00001-of-{m.group('count')}.gguf")
    if not p.exists():
        raise GGUFError(f"No model at {p}", "Check the path, or start the download again.")
    if p.suffix.lower() != ".gguf":
        raise GGUFError(f"{p} is not a .gguf file")
    return p


def sidecars(first: Path) -> dict[str, str | None]:
    folder = first.parent
    names = sorted(q for q in folder.glob("*") if q.is_file())
    mtp = next((q for q in names if SIDECAR_RE.search(q.name) and "mtp" in q.name.lower() and q.suffix.lower() == ".gguf"), None)
    mmproj = next((q for q in names if "mmproj" in q.name.lower() and q.suffix.lower() == ".gguf"), None)
    vocab = next((q for q in names if "mtp" in q.name.lower() and "vocab" in q.name.lower() and q.suffix.lower() == ".txt"), None)
    return {
        "mtp_head": str(mtp) if mtp else None,
        "mtp_vocab": str(vocab) if vocab else None,
        "mmproj": str(mmproj) if mmproj else None,
    }


def read_model(path: str | os.PathLike[str]) -> Model:
    first = resolve_model(path)
    parts = split_parts(first)
    present = [p for p in parts if p.exists()]
    missing = [p for p in parts if not p.exists()]
    if first not in present:
        raise GGUFError(f"The first part is missing: {first.name}", "Wait for the download to finish or pick another model.")
    shards = [read_shard(p) for p in present]
    layout = detect_layout(shards)
    for shard in shards:
        shard.layout = layout
        for t in shard.tensors:
            t.type_name = type_name(t.type_id, layout)
            t.nbytes = tensor_nbytes(t, layout)
    meta: dict[str, Any] = {}
    for shard in shards:
        for key, value in shard.meta.items():
            meta.setdefault(key, value)
    arch = str(meta.get("general.architecture", "unknown"))
    expert = inputs = other = 0
    hist: dict[str, dict[str, int]] = {}
    unknown: set[int] = set()
    unique: dict[str, Tensor] = {}
    where: dict[str, Shard] = {}
    for shard in shards:
        for tensor in shard.tensors:
            if tensor.name not in unique:
                unique[tensor.name] = tensor
                where[tensor.name] = shard
    for tensor in unique.values():
        shard = where[tensor.name]
        present_bytes = min(tensor.nbytes, max(0, shard.pay_load_size - tensor.offset))
        if tensor.nbytes <= 0:
            unknown.add(tensor.type_id)
        if EXPERT_RE.search(tensor.name):
            expert += present_bytes
        elif INPUT_RE.search(tensor.name):
            inputs += present_bytes
        else:
            other += present_bytes
        name = tensor.type_name
        item = hist.setdefault(name, {"tensors": 0, "bytes": 0})
        item["tensors"] += 1
        item["bytes"] += present_bytes

    def num(key: str) -> int | None:
        value = meta.get(f"{arch}.{key}")
        return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None

    name = meta.get("general.name") or SPLIT_RE.sub(r"\g<base>", first.name).removesuffix(".gguf")
    return Model(
        first=first,
        parts=parts,
        missing=missing,
        shards=shards,
        meta=meta,
        layout=layout,
        arch=arch,
        name=str(name),
        size=sum(s.file_size for s in shards),
        tensor_count=len(unique),
        expert_bytes=expert,
        input_bytes=inputs,
        gpu_bytes=other,
        context_length=num("context_length"),
        layers=num("block_count"),
        experts=num("expert_count"),
        experts_used=num("expert_used_count"),
        file_type=meta.get("general.file_type") if isinstance(meta.get("general.file_type"), int) else None,
        type_histogram=hist,
        unknown_types=sorted(unknown),
        incomplete_shards=[Path(s.path).name for s in shards if not s.complete],
    )


def verify_model(model: Model) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool | None, detail: str, hint: str = "") -> None:
        status = "warn" if ok is None else "ok" if ok else "fail"
        checks.append({"name": name, "status": status, "detail": detail, "hint": hint})

    add("first part", model.first.exists(), str(model.first), "Check the path.")
    if model.missing or model.incomplete_shards:
        detail = f"{len(model.missing)} of {len(model.parts)} parts missing"
        if model.incomplete_shards:
            detail += f", {len(model.incomplete_shards)} incomplete"
        if model.missing:
            errors.append(f"Missing model parts: {', '.join(p.name for p in model.missing[:6])}")
        if model.incomplete_shards:
            errors.append(f"Incomplete model parts: {', '.join(model.incomplete_shards[:6])}")
        add("all parts", False, detail, "Wait for the download to finish, then run verify again.")
    else:
        add("all parts", True, f"{len(model.parts)} part(s) present")

    split_counts: set[int] = set()
    archs = set()
    for index, shard in enumerate(model.shards):
        name = Path(shard.path).name
        try:
            split_no = shard.meta.get("split.no")
            split_count = shard.meta.get("split.count")
            if isinstance(split_no, int) and split_no != index:
                warnings.append(f"{name}: split.no is {split_no}, expected {index}")
            if isinstance(split_count, int):
                split_counts.add(split_count)
            shard_arch = shard.meta.get("general.architecture")
            if isinstance(shard_arch, str):
                archs.add(shard_arch)
            intervals = sorted((t.offset, t.offset + t.nbytes, t.name) for t in shard.tensors)
            overlap = False
            for (a0, a1, an), (b0, _b1, bn) in zip(intervals, intervals[1:]):
                if b0 < a1:
                    overlap = True
                    errors.append(f"{name}: tensors {an} and {bn} overlap")
            if not shard.complete:
                expected = shard.data_start + shard.expected_end
                warnings.append(
                    f"{name}: still downloading or truncated ({human_bytes(shard.file_size)} of about "
                    f"{human_bytes(expected)})"
                )
                add(f"{name} layout", False, f"{len(shard.tensors)} tensors, file is incomplete")
                continue
            out_of_file = [t.name for t in shard.tensors if t.offset < 0 or t.offset + t.nbytes > shard.pay_load_size]
            if out_of_file:
                errors.append(f"{name}: {len(out_of_file)} tensor(s) end behind the file")
            if overlap or out_of_file:
                add(f"{name} layout", False, f"{len(shard.tensors)} tensors, offsets do not fit")
            else:
                add(f"{name} layout", True, f"{len(shard.tensors)} tensors, {human_bytes(shard.pay_load_size)} payload")
        except (OSError, GGUFError) as exc:
            errors.append(f"{name}: {exc}")
            add(name, False, str(exc))
    if split_counts and len(split_counts) > 1:
        errors.append(f"Shards disagree about split.count: {sorted(split_counts)}")
    if len(archs) > 1:
        errors.append(f"Shards disagree about the architecture: {sorted(archs)}")
    expected = None
    for shard in model.shards:
        value = shard.meta.get("split.tensors.count")
        if isinstance(value, int):
            expected = value
            break
    if expected is not None and model.complete and model.tensor_count != expected:
        errors.append(f"Header says {expected} tensors; the shards contain {model.tensor_count}")
    if expected is not None:
        add("tensor count", model.tensor_count == expected or not model.complete,
            f"{model.tensor_count} in the files, {expected} in split.tensors.count")
    if model.unknown_types:
        errors.append(f"Unknown or unsupported tensor types: {model.unknown_types}")
        add("tensor types", False, f"unknown ids {model.unknown_types}", "Use an engine that knows these types, or repair the file.")
    else:
        add("tensor types", True, f"{len(model.type_histogram)} types, all known")
    if model.is_moe and model.experts_used and model.experts and model.experts_used > model.experts:
        errors.append("expert_used_count is larger than expert_count")
    side = sidecars(model.first)
    if side["mtp_head"] and not side["mtp_vocab"]:
        warnings.append("An MTP head is present but no MTP vocabulary file was found; MTP stays off.")
    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "checks": checks,
        "layout": model.layout,
        "tensor_count": model.tensor_count,
        "parts": len(model.parts),
        "present": len(model.shards),
        "size": model.size,
        "sidecars": side,
    }
