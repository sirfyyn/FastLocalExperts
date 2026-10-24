"""Write tiny synthetic GGUF files for tests: header, metadata and tensor infos plus a zero-filled data part.

Only what `fle.gguf` reads is produced. Tensors are F16 (type id 1), so a tensor of shape (a, b) takes 2*a*b bytes and
the file stays a few kilobytes. Tests that need a "big" model replace the byte counts on the parsed Model instead of
writing big files.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

GGUF_F16 = 1
_U32, _I32, _F32, _BOOL, _STRING, _ARRAY, _U64 = 4, 5, 6, 7, 8, 9, 10


def _string(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _value(value: Any) -> bytes:
    if isinstance(value, bool):
        return struct.pack("<I?", _BOOL, value)
    if isinstance(value, int):
        return struct.pack("<II", _U32, value) if value < 2**32 else struct.pack("<IQ", _U64, value)
    if isinstance(value, float):
        return struct.pack("<If", _F32, value)
    if isinstance(value, str):
        return struct.pack("<I", _STRING) + _string(value)
    if isinstance(value, list):
        if all(isinstance(x, int) and not isinstance(x, bool) for x in value):
            body = struct.pack("<IQ", _U32, len(value)) + b"".join(struct.pack("<I", x) for x in value)
            return struct.pack("<I", _ARRAY) + body
        if all(isinstance(x, str) for x in value):
            body = struct.pack("<IQ", _STRING, len(value)) + b"".join(_string(x) for x in value)
            return struct.pack("<I", _ARRAY) + body
    raise TypeError(f"unsupported metadata value {value!r}")


def f16_bytes(shape: tuple[int, ...]) -> int:
    total = 1
    for dim in shape:
        total *= dim
    return 2 * total


def default_tensors(layers: int, experts: int) -> list[tuple[str, tuple[int, ...]]]:
    """A tiny tensor set: embeddings, one dense tensor per layer, and expert tensors when `experts` is set."""
    out: list[tuple[str, tuple[int, ...]]] = [("token_embd.weight", (16, 8))]
    for layer in range(layers):
        out.append((f"blk.{layer}.attn_q.weight", (8, 8)))
        if experts:
            out.append((f"blk.{layer}.ffn_up_exps.weight", (8, 4, experts)))
            out.append((f"blk.{layer}.ffn_down_exps.weight", (8, 4, experts)))
    out.append(("output.weight", (8, 16)))
    return out


def write_gguf(
    path: Path | str,
    *,
    arch: str = "testarch",
    name: str | None = "Test Model",
    layers: int = 2,
    experts: int = 0,
    experts_used: int | None = None,
    context: int | None = 4096,
    heads: int = 4,
    heads_kv: int = 2,
    embedding: int = 64,
    tensors: list[tuple[str, tuple[int, ...]]] | None = None,
    extra_meta: dict[str, Any] | None = None,
    truncate_data_by: int = 0,
    alignment: int = 32,
    split: tuple[int, int] | None = None,
) -> Path:
    """Write one GGUF file. `truncate_data_by` cuts bytes off the data part to simulate a running download."""
    path = Path(path)
    meta: dict[str, Any] = {"general.architecture": arch, "general.alignment": alignment}
    if name is not None:
        meta["general.name"] = name
    meta[f"{arch}.block_count"] = layers
    if context is not None:
        meta[f"{arch}.context_length"] = context
    meta[f"{arch}.attention.head_count"] = heads
    meta[f"{arch}.attention.head_count_kv"] = heads_kv
    meta[f"{arch}.embedding_length"] = embedding
    if experts:
        meta[f"{arch}.expert_count"] = experts
        meta[f"{arch}.expert_used_count"] = experts_used if experts_used is not None else min(2, experts)
    if split:
        meta["split.no"], meta["split.count"] = split[0] - 1, split[1]
    meta.update(extra_meta or {})

    specs = tensors if tensors is not None else default_tensors(layers, experts)
    header = b"GGUF" + struct.pack("<I", 3) + struct.pack("<QQ", len(specs), len(meta))
    for key, value in meta.items():
        header += _string(key) + _value(value)
    offset = 0
    data_size = 0
    for tensor_name, shape in specs:
        header += _string(tensor_name) + struct.pack("<I", len(shape))
        header += struct.pack(f"<{len(shape)}Q", *shape)
        header += struct.pack("<IQ", GGUF_F16, offset)
        size = f16_bytes(shape)
        pad = (-size) % alignment
        offset += size + pad
        data_size = offset
    data_start = -(-len(header) // alignment) * alignment
    blob = header + b"\0" * (data_start - len(header)) + b"\0" * max(0, data_size - truncate_data_by)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blob)
    return path


def expert_bytes(layers: int, experts: int) -> int:
    return layers * 2 * f16_bytes((8, 4, experts)) if experts else 0
