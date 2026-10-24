from __future__ import annotations

import pytest
from fle import gguf
from fle.util import FleError
from gguf_fixture import expert_bytes, write_gguf


def test_read_model_dense(make_model):
    model = make_model(arch="llama", name="Dense Test", layers=3, context=8192)
    assert model.arch == "llama"
    assert model.name == "Dense Test"
    assert model.layers == 3
    assert model.context_length == 8192
    assert model.experts is None
    assert model.expert_bytes == 0
    assert not model.is_moe
    assert model.complete
    assert model.layout == "release"
    # token_embd counts as input, the rest as GPU-resident weights
    assert model.input_bytes == 2 * 16 * 8
    assert model.gpu_bytes > 0
    assert model.tensor_count == 1 + 3 + 1


def test_read_model_moe_counts_expert_bytes(make_model):
    model = make_model(arch="moe", experts=8, experts_used=2, layers=2)
    assert model.experts == 8
    assert model.experts_used == 2
    assert model.is_moe
    assert model.expert_bytes == expert_bytes(2, 8)
    assert "F16" in model.type_histogram


def test_name_falls_back_to_file_name(make_model):
    model = make_model(name=None, filename="Some-Model-Q4.gguf")
    assert model.name == "Some-Model-Q4"


def test_kv_bytes_per_token_uses_heads_kv(make_model):
    model = make_model(layers=2, heads=4, heads_kv=2, embedding=64)
    # head_dim 16, 2 kv heads, 2 layers; f16 is 2 bytes for K and for V
    assert model.kv_bytes_per_token("f16", "f16") == pytest.approx(2 * 2 * (16 * 2 + 16 * 2))
    assert model.kv_bytes_per_token("q8_0", "q8_0") < model.kv_bytes_per_token("f16", "f16")


def test_truncated_data_is_incomplete(make_model):
    model = make_model(experts=4, truncate_data_by=64)
    assert not model.complete
    assert model.incomplete_shards == ["model.gguf"]


def test_not_a_gguf(tmp_path):
    bad = tmp_path / "bad.gguf"
    bad.write_bytes(b"NOPE" + b"\0" * 64)
    with pytest.raises(gguf.GGUFError, match="not a GGUF"):
        gguf.read_model(bad)


def test_header_cut_short(tmp_path):
    path = write_gguf(tmp_path / "ok.gguf")
    short = tmp_path / "short.gguf"
    short.write_bytes(path.read_bytes()[:40])
    with pytest.raises(gguf.GGUFError):
        gguf.read_model(short)


def test_missing_model_and_wrong_suffix(tmp_path):
    with pytest.raises(FleError, match="No model"):
        gguf.read_model(tmp_path / "nothing.gguf")
    other = tmp_path / "x.bin"
    other.write_bytes(b"x")
    with pytest.raises(FleError, match="not a .gguf"):
        gguf.read_model(other)


def test_folder_resolves_to_model_not_sidecar(tmp_path):
    write_gguf(tmp_path / "model.gguf")
    write_gguf(tmp_path / "mtp-head.gguf", layers=1)
    model = gguf.read_model(tmp_path)
    assert model.first.name == "model.gguf"


def test_split_model_missing_part(tmp_path):
    write_gguf(tmp_path / "m-00001-of-00002.gguf", split=(1, 2))
    model = gguf.read_model(tmp_path / "m-00001-of-00002.gguf")
    assert [p.name for p in model.missing] == ["m-00002-of-00002.gguf"]
    assert not model.complete
    write_gguf(tmp_path / "m-00002-of-00002.gguf", split=(2, 2), tensors=[("blk.1.attn_q.weight", (8, 8))])
    full = gguf.read_model(tmp_path / "m-00002-of-00002.gguf")  # resolves back to part 1
    assert full.first.name == "m-00001-of-00002.gguf"
    assert full.complete
    assert len(full.shards) == 2


def test_sidecars_found_next_to_model(tmp_path):
    write_gguf(tmp_path / "model.gguf")
    assert gguf.sidecars(tmp_path / "model.gguf") == {"mtp_head": None, "mtp_vocab": None, "mmproj": None}
    write_gguf(tmp_path / "mtp-head.gguf", layers=1)
    (tmp_path / "fle-mtp-vocab-32k.txt").write_text("x", encoding="utf-8")
    write_gguf(tmp_path / "mmproj-f16.gguf", layers=1)
    side = gguf.sidecars(tmp_path / "model.gguf")
    assert side["mtp_head"].endswith("mtp-head.gguf")
    assert side["mtp_vocab"].endswith("fle-mtp-vocab-32k.txt")
    assert side["mmproj"].endswith("mmproj-f16.gguf")


def test_verify_model_ok_and_incomplete(make_model):
    ok = gguf.verify_model(make_model(experts=4))
    assert ok["ok"]
    bad = gguf.verify_model(make_model(experts=4, truncate_data_by=64))
    assert not bad["ok"]
    assert any("Incomplete" in e for e in bad["errors"])
