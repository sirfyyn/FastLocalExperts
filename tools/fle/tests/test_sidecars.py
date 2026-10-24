from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from fle import cli, gguf, sidecars
from fle.util import FleError
from gguf_fixture import write_gguf


def fake_source(payloads: dict[str, bytes], family="Test-Family") -> sidecars.Source:
    import re

    files = tuple(
        sidecars.SidecarFile(kind, "owner/repo", path, len(data), hashlib.sha256(data).hexdigest())
        for (kind, path), data in zip((("mtp_head", "mtp/mtp-head.gguf"), ("mtp_vocab", "mtp/mtp-vocab.txt")),
                                      payloads.values())
    )
    return sidecars.Source(family, re.compile("^known"), files)


HEAD, VOCAB = b"H" * 1000, b"vocab words\n" * 10
PAYLOADS = {"head": HEAD, "vocab": VOCAB}


class FakeFetcher:
    def __init__(self, data_by_name: dict[str, bytes]):
        self.data = data_by_name
        self.calls: list[str] = []

    def __call__(self, url, dest, headers):
        self.calls.append(url)
        name = url.rsplit("/", 1)[-1]
        dest.write_bytes(self.data[name])


@pytest.fixture
def known_model(make_model):
    return make_model(filename="known-model.gguf")


def test_find_source_by_real_table(make_model):
    qwen = make_model(filename="Qwen3.8-Flash-Next-FLE-IQ3_M-00001-of-00001.gguf")
    assert sidecars.find_source(qwen).family == "Qwen3.8-Flash-Next-FLE"
    reap = make_model(filename="Qwen3.8-Flash-Next-FLE-IQ3_M-REAP448.gguf")
    assert sidecars.find_source(reap) is not None
    assert sidecars.find_source(make_model(filename="MyModel.gguf")) is None


def test_real_table_matches_what_gguf_sidecars_detects(tmp_path):
    """The names we download must be recognised again as MTP sidecars next to a model."""
    source = sidecars.KNOWN_SOURCES[0]
    model = tmp_path / "Qwen3.8-Flash-Next-FLE-IQ3_M.gguf"
    write_gguf(model)
    for f in source.files:
        (tmp_path / f.name).write_bytes(b"x")
    side = gguf.sidecars(model)
    assert side["mtp_head"] and side["mtp_vocab"]
    for f in source.files:
        assert f.url.startswith("https://huggingface.co/sirfyyn/") and "/resolve/main/mtp/" in f.url
        assert len(f.sha256) == 64 and f.size > 0


def test_plan_items_missing_present_and_force(known_model):
    source = fake_source(PAYLOADS)
    items = sidecars.plan_items(known_model, source)
    assert [i.action for i in items] == ["download", "download"]
    assert items[0].dest == known_model.first.parent / "mtp-head.gguf"
    # a head under another name still counts as present
    other = known_model.first.parent / "mtp-other-name.gguf"
    write_gguf(other, layers=1)
    items = sidecars.plan_items(known_model, source)
    assert [i.action for i in items] == ["present", "download"]
    assert items[0].dest == other
    forced = sidecars.plan_items(known_model, source, force=True)
    assert [i.action for i in forced] == ["download", "download"]
    (known_model.first.parent / "mtp-vocab.txt").write_text("old")
    forced = sidecars.plan_items(known_model, source, force=True)
    assert forced[1].action == "replace"


def test_fetch_writes_verified_files_atomically(known_model):
    source = fake_source(PAYLOADS)
    fetcher = FakeFetcher({"mtp-head.gguf": HEAD, "mtp-vocab.txt": VOCAB})
    folder = known_model.first.parent
    written = sidecars.fetch(sidecars.plan_items(known_model, source), fetcher=fetcher)
    assert [i.dest.name for i in written] == ["mtp-head.gguf", "mtp-vocab.txt"]
    assert (folder / "mtp-head.gguf").read_bytes() == HEAD
    assert sorted(p.name for p in folder.iterdir() if p.name.startswith(".")) == []  # no temp files left
    assert gguf.sidecars(known_model.first)["mtp_vocab"].endswith("mtp-vocab.txt")


def test_fetch_rejects_a_wrong_checksum_and_leaves_nothing(known_model):
    source = fake_source(PAYLOADS)
    corrupt = b"X" * len(HEAD)  # right size, wrong content
    fetcher = FakeFetcher({"mtp-head.gguf": corrupt, "mtp-vocab.txt": VOCAB})
    with pytest.raises(FleError, match="SHA-256 mismatch"):
        sidecars.fetch(sidecars.plan_items(known_model, source), fetcher=fetcher)
    names = sorted(p.name for p in known_model.first.parent.iterdir())
    assert names == ["known-model.gguf"]


def test_fetch_rejects_a_wrong_size(known_model):
    source = fake_source(PAYLOADS)
    fetcher = FakeFetcher({"mtp-head.gguf": HEAD[:-1], "mtp-vocab.txt": VOCAB})
    with pytest.raises(FleError, match="expected"):
        sidecars.fetch(sidecars.plan_items(known_model, source), fetcher=fetcher)
    assert not (known_model.first.parent / "mtp-head.gguf").exists()


def test_fetch_never_overwrites_without_force(known_model):
    source = fake_source(PAYLOADS)
    target = known_model.first.parent / "mtp-vocab.txt"
    target.write_text("keep me")
    items = sidecars.plan_items(known_model, source)  # vocab counts as present -> not downloaded
    fetcher = FakeFetcher({"mtp-head.gguf": HEAD, "mtp-vocab.txt": VOCAB})
    sidecars.fetch(items, fetcher=fetcher)
    assert target.read_text() == "keep me"
    assert len(fetcher.calls) == 1
    # even an item marked "download" is skipped when the file exists and force is off
    stale = [sidecars.Item(source.files[1], target, "download")]
    assert sidecars.fetch(stale, fetcher=fetcher) == []
    assert target.read_text() == "keep me"
    # with force it is replaced
    forced = sidecars.plan_items(known_model, source, force=True)
    sidecars.fetch(forced, force=True, fetcher=fetcher)
    assert target.read_bytes() == VOCAB


def test_fetch_checks_disk_space(known_model, monkeypatch):
    source = fake_source(PAYLOADS)

    class Usage:
        free = 10

    monkeypatch.setattr(sidecars.shutil, "disk_usage", lambda p: Usage)
    with pytest.raises(FleError, match="Not enough disk space"):
        sidecars.fetch(sidecars.plan_items(known_model, source), fetcher=FakeFetcher({}))


def test_fetch_sends_token_only_when_set(known_model, monkeypatch):
    seen = []
    source = fake_source(PAYLOADS)

    def fetcher(url, dest, headers):
        seen.append(dict(headers))
        dest.write_bytes(HEAD if url.endswith("gguf") else VOCAB)

    sidecars.fetch(sidecars.plan_items(known_model, source), fetcher=fetcher)
    assert seen == [{}, {}]
    monkeypatch.setenv("HF_TOKEN", "t0k")
    seen.clear()
    sidecars.fetch(sidecars.plan_items(known_model, source, force=True), force=True, fetcher=fetcher)
    assert seen[0] == {"Authorization": "Bearer t0k"}


# ---- the command -----------------------------------------------------------------------------------------


def run_cli(capsys, *argv):
    code = cli.main(["fetch-sidecars", *argv])
    return code, capsys.readouterr()


def test_cli_unknown_model_says_no_known_source(make_model, capsys):
    model = make_model(filename="MyModel.gguf")
    code, cap = run_cli(capsys, str(model.first))
    assert code == 1
    assert "no known source for this model" in cap.err


def test_cli_dry_run_lists_and_changes_nothing(make_model, capsys, monkeypatch):
    model = make_model(filename="Qwen3.8-Flash-Next-FLE-IQ3_M.gguf")
    monkeypatch.setattr(sidecars, "urllib_fetcher", lambda *a: pytest.fail("network used in a dry run"))
    code, cap = run_cli(capsys, str(model.first), "--dry-run")
    assert code == 0
    assert "would download" in cap.out and "huggingface.co/sirfyyn/" in cap.out
    assert sorted(p.name for p in model.first.parent.iterdir()) == ["Qwen3.8-Flash-Next-FLE-IQ3_M.gguf"]
    code, cap = run_cli(capsys, str(model.first), "--dry-run", "--json")
    data = json.loads(cap.out)
    assert code == 0 and data["dry_run"] and [i["action"] for i in data["items"]] == ["download", "download"]


def test_cli_download_with_a_fake_fetcher(make_model, capsys, monkeypatch):
    model = make_model(filename="Qwen3.8-Flash-Next-FLE-IQ3_M.gguf")
    real = sidecars.KNOWN_SOURCES[0]
    payloads = {"mtp-Qwen3.8-Flash-Next-FLE-IQ4_NL.gguf": b"h" * 10, "fle-mtp-vocab-32k.txt": b"v" * 5}
    patched = sidecars.Source(real.family, real.pattern, tuple(
        sidecars.SidecarFile(f.kind, f.repo, f.path, len(payloads[f.name]), hashlib.sha256(payloads[f.name]).hexdigest())
        for f in real.files))
    monkeypatch.setattr(sidecars, "KNOWN_SOURCES", (patched,))
    monkeypatch.setattr(sidecars, "urllib_fetcher", lambda url, dest, headers: dest.write_bytes(payloads[url.rsplit("/", 1)[-1]]))
    code, cap = run_cli(capsys, str(model.first))
    assert code == 0 and cap.out.count("wrote ") == 2
    code, cap = run_cli(capsys, str(model.first))  # second time: nothing to do
    assert code == 0 and "nothing to do" in cap.out


def test_cli_failed_checksum_exits_2(make_model, capsys, monkeypatch):
    model = make_model(filename="Qwen3.8-Flash-Next-FLE-IQ3_M.gguf")
    real = sidecars.KNOWN_SOURCES[0]
    tiny = sidecars.Source(real.family, real.pattern, tuple(
        sidecars.SidecarFile(f.kind, f.repo, f.path, 3, "0" * 64) for f in real.files))
    monkeypatch.setattr(sidecars, "KNOWN_SOURCES", (tiny,))
    monkeypatch.setattr(sidecars, "urllib_fetcher", lambda url, dest, headers: dest.write_bytes(b"abc"))
    code, cap = run_cli(capsys, str(model.first))
    assert code == 2 and "SHA-256 mismatch" in cap.err
