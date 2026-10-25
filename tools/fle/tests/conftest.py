"""Shared fixtures: a fake machine, a fake engine and synthetic models. No real GPU, server or network is touched."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pytest

TOOLS = Path(__file__).resolve().parents[2]
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from fle import config, gguf, hardware, util  # noqa: E402
from fle.util import GiB, MiB  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gguf_fixture import write_gguf  # noqa: E402
from helpers import FLE_ENGINE, UPSTREAM_ENGINE, engine_info  # noqa: E402

class FakeMachine:
    """Mutable description of the machine; `hardware` is patched to read from it."""

    def __init__(self) -> None:
        self.ram_total = 128 * GiB
        self.ram_available = 100 * GiB
        self.gpus: list[dict[str, Any]] = [
            {"index": 0, "name": "Fake GPU", "memory_total": 32 * GiB, "memory_used": 2 * GiB,
             "memory_free": 30 * GiB, "vendor": "nvidia"}
        ]
        self.cpu = {"name": "Fake CPU", "physical": 8, "logical": 16}
        self.engines: dict[str, dict[str, Any]] = {
            FLE_ENGINE: engine_info(FLE_ENGINE, "fle"),
            UPSTREAM_ENGINE: engine_info(UPSTREAM_ENGINE, "upstream"),
        }
        self.default_engine = FLE_ENGINE
        self.port_is_free = True
        self.disk_free = 10_000 * GiB

    def set_gpu_free(self, free: int, total: int | None = None) -> None:
        total = total if total is not None else max(free, 32 * GiB)
        self.gpus = [{"index": 0, "name": "Fake GPU", "memory_total": total, "memory_used": total - free,
                      "memory_free": free, "vendor": "nvidia"}]


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for key in list(os.environ):
        if key.startswith("FLE_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    # Output tests compare plain text even when pytest runs with -s on a terminal; colour tests opt in explicitly.
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("TERM", "xterm")
    util.set_color_mode("auto")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("FLE_CONFIG", str(tmp_path / "xdg" / "fle" / "config.toml"))


@pytest.fixture
def machine(monkeypatch: pytest.MonkeyPatch) -> FakeMachine:
    m = FakeMachine()
    monkeypatch.setattr(hardware, "cpu", lambda: dict(m.cpu))
    monkeypatch.setattr(hardware, "memory", lambda: {"total": m.ram_total, "available": m.ram_available})
    monkeypatch.setattr(hardware, "gpus", lambda: [dict(g) for g in m.gpus])
    monkeypatch.setattr(hardware, "port_free", lambda host, port: m.port_is_free)
    monkeypatch.setattr(hardware, "detect_engine", lambda path: dict(m.engines.get(str(path)) or {
        "ok": False, "path": str(path), "problem": f"No file at {path}"}))
    monkeypatch.setattr(hardware, "find_engine", lambda explicit="": m.default_engine)
    monkeypatch.setattr(hardware, "engine_candidates", lambda explicit="": [explicit] * bool(explicit) + list(m.engines))
    monkeypatch.setattr(hardware, "disk_free", lambda path: m.disk_free)
    monkeypatch.setattr(hardware, "summary", lambda model_path=None: {
        "os": "FakeOS", "platform": "linux", "python": "3", "cpu": dict(m.cpu),
        "memory": {"total": m.ram_total, "available": m.ram_available}, "memory_human": "",
        "gpus": [dict(g) for g in m.gpus], "disk_free": m.disk_free, "disk_free_human": ""})
    # os.path.isfile is used for the upstream candidates in cli; make the fake engines look like files.
    real_is_file = Path.is_file
    monkeypatch.setattr(Path, "is_file", lambda self: str(self) in m.engines or real_is_file(self))
    return m


@pytest.fixture
def make_model(tmp_path: Path):
    """make_model(experts=0, sidecars=False, **write_gguf_args) -> gguf.Model in its own folder."""
    counter = {"n": 0}

    def make(*, sidecars: bool = False, filename: str = "model.gguf", folder: str | None = None, **kw: Any) -> gguf.Model:
        counter["n"] += 1
        root = tmp_path / (folder or f"models{counter['n']}")
        root.mkdir(parents=True, exist_ok=True)
        write_gguf(root / filename, **kw)
        if sidecars:
            write_gguf(root / "mtp-head.gguf", name="mtp head", layers=1)
            (root / "fle-mtp-vocab-32k.txt").write_text("a\nb\n", encoding="utf-8")
        return gguf.read_model(root / filename)

    return make


@pytest.fixture
def moe_model(make_model):
    return make_model(experts=8, layers=2, sidecars=False)
