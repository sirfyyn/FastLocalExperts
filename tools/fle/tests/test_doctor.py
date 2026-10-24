from __future__ import annotations

import dataclasses
import json

import pytest
from fle import cli, doctor
from fle.util import GiB
from helpers import FLE_ENGINE, UPSTREAM_ENGINE, settings


def check(name, status, detail="", hint=""):
    return {"name": name, "status": status, "detail": detail, "hint": hint}


def commands(suggestions):
    return {s["profile"]: s["command"] for s in suggestions}


# ---- the pure rule ---------------------------------------------------------------------------------------


def test_no_warnings_no_suggestions():
    assert doctor.suggest([check("RAM", "ok"), check("GPU memory", "ok")], "auto", "/m/x.gguf") == []


def test_no_model_no_suggestions():
    assert doctor.suggest([check("RAM", "warn")], "auto", "") == []


def test_ram_warning_suggests_nvme_with_a_ready_command():
    out = doctor.suggest([check("RAM", "warn", "experts 200 GiB, available 50 GiB")], "auto", "/models/MyModel")
    assert commands(out) == {"nvme": "fle plan /models/MyModel --profile nvme"}
    assert "RAM" in out[0]["reason"]


def test_model_path_with_spaces_is_quoted():
    out = doctor.suggest([check("RAM", "warn")], "auto", "/my models/a b.gguf")
    assert out[0]["command"] == "fle plan '/my models/a b.gguf' --profile nvme"


def test_larger_than_ram_plus_vram_warning_suggests_nvme_once():
    checks = [check("RAM", "warn"), check("plan warning", "warn", "The model is larger than RAM plus VRAM. Use ...")]
    assert [s["profile"] for s in doctor.suggest(checks, "auto", "/m")] == ["nvme"]


def test_little_vram_suggests_balanced_only_for_the_greedy_profiles():
    checks = [check("GPU memory", "warn", "need 30 GiB, 20 GiB free")]
    for profile in ("auto", "max", "debug"):
        assert [s["profile"] for s in doctor.suggest(checks, profile, "/m")] == ["balanced"]
    for profile in ("balanced", "nvme", "off"):
        assert doctor.suggest(checks, profile, "/m") == []


def test_never_suggests_the_profile_already_in_use():
    assert doctor.suggest([check("RAM", "warn")], "nvme", "/m") == []


def test_no_gpu_or_non_fle_engine_suggests_off():
    assert [s["profile"] for s in doctor.suggest([check("GPU", "fail", "no NVIDIA GPU found")], "auto", "/m")] == ["off"]
    flags = [check("engine flags", "fail", "This is not an FLE engine; the FLE cache will be ignored.")]
    assert [s["profile"] for s in doctor.suggest(flags, "auto", "/m")] == ["off"]
    assert doctor.suggest(flags, "off", "/m") == []


# ---- through doctor.run ----------------------------------------------------------------------------------


def test_run_with_big_experts_gives_the_nvme_command(moe_model, machine, monkeypatch):
    big = dataclasses.replace(moe_model, expert_bytes=400 * GiB, size=420 * GiB)
    monkeypatch.setattr(doctor.gguf, "read_model", lambda path: big)
    machine.ram_available = 50 * GiB
    result = doctor.run(str(moe_model.first), settings(engine=FLE_ENGINE))
    assert result["status"] == "warn"
    assert commands(result["suggestions"]) == {"nvme": f"fle plan {moe_model.first} --profile nvme"}


def test_run_with_little_vram_gives_the_balanced_command(moe_model, machine):
    machine.set_gpu_free(3 * GiB, 4 * GiB)
    result = doctor.run(str(moe_model.first), settings(engine=FLE_ENGINE))
    assert "balanced" in commands(result["suggestions"])


def test_healthy_machine_has_an_empty_suggestions_field(moe_model, machine):
    result = doctor.run(str(moe_model.first), settings(engine=FLE_ENGINE))
    assert result["status"] == "ok", result["checks"]
    assert result["suggestions"] == []


def test_cli_json_and_text_output(moe_model, machine, capsys, monkeypatch):
    big = dataclasses.replace(moe_model, expert_bytes=400 * GiB, size=420 * GiB)
    monkeypatch.setattr(doctor.gguf, "read_model", lambda path: big)
    machine.ram_available = 50 * GiB
    assert cli.main(["doctor", str(moe_model.first), "--engine", FLE_ENGINE, "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["suggestions"][0]["profile"] == "nvme"
    assert cli.main(["doctor", str(moe_model.first), "--engine", FLE_ENGINE]) == 0
    out = capsys.readouterr().out
    assert "suggestion:" in out and "--profile nvme" in out
