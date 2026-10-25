"""How the human-readable output looks: the multi-line start command, colour, and dense (non-MoE) models."""

from __future__ import annotations

import dataclasses
import json
import os
import shlex
import stat
import subprocess
import sys

import pytest
from fle import cli, doctor, hardware, plan, util
from fle.util import GiB
from helpers import FLE_ENGINE, settings

ESC = "\033["


def build(model, profile="auto", **kw):
    extra = kw.pop("settings", {})
    return plan.build(settings(profile=profile, **extra), model, engine_path=kw.pop("engine_path", FLE_ENGINE),
                      model_path=str(model.first), **kw)


def joined(block: str) -> str:
    """What a shell sees: backslash-newline continuations removed."""
    return block.replace(" \\\n", " ")


# ---- 1. the start command ----------------------------------------------------------------------------------


def test_block_is_the_same_command_as_the_oneline(moe_model, machine):
    p = build(moe_model, "auto", extra_args=["--note", 'it\'s a "test" with spaces', "--arg=--kv-unified"],
              env_override={"FLE_X": "a b'c"})
    block = p.command_block()
    assert "\n" in block and "\\\n" in block
    expected = [f"{k}={v}" for k, v in p.env.items()] + [p.binary, *p.args]
    assert shlex.split(joined(block)) == expected
    assert shlex.split(joined(p.command_block(oneline=True))) == expected


def test_block_layout_one_env_per_line_then_binary_then_flag_pairs(moe_model, machine):
    p = build(moe_model, "auto")
    lines = p.command_block().split("\n")
    env_lines = lines[: len(p.env)]
    assert [line.split("=")[0] for line in env_lines] == list(p.env)
    assert all(line.endswith(" \\") for line in lines[:-1]) and not lines[-1].endswith("\\")
    assert lines[len(p.env)].strip(" \\") == p.binary
    assert "  -ngl 99 \\" in lines          # a flag with its value shares a line
    assert "  --jinja \\" in lines          # a flag without a value stands alone
    assert "  --fit-target " + str(p.fit_target_mib) + " \\" in lines
    assert max(len(line) for line in lines) < 120 or any(len(x) >= 100 for x in (p.binary, str(p.args)))


def test_block_without_environment_starts_with_the_binary(make_model, machine):
    p = build(make_model(experts=0), "off")
    assert not p.env
    assert p.command_block().split("\n")[0] == shlex.quote(p.binary) + " \\"


def test_block_runs_in_a_shell_and_means_the_same(moe_model, machine, tmp_path):
    if os.name == "nt":
        pytest.skip("POSIX shell")
    script = tmp_path / "fake server.py"  # a space in the path
    script.write_text(
        "#!" + sys.executable + "\nimport json, os, sys\n"
        "print(json.dumps([sys.argv[1:], {k: v for k, v in os.environ.items() if k.startswith('FLE_')}]))\n"
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    p = build(moe_model, "auto", extra_args=["--note", 'it\'s a "test"; $HOME `x`'], env_override={"FLE_X": "a b'c"})
    p.binary = str(script)
    outputs = []
    for text in (p.command_block(), p.command_block(oneline=True)):
        done = subprocess.run(text, shell=True, capture_output=True, text=True, check=True)
        outputs.append(json.loads(done.stdout))
    assert outputs[0] == outputs[1]
    args, env = outputs[0]
    assert args == p.args and args[args.index("--note") + 1] == 'it\'s a "test"; $HOME `x`'
    assert env["FLE_X"] == "a b'c"


def test_json_command_is_unchanged(moe_model, machine, capsys):
    assert cli.main(["plan", str(moe_model.first), "--engine", FLE_ENGINE, "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    p = build(moe_model, "auto", engine_path=FLE_ENGINE)
    assert data["command"] == p.command()
    assert "\\\n  " in data["command"] and data["command"].count("\n") == 1  # the old two-line form


def test_plan_serve_dry_run_print_the_block_and_oneline_keeps_the_old_text(moe_model, machine, capsys):
    base = [str(moe_model.first), "--engine", FLE_ENGINE]
    for argv in (["plan"] + base, ["serve", "--dry-run", "--no-preflight"] + base):
        assert cli.main(argv) == 0
        multi = capsys.readouterr().out
        assert cli.main(argv + ["--oneline"]) == 0
        single = capsys.readouterr().out
        assert "  -ngl 99 \\\n" in multi and "  -ngl 99 \\\n" not in single
        assert multi.count("\n") > single.count("\n")
        assert shlex.split(joined(multi.split("\n\n")[1])) == shlex.split(joined(single.split("\n\n")[1]))
        assert " -ngl 99 " in single  # the long line is back


# ---- 2. colour ---------------------------------------------------------------------------------------------


class Tty:
    def __init__(self, tty=True):
        self.tty = tty

    def isatty(self):
        return self.tty


def test_paint_follows_the_stream_and_the_environment(monkeypatch):
    monkeypatch.delenv("NO_COLOR")
    assert util.paint("x", "red", stream=Tty(True)) == f"{ESC}31mx{ESC}0m"
    assert util.paint("x", "bold", "red", stream=Tty(True)) == f"{ESC}1;31mx{ESC}0m"
    assert util.paint("x", "red", stream=Tty(False)) == "x"
    monkeypatch.setenv("NO_COLOR", "1")
    assert util.paint("x", "red", stream=Tty(True)) == "x"
    monkeypatch.setenv("NO_COLOR", "")  # empty means unset, as no-color.org says
    assert util.paint("x", "red", stream=Tty(True)) != "x"
    monkeypatch.delenv("NO_COLOR")
    monkeypatch.setenv("TERM", "dumb")
    assert util.paint("x", "red", stream=Tty(True)) == "x"


def test_color_modes_override_the_environment(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("TERM", "dumb")
    util.set_color_mode("always")
    assert util.paint("x", "green", stream=Tty(False)) == f"{ESC}32mx{ESC}0m"
    monkeypatch.delenv("NO_COLOR")
    monkeypatch.setenv("TERM", "xterm")
    util.set_color_mode("never")
    assert util.paint("x", "green", stream=Tty(True)) == "x"
    util.set_color_mode("nonsense")
    assert util.use_color(Tty(True)) is True


def test_paint_status_only_knows_ok_warn_fail(monkeypatch):
    util.set_color_mode("always")
    assert util.paint_status("a", "ok").startswith(f"{ESC}32m")
    assert util.paint_status("a", "warn").startswith(f"{ESC}33m")
    assert util.paint_status("a", "fail").startswith(f"{ESC}31m")
    assert util.paint_status("a", "other") == "a"


def busy(moe_model, machine):
    machine.port_is_free = False  # gives the plan a warning
    return [str(moe_model.first), "--engine", FLE_ENGINE]


def test_plain_when_not_a_terminal(moe_model, machine, capsys, monkeypatch):
    monkeypatch.delenv("NO_COLOR")
    assert cli.main(["plan"] + busy(moe_model, machine)) == 0
    out = capsys.readouterr().out
    assert "warnings:" in out and ESC not in out  # capsys is not a terminal


def test_plan_warnings_are_coloured_on_a_terminal_and_when_asked(moe_model, machine, capsys, monkeypatch):
    argv = ["plan"] + busy(moe_model, machine)
    monkeypatch.delenv("NO_COLOR")
    monkeypatch.setattr(util, "is_tty", lambda stream=None: True)
    assert cli.main(argv) == 0
    out = capsys.readouterr().out
    assert f"{ESC}1;33mwarnings:{ESC}0m" in out and f"{ESC}33m  - Port" in out
    assert "notes:" in out and f"{ESC}1;33mnotes" not in out  # only warnings carry colour
    monkeypatch.setenv("NO_COLOR", "1")
    assert cli.main(argv) == 0
    assert ESC not in capsys.readouterr().out
    assert cli.main(argv + ["--color", "always"]) == 0  # an explicit switch beats NO_COLOR
    assert ESC in capsys.readouterr().out
    assert cli.main(argv + ["--color", "never"]) == 0
    assert ESC not in capsys.readouterr().out


def test_json_is_never_coloured(moe_model, machine, capsys):
    argv = ["plan"] + busy(moe_model, machine) + ["--json", "--color", "always"]
    assert cli.main(argv) == 0
    out = capsys.readouterr().out
    assert ESC not in out and json.loads(out)["warnings"]
    assert cli.main(["doctor", str(moe_model.first), "--engine", FLE_ENGINE, "--json", "--color", "always"]) in (0, 1)
    assert ESC not in capsys.readouterr().out


def test_doctor_status_is_coloured(moe_model, machine, capsys):
    machine.set_gpu_free(3 * GiB, 4 * GiB)  # a warning
    assert cli.main(["doctor", str(moe_model.first), "--engine", FLE_ENGINE, "--color", "always"]) == 0
    out = capsys.readouterr().out
    assert f"[{ESC}32mok  {ESC}0m] model" in out
    assert f"[{ESC}33mwarn{ESC}0m] GPU memory" in out
    assert f"fle doctor: {ESC}33mwarn{ESC}0m" in out
    machine.gpus = []
    assert cli.main(["doctor", str(moe_model.first), "--engine", FLE_ENGINE, "--color", "always"]) == 1
    assert f"[{ESC}31mfail{ESC}0m] GPU" in capsys.readouterr().out


def test_doctor_text_is_plain_by_default(moe_model, machine, capsys):
    assert cli.main(["doctor", str(moe_model.first), "--engine", FLE_ENGINE]) == 0
    out = capsys.readouterr().out
    assert "[ok  ] model" in out and ESC not in out


def test_errors_are_coloured_on_stderr_only_when_wanted(machine, capsys):
    assert cli.main(["plan", "--color", "always"]) == 2
    err = capsys.readouterr().err
    assert f"{ESC}1;31mfle:{ESC}0m No model given." in err and f"{ESC}36mhint:{ESC}0m Pass MODEL" in err
    assert cli.main(["plan"]) == 2
    err = capsys.readouterr().err
    assert err.startswith("fle: No model given.\nhint: Pass MODEL") and ESC not in err
    assert cli.main(["plan", "--json", "--color", "always"]) == 2
    assert ESC not in capsys.readouterr().err


def test_color_switch_is_accepted_by_every_command(capsys):
    parser = cli.build_parser()
    for words in (["inspect"], ["verify"], ["doctor"], ["plan"], ["serve"], ["report"], ["bench"], ["config"]):
        tail = ["show"] if words == ["config"] else []  # config takes its switches before the action
        assert parser.parse_args(words + ["--color", "never"] + tail).color == "never"
        assert parser.parse_args(words + tail).color == "auto"
    with pytest.raises(SystemExit):
        parser.parse_args(["plan", "--color", "rainbow"])


# ---- 4. dense models -----------------------------------------------------------------------------------------


def test_inspect_says_dense_instead_of_zero_experts(make_model, moe_model, machine, capsys):
    dense = make_model(experts=0)
    assert cli.main(["inspect", str(dense.first)]) == 0
    out = capsys.readouterr().out
    assert "experts:  dense (no experts)" in out
    assert "experts:  0 B" not in out and "per layer" not in out
    assert cli.main(["inspect", str(moe_model.first)]) == 0
    out = capsys.readouterr().out
    assert "per layer" in out and "dense (no experts)" not in out
    assert out.count("experts:") == 2  # the count and the bytes


def test_missing_cmoe_matters_only_for_moe_models():
    engine = {"ok": True, "flavor": "fle", "has_fit_target": True, "has_cmoe": False}
    assert any("-cmoe" in p for p in hardware.check_engine_flags(engine, "auto", True))
    assert any("-cmoe" in p for p in hardware.check_engine_flags(engine, "auto"))  # model unknown: warn as before
    assert hardware.check_engine_flags(engine, "auto", False) == []
    engine["has_fit_target"] = False
    assert [p for p in hardware.check_engine_flags(engine, "auto", False)] == ["This engine has no --fit-target; use a newer FLE build."]


def test_doctor_does_not_warn_about_cmoe_for_a_dense_model(make_model, moe_model, machine):
    machine.engines[FLE_ENGINE]["has_cmoe"] = False
    dense = doctor.run(str(make_model(experts=0).first), settings(engine=FLE_ENGINE))
    assert not any(c["name"] == "engine flags" for c in dense["checks"]), dense["checks"]
    moe = doctor.run(str(moe_model.first), settings(engine=FLE_ENGINE))
    assert any(c["name"] == "engine flags" and "-cmoe" in c["detail"] for c in moe["checks"])


def test_doctor_names_a_dense_model(make_model, machine, capsys):
    path = str(make_model(experts=0).first)
    assert cli.main(["doctor", path, "--engine", FLE_ENGINE]) == 0
    assert ", dense, " in capsys.readouterr().out


def test_doctor_gpu_memory_wording_for_dense_and_moe(make_model, moe_model, machine):
    machine.set_gpu_free(3 * GiB, 4 * GiB)

    def gpu_memory(model):
        result = doctor.run(str(model.first), settings(engine=FLE_ENGINE))
        return next(c for c in result["checks"] if c["name"] == "GPU memory")

    dense = gpu_memory(make_model(experts=0))
    assert dense["detail"].startswith("weights and KV need") and "nvme" not in dense["hint"]
    moe = gpu_memory(moe_model)
    assert moe["detail"].startswith("non-expert weights and KV need") and "nvme" in moe["hint"]


def test_nvme_notes_for_a_dense_model_are_honest(make_model, moe_model, machine):
    dense = build(make_model(experts=0), "nvme")
    text = " ".join(dense.notes)
    assert "meant for models larger than RAM" in text and "no experts to stream" in text
    assert "page experts" not in text
    assert "--no-warmup" in dense.args and "--load-mode" not in dense.args  # behaviour unchanged
    assert any("page experts in on demand" in n for n in build(moe_model, "nvme").notes)


def test_balanced_and_max_notes_for_a_dense_model(make_model, moe_model, machine):
    dense = make_model(experts=0)
    balanced = " ".join(build(dense, "balanced").notes)
    assert "1536 MiB" in balanced and "30 percent" not in balanced and "dense model has no expert cache" in balanced
    maxed = " ".join(build(dense, "max").notes)
    assert "Max does not add an expert cache" in maxed and "65,536" in maxed
    assert "30 percent of the cache" in " ".join(build(moe_model, "balanced").notes)


def test_oversized_dense_model_does_not_promise_expert_streaming(make_model, moe_model, machine):
    small = make_model(experts=0)
    big = dataclasses.replace(small, size=500 * GiB)
    warning = next(w for w in build(big, "auto").warnings if "larger than RAM plus VRAM" in w)
    assert "no experts to stream" in warning and "stream experts from disk" not in warning
    moe = dataclasses.replace(moe_model, size=500 * GiB, expert_bytes=400 * GiB)
    assert any("stream experts from disk" in w for w in build(moe, "auto").warnings)
    # the doctor suggestion still points at nvme, but says what it really does
    checks = [{"name": "plan warning", "status": "warn", "detail": warning, "hint": ""}]
    reason = doctor.suggest(checks, "auto", "/m", dense=True)[0]["reason"]
    assert "slow for a dense model" in reason
    assert "streams experts" in doctor.suggest(checks, "auto", "/m")[0]["reason"]
