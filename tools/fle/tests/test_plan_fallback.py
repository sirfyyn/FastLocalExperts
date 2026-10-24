"""--fallback at plan time (cli._plan_for)."""

from __future__ import annotations

import pytest
from fle import cli
from fle.util import FleError
from helpers import FLE_ENGINE, UPSTREAM_ENGINE, cli_args as args, settings


def test_plan_for_without_failure_ignores_fallback(moe_model, machine):
    p = cli._plan_for(args(), settings(engine=FLE_ENGINE), str(moe_model.first))
    assert not p.fallback and p.binary == FLE_ENGINE


def test_plan_for_falls_back_to_upstream_when_the_fle_plan_fails(moe_model, machine):
    machine.gpus = []  # the FLE engine needs a GPU -> plan fails
    p = cli._plan_for(args(), settings(engine=FLE_ENGINE), str(moe_model.first))
    assert p.fallback
    assert p.binary == UPSTREAM_ENGINE
    assert p.profile == "off" and p.spec == "none"
    assert not any(k.startswith("FLE_") for k in p.env)
    assert p.notes[0].startswith("FLE plan failed:")
    assert "Falling back to upstream" in p.notes[1]


def test_plan_for_without_fallback_flag_raises(moe_model, machine):
    machine.gpus = []
    with pytest.raises(FleError, match="needs an NVIDIA GPU"):
        cli._plan_for(args(fallback=False), settings(engine=FLE_ENGINE), str(moe_model.first))


def test_plan_for_fallback_without_an_upstream_engine_explains(moe_model, machine):
    machine.gpus = []
    del machine.engines[UPSTREAM_ENGINE]
    with pytest.raises(FleError) as info:
        cli._plan_for(args(), settings(engine=FLE_ENGINE), str(moe_model.first))
    assert "FLE plan failed" in str(info.value)
    assert "No upstream llama.cpp" in info.value.hint


def test_plan_for_never_drops_an_explicit_spec(moe_model, machine):
    machine.gpus = []
    with pytest.raises(FleError, match="needs an NVIDIA GPU"):
        cli._plan_for(args(spec="ngram-mod"), settings(engine=FLE_ENGINE), str(moe_model.first))


def test_upstream_engine_can_come_from_the_environment(moe_model, machine, monkeypatch):
    machine.gpus = []
    machine.engines["/other/llama-server"] = dict(machine.engines[UPSTREAM_ENGINE], path="/other/llama-server")
    del machine.engines[UPSTREAM_ENGINE]
    monkeypatch.setenv("FLE_UPSTREAM_ENGINE", "/other/llama-server")
    assert cli._plan_for(args(), settings(engine=FLE_ENGINE), str(moe_model.first)).binary == "/other/llama-server"
