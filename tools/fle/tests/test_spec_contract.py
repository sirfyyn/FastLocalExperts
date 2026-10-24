"""The shared speculation contract (spec_cases.json) run against the real plan logic.

The same JSON lives in the FLE Studio repo. Do not edit it here; change both copies and both implementations together.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fle import plan
from fle.util import FleError
from helpers import FLE_ENGINE, settings

CASES = json.loads((Path(__file__).with_name("spec_cases.json")).read_text(encoding="utf-8"))["cases"]


def test_contract_file_is_what_we_expect():
    ids = [c["id"] for c in CASES]
    assert len(ids) == len(set(ids)) >= 11
    for case in CASES:
        assert set(case) >= {"id", "spec", "sidecars", "mtp_enabled", "engine_spec_type", "expect"}
        assert set(case["expect"]) in ({"effective"}, {"error"})


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_spec_case(case, make_model, machine):
    model = make_model(experts=8, sidecars=bool(case["sidecars"]))
    machine.engines[FLE_ENGINE]["has_spec_mtp"] = bool(case["engine_spec_type"])
    conf = settings(profile="auto", spec=case["spec"], mtp=bool(case["mtp_enabled"]))
    expect = case["expect"]
    if "error" in expect:
        with pytest.raises(FleError) as info:
            plan.build(conf, model, engine_path=FLE_ENGINE, model_path=str(model.first))
        text = f"{info.value} {info.value.hint}".lower()
        assert expect["error"] in text
    else:
        built = plan.build(conf, model, engine_path=FLE_ENGINE, model_path=str(model.first))
        assert built.spec == expect["effective"]
        has_spec_arg = "--spec-type" in built.args
        assert has_spec_arg == (expect["effective"] != "none")
