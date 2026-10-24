from __future__ import annotations

import pytest
from fle import plan
from fle.util import FleError
from helpers import FLE_ENGINE, settings


def build(model, machine, profile="auto", **kw):
    extra = kw.pop("settings", {})
    return plan.build(settings(profile=profile, **extra), model, engine_path=FLE_ENGINE,
                      model_path=str(model.first), **kw)


def test_no_mtp_flag_with_spec_mtp_contradicts(make_model, machine):
    model = make_model(experts=8, sidecars=True)
    with pytest.raises(FleError, match="contradict"):
        build(model, machine, "auto", spec="mtp", no_mtp=True)
    with pytest.raises(FleError, match="contradict"):
        build(model, machine, "auto", spec="mtp", settings={"mtp": False})


def test_missing_sidecar_error_points_to_fetch_command(make_model, machine):
    model = make_model(experts=8)
    with pytest.raises(FleError) as info:
        build(model, machine, "auto", spec="mtp")
    assert "fle fetch-sidecars" in info.value.hint
    assert str(model.first) in info.value.hint
