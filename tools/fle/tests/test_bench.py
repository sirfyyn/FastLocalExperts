from __future__ import annotations

import pytest
from fle import bench
from fle.util import FleError
from helpers import FakeLlama, reply


def test_run_median_and_warmup_excluded():
    replies = [reply(1.0), reply(40.0), reply(60.0), reply(50.0)]  # first is the warmup
    with FakeLlama(replies=replies) as srv:
        result = bench.run(srv.url, runs=3, warmup=1)
        assert len(srv.requests) == 4
    assert result["model"] == "fake-model"  # taken from /v1/models
    assert [r["warmup"] for r in result["runs"]] == [False] * 3
    assert result["median_tok_s"] == 50.0
    assert (result["min_tok_s"], result["max_tok_s"]) == (40.0, 60.0)
    assert result["truncated"] == 0
    assert srv.requests[0]["temperature"] == 0 and srv.requests[0]["model"] == "fake-model"


def test_run_counts_truncated_runs_and_draft_acceptance():
    replies = [reply(30.0, finish="length", draft_n=10, draft_n_accepted=7), reply(30.0, finish="stop")]
    with FakeLlama(replies=replies) as srv:
        result = bench.run(srv.url, runs=2, warmup=0)
    assert result["truncated"] == 1
    assert result["runs"][0]["accept"] == 0.7
    assert result["runs"][1]["accept"] is None


def test_run_falls_back_to_wall_clock_without_timings():
    body = {"usage": {"completion_tokens": 20}, "choices": [{"finish_reason": "stop"}]}
    with FakeLlama(replies=[body]) as srv:
        result = bench.run(srv.url, runs=1, warmup=0)
    assert result["runs"][0]["tok_s"] > 0
    assert result["runs"][0]["completion_tokens"] == 20


def test_run_explicit_model_skips_models_endpoint():
    with FakeLlama(models=(), models_status=500) as srv:
        result = bench.run(srv.url, model="given", runs=1, warmup=0)
    assert result["model"] == "given"


def test_run_adds_scheme_and_reports_url():
    with FakeLlama() as srv:
        result = bench.run(srv.url.removeprefix("http://") + "/", runs=1, warmup=0)
    assert result["url"] == srv.url


def test_model_errors():
    with FakeLlama(models=()) as srv:
        with pytest.raises(FleError, match="named no model"):
            bench._model(srv.url, "")
    with FakeLlama(models=("   ",)) as srv:
        with pytest.raises(FleError, match="empty model name"):
            bench._model(srv.url, "")
    with FakeLlama(models_status=503) as srv:
        with pytest.raises(FleError, match="HTTP 503"):
            bench._model(srv.url, "")
    with pytest.raises(FleError, match="Cannot reach"):
        bench._model("http://127.0.0.1:9", "")  # discard port, nothing listens


def test_model_explicit_name_is_returned_unchanged():
    assert bench._model("http://unused", "abc") == "abc"
