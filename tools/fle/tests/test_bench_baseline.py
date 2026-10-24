from __future__ import annotations

import json

import pytest
from fle import bench, cli
from fle.util import FleError
from helpers import FakeLlama, reply


# ---- baseline comparison (pure) ----------------------------------------------------------------------------


def result(tok_s, *, model="m", url="http://127.0.0.1:8080", truncated=0):
    return {"median_tok_s": tok_s, "model": model, "url": url, "truncated": truncated}


@pytest.mark.parametrize(
    "now,before,status,delta",
    [
        (100.0, 100.0, "ok", 0.0),
        (95.0, 100.0, "ok", -5.0),
        (90.0, 100.0, "ok", -10.0),  # exactly at the limit is allowed
        (89.0, 100.0, "regression", -11.0),
        (120.0, 100.0, "ok", 20.0),
    ],
)
def test_compare_baseline_thresholds(now, before, status, delta):
    out = bench.compare_baseline(result(now), result(before), 10.0)
    assert out["status"] == status
    assert out["delta_pct"] == pytest.approx(delta)


def test_compare_baseline_custom_threshold():
    assert bench.compare_baseline(result(97.0), result(100.0), 2.0)["status"] == "regression"
    assert bench.compare_baseline(result(97.0), result(100.0), 5.0)["status"] == "ok"


@pytest.mark.parametrize(
    "cur,base,word",
    [
        (result(50, truncated=1), result(50), "current run"),
        (result(50), result(50, truncated=2), "baseline run"),
        (result(50, model="a"), result(50, model="b"), "different model"),
        (result(50, url="https://example.org"), result(50), "kind of URL"),
        (result(0), result(50), "no tokens"),
        (result(50), result(0), "no tokens"),
    ],
)
def test_compare_baseline_skips_what_cannot_be_compared(cur, base, word):
    out = bench.compare_baseline(cur, base, 10.0)
    assert out["status"] == "skipped" and out["delta_pct"] is None
    assert word in out["reason"]


def test_local_urls_are_the_same_kind():
    out = bench.compare_baseline(result(50, url="http://localhost:1"), result(50, url="http://127.0.0.1:2"))
    assert out["status"] == "ok"
    assert bench.url_kind("10.0.0.5:8080") == "http-remote"


def test_load_baseline(tmp_path):
    plain = tmp_path / "a.json"
    plain.write_text(json.dumps(result(42.0)))
    assert bench.load_baseline(str(plain))["median_tok_s"] == 42.0
    both = tmp_path / "b.json"
    both.write_text(json.dumps({"primary": result(41.0), "compare": result(10.0)}))
    assert bench.load_baseline(str(both))["median_tok_s"] == 41.0
    for name, text, word in (("c.json", "{not json", "not valid JSON"), ("d.json", "[1]", "no bench result"),
                             ("missing.json", None, "Cannot read")):
        path = tmp_path / name
        if text is not None:
            path.write_text(text)
        with pytest.raises(FleError, match=word):
            bench.load_baseline(str(path))


# ---- the command: exit codes -------------------------------------------------------------------------------


def run_cli(capsys, *argv):
    code = cli.main(["bench", *argv])
    return code, capsys.readouterr()


def test_cli_baseline_ok_and_regression_exit_codes(tmp_path, capsys):
    with FakeLlama(replies=[reply(50.0)]) as srv:
        out_file = tmp_path / "base.json"
        code, _ = run_cli(capsys, "--url", srv.url, "--runs", "2", "--warmup", "0", "--output", str(out_file))
        assert code == 0
        code, cap = run_cli(capsys, "--url", srv.url, "--runs", "2", "--warmup", "0", "--baseline", str(out_file))
        assert code == 0 and "baseline: 50.00 -> 50.00" in cap.out
        faster = tmp_path / "fast.json"  # a baseline of 100 tok/s makes today's 50 a 50 % regression
        data = json.loads(out_file.read_text())
        data["median_tok_s"] = 100.0
        faster.write_text(json.dumps(data))
        code, cap = run_cli(capsys, "--url", srv.url, "--runs", "2", "--warmup", "0", "--baseline", str(faster))
        assert code == 2 and "REGRESSION" in cap.out
        code, cap = run_cli(capsys, "--url", srv.url, "--runs", "2", "--warmup", "0", "--baseline", str(faster),
                            "--max-regression", "60")
        assert code == 0
        code, cap = run_cli(capsys, "--url", srv.url, "--runs", "1", "--warmup", "0", "--baseline", str(faster),
                            "--json")
        assert code == 2
        assert json.loads(cap.out)["baseline_comparison"]["status"] == "regression"


def test_cli_baseline_skipped_when_truncated(tmp_path, capsys):
    base = tmp_path / "t.json"
    base.write_text(json.dumps(result(500.0, model="fake-model", truncated=1)))
    with FakeLlama() as srv:
        code, cap = run_cli(capsys, "--url", srv.url, "--runs", "1", "--warmup", "0", "--baseline", str(base))
    assert code == 0
    assert "not compared" in cap.out and "truncated" in cap.out


def test_cli_rejects_negative_threshold(capsys):
    code, cap = run_cli(capsys, "--max-regression", "-1")
    assert code == 2 and "must not be negative" in cap.err
