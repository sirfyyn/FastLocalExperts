"""A small OpenAI-compatible benchmark: tokens per second and latency through /v1."""

from __future__ import annotations

import json
import statistics
import time
import urllib.error
import urllib.request
from typing import Any

from .util import FleError, human_int


def _request(url: str, payload: dict[str, Any] | None = None, timeout: float = 600.0) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, method="POST" if data else "GET")
    if data:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:500]
        raise FleError(f"{url} returned HTTP {exc.code}: {body}", "Check the server log.") from None
    except (OSError, ValueError) as exc:
        raise FleError(f"Cannot reach {url}: {exc}", "Start the server first, or pass --url.") from None


def _model(base: str, wanted: str) -> str:
    if wanted:
        return wanted
    body = _request(base.rstrip("/") + "/v1/models")
    items = body.get("data") or []
    if not items:
        raise FleError(f"The server at {base} named no model.", "Pass --model NAME.")
    name = str(items[0].get("id") or "").strip()
    if not name:
        raise FleError(f"The server at {base} returned an empty model name.", "Pass --model NAME.")
    return name


def run(
    url: str,
    model: str = "",
    prompt: str = "Write a short paragraph about why local language models are useful.",
    max_tokens: int = 128,
    runs: int = 3,
    warmup: int = 1,
    timeout: float = 600.0,
) -> dict[str, Any]:
    base = url.rstrip("/")
    if not base.startswith(("http://", "https://")):
        base = "http://" + base
    name = _model(base, model)
    payload = {
        "model": name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max(1, max_tokens),
        "temperature": 0,
        "stream": False,
    }
    samples: list[dict[str, Any]] = []
    for index in range(max(0, warmup) + max(1, runs)):
        started = time.perf_counter()
        body = _request(base + "/v1/chat/completions", payload, timeout=timeout)
        elapsed = time.perf_counter() - started
        usage = body.get("usage") or {}
        timings = body.get("timings") or {}
        choice = (body.get("choices") or [{}])[0]
        completion = int(usage.get("completion_tokens") or timings.get("predicted_n") or 0)
        predicted_ms = float(timings.get("predicted_ms") or 0)
        tok_s = float(timings.get("predicted_per_second") or 0)
        if not tok_s and completion > 0:
            tok_s = completion / elapsed if elapsed > 0 else 0.0
        draft_n = int(timings.get("draft_n") or 0)
        sample = {
            "warmup": index < warmup,
            "latency_ms": elapsed * 1000.0,
            "completion_tokens": completion,
            "prompt_tokens": int(usage.get("prompt_tokens") or timings.get("prompt_n") or 0),
            "tok_s": tok_s,
            "predicted_ms": predicted_ms,
            "prompt_tok_s": float(timings.get("prompt_per_second") or 0),
            "finish_reason": choice.get("finish_reason") or "",
            "accept": round(int(timings.get("draft_n_accepted") or 0) / draft_n, 4) if draft_n else None,
        }
        if not sample["warmup"]:
            samples.append(sample)
    if not samples:
        raise FleError("No measured run finished.", "Check the server.")
    tok_s = [s["tok_s"] for s in samples if s["tok_s"] > 0]
    latency = [s["latency_ms"] for s in samples]
    return {
        "url": base,
        "model": name,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "runs": samples,
        "median_tok_s": statistics.median(tok_s) if tok_s else 0.0,
        "median_latency_ms": statistics.median(latency),
        "min_tok_s": min(tok_s) if tok_s else 0.0,
        "max_tok_s": max(tok_s) if tok_s else 0.0,
        "truncated": sum(1 for sample in samples if sample.get("finish_reason") == "length"),
    }


def print_result(result: dict[str, Any]) -> None:
    print(f"model: {result['model']}  url: {result['url']}")
    print(f"{'run':>3}  {'tokens':>7}  {'tok/s':>8}  {'latency':>10}  {'finish':<8}  {'accept':>7}")
    for i, sample in enumerate(result["runs"], 1):
        accept = "-" if sample.get("accept") is None else f"{sample['accept']:.3f}"
        print(
            f"{i:>3}  {human_int(sample['completion_tokens']):>7}  "
            f"{sample['tok_s']:>8.2f}  {sample['latency_ms']:>8.0f} ms  "
            f"{str(sample.get('finish_reason') or '-'):<8}  {accept:>7}"
        )
    print(
        f"median: {result['median_tok_s']:.2f} tok/s, {result['median_latency_ms']:.0f} ms "
        f"(min {result['min_tok_s']:.2f}, max {result['max_tok_s']:.2f})"
    )
    if result.get("truncated"):
        print(f"warning: {result['truncated']} run(s) hit the token limit; the answers may be cut off.")
