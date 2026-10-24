"""Plain helpers shared by the tests (importable, unlike conftest)."""

from __future__ import annotations

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

TOOLS = Path(__file__).resolve().parents[2]
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from fle import config  # noqa: E402

FLE_ENGINE = "/fake/fle/llama-server"
UPSTREAM_ENGINE = "/fake/upstream/llama-server"


def engine_info(path: str, flavor: str = "fle", *, ok: bool = True, spec_type: bool = True) -> dict[str, Any]:
    return {
        "ok": ok,
        "path": path,
        "version": "1 (abc)",
        "backend": "cuda" if flavor == "fle" else "cpu",
        "flavor": flavor,
        "has_fit_target": flavor == "fle",
        "has_load_mode": flavor == "fle",
        "has_cmoe": True,
        "has_spec_mtp": spec_type,
        "problem": "" if ok else "does not start",
    }


def settings(**kw: Any) -> dict[str, Any]:
    """Settings as cli._settings would produce them: the defaults plus overrides."""
    data = {key: (dict(v) if isinstance(v, dict) else v) for key, v in config.DEFAULTS.items()}
    data.update(kw)
    return data


class FakeLlama:
    """A throwaway local /v1 server. `replies` is consumed one chat completion at a time (the last one repeats)."""

    def __init__(self, models=("fake-model",), replies=None, models_status=200):
        self.models = list(models)
        self.replies = replies or [{"usage": {"completion_tokens": 10}, "timings": {"predicted_per_second": 50.0},
                                    "choices": [{"finish_reason": "stop"}]}]
        self.models_status = models_status
        self.requests: list[dict] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # silence
                pass

            def _send(self, status, body):
                raw = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                if self.path == "/v1/models":
                    if outer.models_status != 200:
                        return self._send(outer.models_status, {"error": "nope"})
                    return self._send(200, {"data": [{"id": m} for m in outer.models]})
                self._send(404, {})

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                outer.requests.append(json.loads(self.rfile.read(length)))
                index = min(len(outer.requests) - 1, len(outer.replies) - 1)
                self._send(200, outer.replies[index])

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()


def reply(tok_s, tokens=10, finish="stop", **timings):
    return {"usage": {"completion_tokens": tokens, "prompt_tokens": 5},
            "timings": {"predicted_per_second": tok_s, **timings}, "choices": [{"finish_reason": finish}]}


def cli_args(**kw: Any) -> argparse.Namespace:
    """A Namespace like the one the plan/serve commands get from argparse."""
    base = dict(extra_arg=[], env=[], gpu_ids="", mtp_head="", mtp_vocab="", no_mtp=False, spec=None, fallback=True,
                config=None, profile=None, engine=None, host=None, port=None, context=None, threads=None,
                gpu_cache_mib=None, fit_target_mib=None, pinned_load=None)
    base.update(kw)
    return argparse.Namespace(**base)
