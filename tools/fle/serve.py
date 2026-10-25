"""Start llama-server and, with --fallback, retry once with upstream llama.cpp if it dies while loading.

The watching is in `Supervisor` so it can be tested with a fake process and a fake health check; nothing here needs a
real server.
"""

from __future__ import annotations

import os
import subprocess
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Protocol

from .util import FleError, eprint, prepare_run_env

READY = "ready"  # /health answered OK
FAILED = "failed"  # the process ended with a non-zero code before it was ready
EXITED = "exited"  # the process ended with code 0 before it was ready
TIMEOUT = "timeout"  # still loading when the wait ran out


class Process(Protocol):
    def poll(self) -> int | None: ...
    def wait(self, timeout: float | None = None) -> int: ...
    def terminate(self) -> None: ...
    def kill(self) -> None: ...


def spawn_process(plan: Any) -> subprocess.Popen[bytes]:
    """Start the plan's server as a child that shares our stdout/stderr."""
    env = os.environ.copy()
    env.update(plan.env)
    prepare_run_env(plan.env)
    try:
        return subprocess.Popen([plan.binary, *plan.args], env=env)
    except OSError as exc:
        raise FleError(f"Cannot start {plan.binary}: {exc}", "Check the path and its shared libraries.") from None


def http_health(host: str, port: int, timeout: float = 2.0) -> bool:
    """True when llama-server's /health answers 200 (it answers 503 while the model is loading)."""
    shown = "127.0.0.1" if host in ("0.0.0.0", "", "::") else host
    shown = f"[{shown}]" if ":" in shown else shown
    try:
        with urllib.request.urlopen(f"http://{shown}:{port}/health", timeout=timeout) as response:
            return response.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


class Supervisor:
    def __init__(
        self,
        spawn: Callable[[Any], Process] = spawn_process,
        health: Callable[[Any], bool] | None = None,
        *,
        ready_timeout: float = 3600.0,
        poll_interval: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        say: Callable[[str], None] = eprint,
    ) -> None:
        self.spawn = spawn
        self.health = health or (lambda plan: http_health(plan.host, plan.port))
        self.ready_timeout = ready_timeout
        self.poll_interval = poll_interval
        self.sleep = sleep
        self.clock = clock
        self.say = say

    def wait_ready(self, proc: Process, plan: Any) -> tuple[str, int | None]:
        """Watch a starting server until it is ready, has ended, or the wait runs out."""
        deadline = self.clock() + self.ready_timeout
        while True:
            code = proc.poll()
            if code is not None:
                return (EXITED if code == 0 else FAILED), code
            if self.health(plan):
                return READY, None
            if self.clock() >= deadline:
                return TIMEOUT, None
            self.sleep(self.poll_interval)

    def _finish(self, proc: Process) -> int:
        try:
            return int(proc.wait())
        except KeyboardInterrupt:
            proc.terminate()
            try:
                return int(proc.wait(timeout=30))
            except Exception:  # noqa: BLE001 - last resort for a stuck child
                proc.kill()
                return 130

    def run(self, plan: Any, make_fallback: Callable[[str], Any]) -> int:
        """Run `plan`; if it fails while loading, build a fallback plan with `make_fallback(reason)` and run that once.

        `make_fallback` raises FleError when there is nothing to fall back to. The fallback is not watched further:
        a second failure is the exit code of the second server.
        """
        proc = self.spawn(plan)
        try:
            state, code = self.wait_ready(proc, plan)
        except KeyboardInterrupt:
            proc.terminate()
            return 130
        if state == READY:
            self.say("fle: the server is ready.")
            return self._finish(proc)
        if state == EXITED:
            return int(code or 0)
        if state == TIMEOUT:
            self.say("fle: still loading after the wait limit; leaving the server running without fallback.")
            return self._finish(proc)
        reason = f"the server exited with code {code} before it was ready"
        try:
            fallback = make_fallback(reason)
        except FleError as exc:
            raise FleError(f"The server failed to start ({reason}), and no fallback is possible: {exc}", exc.hint) from None
        self.say(f"fle: {reason}.")
        self.say(f"fle: restarting once with upstream llama.cpp ({fallback.binary}, profile off, no FLE features).")
        second = self.spawn(fallback)
        return self._finish(second)
