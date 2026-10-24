"""--fallback: at plan time (cli._plan_for) and while the server loads (serve.Supervisor)."""

from __future__ import annotations

import pytest
from fle import cli, serve
from fle.util import FleError
from helpers import FLE_ENGINE, UPSTREAM_ENGINE, cli_args as args, settings


# ---- the supervisor (fake process, fake health) ------------------------------------------------------------


class FakeProc:
    """polls as running for `alive_polls` calls, then reports `code`."""

    def __init__(self, name, code, alive_polls=0, log=None):
        self.name, self.code, self.alive = name, code, alive_polls
        self.log = log if log is not None else []
        self.terminated = False

    def poll(self):
        if self.alive > 0:
            self.alive -= 1
            return None
        return self.code

    def wait(self, timeout=None):
        self.log.append(f"wait {self.name}")
        return self.code if self.code is not None else 0

    def terminate(self):
        self.terminated = True

    def kill(self):
        pass


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def make_supervisor(procs, health, said, clock=None, **kw):
    clock = clock or Clock()
    spawned = []

    def spawn(plan):
        proc = procs[len(spawned)]
        spawned.append((plan, proc))
        return proc

    sup = serve.Supervisor(spawn, health, sleep=clock.sleep, clock=clock, say=said.append, **kw)
    return sup, spawned


class P:
    def __init__(self, binary, fallback=False):
        self.binary, self.fallback, self.host, self.port = binary, fallback, "127.0.0.1", 8080


def test_ready_server_is_just_waited_for():
    said = []
    sup, spawned = make_supervisor([FakeProc("fle", code=0, alive_polls=3)], lambda plan: True, said)
    called = []
    assert sup.run(P("fle-bin"), lambda reason: called.append(reason)) == 0
    assert called == [] and len(spawned) == 1
    assert said == ["fle: the server is ready."]


def test_first_start_fails_second_start_uses_upstream_and_a_hint_is_printed():
    said = []
    first = FakeProc("fle", code=1, alive_polls=2)  # dies while loading
    second = FakeProc("upstream", code=0)
    sup, spawned = make_supervisor([first, second], lambda plan: False, said)
    reasons = []

    def make_fallback(reason):
        reasons.append(reason)
        return P("upstream-bin", fallback=True)

    assert sup.run(P("fle-bin"), make_fallback) == 0
    assert [plan.binary for plan, _ in spawned] == ["fle-bin", "upstream-bin"]
    assert reasons == ["the server exited with code 1 before it was ready"]
    assert said[0] == "fle: the server exited with code 1 before it was ready."
    assert "restarting once with upstream llama.cpp (upstream-bin, profile off" in said[1]


def test_second_failure_is_not_retried_and_returns_its_code():
    said = []
    sup, spawned = make_supervisor([FakeProc("a", 1), FakeProc("b", 7)], lambda plan: False, said)
    assert sup.run(P("fle-bin"), lambda reason: P("up", True)) == 7
    assert len(spawned) == 2


def test_no_fallback_possible_gives_one_error_with_the_reason():
    said = []
    sup, spawned = make_supervisor([FakeProc("a", 9)], lambda plan: False, said)

    def make_fallback(reason):
        raise FleError("No upstream llama.cpp was found.", "Install one.")

    with pytest.raises(FleError) as info:
        sup.run(P("fle-bin"), make_fallback)
    assert "exited with code 9" in str(info.value) and "No upstream llama.cpp" in str(info.value)
    assert info.value.hint == "Install one."
    assert len(spawned) == 1


def test_exit_code_zero_before_ready_is_not_a_failure():
    sup, spawned = make_supervisor([FakeProc("a", 0)], lambda plan: False, [])
    assert sup.run(P("x"), lambda reason: pytest.fail("no fallback expected")) == 0


def test_crash_after_ready_is_not_restarted():
    sup, spawned = make_supervisor([FakeProc("a", 139, alive_polls=2)], lambda plan: True, [])
    assert sup.run(P("x"), lambda reason: pytest.fail("no fallback after ready")) == 139
    assert len(spawned) == 1


def test_timeout_leaves_the_server_running():
    said = []
    clock = Clock()
    proc = FakeProc("a", 0, alive_polls=10**9)
    sup, spawned = make_supervisor([proc], lambda plan: False, said, clock=clock, ready_timeout=5.0, poll_interval=1.0)
    assert sup.run(P("x"), lambda reason: pytest.fail("no fallback on timeout")) == 0
    assert clock.now >= 5.0
    assert any("still loading" in line for line in said)


def test_wait_ready_states():
    sup, _ = make_supervisor([], lambda plan: False, [])
    assert sup.wait_ready(FakeProc("a", 3), None) == (serve.FAILED, 3)
    assert sup.wait_ready(FakeProc("a", 0), None) == (serve.EXITED, 0)
    sup2, _ = make_supervisor([], lambda plan: True, [])
    assert sup2.wait_ready(FakeProc("a", 0, alive_polls=5), None) == (serve.READY, None)


# ---- cli glue ---------------------------------------------------------------------------------------------


def test_serve_supervised_builds_the_upstream_plan(moe_model, machine):
    said = []
    plan = cli._plan_for(args(fallback=False), settings(engine=FLE_ENGINE), str(moe_model.first))
    sup, spawned = make_supervisor([FakeProc("fle", 1), FakeProc("up", 0)], lambda p: False, said)
    code = cli._serve_supervised(args(), settings(engine=FLE_ENGINE), plan, sup)
    assert code == 0
    first, second = (p for p, _ in spawned)
    assert first.binary == FLE_ENGINE and not first.fallback
    assert second.binary == UPSTREAM_ENGINE and second.fallback and second.profile == "off"
    assert second.notes[0].startswith("The FLE server failed: the server exited with code 1")
    assert any("restarting once" in line for line in said)


def test_serve_supervised_refuses_to_drop_an_explicit_spec(moe_model, machine):
    plan = cli._plan_for(args(fallback=False, spec="ngram-mod"), settings(engine=FLE_ENGINE), str(moe_model.first))
    sup, spawned = make_supervisor([FakeProc("fle", 1)], lambda p: False, [])
    with pytest.raises(FleError, match="--spec ngram-mod"):
        cli._serve_supervised(args(spec="ngram-mod"), settings(engine=FLE_ENGINE), plan, sup)
    assert len(spawned) == 1


def test_serve_command_uses_the_supervisor_only_with_fallback(moe_model, machine, monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "_serve_supervised", lambda a, s, plan, sup=None: calls.append(plan.binary) or 0)
    monkeypatch.setattr(cli.os, "execvpe", lambda *a: calls.append("exec"))
    base = ["serve", str(moe_model.first), "--engine", FLE_ENGINE, "--no-preflight"]
    assert cli.main(base + ["--fallback"]) == 0
    assert calls == [FLE_ENGINE]
    calls.clear()
    cli.main(base)
    assert calls == ["exec"]
