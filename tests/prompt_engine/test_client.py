"""The client: retry what can pass, raise what cannot, never use other wording."""
from __future__ import annotations

import pytest

from services.prompt_engine import client as C
from services.prompt_engine import registry as R
from services.prompt_engine import templates as T


class _Clock:
    def __init__(self):
        self.now = 0.0
        self.waits = []

    def sleep(self, seconds):
        self.waits.append(seconds)
        self.now += seconds

    def clock(self):
        return self.now


@pytest.fixture()
def timing(monkeypatch):
    clock = _Clock()
    monkeypatch.setattr(C.db, "close", lambda: None)
    monkeypatch.setattr(C, "_allowed_seconds", lambda: 30.0)
    return clock


def _failing(times, then):
    state = {"left": times, "calls": 0}

    def call(request=None):
        state["calls"] += 1
        if state["left"] > 0:
            state["left"] -= 1
            raise ConnectionError("database not answering")
        return then

    call.state = state
    return call


def test_a_failure_that_can_pass_is_tried_again_until_it_completes(timing, monkeypatch):
    call = _failing(3, {"instructions": "x", "manifest": {"hash": "h"}})
    monkeypatch.setattr(C._render, "render", call)

    out = C.render("p", trace=C.trace("run-1", "s"), sleep=timing.sleep, clock=timing.clock)

    assert out["instructions"] == "x" and call.state["calls"] == 4
    assert timing.waits == [0.1, 0.2, 0.4]


def test_when_the_time_allowed_runs_out_it_raises_and_gives_no_wording(timing, monkeypatch):
    call = _failing(10_000, {"instructions": "never"})
    monkeypatch.setattr(C._render, "render", call)

    with pytest.raises(C.PromptUnavailable) as e:
        C.render("p", trace=C.trace("run-1", "s"), sleep=timing.sleep, clock=timing.clock)

    assert e.value.code == "RENDER_UNAVAILABLE"
    assert timing.now <= 30.0 and max(timing.waits) == 2.0
    assert isinstance(e.value.__cause__, ConnectionError)


def test_the_caller_may_allow_its_own_time(timing, monkeypatch):
    monkeypatch.setattr(C._render, "render", _failing(10_000, None))
    with pytest.raises(C.PromptUnavailable):
        C.render("p", trace=C.trace("r", "s"), seconds=1.0, sleep=timing.sleep, clock=timing.clock)
    assert timing.now <= 1.0


@pytest.mark.parametrize("refusal", [
    T.VarMissing("no value for: x"), T.VarType("x"), T.VarUndeclared("x"),
    T.FragmentMissing("x"), R.PointerUnresolved("x"), R.VersionNotFound("x"),
])
def test_a_refusal_is_raised_at_once_and_never_tried_again(timing, monkeypatch, refusal):
    calls = []

    def call(request):
        calls.append(1)
        raise refusal

    monkeypatch.setattr(C._render, "render", call)
    with pytest.raises(type(refusal)):
        C.render("p", trace=C.trace("r", "s"), sleep=timing.sleep, clock=timing.clock)
    assert len(calls) == 1 and timing.waits == []


def test_the_usual_call_does_not_look_up_the_time_allowed(monkeypatch):
    monkeypatch.setattr(C, "_allowed_seconds", lambda: pytest.fail("looked up on a clean call"))
    monkeypatch.setattr(C._render, "render", lambda request: {"instructions": "x"})
    assert C.render("p", trace=C.trace("r", "s"))["instructions"] == "x"


def test_what_is_passed_reaches_the_engine_as_the_contracts_request(monkeypatch):
    seen = {}
    monkeypatch.setattr(C._render, "render", lambda request: seen.update(request) or {})

    C.render("claims-triage", vars={"region": "EU"}, allowed_tools=("lookup",),
             obligations=["redact:customer.ssn"],
             trace=C.trace("run-7", "triage", attempt=2, decision_id="d-9"))

    assert seen["prompt_id"] == "claims-triage" and seen["vars"] == {"region": "EU"}
    assert seen["allowed_tools"] == ["lookup"] and seen["obligations"] == ["redact:customer.ssn"]
    assert seen["trace"] == {"run_id": "run-7", "step_id": "triage", "attempt": 2,
                             "decision_id": "d-9"}


def test_a_caller_with_no_run_of_its_own_still_gets_a_findable_run_id():
    a, b = C.trace(None, "s"), C.trace(None, "s")
    assert a["run_id"].startswith("run-") and a["run_id"] != b["run_id"]


def test_in_a_process_with_no_real_database_nothing_is_in_use(monkeypatch):
    monkeypatch.setattr(C.db, "available", lambda: False)
    monkeypatch.setattr(C.registry, "resolve", lambda *a: pytest.fail("touched the registry"))
    assert C.in_use("p") is None


def test_in_use_is_none_for_a_prompt_the_registry_does_not_have_in_use(monkeypatch, timing):
    monkeypatch.setattr(C.db, "available", lambda: True)

    def gone(*a):
        raise R.PointerUnresolved("x")

    monkeypatch.setattr(C.registry, "resolve", gone)
    assert C.in_use("p", sleep=timing.sleep, clock=timing.clock) is None
