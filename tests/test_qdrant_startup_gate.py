"""The startup gate that stops the API overtaking its vector store.

The bug these cover: on a cold boot the API built its Qdrant client, created
the learning collection and synced the static policy corpus before the Qdrant
container was listening. Every failure was caught and logged, so the box came
up healthy with an empty vector store and nothing ever retried.
"""

import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from src.services import qdrant_health


@pytest.fixture
def no_sleep(monkeypatch):
    """Collect the sleeps instead of serving them, so the clock is ours."""
    slept = []
    monkeypatch.setattr(qdrant_health.time, "sleep", lambda s: slept.append(s))
    return slept


def _health_sequence(monkeypatch, answers):
    """Make is_qdrant_healthy return each answer in turn, then the last one."""
    calls = {"n": 0}

    def fake(url="http://localhost:6333", api_key=None):
        i = min(calls["n"], len(answers) - 1)
        calls["n"] += 1
        return answers[i]

    monkeypatch.setattr(qdrant_health, "is_qdrant_healthy", fake)
    return calls


def _never_recover(monkeypatch):
    """Record whether recovery was reached, and refuse to actually restart."""
    reached = {"ensure": 0}

    def fake_ensure(url=None, max_wait=120, api_key=None):
        reached["ensure"] += 1
        return False

    monkeypatch.setattr(qdrant_health, "ensure_qdrant_available", fake_ensure)
    return reached


# ---------------------------------------------------------------------------
# await_qdrant_ready
# ---------------------------------------------------------------------------

def test_already_up_returns_immediately(monkeypatch, no_sleep):
    _health_sequence(monkeypatch, [True])
    reached = _never_recover(monkeypatch)

    assert qdrant_health.await_qdrant_ready(wait_seconds=90) is True
    assert no_sleep == [], "a healthy Qdrant must not delay the boot at all"
    assert reached["ensure"] == 0


def test_waits_for_a_late_arriving_qdrant(monkeypatch, no_sleep):
    # Down for the first two checks, listening on the third.
    _health_sequence(monkeypatch, [False, False, True])
    reached = _never_recover(monkeypatch)

    assert qdrant_health.await_qdrant_ready(wait_seconds=90, grace_seconds=30) is True
    assert no_sleep, "the gate must actually wait rather than fall straight through"
    assert reached["ensure"] == 0, "a container still booting must not be restarted"


def test_escalates_to_recovery_only_after_the_grace_period(monkeypatch):
    """The grace period comes first; restarting a booting container just
    lengthens the outage."""
    _health_sequence(monkeypatch, [False])
    order = []

    def fake_sleep(_s):
        order.append("poll")

    def fake_ensure(url=None, max_wait=120, api_key=None):
        order.append("recover")
        return True

    import itertools
    clock = itertools.count(0.0, 5.0)  # 5s per call to monotonic()
    monkeypatch.setattr(qdrant_health.time, "sleep", fake_sleep)
    monkeypatch.setattr(qdrant_health.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(qdrant_health, "ensure_qdrant_available", fake_ensure)

    assert qdrant_health.await_qdrant_ready(wait_seconds=90, grace_seconds=30) is True
    assert "recover" in order, "a Qdrant that never arrives must trigger recovery"
    assert order.index("poll") < order.index("recover")


def test_reports_failure_when_qdrant_never_comes_back(monkeypatch, no_sleep):
    _health_sequence(monkeypatch, [False])
    reached = _never_recover(monkeypatch)
    import itertools
    clock = itertools.count(0.0, 10.0)
    monkeypatch.setattr(qdrant_health.time, "monotonic", lambda: next(clock))

    assert qdrant_health.await_qdrant_ready(wait_seconds=60, grace_seconds=20) is False
    assert reached["ensure"] == 1


def test_wait_of_zero_disables_the_gate(monkeypatch, no_sleep):
    _health_sequence(monkeypatch, [False])
    reached = _never_recover(monkeypatch)

    assert qdrant_health.await_qdrant_ready(wait_seconds=0) is False
    assert no_sleep == []
    assert reached["ensure"] == 0, "a disabled gate must not restart anything either"


# ---------------------------------------------------------------------------
# The call site: AgentNick must consult the gate before it uses the client.
# ---------------------------------------------------------------------------

def _agent_nick_class():
    from agents.base_agent import AgentNick

    return AgentNick


def test_startup_blocks_on_the_gate_and_survives_a_dead_store(monkeypatch, caplog):
    AgentNick = _agent_nick_class()
    seen = {}

    def fake_await(url, api_key=None, wait_seconds=None, grace_seconds=None):
        seen.update(url=url, wait=wait_seconds, grace=grace_seconds)
        return False

    monkeypatch.setattr(qdrant_health, "await_qdrant_ready", fake_await)

    fake_self = SimpleNamespace(
        settings=SimpleNamespace(
            qdrant_startup_wait_seconds=45,
            qdrant_startup_grace_seconds=15,
            qdrant_url="http://localhost:6333",
            qdrant_api_key=None,
        )
    )

    with caplog.at_level("ERROR"):
        AgentNick._await_qdrant_ready(fake_self)

    assert seen == {"url": "http://localhost:6333", "wait": 45, "grace": 15}
    # It must not raise: a vector store being down cannot stop the API booting.
    assert "will NOT be ingested" in caplog.text


def test_a_failing_readiness_check_never_stops_the_boot(monkeypatch):
    AgentNick = _agent_nick_class()

    def exploding(*_a, **_k):
        raise RuntimeError("health check itself is broken")

    monkeypatch.setattr(qdrant_health, "await_qdrant_ready", exploding)

    fake_self = SimpleNamespace(
        settings=SimpleNamespace(
            qdrant_startup_wait_seconds=30,
            qdrant_startup_grace_seconds=10,
            qdrant_url="http://localhost:6333",
            qdrant_api_key=None,
        )
    )

    AgentNick._await_qdrant_ready(fake_self)  # must not raise


def test_the_gate_runs_before_anything_that_uses_qdrant():
    """Ordering is the whole fix.

    Both Qdrant consumers in the constructor -- the learning repository and
    the static policy sync -- swallow their own connection errors, so a gate
    placed after either of them would pass every other test here while the
    box still booted with an empty vector store. This asserts the position,
    not just the existence, of the call.
    """
    import inspect

    src = inspect.getsource(_agent_nick_class().__init__)

    gate = src.find("self._await_qdrant_ready()")
    assert gate != -1, "AgentNick.__init__ no longer consults the readiness gate"

    for consumer in ("LearningRepository(", "_initialise_static_policy_corpus("):
        at = src.find(consumer)
        assert at != -1, f"{consumer} moved out of __init__ -- re-check the gate"
        assert gate < at, (
            f"the readiness gate must run before {consumer}; a consumer that "
            "runs first fails silently against a Qdrant that is not up yet"
        )
