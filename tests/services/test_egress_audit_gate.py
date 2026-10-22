"""1.20h: under enforcement, a call whose audit row cannot be queued does not leave (design.md:28).

The audit writer never raises and never blocks (a logging problem must not fail a shadow call), so a
full queue silently DROPS rows and counts them. For a purpose the Admin has chosen to ENFORCE that is
not acceptable: an unrecorded external call is the thing enforcement exists to prevent. So when the
purpose is enforced and the writer has no capacity, the call is refused and the refusal is logged at
ERROR (it cannot be recorded either). Shadow purposes and local destinations are unaffected.
"""
from __future__ import annotations

import logging

import pytest

from src.services import egress, egress_log
from src.services import egress_policy as ep
from src.services.egress import EgressDenied, Purpose

RULES = {"allow": [{"purpose": "fx_rates", "hosts": ["open.er-api.com"], "agents": ["*"]},
                   {"purpose": "supplier_research", "hosts": ["*"], "agents": ["*"]}],
         "deny": [], "enforce": ["fx_rates", "supplier_research"]}


class _Resp:
    status_code, headers, text = 200, {}, ""


@pytest.fixture
def world(monkeypatch):
    calls, events = [], []
    monkeypatch.setattr(egress.requests, "request", lambda m, u, **kw: calls.append((m, u)) or _Resp())
    monkeypatch.setattr(egress, "_resolve", lambda host: ["127.0.0.1"] if host == "localhost" else ["93.184.216.34"])
    monkeypatch.setattr(egress_log, "record", lambda **kw: events.append(kw))
    monkeypatch.setattr(ep, "_load_row", lambda: ep.Policy(version=1, rules=RULES))
    ep.invalidate()
    yield calls, events
    ep.invalidate(forget_last_good=True)


def test_has_capacity_reads_the_queue(monkeypatch):
    assert egress_log.has_capacity() is True
    monkeypatch.setattr(egress_log._queue, "full", lambda: True)
    assert egress_log.has_capacity() is False
    monkeypatch.setattr(egress_log._queue, "full", lambda: False)
    monkeypatch.setattr(egress_log, "_enabled", False)
    assert egress_log.has_capacity() is False, "a disabled writer records nothing, so it has no capacity"


def test_an_enforced_call_that_cannot_be_audited_does_not_leave(world, monkeypatch, caplog):
    calls, events = world
    monkeypatch.setattr(egress_log, "has_capacity", lambda: False)
    with caplog.at_level(logging.ERROR):
        assert egress.get("https://open.er-api.com/v6/latest/GBP", purpose=Purpose.FX_RATES, agent="fx") is None
    assert calls == []
    assert any("audit" in r.getMessage() and "open.er-api.com" in r.getMessage() for r in caplog.records)


def test_a_shadow_call_still_leaves_when_the_writer_is_full(world, monkeypatch):
    calls, events = world
    monkeypatch.setattr(egress_log, "has_capacity", lambda: False)
    assert egress.get("https://api.ollama.com/x", purpose=Purpose.MODEL_INFERENCE) is not None
    assert len(calls) == 1


def test_a_local_call_is_never_held_up_by_the_writer(world, monkeypatch):
    calls, events = world
    monkeypatch.setattr(egress_log, "has_capacity", lambda: False)
    assert egress.get("http://localhost:11434/api/tags", purpose=Purpose.FX_RATES, require_global=False) is not None


def test_an_observed_enforced_call_is_refused_the_same_way(world, monkeypatch):
    calls, events = world
    monkeypatch.setattr(egress_log, "has_capacity", lambda: False)
    ran = []
    with pytest.raises(EgressDenied):
        with egress.observe(Purpose.SUPPLIER_RESEARCH, "html.duckduckgo.com", agent="supplier_research", payload="q"):
            ran.append(1)
    assert ran == []


def test_with_capacity_an_enforced_allowed_call_goes_out(world):
    calls, events = world
    assert egress.get("https://open.er-api.com/v6/latest/GBP", purpose=Purpose.FX_RATES) is not None
    assert len(calls) == 1 and events[-1]["outcome"] == "http_200"
