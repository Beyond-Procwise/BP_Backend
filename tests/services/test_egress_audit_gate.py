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
    # Process-wide writer state: earlier test files let the REAL writer thread flush against the dead
    # test database, which sets _writer_failing for the whole process. Start each test from "healthy".
    monkeypatch.setattr(egress_log, "_writer_failing", False)
    ep.invalidate()
    yield calls, events
    ep.invalidate(forget_last_good=True)


class _Q:
    def __init__(self, full):
        self._full = full

    def full(self):
        return self._full


def test_has_capacity_reads_the_queue(monkeypatch):
    monkeypatch.setattr(egress_log, "_writer_failing", False)
    monkeypatch.setattr(egress_log, "_queue", _Q(full=False))      # the module's queue, never patched in place
    assert egress_log.has_capacity() is True
    monkeypatch.setattr(egress_log, "_queue", _Q(full=True))
    assert egress_log.has_capacity() is False
    monkeypatch.setattr(egress_log, "_queue", _Q(full=False))
    monkeypatch.setattr(egress_log, "_enabled", False)
    assert egress_log.has_capacity() is False, "a disabled writer records nothing, so it has no capacity"


def test_an_enforced_call_that_cannot_be_audited_does_not_leave(world, monkeypatch, caplog):
    calls, events = world
    monkeypatch.setattr(egress_log, "capacity", lambda: (False, "the egress audit queue is full"))
    with caplog.at_level(logging.ERROR):
        assert egress.get("https://open.er-api.com/v6/latest/GBP", purpose=Purpose.FX_RATES, agent="fx") is None
    assert calls == []
    assert any("audit" in r.getMessage() and "open.er-api.com" in r.getMessage() for r in caplog.records)


def test_a_shadow_call_still_leaves_when_the_writer_is_full(world, monkeypatch):
    calls, events = world
    monkeypatch.setattr(egress_log, "capacity", lambda: (False, "the egress audit queue is full"))
    assert egress.get("https://api.ollama.com/x", purpose=Purpose.MODEL_INFERENCE) is not None
    assert len(calls) == 1


def test_a_local_call_is_never_held_up_by_the_writer(world, monkeypatch):
    calls, events = world
    monkeypatch.setattr(egress_log, "capacity", lambda: (False, "the egress audit queue is full"))
    assert egress.get("http://localhost:11434/api/tags", purpose=Purpose.FX_RATES, require_global=False) is not None


def test_an_observed_enforced_call_is_refused_the_same_way(world, monkeypatch):
    calls, events = world
    monkeypatch.setattr(egress_log, "capacity", lambda: (False, "the egress audit queue is full"))
    ran = []
    with pytest.raises(EgressDenied):
        with egress.observe(Purpose.SUPPLIER_RESEARCH, "html.duckduckgo.com", agent="supplier_research", payload="q"):
            ran.append(1)
    assert ran == []


def test_with_capacity_an_enforced_allowed_call_goes_out(world, monkeypatch):
    calls, events = world
    # Pinned, not inferred: the real writer thread may flip the flag at any moment while the dead
    # database is unreachable; this test is about what happens WITH capacity.
    monkeypatch.setattr(egress_log, "capacity", lambda: (True, ""))
    assert egress.get("https://open.er-api.com/v6/latest/GBP", purpose=Purpose.FX_RATES) is not None
    assert len(calls) == 1 and events[-1]["outcome"] == "http_200"


# ---- the honest residual (review of 8f8a17af): a failing WRITER, and the refusals themselves ----

def _conn_that(fails: bool):
    class _Cur:
        def executemany(self, sql, rows):
            if fails:
                raise RuntimeError("relation does not exist")

    class _Conn:
        autocommit = True

        def cursor(self):
            return _Cur()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False
    return _Conn()


def test_a_failing_writer_removes_capacity_until_a_flush_succeeds(monkeypatch):
    import src.services.db as db
    monkeypatch.setattr(egress_log, "_writer_failing", False)     # process-wide flag: restored at teardown
    monkeypatch.setattr(egress_log, "_take_drops", lambda: 0)
    monkeypatch.setattr(egress_log, "_take_refused", lambda: 0)
    event = {"occurred_at": None, "purpose": "fx_rates", "destination": "x", "method": "GET", "outcome": "http_200",
             "detail": None, "payload_sha256": None, "payload_bytes": None}
    monkeypatch.setattr(db, "get_conn", lambda: _conn_that(fails=True))
    assert egress_log.flush([dict(event)]) == 0
    ok, why = egress_log.capacity()
    assert ok is False and "writer" in why and egress_log.has_capacity() is False
    monkeypatch.setattr(db, "get_conn", lambda: _conn_that(fails=False))
    assert egress_log.flush([dict(event)]) == 1
    assert egress_log.has_capacity() is True


def test_a_switched_off_writer_says_so_not_full(monkeypatch):
    monkeypatch.setattr(egress_log, "_enabled", False)
    ok, why = egress_log.capacity()
    assert ok is False and "switched off" in why


def test_refused_unaudited_calls_become_their_own_row_when_the_writer_recovers(monkeypatch):
    written = {}

    class _Cur:
        def executemany(self, sql, rows):
            written["rows"] = rows

    class _Conn:
        autocommit = True

        def cursor(self):
            return _Cur()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False
    import src.services.db as db
    monkeypatch.setattr(db, "get_conn", lambda: _Conn())
    monkeypatch.setattr(egress_log, "_take_drops", lambda: 0)
    monkeypatch.setattr(egress_log, "_refused_unaudited", 0)     # process-wide counter: start clean
    egress_log.note_refused_unaudited()
    egress_log.note_refused_unaudited()
    egress_log.flush([])
    rows = written["rows"]
    assert [r[5] for r in rows] == ["calls_refused_unaudited"]
    assert "2 enforced call(s) refused" in rows[0][6]


def test_the_gate_counts_what_it_refuses(world, monkeypatch):
    calls, events = world
    monkeypatch.setattr(egress_log, "capacity", lambda: (False, "the egress audit queue is full"))
    counted = []
    monkeypatch.setattr(egress_log, "note_refused_unaudited", lambda: counted.append(1))
    assert egress.get("https://open.er-api.com/v6/latest/GBP", purpose=Purpose.FX_RATES) is None
    assert counted == [1]
