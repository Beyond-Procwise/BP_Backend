"""1.20i: the audit writer makes sure its table has today's columns before its first write, and a batch
it cannot write is admitted as a gap, not lost in silence.

At the 2026-10-22 restart the first three outbound calls were flushed BEFORE main.py's schema step added
the new columns: the INSERT failed on "column agent does not exist", the batch was discarded with a
DEBUG line, and the writer sat in its failing state. Two rules now: (1) the writer runs ensure_schema()
itself, once, before its first INSERT, so it never depends on who called what first at startup;
(2) a batch lost to a failed write is counted as dropped, so the next successful flush writes the
same ``events_dropped`` row a full queue would.
"""
from __future__ import annotations

import pytest

from src.services import egress_log


def _event(**kw):
    e = {"occurred_at": None, "purpose": "fx_rates", "destination": "x", "method": "GET", "outcome": "http_200",
         "detail": None, "payload_sha256": None, "payload_bytes": None}
    e.update(kw)
    return e


class _Cur:
    def __init__(self, log, fail_insert=False):
        self.log, self.fail_insert = log, fail_insert

    def execute(self, sql, params=None):
        self.log.append(("execute", " ".join(sql.split())[:40]))

    def executemany(self, sql, rows):
        self.log.append(("executemany", len(rows)))
        if self.fail_insert:
            raise RuntimeError('column "agent" of relation "bp_egress_event" does not exist')


class _Conn:
    autocommit = True

    def __init__(self, log, fail_insert=False):
        self.log, self.fail_insert = log, fail_insert

    def cursor(self):
        return _Cur(self.log, self.fail_insert)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    """Process-wide writer state starts clean, and the REAL writer thread (started by earlier test
    files) is given an empty queue of its own so it cannot flush through our fake connection.
    Known residual: that thread's idle tick still calls _take_drops(); if a 2 s tick lands between
    two flushes of a test that carries a drop count, it can steal the count (microsecond window)."""
    import queue
    monkeypatch.setattr(egress_log, "_writer_failing", False)
    monkeypatch.setattr(egress_log, "_schema_ready", False)
    monkeypatch.setattr(egress_log, "_dropped", 0)
    monkeypatch.setattr(egress_log, "_refused_unaudited", 0)
    monkeypatch.setattr(egress_log, "_queue", queue.Queue(maxsize=10))
    monkeypatch.setattr(egress_log, "_take_refused", lambda: 0)


def test_the_first_flush_runs_the_schema_before_it_writes(monkeypatch):
    import src.services.db as db
    log = []
    monkeypatch.setattr(db, "get_conn", lambda: _Conn(log))
    monkeypatch.setattr(egress_log, "_take_drops", lambda: 0)
    assert egress_log.flush([_event()]) == 1
    kinds = [k for k, _ in log]
    assert kinds.index("execute") < kinds.index("executemany"), "DDL (ALTER ... ADD COLUMN IF NOT EXISTS) first"
    assert log[0][1].startswith("CREATE SCHEMA IF NOT EXISTS proc")
    log.clear()
    assert egress_log.flush([_event()]) == 1
    assert [k for k, _ in log] == ["executemany"], "once per process, not once per flush"


def test_a_failed_schema_step_is_retried_on_the_next_flush(monkeypatch):
    import src.services.db as db
    calls = []

    def boom():
        calls.append(1)
        raise RuntimeError("db down")
    monkeypatch.setattr(db, "get_conn", boom)
    assert egress_log.flush([_event()]) == 0
    assert egress_log._schema_ready is False
    log = []
    monkeypatch.setattr(db, "get_conn", lambda: _Conn(log))
    assert egress_log.flush([_event()]) >= 1          # the event, plus the gap row for the one lost above
    assert [k for k, _ in log][0] == "execute", "the schema step ran on the retry"


def test_a_batch_lost_to_a_failed_write_is_admitted_as_a_gap_row(monkeypatch):
    import src.services.db as db
    monkeypatch.setattr(egress_log, "_schema_ready", True)
    log_fail = []
    monkeypatch.setattr(db, "get_conn", lambda: _Conn(log_fail, fail_insert=True))
    assert egress_log.flush([_event(), _event(), _event()]) == 0
    assert egress_log._writer_failing is True
    written = []

    class _Cur2:
        def execute(self, *a):
            pass

        def executemany(self, sql, rows):
            written.extend(rows)

    class _Conn2(_Conn):
        def cursor(self):
            return _Cur2()
    monkeypatch.setattr(db, "get_conn", lambda: _Conn2([]))
    assert egress_log.flush([_event()]) >= 1
    gap = [r for r in written if r[5] == "events_dropped"]
    assert len(gap) == 1 and "3 event(s)" in gap[0][6], "the three lost rows are admitted when the writer recovers"
    assert egress_log._writer_failing is False


def test_the_lost_count_survives_an_outage_longer_than_one_tick(monkeypatch):
    """fail (3 lost) -> fail again with only the gap row -> succeed: the 3 are still admitted."""
    import src.services.db as db
    monkeypatch.setattr(egress_log, "_schema_ready", True)
    monkeypatch.setattr(db, "get_conn", lambda: _Conn([], fail_insert=True))
    assert egress_log.flush([_event(), _event(), _event()]) == 0
    assert egress_log.flush([]) == 0, "the idle tick: the gap row alone, and it fails too"
    written = []

    class _Cur2:
        def execute(self, *a):
            pass

        def executemany(self, sql, rows):
            written.extend(rows)

    class _Conn2(_Conn):
        def cursor(self):
            return _Cur2()
    monkeypatch.setattr(db, "get_conn", lambda: _Conn2([]))
    assert egress_log.flush([]) == 1
    assert [r[6] for r in written if r[5] == "events_dropped"][0].startswith("3 event(s)")
