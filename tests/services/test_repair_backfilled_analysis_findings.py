"""Repair for analyses the sweep closed with findings it captured too late.

Before the history rule existed, sweep pass 1 created events for old sessions
and pass 2 froze them by running a capture THEN — days after the session had
actually ended. The numbers that landed in `findings` are today's data wearing
a July date. This strips them back to NULL so the UI's honest "findings were
not captured" path shows instead.

Drives a fake connection: the suite never needs a database.
"""
import importlib

mod = importlib.import_module("scripts.repair_backfilled_analysis_findings")


class FakeCursor:
    def __init__(self, results):
        self._results = list(results)
        self.calls = []

    def execute(self, sql, params=None):
        self.calls.append((" ".join(sql.split()), params))

    def fetchall(self):
        return self._results.pop(0) if self._results else []

    def fetchone(self):
        return self._results.pop(0) if self._results else None


class FakeConn:
    def __init__(self, results=()):
        self.cur = FakeCursor(results)
        self.committed = False

    def cursor(self):
        return self.cur

    def commit(self):
        self.committed = True

    def rollback(self):
        pass


def _sql(conn):
    return " ".join(s for s, _ in conn.cur.calls)


def test_only_touches_analyses_closed_long_after_their_session_ended():
    """An analysis frozen seconds after its session resolved holds REAL
    findings and must be left alone. The lateness of the close is the only
    thing that distinguishes the two."""
    conn = FakeConn([[]])

    mod.repair(conn=conn)

    sql = _sql(conn)
    assert "a.completed_at >" in sql
    assert "minutes" in sql
    assert "findings IS NOT NULL" in sql


def test_nulls_the_captured_figures_and_redates_the_close():
    conn = FakeConn([[("aid-1",), ("aid-2",)]])

    got = mod.repair(conn=conn)

    update = next(s for s, _ in conn.cur.calls if "UPDATE proc.bp_analysis" in s)
    assert "findings = NULL" in update
    assert "value_found = NULL" in update
    assert "currency = NULL" in update
    assert "completed_at =" in update and "resolved_at" in update
    assert got["repaired"] == 2


def test_never_discards_the_document_count():
    """session_document_outcome is a durable record of what was really
    uploaded. It is the one figure that is not a reconstruction."""
    conn = FakeConn([[]])

    mod.repair(conn=conn)

    assert "document_count" not in _sql(conn)


def test_dry_run_writes_nothing_but_still_reports_what_it_would_do():
    conn = FakeConn([[("aid-1",)]])

    got = mod.repair(dry_run=True, conn=conn)

    assert not any("UPDATE" in s for s, _ in conn.cur.calls)
    assert got["repaired"] == 1


def test_is_idempotent_a_second_run_finds_nothing():
    """After the first run findings IS NULL, so the predicate excludes the
    same rows — running it twice is safe."""
    conn = FakeConn([[]])

    got = mod.repair(conn=conn)

    assert got["repaired"] == 0
