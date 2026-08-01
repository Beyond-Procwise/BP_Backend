"""analysis_store writes. Every test drives a fake psycopg2 connection so the
suite never needs a database."""
import uuid

import pytest

from src.services import analysis_store


class FakeCursor:
    """Records every statement and replays queued results in order."""

    def __init__(self, results):
        self._results = list(results)
        self.calls = []

    def execute(self, sql, params=None):
        self.calls.append((" ".join(sql.split()), params))

    def fetchone(self):
        return self._results.pop(0) if self._results else None

    def fetchall(self):
        return self._results.pop(0) if self._results else []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeConn:
    def __init__(self, results=()):
        self.cur = FakeCursor(results)
        self.committed = False
        self.rolled_back = False

    def cursor(self):
        return self.cur

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True


def test_start_returns_the_new_analysis_id():
    new_id = uuid.uuid4()
    conn = FakeConn(results=[(new_id,)])

    got = analysis_store.start(session_id="ses-1", name="Q3 renewal",
                               mode="new", conn=conn)

    assert got == str(new_id)


def test_start_is_idempotent_on_session_id():
    """A second call for the same upload must return the SAME id and must not
    overwrite the name the user chose."""
    existing = uuid.uuid4()
    conn = FakeConn(results=[(existing,)])

    got = analysis_store.start(session_id="ses-1", name="", conn=conn)

    sql, _ = conn.cur.calls[0]
    assert "ON CONFLICT (session_id) DO UPDATE" in sql
    assert "name = COALESCE(proc.bp_analysis.name, EXCLUDED.name)" in sql
    assert got == str(existing)


def test_start_rejects_a_blank_session_id():
    with pytest.raises(ValueError, match="session_id is required"):
        analysis_store.start(session_id="  ", conn=FakeConn())


def test_start_rejects_an_unknown_mode():
    with pytest.raises(ValueError, match="mode must be one of"):
        analysis_store.start(session_id="ses-1", mode="sideways",
                             conn=FakeConn())
