"""analysis_store writes. Every test drives a fake psycopg2 connection so the
suite never needs a database."""
import uuid
from unittest.mock import patch

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
        self.autocommit = None

    def cursor(self):
        return self.cur

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_start_returns_the_new_analysis_id():
    new_id = uuid.uuid4()
    conn = FakeConn(results=[(new_id,)])

    got = analysis_store.start(session_id="ses-1", name="Q3 renewal",
                               mode="new", created_by="user@example.com", conn=conn)

    assert got == str(new_id)
    # Verify params reach the SQL
    sql, params = conn.cur.calls[0]
    assert params == ("ses-1", "Q3 renewal", "new", "user@example.com")


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


def test_start_strips_whitespace_from_session_id():
    """session_id gets .strip() before reaching the SQL params."""
    new_id = uuid.uuid4()
    conn = FakeConn(results=[(new_id,)])

    analysis_store.start(session_id="  ses-1  ", name="test", conn=conn)

    sql, params = conn.cur.calls[0]
    assert params[0] == "ses-1"  # First param is session_id, stripped


def test_txn_does_not_commit_when_caller_supplies_connection():
    """When conn is provided, _txn must yield it without commit/rollback."""
    new_id = uuid.uuid4()
    conn = FakeConn(results=[(new_id,)])

    analysis_store.start(session_id="ses-1", name="test", conn=conn)

    assert conn.committed is False
    assert conn.rolled_back is False


def test_txn_owns_connection_and_commits_on_success():
    """When conn is None, _txn owns the connection, sets autocommit=False, and commits."""
    new_id = uuid.uuid4()
    fake_conn = FakeConn(results=[(new_id,)])

    with patch("src.services.analysis_store.get_conn") as mock_get_conn:
        mock_get_conn.return_value.__enter__ = lambda self: fake_conn
        mock_get_conn.return_value.__exit__ = lambda self, *a: None

        analysis_store.start(session_id="ses-1", name="test")

    assert fake_conn.autocommit is False
    assert fake_conn.committed is True
    assert fake_conn.rolled_back is False


def test_txn_owns_connection_and_rollback_on_exception():
    """When conn is None and an exception occurs, _txn rolls back and re-raises."""
    fake_conn = FakeConn()
    fake_conn.cursor = lambda: FakeCursorThatRaises()

    with patch("src.services.analysis_store.get_conn") as mock_get_conn:
        mock_get_conn.return_value.__enter__ = lambda self: fake_conn
        mock_get_conn.return_value.__exit__ = lambda self, *a: None

        with pytest.raises(RuntimeError, match="Database error"):
            analysis_store.start(session_id="ses-1", name="test")

    assert fake_conn.autocommit is False
    assert fake_conn.rolled_back is True
    assert fake_conn.committed is False


class FakeCursorThatRaises:
    """A cursor that raises an exception on execute()."""
    def execute(self, sql, params=None):
        raise RuntimeError("Database error")
