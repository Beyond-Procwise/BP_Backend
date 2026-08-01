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


def test_freeze_copies_session_documents_onto_the_analysis():
    aid = uuid.uuid4()
    conn = FakeConn(results=[
        (aid,),          # SELECT the running analysis
        # (the documents INSERT has no RETURNING, so it never fetches)
        [("D-1",)],      # SELECT DISTINCT deal_id
        (2,),            # _allocate_version -> next version for D-1
    ])

    analysis_store.freeze("ses-1", findings={"deal": {}}, conn=conn)

    sqls = [s for s, _ in conn.cur.calls]
    assert any("INSERT INTO proc.bp_analysis_document" in s for s in sqls)
    assert any("proc.session_document_outcome" in s for s in sqls)


def test_freeze_allocates_version_per_deal_not_globally():
    """The same run may be v3 for one deal and v1 for another."""
    aid = uuid.uuid4()
    conn = FakeConn(results=[
        (aid,),
        [("D-1",), ("D-2",)],
        (3,),            # D-1 already had two analyses
        (1,),            # D-2 has none
    ])

    analysis_store.freeze("ses-1", findings={}, conn=conn)

    link_params = [p for s, p in conn.cur.calls
                   if "INSERT INTO proc.bp_analysis_deal" in s]
    assert [(p[1], p[2]) for p in link_params] == [("D-1", 3), ("D-2", 1)]


def test_freeze_clears_is_latest_on_the_deals_previous_versions():
    aid = uuid.uuid4()
    conn = FakeConn(results=[(aid,), [("D-1",)], (2,)])

    analysis_store.freeze("ses-1", findings={}, conn=conn)

    sqls = [s for s, _ in conn.cur.calls]
    assert any("SET is_latest = false" in s and "deal_id = %s" in s
               for s in sqls)


def test_freeze_is_a_noop_when_nothing_is_running():
    """Idempotent: freezing an already-complete analysis must not touch it.

    Nothing beyond the initial lookup may run - no document copy, no deal
    link, no completion update - because a second freeze() call landing on
    the same session (the live listener and the scheduled sweep can both
    call freeze()) must be a true no-op, not merely "returns None"."""
    conn = FakeConn(results=[None])

    assert analysis_store.freeze("ses-1", findings={}, conn=conn) is None

    # Exactly one statement: the SELECT ... FOR UPDATE lookup. Nothing else.
    assert len(conn.cur.calls) == 1, (
        f"expected exactly 1 statement, got {len(conn.cur.calls)}: "
        f"{[s for s, _ in conn.cur.calls]}"
    )
    sql, _ = conn.cur.calls[0]
    assert "SELECT analysis_id FROM proc.bp_analysis" in sql
    assert "FOR UPDATE" in sql

    sqls = [s for s, _ in conn.cur.calls]
    assert not any("INSERT INTO proc.bp_analysis_document" in s for s in sqls)
    assert not any("INSERT INTO proc.bp_analysis_deal" in s for s in sqls)
    assert not any("UPDATE proc.bp_analysis" in s and "status = 'complete'" in s
                   for s in sqls)


def test_freeze_marks_the_analysis_complete_with_its_headline_figures():
    aid = uuid.uuid4()
    conn = FakeConn(results=[(aid,), []])

    analysis_store.freeze("ses-1", findings={"a": 1}, document_count=4,
                          value_found=12400, currency="GBP", conn=conn)

    final = [(s, p) for s, p in conn.cur.calls
             if "UPDATE proc.bp_analysis" in s and "status = 'complete'" in s]
    assert len(final) == 1
    assert 4 in final[0][1] and "GBP" in final[0][1]


def test_sweep_creates_events_for_sessions_that_never_got_one(monkeypatch):
    """The UI's POST is an optimisation. If the browser closed before it fired,
    the sweep must still produce the event — just without the chosen name."""
    conn = FakeConn(results=[
        [("ses-9", "Renewal deal")],   # sessions with no bp_analysis row
        [],                            # sessions running but resolved
        [],                            # sessions running past the stale cap
    ])
    started = []
    monkeypatch.setattr(analysis_store, "start",
                        lambda **kw: started.append(kw) or "aid")

    got = analysis_store.sweep(conn=conn)

    assert got["created"] == 1
    assert started[0]["session_id"] == "ses-9"
    assert started[0]["name"] == "Renewal deal"


def test_sweep_freezes_a_resolved_but_stuck_analysis(monkeypatch):
    conn = FakeConn(results=[[], [("ses-9",)], []])
    frozen = []
    monkeypatch.setattr(analysis_store, "_freeze_one",
                        lambda sid: frozen.append(sid))

    got = analysis_store.sweep(conn=conn)

    assert got["frozen"] == 1 and frozen == ["ses-9"]


def test_sweep_fails_an_analysis_whose_session_never_resolved():
    conn = FakeConn(results=[[], [], [("ses-9",)]])

    got = analysis_store.sweep(conn=conn)

    assert got["failed"] == 1
    sqls = [s for s, _ in conn.cur.calls]
    assert any("status = 'failed'" in s for s in sqls)
    assert any("session did not resolve" in str(p) for _, p in conn.cur.calls)


def test_sweep_stale_cap_defaults_to_sixty_minutes():
    """The UI gives up narrating after 6 minutes (REPORT_WAIT_CAP_MS). Failing
    an analysis at that point would kill slow-but-working large uploads."""
    import inspect
    sig = inspect.signature(analysis_store.sweep)
    assert sig.parameters["stale_minutes"].default == 60


def test_freeze_one_scopes_discrepancies_by_file_path(monkeypatch):
    """_freeze_one must pass file_paths through to capture(), or every swept
    analysis silently freezes with zero discrepancies (bp_extraction_discrepancy
    has no deal_id column)."""
    from src.services import analysis_findings

    captured = {}

    def fake_capture(deal_ids, *, file_paths=None, conn=None):
        captured["deal_ids"] = deal_ids
        captured["file_paths"] = file_paths
        return {}

    monkeypatch.setattr(analysis_store, "deal_ids_for_session",
                        lambda sid: ["D-1"])
    monkeypatch.setattr(analysis_store, "file_paths_for_session",
                        lambda sid: ["documents/invoice/x.xlsx"])
    monkeypatch.setattr(analysis_store, "document_count_for_session",
                        lambda sid: 1)
    monkeypatch.setattr(analysis_findings, "capture", fake_capture)
    monkeypatch.setattr(analysis_findings, "headline", lambda findings: (None, None))
    frozen = {}
    monkeypatch.setattr(analysis_store, "freeze",
                        lambda sid, **kw: frozen.update(kw))

    analysis_store._freeze_one("ses-9")

    assert captured["deal_ids"] == ["D-1"]
    assert captured["file_paths"] == ["documents/invoice/x.xlsx"]
