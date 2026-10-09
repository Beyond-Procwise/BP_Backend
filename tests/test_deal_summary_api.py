"""GET /deals/{deal_id}/summary: a draft (upload) summary is re-rendered from current
facts before it is served.

The endpoint stopped generating summaries on request in fddd763b (it serves the stored
row, or "Summary not available"; tests/api/test_deal_summary_endpoints.py covers that).
What these tests cover is the step in front of the read: the upload's deterministic
summary was written once, so an upload kept reading "87 warnings" after a rule closed
all 87 (2026-10-09). The read now re-renders it first, and a failure there must never
cost the user the stored text.
"""
from datetime import datetime, timezone

import src.api.routers.deal_summary as router_mod
import src.services.db as db
import src.services.session_postprocess as sp

_TS = datetime(2026, 10, 9, tzinfo=timezone.utc)


class _Store:
    """One bp_analysis_summary row, readable and rewritable through a fake cursor."""

    def __init__(self, summary, model):
        self.summary, self.model = summary, model
        self.rolled_back = False
        self.updates = []


class _Cur:
    def __init__(self, store, session_id="ses-1"):
        self._store, self._session_id, self._last = store, session_id, None

    def execute(self, sql, params=()):
        s = " ".join(sql.split()).lower()
        if s.startswith("update proc.bp_analysis_summary"):
            self._store.summary, self._store.model = params[0], params[1]
            self._store.updates.append(params)
            self._last = None
        elif "from proc.process_monitor" in s:
            self._last = (self._session_id,) if self._session_id else None
        elif "generated_at from proc.bp_analysis_summary" in s:
            self._last = (self._store.summary, self._store.model, _TS)
        elif "from proc.bp_analysis_summary" in s:
            self._last = (self._store.summary, self._store.model)
        else:
            raise AssertionError(f"unexpected SQL: {s}")

    def fetchone(self):
        return self._last


class _Conn:
    def __init__(self, store, **kw):
        self._store, self._kw = store, kw

    def cursor(self):
        return _Cur(self._store, **self._kw)

    def commit(self):
        pass

    def rollback(self):
        self._store.rolled_back = True

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _serve(monkeypatch, store, **kw):
    monkeypatch.setattr(db, "get_conn", lambda: _Conn(store, **kw))


def _render_as(monkeypatch, text):
    monkeypatch.setattr(sp, "session_summary_facts", lambda cur, deal, sid: {"sid": sid})
    monkeypatch.setattr(sp, "compose_session_summary", lambda facts: text)


# ---- through the endpoint ---------------------------------------------------------------

def test_a_stale_draft_summary_is_served_as_it_reads_today(monkeypatch):
    store = _Store("Findings: 87 warnings.", "session-postprocess/deterministic-v1")
    _serve(monkeypatch, store)
    _render_as(monkeypatch, "There are no open findings.")
    res = router_mod.get_deal_summary("BATCH-1")
    assert res["summary"] == "There are no open findings."
    assert res["model"] == sp.SUMMARY_MODEL
    assert store.summary == "There are no open findings."   # stored, not only served


def test_a_refresh_failure_still_serves_the_stored_summary(monkeypatch):
    store = _Store("Findings: 87 warnings.", "session-postprocess/deterministic-v1")
    _serve(monkeypatch, store)

    def boom(conn, deal_id):
        raise RuntimeError("facts query failed")
    monkeypatch.setattr(sp, "refresh_session_summary", boom)
    res = router_mod.get_deal_summary("BATCH-1")
    assert res["summary"] == "Findings: 87 warnings."
    assert store.rolled_back


def test_a_confirmed_deals_narrative_is_never_rewritten(monkeypatch):
    store = _Store("This deal involves Acme.", "BeyondProcwise/AgentNick:unified")
    _serve(monkeypatch, store)
    _render_as(monkeypatch, "SHOULD NOT APPEAR")
    res = router_mod.get_deal_summary("DEAL-1")
    assert res["summary"] == "This deal involves Acme."
    assert store.updates == []


# ---- refresh_session_summary itself ----------------------------------------------------

def test_an_unchanged_current_summary_is_not_rewritten(monkeypatch):
    store = _Store("Same text.", sp.SUMMARY_MODEL)
    _render_as(monkeypatch, "Same text.")
    assert sp.refresh_session_summary(_Conn(store), "BATCH-1") == "Same text."
    assert store.updates == []


def test_an_older_version_is_retagged_even_when_the_text_matches(monkeypatch):
    store = _Store("Same text.", "session-postprocess/deterministic-v1")
    _render_as(monkeypatch, "Same text.")
    sp.refresh_session_summary(_Conn(store), "BATCH-1")
    assert store.model == sp.SUMMARY_MODEL


def test_a_draft_with_no_session_is_left_alone(monkeypatch):
    store = _Store("Findings: 87 warnings.", "session-postprocess/deterministic-v1")
    _render_as(monkeypatch, "SHOULD NOT APPEAR")
    assert sp.refresh_session_summary(_Conn(store, session_id=None), "BATCH-1") is None
    assert store.updates == []
