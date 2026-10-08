"""/health reports the two signals whose silence has already cost us.

Findings writes were rejected for 65 days in bp_sqldb before anyone noticed,
and the document vector store sat empty for five weeks after Qdrant Cloud
died. Both looked, from every existing surface, exactly like "nothing to
report". /health now says when the newest detection finding was written and
how many points the document collection holds, so an operator (or a probe)
can see the silence.
"""
from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from src.services import health_signals


class _Cur:
    def __init__(self, row):
        self._row = row
        self.sql = ""

    def execute(self, sql, params=()):
        self.sql = sql

    def fetchone(self):
        return self._row


def test_last_finding_written_is_the_newest_detected_at_as_iso():
    ts = datetime(2026, 9, 24, 12, 43, 4, tzinfo=timezone.utc)
    cur = _Cur(row=(ts,))

    assert health_signals.last_finding_written(cur) == "2026-09-24T12:43:04+00:00"
    assert "bp_detection_finding" in cur.sql
    assert "detected_at" in cur.sql


def test_last_finding_written_is_none_when_no_finding_exists():
    assert health_signals.last_finding_written(_Cur(row=(None,))) is None


def test_vector_store_points_asks_for_an_exact_count():
    calls = []

    class _Client:
        def count(self, collection_name, exact):
            calls.append((collection_name, exact))
            return SimpleNamespace(count=0)

    assert health_signals.vector_store_points(_Client(), "procwise_document_embeddings") == 0
    assert calls == [("procwise_document_embeddings", True)]


def test_collection_name_falls_back_to_the_default_when_blank():
    assert health_signals.collection_name(SimpleNamespace(qdrant_collection_name="  ")) == (
        "procwise_document_embeddings"
    )
    assert health_signals.collection_name(SimpleNamespace(qdrant_collection_name="docs v2")) == "docs_v2"


def _health_with(monkeypatch, *, agent_nick, last_finding=None):
    """Call the route function directly, as the lifespan test does, so no
    Ollama or Neo4j is needed. main imports the signals module through the
    ``services`` root, so patches go on main's own reference to it."""
    from src.api import main

    monkeypatch.setattr(main.app.state, "agent_nick", agent_nick, raising=False)
    monkeypatch.setattr(main.app.state, "extraction_v3_schemas", {}, raising=False)
    if last_finding is not None:
        monkeypatch.setattr(main.health_signals, "last_finding_written", last_finding)
    return main.health()


def test_health_reports_the_vector_store_point_count(monkeypatch):
    class _Client:
        def count(self, collection_name, exact):
            return SimpleNamespace(count=1234)

    body = _health_with(monkeypatch, agent_nick=SimpleNamespace(qdrant_client=_Client()))

    assert body["vector_store"]["points"] == 1234
    assert body["vector_store"]["collection"]


def test_health_reports_unavailable_not_an_error_when_signals_cannot_be_read(monkeypatch):
    """Under pytest the database is a fake that rejects this query, and there is
    no agent (so no vector client). Neither may take /health down."""
    def _boom(cur):
        raise RuntimeError("database away")

    body = _health_with(monkeypatch, agent_nick=None, last_finding=_boom)

    assert body["last_finding_written"] == "unavailable"
    assert body["vector_store"]["points"] == "unavailable"


def test_health_reports_the_last_finding_timestamp(monkeypatch):
    body = _health_with(
        monkeypatch, agent_nick=None, last_finding=lambda cur: "2026-09-24T12:43:04+00:00"
    )

    assert body["last_finding_written"] == "2026-09-24T12:43:04+00:00"


# --- last triage run ---------------------------------------------------------
# "Last finding written" alone looks stale on a healthy system: triage dedupes by
# fingerprint, so a finding it already holds is updated in place, never written
# again, and an unchanged corpus produces no new rows for weeks. The liveness
# signal is when triage last FINISHED and how many deals it failed.


def test_last_triage_run_reports_finish_time_and_failed_deal_count():
    ts = datetime(2026, 10, 8, 8, 47, 13, tzinfo=timezone.utc)
    cur = _Cur(row=(ts, {"D-1": "boom", "D-2": "boom"}))

    assert health_signals.last_triage_run(cur) == {
        "finished_at": "2026-10-08T08:47:13+00:00",
        "failed_deals": 2,
    }
    assert "bp_triage_run" in cur.sql
    assert "finished_at IS NOT NULL" in cur.sql
    assert "rolled_back_at IS NULL" in cur.sql


def test_last_triage_run_is_none_when_triage_has_never_finished():
    assert health_signals.last_triage_run(_Cur(row=None)) is None


def test_health_reports_unavailable_triage_when_the_database_cannot_answer(monkeypatch):
    from src.api import main

    def _boom(cur):
        raise RuntimeError("database away")

    monkeypatch.setattr(main.health_signals, "last_triage_run", _boom)
    body = _health_with(monkeypatch, agent_nick=None)

    assert body["last_triage_run"] == "unavailable"
