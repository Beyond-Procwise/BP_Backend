"""The write side: callers write SOURCE rows / queue jobs, never embed or upsert vectors inline."""
import contextlib
import json
import os
import threading
import types

import pytest

from src.services.vector_store import DEFAULT_TENANT, PgVectorStore, sources
from src.services.vector_store.profiles import PROFILES
from tests.services.vector_store.helpers import HashEmbedder
from tests.services.vector_store.test_pgvector_store_unit import Rec

PROFILE = PROFILES["bge-large-en-v1.5@1"]


# --- sources.py (unit) ----------------------------------------------------------------------------------------------------------

def test_put_source_upserts_the_source_row_and_strips_unindexable_metadata():
    rec = Rec()
    sources.put_source("t", "uploaded", "d1", "some text", {"s3_key": "b/x", "uploaded_by": "u", "keep": 1}, connect=rec)
    sql, params = rec.executed[0]
    assert "insert into proc.bp_vector_source" in sql.lower() and "on conflict (tenant_id, collection, source_id) do update" in sql.lower()
    assert params[:4] == ("t", "uploaded", "d1", "some text")
    meta = json.loads(params[4])
    assert meta == {"keep": 1}


@pytest.mark.parametrize("bad", [("", "c", "s", "x"), (None, "c", "s", "x"), ("t", "", "s", "x"), ("t", "c", "", "x"), ("t", "c", "s", ""), ("t", "c", "s", "   ")])
def test_put_source_refuses_a_missing_tenant_collection_id_or_text(bad):
    rec = Rec()
    with pytest.raises(ValueError):
        sources.put_source(*bad, connect=rec)
    assert rec.executed == []


def test_delete_source_is_scoped_to_tenant_and_collection():
    rec = Rec()
    sources.delete_source("t", "uploaded", "d1", connect=rec)
    sql, params = rec.executed[0]
    assert "delete from proc.bp_vector_source" in sql.lower() and params == ("t", "uploaded", "d1")
    with pytest.raises(ValueError):
        sources.delete_source("", "uploaded", "d1", connect=rec)


class StatusRec(Rec):
    def __init__(self, statuses):
        super().__init__()
        self.statuses = statuses

    def cursor(self):
        rec = self
        base = super().cursor()

        class C:
            description = [("x",)]
            def __enter__(s): return s
            def __exit__(s, *a): return False
            def execute(s, sql, params=()):
                rec.executed.append((sql, params))
            def fetchall(s):
                return [(x,) for x in rec.statuses]
        return C()


@pytest.mark.parametrize("statuses,expected", [([], "done"), (["done"], "done"), (["pending"], "pending"), (["processing", "done"], "pending"),
                                               (["failed"], "failed"), (["failed", "pending"], "pending")])
def test_wait_until_embedded_reports_where_the_job_stands(statuses, expected):
    assert sources.wait_until_embedded("t", "uploaded", "d1", timeout_s=0, poll_s=0, connect=StatusRec(statuses)) == expected


# --- RAGService.upsert_payloads under pgvector -----------------------------------------------------------------------------------

def rag(monkeypatch, calls):
    from src.services.rag_service import RAGService
    s = RAGService.__new__(RAGService)
    s.client = None
    s.embedder = None                                  # an inline embed would crash: it must not happen
    s.primary_collection, s.uploaded_collection = "procwise_document_embeddings", "uploaded_documents"
    s.static_policy_collection, s.learning_collection = "static_policy", "learning"
    monkeypatch.setattr(sources, "put_source", lambda *a, **k: calls.append((a, k)))
    return s


def test_upsert_payloads_writes_a_source_row_and_never_embeds_or_touches_qdrant(monkeypatch):
    calls = []
    rag(monkeypatch, calls).upsert_payloads([{"content": "hello world", "record_id": "r1", "document_type": "invoice", "s3_key": "b/x"}])
    (tenant, collection, source_id, content, meta), _ = calls[0]
    assert (tenant, collection, source_id, content) == (DEFAULT_TENANT, "documents", "r1", "hello world")
    assert meta["document_type"] == "invoice" and meta["record_id"] == "r1" and "s3_key" not in meta and "content" not in meta


def test_upsert_payloads_keeps_the_collection_it_was_asked_for(monkeypatch):
    calls = []
    rag(monkeypatch, calls).upsert_payloads([{"content": "x", "record_id": "r2"}], collection_name="learning")
    assert calls[0][0][1] == "learning"


def test_upsert_payloads_gives_an_id_when_the_payload_has_none_and_skips_what_is_not_a_payload(monkeypatch):
    calls = []
    s = rag(monkeypatch, calls)
    s.upsert_payloads([{"content": "no id here"}, "not a dict", 42])
    assert len(calls) == 1 and calls[0][0][2] and calls[0][0][3] == "no id here"


def test_a_payload_with_no_text_is_indexed_by_its_metadata_exactly_as_the_qdrant_path_did(monkeypatch):
    """Parity: with no usable text the old path embedded the JSON of the payload, so a metadata-only record stays searchable."""
    calls = []
    rag(monkeypatch, calls).upsert_payloads([{"content": "   ", "record_id": "r9", "document_type": "note"}])
    (_, _, source_id, content, meta), _ = calls[0]
    assert source_id == "r9" and json.loads(content) == {"record_id": "r9", "document_type": "note"} and meta["document_type"] == "note"


def test_a_failed_source_write_is_logged_and_the_caller_carries_on(monkeypatch):
    from src.services.rag_service import RAGService
    s = RAGService.__new__(RAGService)
    s.client, s.embedder = None, None
    s.primary_collection, s.uploaded_collection, s.static_policy_collection, s.learning_collection = "p", "u", "s", "l"

    def boom(*a, **k):
        raise RuntimeError("db down")
    monkeypatch.setattr(sources, "put_source", boom)
    s.upsert_payloads([{"content": "x", "record_id": "r"}])          # must not raise: document processing continues


# --- the live extraction path ----------------------------------------------------------------------------------------------------

class RawCur:
    def __init__(self, text):
        self.text, self.executed = text, []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=()):
        self.executed.append(sql)

    def fetchone(self):
        return (json.dumps({"full_text": self.text}),)


def test_embed_document_queues_a_job_instead_of_embedding_when_the_backend_is_pgvector(monkeypatch):
    from src.services.extraction import embed
    queued = []
    monkeypatch.setattr("src.services.vector_store.outbox.enqueue", lambda cur, *a, **k: queued.append(a) or True)

    class Conn:
        autocommit = True
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def cursor(self): return RawCur("w " * 400)

    monkeypatch.setattr(embed, "_queue_connect", lambda: Conn())
    nick = types.SimpleNamespace(settings=None, embedding_model=None, qdrant_client=None)         # no model, no client: neither may be used
    n = embed.embed_document(nick, RawCur("w " * 400), doc_type="invoice", doc_pk="INV-1", raw_id=1)
    assert queued == [(DEFAULT_TENANT, "documents", "invoice:INV-1")] and n == 2             # 400 words -> 2 chunks, reported as before


def test_embed_document_with_no_text_queues_nothing(monkeypatch):
    from src.services.extraction import embed
    monkeypatch.setattr("src.services.vector_store.outbox.enqueue", lambda *a, **k: pytest.fail("queued a job for a document with no text"))
    assert embed.embed_document(types.SimpleNamespace(settings=None), RawCur(""), doc_type="invoice", doc_pk="INV-1", raw_id=1) == 0


# --- the scheduler runs the worker only on pgvector ------------------------------------------------------------------------------

def _scheduler():
    from src.services.backend_scheduler import BackendScheduler
    s = BackendScheduler.__new__(BackendScheduler)
    s._jobs, s._lock, s.agent_nick = {}, threading.RLock(), types.SimpleNamespace(embedding_model=HashEmbedder())
    return s


def test_the_vector_worker_job_is_registered_once_and_runs_every_few_seconds():
    s = _scheduler()
    s._register_vector_worker_job()
    s._register_vector_worker_job()
    assert len(s._jobs) == 1 and "vector_embedding_worker" in s._jobs and s._jobs["vector_embedding_worker"].interval.total_seconds() <= 15


# --- live: the trigger, the worker and delete, end to end -------------------------------------------------------------------------

_LIVE = os.environ.get("PROCWISE_TEST_LIVE_DB", "").strip().lower() in ("1", "true", "yes", "on")
live = pytest.mark.skipif(not _LIVE, reason="needs PROCWISE_TEST_LIVE_DB=1")


@pytest.fixture()
def conn():
    from src.services.db import get_conn
    with get_conn() as c:
        c.autocommit = False
        try:
            yield c
        finally:
            c.rollback()


@live
def test_a_written_source_is_embedded_by_the_worker_and_removed_when_deleted(conn):
    from src.services.vector_store.worker import VectorWorker
    connect = lambda: contextlib.nullcontext(conn)
    store = PgVectorStore(connect, profile=PROFILE)
    worker = VectorWorker(connect, HashEmbedder(), store_factory=lambda p: PgVectorStore(connect, profile=p or PROFILE))
    sources.put_source("t1", "uploaded", "doc-1", "quarterly supplier review minutes", {"document_type": "minutes"}, connect=connect)
    assert sources.wait_until_embedded("t1", "uploaded", "doc-1", timeout_s=0, poll_s=0, connect=connect) == "pending"
    worker.run_once()
    assert sources.wait_until_embedded("t1", "uploaded", "doc-1", timeout_s=0, poll_s=0, connect=connect) == "done"
    assert store.count("t1", collection="uploaded") == 1
    sources.delete_source("t1", "uploaded", "doc-1", connect=connect)
    worker.run_once()
    assert store.count("t1") == 0


# --- several chunks of one record must not overwrite each other (found while reading the upload path) ---------------------------------

def test_chunks_that_share_a_record_id_become_separate_sources_and_keep_their_chunk_ids(monkeypatch):
    calls = []
    rag(monkeypatch, calls).upsert_payloads([
        {"content": "first chunk text", "record_id": "doc-1", "chunk_id": 0, "document_type": "uploaded_document"},
        {"content": "second chunk text", "record_id": "doc-1", "chunk_id": 1, "document_type": "uploaded_document"}])
    ids = [c[0][2] for c in calls]
    assert ids == ["doc-1:0", "doc-1:1"] and len(set(ids)) == 2
    assert [c[0][4]["chunk_id"] for c in calls] == [0, 1] and all(c[0][4]["record_id"] == "doc-1" for c in calls)


def test_a_payload_with_no_chunk_id_keeps_its_record_id_as_the_source_id(monkeypatch):
    calls = []
    rag(monkeypatch, calls).upsert_payloads([{"content": "whole record", "record_id": "rec-7"}])
    assert calls[0][0][2] == "rec-7"


# --- sources helpers for policy sync and uploads -----------------------------------------------------------------------------------

def test_delete_sources_matching_deletes_by_metadata_containment_within_tenant_and_collection():
    rec = Rec()
    sources.delete_sources_matching("t", "policy", {"source_path": "s3://b/policy.pdf"}, connect=rec)
    sql, params = rec.executed[0]
    assert "delete from proc.bp_vector_source" in sql.lower() and "metadata @> %s::jsonb" in sql and params[:2] == ("t", "policy")
    assert json.loads(params[2]) == {"source_path": "s3://b/policy.pdf"}


@pytest.mark.parametrize("meta", [{}, None])
def test_delete_sources_matching_refuses_an_empty_match_so_it_can_never_delete_a_whole_collection(meta):
    rec = Rec()
    with pytest.raises(ValueError):
        sources.delete_sources_matching("t", "policy", meta, connect=rec)
    assert rec.executed == []


def test_find_source_metadata_returns_the_first_match_or_none():
    class Row(Rec):
        def cursor(self):
            rec = self

            class C:
                description = [("metadata",)]
                def __enter__(s): return s
                def __exit__(s, *a): return False
                def execute(s, sql, params=()): rec.executed.append((sql, params))
                def fetchone(s): return ({"s3_etag": "abc"},) if rec.rows else None
            return C()
    assert sources.find_source_metadata("t", "policy", {"source_path": "p"}, connect=Row(rows=[1])) == {"s3_etag": "abc"}
    assert sources.find_source_metadata("t", "policy", {"source_path": "p"}, connect=Row()) is None
    with pytest.raises(ValueError):
        sources.find_source_metadata("t", "policy", {}, connect=Row())


def test_wait_until_embedded_accepts_several_source_ids():
    rec = StatusRec(["pending"])
    assert sources.wait_until_embedded("t", "uploaded", ["d:0", "d:1"], timeout_s=0, poll_s=0, connect=rec) == "pending"
    sql, params = rec.executed[0]
    assert "source_id = any(%s)" in sql.lower() and ["d:0", "d:1"] in params
