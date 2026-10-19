"""The learning repository under VECTOR_BACKEND=pgvector: events are source rows, reads come back as the same payload dicts."""
import contextlib
import os
import types

import pytest

from src.services import learning_repository as lr
from src.services.vector_store import DEFAULT_TENANT, PgVectorStore, VectorHit, sources
from src.services.vector_store.profiles import PROFILES
from tests.services.vector_store.helpers import HashEmbedder
from tests.services.vector_store.test_pgvector_store_unit import PROFILE, Rec, VEC, sql_of


@pytest.fixture(autouse=True)
def pg(monkeypatch):
    monkeypatch.setenv("VECTOR_BACKEND", "pgvector")


def repo(monkeypatch, put=None, store=None):
    r = lr.LearningRepository.__new__(lr.LearningRepository)
    r.agent_nick, r.settings, r.qdrant_client, r.embedder = None, types.SimpleNamespace(learning_collection_name="learning"), None, None
    r.collection_name = "learning"
    if put is not None:
        monkeypatch.setattr(sources, "put_source", lambda *a, **k: put.append((a, k)))
    if store is not None:
        monkeypatch.setattr("src.services.vector_store.get_vector_store", lambda: store)
    return r


def test_a_learning_event_becomes_a_source_row_without_a_qdrant_client_or_embedder(monkeypatch):
    put = []
    pid = repo(monkeypatch, put).record_learning(workflow_id="wf-1", source="workflow", event_type="workflow_complete",
                                                 metadata={"status": "ok"}, tags=["b", "a", "a"])
    (tenant, collection, source_id, content, meta), _ = put[0]
    assert pid == source_id and (tenant, collection) == (DEFAULT_TENANT, "learning") and content and content == meta["summary"]
    assert meta["document_type"] == "learning" and meta["workflow_id"] == "wf-1" and meta["tags"] == ["a", "b"] and meta["metadata"]["status"] == "ok"


def test_a_model_plan_becomes_a_source_row_with_its_text_as_the_content(monkeypatch):
    put = []
    repo(monkeypatch, put).record_model_plan(model_name="AgentNick", plan_text="  warm, brief, specific  ", plan_metadata={"v": 1})
    (_, collection, _, content, meta), _ = put[0]
    assert collection == "learning" and content == "warm, brief, specific" and meta["document_type"] == "model_plan" and meta["model_name"] == "AgentNick"
    assert meta["plan_text"] == "warm, brief, specific" and "summary" not in meta


def test_an_empty_event_or_plan_writes_nothing(monkeypatch):
    put = []
    r = repo(monkeypatch, put)
    assert r.record_model_plan(model_name="m", plan_text="   ") is None and put == []


def test_a_failed_write_is_logged_and_returns_none(monkeypatch):
    r = repo(monkeypatch)
    monkeypatch.setattr(sources, "put_source", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("db down")))
    assert r.record_learning(workflow_id="w", source="s", event_type="e", metadata={"a": 1}) is None


class FakeListStore:
    def __init__(self, hits):
        self.hits, self.calls = hits, []

    def list(self, tenant_id, *, collection, filters=None, limit=100, offset=0, newest_first=False):
        self.calls.append({"tenant": tenant_id, "collection": collection, "filters": filters, "limit": limit, "newest_first": newest_first})
        return self.hits


def hit(**payload):
    return VectorHit(id="learning:x:0", source_id="x", chunk_idx=0, collection="learning", score=None, content="c", payload=payload)


def test_recent_learnings_are_listed_newest_first_for_the_workflow_as_the_original_payloads(monkeypatch):
    store = FakeListStore([hit(document_type="learning", workflow_id="wf-1", summary="did a thing", metadata={"a": 1}, content="did a thing",
                               chunk_id=0, chunk_index=0)])
    out = repo(monkeypatch, store=store).get_recent_learnings(workflow_id="wf-1", limit=5)
    assert store.calls == [{"tenant": DEFAULT_TENANT, "collection": "learning", "limit": 5, "newest_first": True,
                            "filters": {"document_type": "learning", "metadata": {"workflow_id": "wf-1"}}}]
    assert out == [{"document_type": "learning", "workflow_id": "wf-1", "summary": "did a thing", "metadata": {"a": 1}}]   # no store bookkeeping keys


def test_recent_learnings_without_a_workflow_filter_only_on_the_type(monkeypatch):
    store = FakeListStore([])
    repo(monkeypatch, store=store).get_recent_learnings(limit=3)
    assert store.calls[0]["filters"] == {"document_type": "learning"}


def test_model_plans_are_filtered_by_model_and_sorted_by_created_at_without_bookkeeping_keys(monkeypatch):
    store = FakeListStore([hit(document_type="model_plan", model_name="M", plan_text="old", created_at="2026-01-01", summary="old", content="old"),
                           hit(document_type="model_plan", model_name="M", plan_text="new", created_at="2026-02-01", summary="new", content="new")])
    out = repo(monkeypatch, store=store).fetch_model_plans(model_name="M", limit=5)
    assert [p["plan_text"] for p in out] == ["new", "old"] and all("summary" not in p and "content" not in p for p in out)
    assert store.calls[0]["filters"] == {"document_type": "model_plan", "metadata": {"model_name": "M"}}


def test_a_failed_listing_returns_nothing_not_an_exception(monkeypatch):
    class Boom:
        def list(self, *a, **k): raise RuntimeError("db down")
    assert repo(monkeypatch, store=Boom()).get_recent_learnings(limit=3) == []


def test_the_repository_can_be_built_on_pgvector_without_touching_qdrant():
    nick = types.SimpleNamespace(settings=types.SimpleNamespace(learning_collection_name="learning"), qdrant_client=None, embedding_model=None)
    lr.LearningRepository(nick)


# --- store.list newest_first and the worker keeping a source's own summary ------------------------------------------------------

def test_list_can_return_newest_first():
    rec = Rec()
    PgVectorStore(rec, profile=PROFILE).list("t", collection="learning", newest_first=True)
    assert "order by vector_id desc" in " ".join(sql_of(rec, "from proc.bp_vector_chunk")[0].lower().split())
    rec2 = Rec()
    PgVectorStore(rec2, profile=PROFILE).list("t", collection="learning")
    assert "order by vector_id limit" in " ".join(sql_of(rec2, "from proc.bp_vector_chunk")[0].lower().split())


_LIVE = os.environ.get("PROCWISE_TEST_LIVE_DB", "").strip().lower() in ("1", "true", "yes", "on")


@pytest.mark.skipif(not _LIVE, reason="needs PROCWISE_TEST_LIVE_DB=1")
def test_a_learning_event_round_trips_through_the_worker_and_the_store_unchanged():
    from src.services.db import get_conn
    from src.services.vector_store.worker import VectorWorker
    with get_conn() as c:
        c.autocommit = False
        try:
            connect = lambda: contextlib.nullcontext(c)
            store = PgVectorStore(connect, profile=PROFILE)
            original = {"document_type": "learning", "source": "workflow", "event_type": "workflow_complete", "workflow_id": "wf-live",
                        "summary": "workflow wf-live finished with status ok", "metadata": {"status": "ok", "n": 3}, "tags": ["a", "b"]}
            sources.put_source("t-live", "learning", "evt-1", original["summary"], original, connect=connect)
            VectorWorker(connect, HashEmbedder(), store_factory=lambda p: PgVectorStore(connect, profile=p or PROFILE)).run_once()
            hits = store.list("t-live", collection="learning", filters={"document_type": "learning", "metadata": {"workflow_id": "wf-live"}})
            assert len(hits) == 1
            back = {k: v for k, v in hits[0].payload.items() if k not in ("content", "chunk_id", "chunk_index")}
            assert back == original                         # including the nested metadata, the tags and the full summary
        finally:
            c.rollback()
