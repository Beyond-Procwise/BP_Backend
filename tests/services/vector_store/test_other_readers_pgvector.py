"""BaseAgent.vector_search and the qwen30b training retriever under pgvector: same hit shape, tenant-scoped, no Qdrant client."""
import types

import numpy as np
import pytest

from src.services.vector_store import DEFAULT_TENANT, VectorHit
from src.services.vector_store.compat import search_hits


@pytest.fixture(autouse=True)
def pg(monkeypatch):
    monkeypatch.setenv("VECTOR_BACKEND", "pgvector")


class Store:
    def __init__(self, hits=None, boom=False):
        self.hits, self.boom, self.calls = hits or [], boom, []

    def search(self, tenant_id, query_embedding, filters, k, *, collection=None):
        self.calls.append({"tenant": tenant_id, "dim": len(query_embedding), "filters": filters, "k": k, "collection": collection})
        if self.boom:
            raise RuntimeError("db down")
        return self.hits


def hit(i, score=0.9, **payload):
    return VectorHit(id=f"documents:s{i}:0", source_id=f"s{i}", chunk_idx=0, collection="documents", score=score, content="text", payload=payload)


def test_search_hits_returns_qdrant_shaped_hits_for_the_default_tenant():
    store = Store([hit(1, 0.8, record_id="r1", content="text")])
    out = search_hits([0.1] * 8, "documents", 3, store=store)
    assert store.calls == [{"tenant": DEFAULT_TENANT, "dim": 8, "filters": {}, "k": 3, "collection": "documents"}]
    assert out[0].id == "documents:s1:0" and out[0].score == 0.8 and out[0].payload["record_id"] == "r1" and out[0].vector is None


def test_search_hits_accepts_numpy_vectors_and_filters():
    store = Store()
    search_hits(np.array([0.5, 0.5]), "uploaded", 2, filters={"session_id": "s"}, store=store)
    assert store.calls[0]["dim"] == 2 and store.calls[0]["filters"] == {"session_id": "s"}


# --- BaseAgent.vector_search -----------------------------------------------------------------------------------------------------

def test_vector_search_uses_the_store_and_needs_no_qdrant_client(monkeypatch):
    from src.agents.base_agent import BaseAgent
    store = Store([hit(1, record_id="r1")])
    monkeypatch.setattr("src.services.vector_store.get_vector_store", lambda: store)

    class Embedder:
        def encode(self, text, normalize_embeddings=True):
            return np.array([0.1] * 1024)
    me = types.SimpleNamespace(agent_nick=types.SimpleNamespace(qdrant_client=None, embedding_model=Embedder()),
                               settings=types.SimpleNamespace(qdrant_collection_name="procwise_document_embeddings"))
    out = BaseAgent.vector_search(me, "payment terms", top_k=4)
    assert out and out[0].payload["record_id"] == "r1"
    assert store.calls[0]["k"] == 4 and store.calls[0]["collection"] == "documents" and store.calls[0]["dim"] == 1024


def test_vector_search_without_an_embedder_returns_nothing(monkeypatch):
    from src.agents.base_agent import BaseAgent
    me = types.SimpleNamespace(agent_nick=types.SimpleNamespace(qdrant_client=None, embedding_model=None), settings=types.SimpleNamespace(qdrant_collection_name="p"))
    assert BaseAgent.vector_search(me, "q") == []


def test_vector_search_swallows_a_store_failure(monkeypatch):
    from src.agents.base_agent import BaseAgent
    monkeypatch.setattr("src.services.vector_store.get_vector_store", lambda: Store(boom=True))
    me = types.SimpleNamespace(agent_nick=types.SimpleNamespace(qdrant_client=None, embedding_model=types.SimpleNamespace(encode=lambda *a, **k: np.array([0.1] * 4))),
                               settings=types.SimpleNamespace(qdrant_collection_name="p"))
    assert BaseAgent.vector_search(me, "q") == []


# --- the qwen30b training retriever ----------------------------------------------------------------------------------------------

def test_the_training_retriever_searches_the_store_for_each_requested_collection(monkeypatch):
    from src.services.rag_qwen30b import RAGQwen30b
    store = Store([hit(1, 0.7, record_id="r1", content="alpha text"), hit(2, 0.6, record_id="r2", content="beta text")])
    monkeypatch.setattr("src.services.vector_store.get_vector_store", lambda: store)
    r = RAGQwen30b.__new__(RAGQwen30b)
    r.embedder = types.SimpleNamespace(encode=lambda q, normalize_embeddings=True: np.array([0.1] * 1024))
    r.primary_collection, r.uploaded_collection, r._reranker = "procwise_document_embeddings", "uploaded_documents", None
    r.telemetry = {"dense_count": 0, "reranked_count": 0, "deduped_count": 0, "capped_count": 0, "packed_chars": 0}
    r._embed_batch_size = 8
    out = r.retrieve_and_rerank("alpha", top_k=5, rerank_top_n=2, collections=["procwise_document_embeddings", "uploaded_documents"])
    assert {c["collection"] for c in store.calls} == {"documents", "uploaded"}
    assert out and r.telemetry["dense_count"] == 4
