"""RAGService.search routing on the vector store.

These replace two Qdrant-double tests from tests/test_rag_service.py. The behaviour is the same, only the thing
asked is now the VectorStore: an uploaded-collection query carries the session, and a target document is not
crowded out by a higher-scoring policy.
"""
from types import SimpleNamespace

from src.services.rag_service import RAGService
from src.services.vector_store import DEFAULT_TENANT, VectorHit


class DummyEmbed:
    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        if isinstance(texts, list):
            return [[0.1] for _ in texts]
        return [0.1]


class DummyCrossEncoder:
    def __init__(self, *args, **kwargs):
        pass

    def predict(self, pairs):
        return [0.5 for _ in pairs]


class RoutingStore:
    """Answers per logical collection and records every question it was asked."""

    def __init__(self, by_collection=None):
        self.by_collection = by_collection or {}
        self.calls = []

    def search(self, tenant_id, query_embedding, filters, k, *, collection=None):
        self.calls.append({"tenant": tenant_id, "filters": filters, "k": k, "collection": collection})
        return list(self.by_collection.get(collection, []))


def _hit(collection, record_id, score, **payload):
    return VectorHit(id=f"{collection}:{record_id}:0", source_id=record_id, chunk_idx=0, collection=collection,
                     score=score, content=payload.get("summary", "text"), payload={"record_id": record_id, **payload})


def _rag(store):
    nick = SimpleNamespace(device="cpu", settings=SimpleNamespace(qdrant_collection_name="c", reranker_model="x"),
                           embedding_model=DummyEmbed())
    rag = RAGService(nick, cross_encoder_cls=DummyCrossEncoder)
    rag._vector_store = store
    return rag


def test_the_uploaded_collection_is_asked_with_the_session_and_no_other_collection_is():
    store = RoutingStore()
    _rag(store).search("session scoped query", session_id="session-abc")

    uploaded = [c for c in store.calls if c["collection"] == "uploaded"]
    assert uploaded, "Uploaded collection should be queried"
    assert all((c["filters"] or {}).get("session_id") == "session-abc" for c in uploaded)

    others = [c for c in store.calls if c["collection"] != "uploaded"]
    assert others, "Primary collections should also be queried"
    assert all("session_id" not in (c["filters"] or {}) for c in others)
    assert {c["tenant"] for c in store.calls} == {DEFAULT_TENANT}


def test_a_target_document_is_returned_alongside_a_higher_scoring_policy():
    store = RoutingStore({
        "policy": [_hit("policy", "policy-1", 0.92, document_type="policy", source_type="Policy", summary="Policy guidance.")],
        "documents": [_hit("documents", "invoice-77", 0.55, document_type="Invoice", source_type="Invoice", summary="Invoice 77 details.")],
    })
    hits = _rag(store).search("Need invoice 77 details and PO reference")

    assert hits, "RAG service should return results"
    assert hits[0].payload["document_type"].lower() == "invoice"
    assert any(h.payload["document_type"].lower() == "invoice" for h in hits)
