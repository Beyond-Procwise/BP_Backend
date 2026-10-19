"""RAGService reads through the VectorStore: same results shape, tenant-scoped."""
from types import SimpleNamespace

import pytest
from src.services.vector_store.filter_models import models

from src.services.rag_service import RAGService
from src.services.vector_store import DEFAULT_TENANT, VectorHit
from src.services.vector_store.filters import to_store_filter


def cond(key, value=None, any_=None):
    return models.FieldCondition(key=key, match=models.MatchValue(value=value) if any_ is None else models.MatchAny(any=any_))


# --- the converter ---------------------------------------------------------------------------------------------------------------

def test_no_filter_converts_to_no_conditions():
    assert to_store_filter(None) == {}
    assert to_store_filter(models.Filter(must=[])) == {}


def test_must_conditions_on_known_columns_become_direct_filters():
    f = models.Filter(must=[cond("document_type", "invoice"), cond("document_id", any_=["d1", "d2"]), cond("session_id", "s1")])
    assert to_store_filter(f) == {"document_type": "invoice", "document_id": ["d1", "d2"], "session_id": "s1"}


def test_a_payload_key_with_no_column_filters_the_stored_metadata_like_qdrant_did():
    f = models.Filter(must=[cond("supplier_name", "Acme"), cond("region", any_=["uk", "us"])])
    out = to_store_filter(f)
    assert out == {"metadata": {"supplier_name": "Acme"}, "metadata_in": {"region": ["uk", "us"]}}


def test_a_should_group_becomes_an_or_group():
    f = models.Filter(must=[cond("document_type", "quote")],
                      should=[models.Filter(should=[cond("supplier_id", any_=["S1"]), cond("supplier_name", any_=["Acme"])])])
    out = to_store_filter(f)
    assert out["document_type"] == "quote" and "any_of" in out


def test_must_not_is_refused_rather_than_silently_dropped():
    with pytest.raises(ValueError):
        to_store_filter(models.Filter(must_not=[cond("document_type", "quote")]))


def test_a_condition_on_tenant_id_is_refused():
    with pytest.raises(ValueError):
        to_store_filter(models.Filter(must=[cond("tenant_id", "other")]))


# --- RAGService ---------------------------------------------------------------------------------------------------

class FakeStore:
    def __init__(self, hits=None, boom=False):
        self.calls, self.hits, self.boom = [], hits or [], boom

    def search(self, tenant_id, query_embedding, filters, k, *, collection=None):
        self.calls.append({"tenant": tenant_id, "filters": filters, "k": k, "collection": collection, "dim": len(query_embedding)})
        if self.boom:
            raise RuntimeError("db down")
        return list(self.hits)


def svc(store):
    s = RAGService.__new__(RAGService)
    s.primary_collection, s.uploaded_collection = "procwise_document_embeddings", "uploaded_documents"
    s.static_policy_collection, s.learning_collection = "static_policy", "learning"
    s._vector_store = store
    s._search_workers = 2
    return s


def hit(i, score, **payload):
    return VectorHit(id=f"documents:s{i}:0", source_id=f"s{i}", chunk_idx=0, collection="documents", score=score, content="text", payload=payload)


def test_a_query_goes_to_the_store_for_the_right_tenant_collection_and_filters():
    store = FakeStore([hit(1, 0.9, content="text", record_id="r1")])
    out = svc(store)._query_collection("procwise_document_embeddings", [0.1] * 1024,
                                              query_filter=models.Filter(must=[cond("document_type", "invoice")]), limit=7)
    assert store.calls == [{"tenant": DEFAULT_TENANT, "filters": {"document_type": "invoice"}, "k": 7, "collection": "documents", "dim": 1024}]
    assert len(out) == 1 and out[0].score == 0.9 and out[0].id == "documents:s1:0"
    assert out[0].payload["record_id"] == "r1" and out[0].payload["collection_name"] == "procwise_document_embeddings"


@pytest.mark.parametrize("name,logical", [("procwise_document_embeddings", "documents"), ("uploaded_documents", "uploaded"),
                                          ("static_policy", "policy"), ("learning", "learning")])
def test_each_collection_name_maps_to_its_logical_collection(name, logical):
    store = FakeStore()
    svc(store)._query_collection(name, [0.0] * 1024, query_filter=None, limit=3)
    assert store.calls[0]["collection"] == logical


def test_an_unknown_collection_returns_nothing_and_never_queries():
    store = FakeStore()
    assert svc(store)._query_collection("something_else", [0.0] * 1024, query_filter=None, limit=3) == [] and store.calls == []


def test_a_store_failure_returns_no_hits_for_that_collection_not_an_exception():
    assert svc(FakeStore(boom=True))._query_collection("static_policy", [0.0] * 1024, query_filter=None, limit=3) == []


def test_the_parallel_search_works_with_no_qdrant_client():
    store = FakeStore([hit(1, 0.8)])
    out = svc(store)._search_collections_parallel([("procwise_document_embeddings", None), ("static_policy", None)], [0.0] * 1024, limit=4)
    assert set(out) == {"procwise_document_embeddings", "static_policy"} and len(store.calls) == 2
    assert {c["collection"] for c in store.calls} == {"documents", "policy"}
