"""Supplier relationships under pgvector: the same lookups through the VectorStore, with no Qdrant client and no payload indexes to manage."""
import types

import pytest

from src.services import supplier_relationship_service as srs
from src.services.vector_store import DEFAULT_TENANT, VectorHit


class FakeListStore:
    def __init__(self, hits=None, boom=False):
        self.hits, self.calls, self.boom = hits or [], [], boom

    def list(self, tenant_id, *, collection, filters=None, limit=100, offset=0, newest_first=False):
        self.calls.append({"tenant": tenant_id, "collection": collection, "filters": filters, "limit": limit})
        if self.boom:
            raise RuntimeError("db down")
        return self.hits


def hit(**payload):
    return VectorHit(id="documents:x:0", source_id="x", chunk_idx=0, collection="documents", score=None, content="c", payload=payload)


def service(monkeypatch, store):
    s = srs.SupplierRelationshipService.__new__(srs.SupplierRelationshipService)
    s.agent_nick, s.settings, s.rag_service = None, types.SimpleNamespace(qdrant_collection_name="procwise_document_embeddings"), None
    s.collection_name, s.client = "procwise_document_embeddings", None
    s._fallback_cache_limit = 8
    from collections import OrderedDict
    s._fallback_cache, s._fallback_negative_cache = OrderedDict(), OrderedDict()
    monkeypatch.setattr("src.services.vector_store.get_vector_store", lambda: store)
    return s


def test_a_supplier_is_found_by_id_through_the_store_with_no_qdrant_client(monkeypatch):
    store = FakeListStore([hit(document_type="supplier_relationship", supplier_id="S1", supplier_name="Acme", supplier_name_normalized="acme")])
    out = service(monkeypatch, store).fetch_relationship(supplier_id="S1", limit=1)
    assert out and out[0]["supplier_id"] == "S1"
    assert store.calls == [{"tenant": DEFAULT_TENANT, "collection": "documents", "filters": {"supplier_id": "S1"}, "limit": 1}]


def test_a_supplier_is_found_by_normalised_name_as_a_metadata_match(monkeypatch):
    store = FakeListStore([hit(supplier_name="Acme Ltd", supplier_name_normalized="acme ltd")])
    service(monkeypatch, store).fetch_relationship(supplier_name="Acme Ltd", limit=1)
    assert store.calls[0]["filters"] == {"metadata": {"supplier_name_normalized": "acme ltd"}}


def test_id_and_name_together_narrow_both(monkeypatch):
    store = FakeListStore()
    service(monkeypatch, store).fetch_relationship(supplier_id="S1", supplier_name="Acme", limit=1)
    assert store.calls[0]["filters"] == {"supplier_id": "S1", "metadata": {"supplier_name_normalized": "acme"}}


def test_no_id_and_no_name_asks_nothing(monkeypatch):
    store = FakeListStore()
    assert service(monkeypatch, store).fetch_relationship(limit=1) == [] and store.calls == []


def test_a_store_failure_returns_nothing_not_an_exception(monkeypatch):
    assert service(monkeypatch, FakeListStore(boom=True)).fetch_relationship(supplier_id="S1", limit=1) == []


def test_the_overview_is_read_by_its_document_type(monkeypatch):
    store = FakeListStore([hit(document_type="supplier_relationship_overview", total_suppliers=3)])
    out = service(monkeypatch, store).fetch_overview()
    assert out["total_suppliers"] == 3
    assert store.calls[0]["filters"] == {"document_type": "supplier_relationship_overview"} and store.calls[0]["limit"] == 1


def test_a_failed_overview_read_returns_none(monkeypatch):
    assert service(monkeypatch, FakeListStore(boom=True)).fetch_overview() is None


def test_the_service_builds_on_pgvector_without_qdrant_or_payload_indexes(monkeypatch):
    nick = types.SimpleNamespace(settings=types.SimpleNamespace(qdrant_collection_name="p"), qdrant_client=None)
    monkeypatch.setattr(srs.rag_module, "RAGService", lambda n: (_ for _ in ()).throw(RuntimeError("no rag in this test")))
    srs.SupplierRelationshipService(nick)


def test_a_found_supplier_is_cached_so_the_second_lookup_does_not_touch_the_store(monkeypatch):
    store = FakeListStore([hit(document_type="supplier_relationship", supplier_id="S1", supplier_name="Acme", supplier_name_normalized="acme")])
    svc = service(monkeypatch, store)
    first = svc.fetch_relationship(supplier_id="S1")
    assert first and len(store.calls) == 1
    assert svc.fetch_relationship(supplier_id="S1") == first
    assert len(store.calls) == 1


def test_a_supplier_that_was_not_found_is_remembered_so_it_is_not_asked_for_again(monkeypatch):
    store = FakeListStore([])
    svc = service(monkeypatch, store)
    assert svc.fetch_relationship(supplier_id="S1") == []
    assert len(store.calls) == 1
    assert svc.fetch_relationship(supplier_id="S1") == []
    assert len(store.calls) == 1
