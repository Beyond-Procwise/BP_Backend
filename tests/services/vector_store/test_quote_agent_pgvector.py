"""QuoteEvaluationAgent's vector-side quote fetch under pgvector: same filters, through the store, no Qdrant client."""
import types

import pytest

from src.agents import quote_evaluation_agent as qea
from src.services.vector_store import DEFAULT_TENANT, VectorHit


class Store:
    def __init__(self, batches):
        self.batches, self.calls = list(batches), []

    def list(self, tenant_id, *, collection, filters=None, limit=100, offset=0, newest_first=False):
        self.calls.append({"tenant": tenant_id, "collection": collection, "filters": filters, "limit": limit})
        return self.batches.pop(0) if self.batches else []


def hit(i, **payload):
    return VectorHit(id=f"documents:q{i}:0", source_id=f"q{i}", chunk_idx=0, collection="documents", score=None, content="c", payload=payload)


def agent(monkeypatch, store):
    a = qea.QuoteEvaluationAgent.__new__(qea.QuoteEvaluationAgent)
    a.agent_nick = types.SimpleNamespace(qdrant_client=None)
    a.settings = types.SimpleNamespace(qdrant_collection_name="procwise_document_embeddings")
    monkeypatch.setattr("src.services.vector_store.get_vector_store", lambda: store)
    return a


def test_quotes_are_listed_through_the_store_with_the_same_filters_as_before(monkeypatch):
    store = Store([[hit(1, supplier_name="Acme", supplier_id="S1", total_amount=100, quote_id="Q1")]])
    out = agent(monkeypatch, store)._fetch_quotes(["Acme"], ["S1"], "it")
    assert out[0]["quote_id"] == "Q1" and out[0]["total_amount"] == 100 and out[0]["supplier_name"] == "Acme"
    f = store.calls[0]
    assert (f["tenant"], f["collection"], f["limit"]) == (DEFAULT_TENANT, "documents", 100)
    assert f["filters"]["document_type"] == "quote" and f["filters"]["metadata"] == {"product_category": "it"}
    assert f["filters"]["any_of"] == [{"metadata_in": {"supplier_name": ["Acme"]}}, {"supplier_id": ["S1"]}]


def test_a_single_supplier_condition_is_a_plain_filter(monkeypatch):
    store = Store([[hit(1, quote_id="Q1")]])
    agent(monkeypatch, store)._fetch_quotes([], ["S1"], None)
    assert store.calls[0]["filters"] == {"document_type": "quote", "supplier_id": ["S1"]}


def test_no_hits_retries_without_the_supplier_filter_as_before(monkeypatch):
    store = Store([[], [hit(1, quote_id="Q9")]])
    out = agent(monkeypatch, store)._fetch_quotes(["Acme"], [], None)
    assert [c["filters"] for c in store.calls] == [{"document_type": "quote", "metadata_in": {"supplier_name": ["Acme"]}}, {"document_type": "quote"}]
    assert out[0]["quote_id"] == "Q9"


def test_a_failing_store_returns_no_quotes(monkeypatch):
    class Boom:
        def list(self, *a, **k): raise RuntimeError("db down")
    assert agent(monkeypatch, Boom())._fetch_quotes(["Acme"], [], None) == []
