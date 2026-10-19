"""Knowledge-graph persistence under pgvector writes the SAME records the Qdrant path wrote, as source rows, with no inline embedding."""
import types

import pytest

from src.services.vector_store import DEFAULT_TENANT, sources
from tests.test_data_flow_manager import DummyNick, _tables_fixture
from services.data_flow_manager import DataFlowManager

TABLE_MAP = {"contracts": "proc.bp_contracts", "supplier_master": "proc.bp_supplier", "purchase_orders": "proc.bp_purchase_order_trgt",
             "purchase_order_lines": "proc.bp_po_line_items_trgt", "invoices": "proc.bp_invoice_trgt", "invoice_lines": "proc.bp_invoice_line_items_trgt",
             "product_mapping": "proc.cat_product_mapping", "quotes": "proc.bp_quote_trgt", "quote_lines": "proc.bp_quote_line_items_trgt"}


def _relations_and_graph():
    nick = DummyNick(enable_learning=True)
    return DataFlowManager(nick).build_data_flow_map(_tables_fixture(), table_name_map=TABLE_MAP)


def test_pgvector_persists_exactly_the_records_the_qdrant_path_did(monkeypatch):
    relations, graph = _relations_and_graph()

    monkeypatch.setenv("VECTOR_BACKEND", "qdrant")
    qnick = DummyNick(enable_learning=True)
    DataFlowManager(qnick).persist_knowledge_graph(relations, graph)
    qdrant = {str(p.id): dict(p.payload) for e in qnick.qdrant_client.upserts for p in e["points"]}
    assert qdrant

    monkeypatch.setenv("VECTOR_BACKEND", "pgvector")
    written = []
    monkeypatch.setattr(sources, "put_source", lambda *a, **k: written.append(a))
    pnick = types.SimpleNamespace(settings=DummyNick(enable_learning=True).settings, qdrant_client=None, embedding_model=None)
    DataFlowManager(pnick).persist_knowledge_graph(relations, graph)

    pg = {a[2]: a[4] for a in written}
    assert set(pg) == set(qdrant)                                                  # the same deterministic ids
    assert all(pg[i] == qdrant[i] for i in qdrant)                                 # the same payloads
    assert all(a[0] == DEFAULT_TENANT and a[1] == "knowledge_graph" and a[3].strip() for a in written)


def test_pgvector_persistence_needs_neither_a_qdrant_client_nor_an_embedder(monkeypatch):
    relations, graph = _relations_and_graph()
    monkeypatch.setenv("VECTOR_BACKEND", "pgvector")
    written = []
    monkeypatch.setattr(sources, "put_source", lambda *a, **k: written.append(a))
    nick = types.SimpleNamespace(settings=DummyNick(enable_learning=True).settings, qdrant_client=None, embedding_model=None)
    DataFlowManager(nick).persist_knowledge_graph(relations, graph)
    assert written


def test_a_failed_source_write_is_logged_and_never_raised(monkeypatch):
    relations, graph = _relations_and_graph()
    monkeypatch.setenv("VECTOR_BACKEND", "pgvector")
    monkeypatch.setattr(sources, "put_source", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("db down")))
    nick = types.SimpleNamespace(settings=DummyNick(enable_learning=True).settings, qdrant_client=None, embedding_model=None)
    DataFlowManager(nick).persist_knowledge_graph(relations, graph)


def test_learning_disabled_still_writes_nothing_on_pgvector(monkeypatch):
    relations, graph = _relations_and_graph()
    monkeypatch.setenv("VECTOR_BACKEND", "pgvector")
    written = []
    monkeypatch.setattr(sources, "put_source", lambda *a, **k: written.append(a))
    nick = types.SimpleNamespace(settings=DummyNick(enable_learning=False).settings, qdrant_client=None, embedding_model=None)
    DataFlowManager(nick).persist_knowledge_graph(relations, graph)
    assert written == []
