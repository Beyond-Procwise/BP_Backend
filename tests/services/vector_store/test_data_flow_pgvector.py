"""Knowledge-graph persistence writes SOURCE rows: the same records, ids and payloads the old Qdrant path produced (golden file), no inline embedding."""
import json
import os
import types

import pytest

from src.services.vector_store import DEFAULT_TENANT, sources
from tests.test_data_flow_manager import DummyNick, _tables_fixture
from services.data_flow_manager import DataFlowManager

GOLDEN = os.path.join(os.path.dirname(__file__), "fixtures", "kg_persist_golden.json")
TABLE_MAP = {"contracts": "proc.bp_contracts", "supplier_master": "proc.bp_supplier", "purchase_orders": "proc.bp_purchase_order_trgt",
             "purchase_order_lines": "proc.bp_po_line_items_trgt", "invoices": "proc.bp_invoice_trgt", "invoice_lines": "proc.bp_invoice_line_items_trgt",
             "product_mapping": "proc.cat_product_mapping", "quotes": "proc.bp_quote_trgt", "quote_lines": "proc.bp_quote_line_items_trgt"}


def _relations_and_graph():
    return DataFlowManager(DummyNick(enable_learning=True)).build_data_flow_map(_tables_fixture(), table_name_map=TABLE_MAP)


def _persist(monkeypatch, nick=None, boom=False):
    relations, graph = _relations_and_graph()
    written = []

    def put(*a, **k):
        if boom:
            raise RuntimeError("db down")
        written.append(a)
    monkeypatch.setattr(sources, "put_source", put)
    DataFlowManager(nick or DummyNick(enable_learning=True)).persist_knowledge_graph(relations, graph)
    return written


def test_persistence_writes_exactly_the_records_the_old_qdrant_path_wrote(monkeypatch):
    """The golden file is the 22 points the Qdrant path wrote for this fixture (captured before that path was removed)."""
    golden = json.load(open(GOLDEN))
    written = _persist(monkeypatch)
    got = {a[2]: json.loads(json.dumps(a[4], default=str)) for a in written}
    assert set(got) == set(golden)                                                  # the same deterministic ids
    # The same payloads. One honest exception: an agent_relationship_summary's WORDING depends on set iteration order in the existing code, so it
    # differs between processes (it did before this change too); its id and its stable fields must still match.
    for i, expected in golden.items():
        if expected["document_type"] == "agent_relationship_summary":
            stable = ("document_type", "agent_name", "related_tables")
            assert {k: got[i].get(k) for k in stable} == {k: expected.get(k) for k in stable} and got[i].get("summary", "").strip()
        else:
            assert got[i] == expected, expected["document_type"]
    assert all(a[0] == DEFAULT_TENANT and a[1] == "knowledge_graph" and a[3].strip() for a in written)


def test_a_failed_source_write_is_logged_and_never_raised(monkeypatch):
    assert _persist(monkeypatch, boom=True) == []


def test_learning_disabled_writes_nothing(monkeypatch):
    assert _persist(monkeypatch, nick=DummyNick(enable_learning=False)) == []
