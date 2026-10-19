import json
import pytest
from types import SimpleNamespace

import pandas as pd

import services.data_flow_manager as data_flow_module

from services.data_flow_manager import DataFlowManager


class DummyNick:
    """No Qdrant client and no embedder: persistence only stores source rows, the vector worker embeds."""

    def __init__(self, *, enable_learning: bool = True):
        self.settings = SimpleNamespace(
            enable_learning=enable_learning,
            knowledge_graph_collection_name="procwise_knowledge_graph",
        )


@pytest.fixture()
def written(monkeypatch):
    """The vector-store source rows persistence writes: (tenant, collection, source_id, text, metadata)."""
    from src.services.vector_store import sources
    rows = []
    monkeypatch.setattr(sources, "put_source", lambda *a, **k: rows.append(a))
    return rows


def _tables_fixture():
    contracts = pd.DataFrame(
        [
            {"contract_id": "CO1", "supplier_id": "SI1"},
            {"contract_id": "CO2", "supplier_id": "SI2"},
        ]
    )
    supplier_master = pd.DataFrame(
        [
            {"supplier_id": "SI1", "supplier_name": "Supplier One"},
            {"supplier_id": "SI2", "supplier_name": "Supplier Two"},
        ]
    )
    purchase_orders = pd.DataFrame(
        [
            {"po_id": "PO1", "contract_id": "CO1", "supplier_id": "SI1"},
            {"po_id": "PO2", "contract_id": "CO2", "supplier_id": "SI2"},
        ]
    )
    purchase_order_lines = pd.DataFrame(
        [
            {"po_id": "PO1", "item_description": "Widget Service"},
            {"po_id": "PO2", "item_description": "Widget Service"},
        ]
    )
    invoices = pd.DataFrame(
        [
            {"invoice_id": "INV1", "po_id": "PO1"},
        ]
    )
    invoice_lines = pd.DataFrame(
        [
            {"invoice_id": "INV1", "po_id": "PO1"},
        ]
    )
    product_mapping = pd.DataFrame(
        [
            {"product": "Widget Service", "category_level_2": "Services"},
        ]
    )
    quotes = pd.DataFrame(
        [
            {"quote_id": "Q1", "po_id": "PO1"},
        ]
    )
    quote_lines = pd.DataFrame(
        [
            {"quote_id": "Q1", "line_total": 100.0},
        ]
    )

    return {
        "contracts": contracts,
        "supplier_master": supplier_master,
        "purchase_orders": purchase_orders,
        "purchase_order_lines": purchase_order_lines,
        "invoices": invoices,
        "invoice_lines": invoice_lines,
        "product_mapping": product_mapping,
        "quotes": quotes,
        "quote_lines": quote_lines,
    }


def test_data_flow_manager_builds_graph_and_persists(written):
    nick = DummyNick(enable_learning=True)
    manager = DataFlowManager(nick)
    tables = _tables_fixture()

    table_name_map = {
        "contracts": "proc.bp_contracts",
        "supplier_master": "proc.bp_supplier",
        "purchase_orders": "proc.bp_purchase_order_trgt",
        "purchase_order_lines": "proc.bp_po_line_items_trgt",
        "invoices": "proc.bp_invoice_trgt",
        "invoice_lines": "proc.bp_invoice_line_items_trgt",
        "product_mapping": "proc.cat_product_mapping",
        "quotes": "proc.bp_quote_trgt",
        "quote_lines": "proc.bp_quote_line_items_trgt",
    }

    relations, graph = manager.build_data_flow_map(tables, table_name_map=table_name_map)

    assert relations
    assert any(
        rel["status"] == "linked"
        for rel in relations
        if rel["source_table"] == "proc.bp_contracts" and rel["target_table"] == "proc.bp_supplier"
    )
    assert graph["nodes"]["proc.bp_contracts"]["row_count"] == 2
    assert graph["paths"]
    for path in graph["paths"]:
        assert path["canonical"]

    flows = graph.get("supplier_flows")
    assert isinstance(flows, list) and flows
    first_flow = flows[0]
    assert first_flow.get("supplier_id")
    assert "coverage_ratio" in first_flow
    assert "categories" in first_flow
    if first_flow.get("categories"):
        assert first_flow.get("primary_category")

    manager.persist_knowledge_graph(relations, graph)

    assert written and all(row[0] == "default" and row[1] == "knowledge_graph" and row[3].strip() for row in written)
    payload_types = {row[4].get("document_type") for row in written}
    assert "supplier_flow" in payload_types
    assert "agent_relationship_summary" in payload_types

    supplier_rows = [row for row in written if row[4].get("document_type") == "supplier_flow"]
    assert supplier_rows
    for row in supplier_rows:
        payload = row[4]
        summary = payload.get("summary")
        assert isinstance(summary, str) and summary.strip()
        mapping_summary = payload.get("mapping_summary")
        assert isinstance(mapping_summary, list) and mapping_summary
        assert all("proc." not in statement for statement in mapping_summary)
        if payload.get("categories"):
            assert payload.get("primary_category")


def test_persist_knowledge_graph_skips_when_learning_disabled(written):
    nick = DummyNick(enable_learning=False)
    manager = DataFlowManager(nick)

    relations = [
        {
            "source_table": "proc.contracts",
            "target_table": "proc.supplier",
            "relationship_type": "references",
            "status": "linked",
        }
    ]
    graph = {"paths": [], "supplier_flows": []}

    manager.persist_knowledge_graph(relations, graph)

    assert written == []


def test_supplier_flow_payload_is_trimmed_below_limit(monkeypatch):
    nick = DummyNick()
    manager = DataFlowManager(nick)

    oversized_payload = {
        "document_type": "supplier_flow",
        "supplier_id": "S-123",
        "supplier_name": "Mega Supplier Incorporated",
        "coverage_ratio": 0.92,
        "summary": "A" * 12000,
        "mapping_summary": [
            f"Mapping statement {idx} " + ("B" * 600) for idx in range(40)
        ],
        "products": {
            "unique_items": 4096,
            "top_items": [
                {
                    "description": "Item description " + ("C" * 500),
                    "spend_gbp": 125000.45,
                }
                for _ in range(10)
            ],
        },
        "quotes": {
            "count": 200,
            "total_value_gbp": 987654.32,
            "quote_ids": [f"Q{idx}" for idx in range(120)],
        },
        "purchase_orders": {
            "count": 150,
            "total_value_gbp": 543210.98,
            "po_ids": [f"PO{idx}" for idx in range(160)],
            "latest_order_date": "2025-01-31",
        },
        "invoices": {
            "count": 80,
            "total_value_gbp": 321098.76,
            "invoice_ids": [f"INV{idx}" for idx in range(130)],
            "latest_invoice_date": "2025-02-15",
        },
    }

    sanitized = manager._sanitize_payload(oversized_payload)

    monkeypatch.setattr(data_flow_module, "_POINT_PAYLOAD_SOFT_LIMIT", 2000)
    monkeypatch.setattr(data_flow_module, "_POINT_PAYLOAD_HARD_LIMIT", 3000)
    trimmed = manager._enforce_payload_limits(dict(sanitized))

    assert manager._payload_json_size(trimmed) <= data_flow_module._POINT_PAYLOAD_HARD_LIMIT
    assert len(trimmed.get("mapping_summary", [])) <= 12
    products = trimmed.get("products")
    if isinstance(products, dict):
        assert len(products.get("top_items", [])) <= 3
