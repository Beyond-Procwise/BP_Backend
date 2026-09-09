import os
import sys
import types
from typing import Any

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engines.query_engine import PROCUREMENT_CATEGORY_FIELDS, QueryEngine


class DummyCursor:
    def __init__(self, cols):
        self._cols = cols

    def execute(self, sql, params):
        pass

    def fetchall(self):
        return [(c,) for c in self._cols]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass


class DummyConn:
    def __init__(self, cols):
        self._cols = cols

    def cursor(self):
        return DummyCursor(self._cols)


def test_quantity_expression_defaults_to_one_when_missing():
    engine = QueryEngine(agent_nick=types.SimpleNamespace())
    conn = DummyConn(["supplier_id"])  # no quantity column
    assert engine._quantity_expression(conn, "schema", "table", "li") == "1"


def test_quantity_expression_detects_quantity_column():
    engine = QueryEngine(agent_nick=types.SimpleNamespace())
    conn = DummyConn(["quantity", "other"])
    assert (
        engine._quantity_expression(conn, "schema", "table", "li")
        == "COALESCE(li.quantity, 1)"
    )


def test_fetch_supplier_data_uses_line_items(monkeypatch):
    calls = []
    engine = QueryEngine(
        agent_nick=types.SimpleNamespace(get_db_connection=lambda: DummyContext())
    )

    def fake_price(conn, schema, table, alias):
        calls.append(("price", schema, table, alias))
        return "0"

    def fake_qty(conn, schema, table, alias):
        calls.append(("qty", schema, table, alias))
        return "1"

    def fake_cols(conn, schema, table):
        if (schema, table) == ("proc", "bp_supplier"):
            calls.append(("cols", schema, table))
        return []

    monkeypatch.setattr(engine, "_price_expression", fake_price)
    monkeypatch.setattr(engine, "_quantity_expression", fake_qty)
    monkeypatch.setattr(engine, "_get_columns", fake_cols)
    monkeypatch.setattr(
        pd, "read_sql", lambda sql, conn: pd.DataFrame({"supplier_id": []})
    )

    engine.fetch_supplier_data()

    assert ("price", "proc", "bp_po_line_items_trgt", "li") in calls
    assert ("qty", "proc", "bp_po_line_items_trgt", "li") in calls
    assert ("price", "proc", "bp_invoice_line_items_trgt", "ili") in calls
    assert ("qty", "proc", "bp_invoice_line_items_trgt", "ili") in calls
    assert ("cols", "proc", "bp_supplier") in calls


def test_fetch_supplier_data_uses_delivery_lead_time(monkeypatch):
    engine = QueryEngine(
        agent_nick=types.SimpleNamespace(get_db_connection=lambda: DummyContext())
    )
    monkeypatch.setattr(engine, "_price_expression", lambda *a, **k: "0")
    monkeypatch.setattr(engine, "_quantity_expression", lambda *a, **k: "1")
    monkeypatch.setattr(
        engine,
        "_get_columns",
        lambda conn, schema, table: [
            "delivery_lead_time_days",
            "trading_name",
            "legal_structure",
        ],
    )
    captured = {}

    def fake_read_sql(sql, conn):
        captured["sql"] = sql
        return pd.DataFrame({"supplier_id": []})

    monkeypatch.setattr(pd, "read_sql", fake_read_sql)

    engine.fetch_supplier_data()

    assert "delivery_lead_time_days" in captured["sql"]
    assert "on_time_pct" in captured["sql"]
    # ensure string values are guarded by a numeric regex and cast
    assert "~ '^-?\\d+(\\.\\d+)?$'" in captured["sql"]
    assert "::numeric" in captured["sql"]

    # ensure additional supplier fields are projected explicitly
    assert "s.trading_name" in captured["sql"]
    assert "s.legal_structure" in captured["sql"]


def test_fetch_supplier_data_defaults_on_time_pct(monkeypatch):
    engine = QueryEngine(
        agent_nick=types.SimpleNamespace(get_db_connection=lambda: DummyContext())
    )
    monkeypatch.setattr(engine, "_price_expression", lambda *a, **k: "0")
    monkeypatch.setattr(engine, "_quantity_expression", lambda *a, **k: "1")
    monkeypatch.setattr(engine, "_get_columns", lambda conn, schema, table: [])
    captured = {}

    def fake_read_sql(sql, conn):
        captured["sql"] = sql
        return pd.DataFrame({"supplier_id": []})

    monkeypatch.setattr(pd, "read_sql", fake_read_sql)

    engine.fetch_supplier_data()

    assert "0.0 AS on_time_pct" in captured["sql"]


def test_fetch_supplier_data_filters_candidates(monkeypatch):
    engine = QueryEngine(
        agent_nick=types.SimpleNamespace(get_db_connection=lambda: DummyContext())
    )
    monkeypatch.setattr(engine, "_price_expression", lambda *a, **k: "0")
    monkeypatch.setattr(engine, "_quantity_expression", lambda *a, **k: "1")
    monkeypatch.setattr(engine, "_get_columns", lambda *a, **k: [])

    sample = pd.DataFrame(
        {
            "supplier_id": ["S1", "S2"],
            "supplier_name": ["Alpha", "Beta"],
            "po_spend": [10.0, 20.0],
            "invoice_spend": [5.0, 6.0],
            "total_spend": [15.0, 26.0],
            "invoice_count": [1, 2],
            "on_time_pct": [1.0, 0.5],
        }
    )

    monkeypatch.setattr(pd, "read_sql", lambda sql, conn: sample.copy())

    result = engine.fetch_supplier_data({"supplier_candidates": ["S2"]})

    assert list(result["supplier_id"]) == ["S2"]
    assert list(result["supplier_name"]) == ["Beta"]


def test_fetch_supplier_data_uses_directory_fallback(monkeypatch):
    engine = QueryEngine(
        agent_nick=types.SimpleNamespace(get_db_connection=lambda: DummyContext())
    )
    monkeypatch.setattr(engine, "_price_expression", lambda *a, **k: "0")
    monkeypatch.setattr(engine, "_quantity_expression", lambda *a, **k: "1")
    monkeypatch.setattr(engine, "_get_columns", lambda *a, **k: [])

    sample = pd.DataFrame(
        {
            "supplier_id": ["S1"],
            "supplier_name": ["Alpha"],
            "po_spend": [10.0],
            "invoice_spend": [5.0],
            "total_spend": [15.0],
            "invoice_count": [1],
            "on_time_pct": [1.0],
        }
    )

    monkeypatch.setattr(pd, "read_sql", lambda sql, conn: sample.copy())

    payload = {
        "supplier_candidates": ["S3"],
        "supplier_directory": [
            {"supplier_id": "S3", "supplier_name": "Gamma Corp"}
        ],
    }

    result = engine.fetch_supplier_data(payload)

    assert list(result["supplier_id"]) == ["S3"]
    assert result.loc[0, "supplier_name"] == "Gamma Corp"
    assert set(result.columns) == set(sample.columns)


def test_fetch_procurement_flow_builds_expected_query(monkeypatch):
    engine = QueryEngine(
        agent_nick=types.SimpleNamespace(get_db_connection=lambda: DummyContext())
    )
    queries = []

    base_df = pd.DataFrame(
        {
            "supplier_id": [1],
            "supplier_name": ["Acme"],
            "po_id": [10],
            "po_line_id": [100],
            "item_description": ["Widget"],
            "invoice_id": [20],
            "invoice_line_id": [200],
        }
    )
    category_df = pd.DataFrame(
        {
            "product": ["Widget"],
            "category_level_1": ["Goods"],
            "category_level_2": ["Hardware"],
            "category_level_3": ["Components"],
            "category_level_4": ["Widgets"],
            "category_level_5": ["Widget Type"],
        }
    )

    def fake_read_sql(sql, conn):
        queries.append(sql)
        if "proc.cat_product_mapping" in sql:
            return category_df
        return base_df

    monkeypatch.setattr(pd, "read_sql", fake_read_sql)
    # The category catalog is present in this scenario, so enrichment runs.
    monkeypatch.setattr(engine, "_table_exists", lambda *a, **k: True)

    df = engine.fetch_procurement_flow()

    assert queries, "No SQL queries captured"
    assert any("proc.cat_product_mapping" in q for q in queries)

    main_query = next(q for q in queries if "proc.cat_product_mapping" not in q)
    for table in [
        "proc.bp_contracts",
        "proc.bp_supplier",
        "proc.bp_purchase_order_trgt",
        "proc.bp_po_line_items_trgt",
        "proc.bp_invoice_trgt",
        "proc.bp_invoice_line_items_trgt",
    ]:
        assert table in main_query

    assert not df.empty
    assert df.loc[0, "product"] == "Widget"


def test_fetch_procurement_flow_embeds_summary(monkeypatch):
    engine = QueryEngine(
        agent_nick=types.SimpleNamespace(get_db_connection=lambda: DummyContext())
    )
    base_df = pd.DataFrame(
        {
            "supplier_id": [1],
            "supplier_name": ["Acme"],
            "po_id": [10],
            "po_line_id": [100],
            "item_description": ["Widget"],
            "invoice_id": [20],
            "invoice_line_id": [200],
        }
    )
    category_df = pd.DataFrame(
        {
            "product": ["Widget"],
            "category_level_1": ["Goods"],
            "category_level_2": ["Hardware"],
            "category_level_3": ["Components"],
            "category_level_4": ["Widgets"],
            "category_level_5": ["Widget Type"],
        }
    )

    def fake_read_sql(sql, conn):
        if "proc.cat_product_mapping" in sql:
            return category_df
        return base_df

    monkeypatch.setattr(pd, "read_sql", fake_read_sql)
    # The category catalog is present in this scenario, so enrichment runs.
    monkeypatch.setattr(engine, "_table_exists", lambda *a, **k: True)

    called = {}

    def fake_embed(df):
        called["df"] = df

    monkeypatch.setattr(engine, "_embed_procurement_summary", fake_embed)

    engine.fetch_procurement_flow(embed=True)

    expected_df = base_df.copy()
    for field in PROCUREMENT_CATEGORY_FIELDS:
        expected_df[field] = category_df.loc[0, field]

    pd.testing.assert_frame_equal(
        called["df"].reset_index(drop=True),
        expected_df.reset_index(drop=True),
        check_like=True,
    )


def test_fetch_procurement_flow_handles_missing_category_mapping(monkeypatch):
    engine = QueryEngine(
        agent_nick=types.SimpleNamespace(get_db_connection=lambda: DummyContext())
    )

    base_df = pd.DataFrame(
        {
            "supplier_id": [1],
            "supplier_name": ["Acme"],
            "po_id": [10],
            "po_line_id": [100],
            "item_description": ["Widget"],
            "invoice_id": [20],
            "invoice_line_id": [200],
        }
    )
    empty_category_df = pd.DataFrame(columns=PROCUREMENT_CATEGORY_FIELDS)

    def fake_read_sql(sql, conn):
        if "proc.cat_product_mapping" in sql:
            return empty_category_df
        return base_df

    monkeypatch.setattr(pd, "read_sql", fake_read_sql)

    df = engine.fetch_procurement_flow()

    for field in PROCUREMENT_CATEGORY_FIELDS:
        assert df[field].isna().all()


# ---------------------------------------------------------------------------
# P5: proc.cat_product_mapping was never built. Column introspection against
# a genuinely missing table must not be logged as an ERROR+traceback (that
# is _table_exists's own, expected "does the table exist?" check working as
# designed) — and the absence must be surfaced once via
# services.capability_status rather than silently disappearing.
# ---------------------------------------------------------------------------

import logging

import sqlalchemy


class _NoCursorConn:
    """Mimics the shape _pandas_reader actually hands back in production:
    a SQLAlchemy Connection, which has no ``.cursor`` attribute — forcing
    _get_columns down the ``sqlalchemy.inspect`` introspection path."""


def test_get_columns_missing_table_does_not_log_error_traceback(monkeypatch, caplog):
    class _RaisingInspector:
        def get_columns(self, table, schema=None):
            raise sqlalchemy.exc.NoSuchTableError(f"{schema}.{table}")

    monkeypatch.setattr(sqlalchemy, "inspect", lambda conn: _RaisingInspector())

    engine = QueryEngine(agent_nick=types.SimpleNamespace())
    with caplog.at_level(logging.DEBUG):
        cols = engine._get_columns(_NoCursorConn(), "proc", "cat_product_mapping")

    assert cols == []
    error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert error_records == [], [r.getMessage() for r in error_records]
    assert "Traceback" not in caplog.text


def test_get_columns_missing_table_marks_capability_degraded(monkeypatch):
    import importlib

    import services.capability_status as capability_status

    importlib.reload(capability_status)

    class _RaisingInspector:
        def get_columns(self, table, schema=None):
            raise sqlalchemy.exc.NoSuchTableError(f"{schema}.{table}")

    monkeypatch.setattr(sqlalchemy, "inspect", lambda conn: _RaisingInspector())

    engine = QueryEngine(agent_nick=types.SimpleNamespace())
    engine._get_columns(_NoCursorConn(), "proc", "cat_product_mapping")

    degraded = {d["capability"]: d["reason"] for d in capability_status.get_degraded()}
    assert "product_category_enrichment" in degraded
    assert "cat_product_mapping" in degraded["product_category_enrichment"]

    importlib.reload(capability_status)


def test_get_columns_genuine_introspection_failure_still_logged(monkeypatch, caplog):
    """A real, unexpected introspection failure (not "table doesn't exist")
    must still be logged loudly — only the expected/handled missing-relation
    case is quieted."""

    class _BrokenInspector:
        def get_columns(self, table, schema=None):
            raise RuntimeError("connection reset by peer")

    monkeypatch.setattr(sqlalchemy, "inspect", lambda conn: _BrokenInspector())

    engine = QueryEngine(agent_nick=types.SimpleNamespace())
    with caplog.at_level(logging.DEBUG):
        cols = engine._get_columns(_NoCursorConn(), "proc", "bp_supplier")

    assert cols == []
    error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert error_records, "a genuinely unexpected error must still be logged"


def test_assign_procurement_categories_uses_vector_fallback():
    engine = QueryEngine(agent_nick=types.SimpleNamespace())

    df = pd.DataFrame(
        {
            "supplier_id": ["S1", "S1"],
            "supplier_name": ["Alpha", "Alpha"],
            "po_id": ["PO-1", "PO-2"],
            "po_line_id": [101, 202],
            "item_description": ["Heavy duty bolt", "Cable tie pack"],
            "invoice_id": ["INV-1", "INV-2"],
            "invoice_line_id": [501, 502],
        }
    )

    category_df = pd.DataFrame(
        {
            "product": ["Bolt", "Cable Tie"],
            "category_level_1": ["Hardware", "Hardware"],
            "category_level_2": ["Fasteners", "Fasteners"],
            "category_level_3": [None, None],
            "category_level_4": [None, None],
            "category_level_5": [None, None],
        }
    )

    result = engine._assign_procurement_categories(df.copy(), category_df)

    assert list(result["product"]) == ["Bolt", "Cable Tie"]


def test_train_procurement_context_embeds_schema(monkeypatch):
    class DummyConnection:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

    agent = types.SimpleNamespace(
        get_db_connection=lambda: DummyConnection(),
        settings=types.SimpleNamespace(
            qdrant_collection_name="procwise_document_embeddings",
            knowledge_graph_collection_name="procwise_knowledge_graph",
        ),
    )
    engine = QueryEngine(agent_nick=agent)

    # stub column discovery across all procurement tables
    monkeypatch.setattr(
        engine,
        "_get_columns",
        lambda conn, schema, table: [f"{table}_c1", f"{table}_c2"],
    )

    sample_frames = {
        "proc.bp_contracts": pd.DataFrame({"contract_id": ["CO1"], "supplier_id": ["SI1"]}),
        "proc.bp_supplier": pd.DataFrame({"supplier_id": ["SI1"], "supplier_name": ["Acme"]}),
        "proc.bp_purchase_order_trgt": pd.DataFrame({"po_id": ["PO1"], "supplier_id": ["SI1"]}),
        "proc.bp_po_line_items_trgt": pd.DataFrame({"po_id": ["PO1"], "item_description": ["Widget"]}),
        "proc.bp_invoice_trgt": pd.DataFrame({"invoice_id": ["IN1"], "po_id": ["PO1"]}),
        "proc.bp_invoice_line_items_trgt": pd.DataFrame({"invoice_id": ["IN1"], "po_id": ["PO1"]}),
        "proc.cat_product_mapping": pd.DataFrame({"product": ["Widget"], "category_level_2": ["Hardware"]}),
        "proc.bp_quote_trgt": pd.DataFrame({"quote_id": ["Q1"], "po_id": ["PO1"]}),
        "proc.bp_quote_line_items_trgt": pd.DataFrame({"quote_id": ["Q1"], "line_total": [100.0]}),
    }

    def fake_read_sql(sql, conn):
        for canonical, df in sample_frames.items():
            if canonical in sql:
                return df.copy()
        return pd.DataFrame()

    captured: dict[str, Any] = {}

    class DummyRAG:
        def __init__(self, *args, **kwargs):
            pass

        def upsert_texts(self, texts, metadata=None):
            captured["texts"] = texts
            captured["metadata"] = metadata

    class DummyManager:
        def __init__(self, *args, **kwargs):
            captured["collection"] = kwargs.get("collection_name")

        def build_data_flow_map(self, tables, table_name_map=None):
            captured["tables"] = tables
            captured["table_name_map"] = table_name_map
            return ([{"status": "linked", "relationship_type": "references"}], {"paths": [], "supplier_flows": []})

        def persist_knowledge_graph(self, relations, graph):
            captured["persist"] = (relations, graph)

    def fake_flow(self, embed=False, supplier_ids=None, supplier_names=None):
        captured["embed"] = embed
        return pd.DataFrame()

    monkeypatch.setattr(QueryEngine, "fetch_procurement_flow", fake_flow)

    import services.rag_service as rag_module
    import services.data_flow_manager as df_module
    import engines.query_engine as qe_module
    import services.procurement_knowledge_service as pk_module

    class DummyKnowledgeService:
        def __init__(self, *args, **kwargs):
            pass

        def load_briefs(self):
            captured["knowledge_loaded"] = True
            return [types.SimpleNamespace(identifier="brief-1", title="t", summary="s")]

        def embed_briefs(self, briefs):
            captured["knowledge_embedded"] = [b.identifier for b in briefs]

    monkeypatch.setattr(rag_module, "RAGService", DummyRAG)
    monkeypatch.setattr(df_module, "DataFlowManager", DummyManager)
    monkeypatch.setattr(qe_module, "read_sql_compat", fake_read_sql)
    monkeypatch.setattr(pk_module, "ProcurementKnowledgeService", DummyKnowledgeService)

    engine.train_procurement_context()

    assert captured["metadata"]["record_id"] == "procurement_schema"
    assert captured["metadata"]["document_type"] == "procurement_schema"
    assert any("contracts_c1" in text for text in captured["texts"])
    assert isinstance(captured["tables"], dict) and "contracts" in captured["tables"]
    assert captured["table_name_map"]["contracts"] == "proc.bp_contracts"
    assert captured["persist"][0][0]["status"] == "linked"
    assert captured["embed"] is True
    assert captured["knowledge_loaded"] is True
    assert captured["knowledge_embedded"] == ["brief-1"]


class DummyContext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        pass


def test_fetch_invoice_data_survives_missing_supplier_name_column(monkeypatch):
    """proc.bp_invoice_trgt has supplier_id but NO supplier_name column.

    ``i.*`` therefore never yields a ``supplier_name`` column; only the
    ``supplier_lookup`` CTE join contributes ``supplier_name_master``. The old
    code did ``df.get("supplier_name")`` (returns ``None`` when absent) into
    ``combine_first`` -- which raises ``AttributeError`` on a bare ``None``.
    This must not crash, and the resulting supplier_name must come from the
    master lookup.
    """
    import engines.query_engine as qe_module

    engine = QueryEngine(agent_nick=types.SimpleNamespace(get_db_connection=lambda: DummyContext()))

    # Mirrors what the real SQL returns: i.* (no supplier_name) plus the
    # supplier_lookup join columns.
    fake_df = pd.DataFrame(
        {
            "invoice_id": ["INV-1"],
            "supplier_id": ["SUP-1"],
            "supplier_id_lookup": ["SUP-1"],
            "supplier_name_master": ["Acme Ltd"],
        }
    )

    monkeypatch.setattr(qe_module, "read_sql_compat", lambda sql, conn, params=None: fake_df.copy())

    df = engine.fetch_invoice_data()

    assert df.loc[0, "supplier_name"] == "Acme Ltd"
    assert "supplier_name_master" not in df.columns


# ---------------------------------------------------------------------------
# Restricted supplier columns
#
# ``fetch_supplier_data`` feeds ``input_data["supplier_data"]`` -- the shared
# workflow blackboard every agent reads from. Ranking competing quotes needs no
# banking detail, so the default projection must not carry it: an IBAN placed on
# the blackboard is an IBAN in every agent's prompt context and every serialised
# run record.
# ---------------------------------------------------------------------------

_BANK_COLUMNS = ("bank_name", "bank_account_number", "bank_swift", "bank_iban")

# What ``proc.bp_supplier`` actually holds. Deliberately spelled out here rather
# than imported from the engine, so these tests fail on the projection changing
# rather than passing vacuously alongside it.
_SUPPLIER_MASTER_COLUMNS = (
    "supplier_id",
    "supplier_name",
    "trading_name",
    "risk_score",
    "delivery_lead_time_days",
    "default_currency",
    "bank_name",
    "bank_account_number",
    "bank_swift",
    "bank_iban",
    "contact_name_1",
    "contact_email_1",
)


def _supplier_frame(monkeypatch, **kwargs):
    """Build a supplier frame against a database that returns what was projected.

    The fake reader answers with exactly the columns the generated SQL selects
    from the supplier alias, which is what a real database would do -- so a
    column reaching the frame means the projection asked for it.
    """

    import re

    import engines.query_engine as qe_module

    engine = QueryEngine(
        agent_nick=types.SimpleNamespace(get_db_connection=lambda: DummyContext())
    )
    monkeypatch.setattr(engine, "_price_expression", lambda *a, **k: "0")
    monkeypatch.setattr(engine, "_quantity_expression", lambda *a, **k: "1")
    monkeypatch.setattr(
        engine,
        "_get_columns",
        lambda conn, schema, table: list(_SUPPLIER_MASTER_COLUMNS),
    )

    captured = {}

    def fake_read_sql(sql, conn, params=None):
        captured["sql"] = sql
        projected = list(dict.fromkeys(re.findall(r"\bs\.([a-z0-9_]+)", sql)))
        return pd.DataFrame({column: [] for column in projected})

    monkeypatch.setattr(qe_module, "read_sql_compat", fake_read_sql)

    return engine.fetch_supplier_data(**kwargs), captured["sql"]


def test_supplier_frame_omits_bank_columns_by_default(monkeypatch):
    """The default frame carries no banking detail, and never asks for any."""

    df, sql = _supplier_frame(monkeypatch)

    leaked = [column for column in _BANK_COLUMNS if column in df.columns]
    assert leaked == [], f"banking columns reached the shared frame: {leaked}"

    # Not merely dropped after the fact -- the query must not select them.
    projected = [column for column in _BANK_COLUMNS if column in sql]
    assert projected == [], f"banking columns projected in SQL: {projected}"


def test_supplier_frame_includes_bank_columns_when_explicitly_requested(monkeypatch):
    """A caller that genuinely needs banking detail must ask for it by name."""

    df, sql = _supplier_frame(monkeypatch, include_restricted=True)

    missing = [column for column in _BANK_COLUMNS if column not in df.columns]
    assert missing == [], f"opt-in did not return: {missing}"


def test_supplier_frame_still_carries_contact_columns(monkeypatch):
    """Contact columns stay in the default projection -- for now.

    ``SupplierRankingAgent._prepare_ranking_entry`` reads ``contact_name_1`` /
    ``contact_email_1`` off this frame and republishes them as ``contact_name`` /
    ``contact_email`` on each ranking entry; ``EmailDraftingAgent._resolve_receiver``
    then takes ``contact_email`` as the RFQ recipient address. Restricting them
    here would silently blank out who RFQs are addressed to, so it waits on the
    email path resolving its own recipient by ``supplier_id``.

    This test exists to make that deferral visible: when the email path is fixed,
    it should fail, and the columns should move to SUPPLIER_FIELDS_RESTRICTED.
    """

    df, _ = _supplier_frame(monkeypatch)

    assert "contact_name_1" in df.columns
    assert "contact_email_1" in df.columns
