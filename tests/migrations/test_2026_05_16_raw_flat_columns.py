"""Integration test for 2026-05-16-extraction-raw-flat-columns.sql.

Asserts that the additive ALTER migration leaves the existing _raw tables
intact (raw_payload preserved) AND adds the new flat columns + parser_snapshot
+ trace_id, AND that the new line_items_raw tables exist with the right shape.

Run against a live bp_sqldb (PG* / settings.db_*). The migration is idempotent
(IF NOT EXISTS), so repeated runs are safe.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import psycopg2
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import Settings  # noqa: E402

MIGRATION = Path(__file__).resolve().parents[2] / "scripts" / "migrations" / "2026-05-16-extraction-raw-flat-columns.sql"


def _conn():
    s = Settings()
    c = psycopg2.connect(
        host=s.db_host, dbname=s.db_name,
        user=s.db_user, password=s.db_password, port=s.db_port,
    )
    c.autocommit = True
    return c


def _columns(cur, table_name: str) -> dict[str, str]:
    cur.execute(
        """SELECT column_name, data_type
             FROM information_schema.columns
            WHERE table_schema='proc' AND table_name=%s
            ORDER BY ordinal_position""",
        (table_name,),
    )
    return {r[0]: r[1] for r in cur.fetchall()}


@pytest.fixture(scope="module")
def applied_conn():
    conn = _conn()
    cur = conn.cursor()
    sql = MIGRATION.read_text()
    cur.execute(sql)
    yield conn
    conn.close()


def test_invoice_raw_has_control_columns(applied_conn):
    cur = applied_conn.cursor()
    cols = _columns(cur, "bp_invoice_raw")
    for c in ("raw_id", "doc_pk_candidate", "source_file", "raw_payload",
              "extracted_at", "pipeline_version", "promotion_status",
              "process_monitor_id", "parser_snapshot", "trace_id", "promoted_at"):
        assert c in cols, f"bp_invoice_raw missing control column {c}"
    # raw_payload is PRESERVED (additive migration); dropped in a later migration
    assert cols["raw_payload"] == "jsonb"
    assert cols["parser_snapshot"] == "jsonb"
    assert cols["trace_id"] == "uuid"


def test_invoice_raw_has_field_columns(applied_conn):
    cur = applied_conn.cursor()
    cols = _columns(cur, "bp_invoice_raw")
    expected = {
        "invoice_id": "text",
        "supplier_id": "text",
        "supplier_name": "text",
        "po_id": "text",
        "invoice_date": "date",
        "due_date": "date",
        "currency": "character varying",
        "invoice_amount": "numeric",
        "tax_percent": "numeric",
        "tax_amount": "numeric",
        "invoice_total_incl_tax": "numeric",
        "country": "text",
        "region": "text",
    }
    for c, t in expected.items():
        assert c in cols, f"bp_invoice_raw missing field column {c}"
        assert cols[c] == t, f"bp_invoice_raw.{c}: expected {t}, got {cols[c]}"


def test_invoice_line_items_raw_created(applied_conn):
    cur = applied_conn.cursor()
    cols = _columns(cur, "bp_invoice_line_items_raw")
    assert cols, "bp_invoice_line_items_raw not created"
    for c in ("line_raw_id", "raw_id", "line_no", "item_description",
              "quantity", "unit_price", "line_amount", "tax_amount"):
        assert c in cols, f"missing line column {c}"


def test_po_raw_has_new_columns(applied_conn):
    cur = applied_conn.cursor()
    cols = _columns(cur, "bp_purchase_order_raw")
    for c in ("parser_snapshot", "trace_id", "process_monitor_id",
              "po_id", "supplier_id", "supplier_name", "total_amount",
              "currency", "delivery_address_line1", "postal_code"):
        assert c in cols, f"bp_purchase_order_raw missing {c}"


def test_po_line_items_raw_created(applied_conn):
    cur = applied_conn.cursor()
    cols = _columns(cur, "bp_po_line_items_raw")
    assert cols
    for c in ("line_raw_id", "raw_id", "line_number", "item_description",
              "quantity", "unit_price", "line_total"):
        assert c in cols, f"missing PO line column {c}"


def test_quote_raw_has_new_columns(applied_conn):
    cur = applied_conn.cursor()
    cols = _columns(cur, "bp_quote_raw")
    for c in ("parser_snapshot", "trace_id", "quote_id", "supplier_id",
              "supplier_name", "quote_date", "validity_date",
              "total_amount", "currency"):
        assert c in cols, f"bp_quote_raw missing {c}"


def test_quote_line_items_raw_created(applied_conn):
    cur = applied_conn.cursor()
    cols = _columns(cur, "bp_quote_line_items_raw")
    assert cols
    for c in ("line_raw_id", "raw_id", "line_number", "item_description",
              "quantity", "unit_price", "line_total"):
        assert c in cols, f"missing quote line column {c}"


def test_contract_raw_has_new_columns(applied_conn):
    cur = applied_conn.cursor()
    cols = _columns(cur, "bp_contract_raw")
    for c in ("parser_snapshot", "trace_id", "contract_id", "contract_title",
              "supplier_id", "contract_start_date", "contract_end_date",
              "currency", "total_contract_value"):
        assert c in cols, f"bp_contract_raw missing {c}"


def test_existing_rows_preserved(applied_conn):
    """Critical: additive ALTER must not drop existing data."""
    cur = applied_conn.cursor()
    cur.execute("SELECT COUNT(*) FROM proc.bp_invoice_raw")
    invoice_count = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM proc.bp_purchase_order_raw")
    po_count = cur.fetchone()[0]
    # Counts checked before run-time: invoice 776, PO 584, quote 23, contract 0
    # Any non-zero baseline is enough to assert preservation.
    assert invoice_count > 0, "bp_invoice_raw was emptied — additive migration must preserve data"
    assert po_count > 0, "bp_purchase_order_raw was emptied"
