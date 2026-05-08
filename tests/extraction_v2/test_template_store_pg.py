"""Tests for PostgresTemplateStore using an in-process fake connection.

These tests verify:
  - upsert/get round-trips a VendorTemplate (incl. JSONB hints)
  - record_correction creates-or-updates and increments correction_count
  - record_success increments success_count and stamps last_used_at
  - record_line_item_hints persists LineItemHints
  - the implementation does not assume cursor return ordering

We run against an in-memory fake-Postgres that supports the SQL shapes
the store actually emits. End-to-end against a real Postgres lives in
the regression test suite (separate task).
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.services.extraction_v2.template_store import (  # noqa: E402
    FieldHint, LineItemHints, VendorTemplate,
)
from src.services.extraction_v2.template_store_pg import (  # noqa: E402
    PostgresTemplateStore,
)


class _FakeCursor:
    def __init__(self, table: dict):
        self._table = table
        self._fetched: Any = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def execute(self, sql: str, params=None):  # noqa: D401 - mimic psycopg2 API
        sql_norm = " ".join(sql.split()).lower()
        params = params or ()
        if sql_norm.startswith("create schema") or sql_norm.startswith("create table"):
            return
        if sql_norm.startswith("create index"):
            return
        if "insert into proc.bp_extraction_template" in sql_norm:
            (fp, vendor, doc_type, hints_json, line_hints_json,
             success, corrections, created_at, last_used) = params
            row = {
                "fingerprint": fp,
                "vendor_name": vendor,
                "doc_type": doc_type,
                "field_hints": hints_json,
                "line_item_hints": line_hints_json,
                "success_count": success,
                "correction_count": corrections,
                "created_at": created_at or "2026-05-04T00:00:00+00:00",
                "last_used_at": last_used,
            }
            self._table[fp] = row
            return
        if "update proc.bp_extraction_template" in sql_norm and "success_count = success_count + 1" in sql_norm:
            (fp,) = params
            row = self._table.get(fp)
            if row:
                row["success_count"] = int(row["success_count"]) + 1
                row["last_used_at"] = "now"
            return
        if "select fingerprint, vendor_name" in sql_norm:
            (fp,) = params
            row = self._table.get(fp)
            if not row:
                self._fetched = None
                return
            self._fetched = (
                row["fingerprint"], row["vendor_name"], row["doc_type"],
                row["field_hints"], row["line_item_hints"],
                row["success_count"], row["correction_count"],
                row["created_at"], row["last_used_at"],
            )
            return
        raise AssertionError(f"unexpected SQL: {sql_norm}")

    def fetchone(self):
        return self._fetched


class _FakeConn:
    def __init__(self, table: dict):
        self._table = table
        self.committed = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def cursor(self):
        return _FakeCursor(self._table)

    def commit(self):
        self.committed += 1


def _store_factory():
    table: dict = {}

    def get_conn():
        return _FakeConn(table)

    return PostgresTemplateStore(get_conn), table


def test_upsert_and_get_roundtrip_includes_field_hints_and_line_item_hints():
    store, table = _store_factory()
    tpl = VendorTemplate(
        fingerprint="abc123",
        vendor_name="ACME Co",
        doc_type="Invoice",
        field_hints={
            "supplier_name": FieldHint(
                field="supplier_name", value="ACME Co Ltd",
                confidence=0.99, label="From:",
            ),
        },
        line_item_hints=LineItemHints(
            header_anchors=["Description", "Qty", "Unit Price"],
            column_map={"Item Description": "item_description"},
            expected_min_rows=1,
        ),
    )
    store.upsert(tpl)
    assert "abc123" in table

    got = store.get("abc123")
    assert got is not None
    assert got.vendor_name == "ACME Co"
    assert got.doc_type == "Invoice"
    assert "supplier_name" in got.field_hints
    assert got.field_hints["supplier_name"].value == "ACME Co Ltd"
    assert got.field_hints["supplier_name"].confidence == 0.99
    assert got.line_item_hints is not None
    assert got.line_item_hints.column_map == {"Item Description": "item_description"}
    assert got.line_item_hints.expected_min_rows == 1


def test_get_returns_none_for_unknown_fingerprint():
    store, _ = _store_factory()
    assert store.get("unknown") is None


def test_record_correction_creates_template_when_absent_and_bumps_corrections():
    store, _ = _store_factory()
    store.record_correction(
        fingerprint="def456", field="supplier_name", value="Aquarius Marketing Ltd",
        confidence=0.95, doc_type="Quote", vendor_name="Aquarius",
    )
    got = store.get("def456")
    assert got is not None
    assert got.vendor_name == "Aquarius"
    assert got.field_hints["supplier_name"].value == "Aquarius Marketing Ltd"
    assert got.correction_count == 1


def test_record_correction_updates_existing_template_in_place():
    store, _ = _store_factory()
    tpl = VendorTemplate(fingerprint="ghi789", vendor_name="Existing",
                         doc_type="Invoice")
    store.upsert(tpl)
    store.record_correction(
        fingerprint="ghi789", field="buyer_name", value="WidgetCo",
    )
    got = store.get("ghi789")
    assert got is not None
    assert got.vendor_name == "Existing"
    assert "buyer_name" in got.field_hints
    assert got.correction_count == 1


def test_record_success_increments_success_count():
    store, table = _store_factory()
    store.upsert(VendorTemplate(fingerprint="jkl000", vendor_name="V",
                                doc_type="Invoice"))
    store.record_success("jkl000", fields_committed=("supplier_name",))
    got = store.get("jkl000")
    assert got is not None
    assert got.success_count == 1


def test_record_line_item_hints_persists_to_jsonb_column():
    store, table = _store_factory()
    hints = LineItemHints(
        header_anchors=["Item", "Qty", "Total"],
        column_map={"Item": "item_description", "Qty": "quantity",
                    "Total": "line_total"},
        expected_min_rows=2,
    )
    store.record_line_item_hints(
        "mno111", hints, doc_type="Invoice", vendor_name="Nexaspark",
    )
    got = store.get("mno111")
    assert got is not None
    assert got.line_item_hints is not None
    assert got.line_item_hints.expected_min_rows == 2
    assert got.line_item_hints.column_map["Total"] == "line_total"
    # The JSONB column actually stores a JSON string we can decode
    raw = table["mno111"]["line_item_hints"]
    parsed = json.loads(raw)
    assert parsed["header_anchors"] == ["Item", "Qty", "Total"]
