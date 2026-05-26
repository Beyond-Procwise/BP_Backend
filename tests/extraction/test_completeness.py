# tests/extraction/test_completeness.py
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.services.extraction.completeness import assess, line_sum  # noqa: E402


def test_complete_invoice_with_reconciling_lines():
    cols = {"invoice_id": "INV1", "invoice_amount": 100.0}
    lines = [{"line_amount": 60.0}, {"line_amount": 40.0}]
    r = assess("invoice", cols, lines, has_line_schema=True, missing_required=[])
    assert r.is_complete is True
    assert r.status == "complete"
    assert r.gaps == []


def test_missing_required_header_field():
    cols = {"invoice_amount": 100.0}
    lines = [{"line_amount": 100.0}]
    r = assess("invoice", cols, lines, has_line_schema=True, missing_required=["invoice_id"])
    assert r.is_complete is False
    assert r.status == "missing_required"


def test_no_line_items_when_expected():
    cols = {"invoice_id": "INV1", "invoice_amount": 100.0}
    r = assess("invoice", cols, [], has_line_schema=True, missing_required=[])
    assert r.status == "no_line_items"
    assert r.is_complete is False


def test_line_sum_mismatch_flagged():
    cols = {"invoice_id": "INV1", "invoice_amount": 100.0}
    lines = [{"line_amount": 30.0}]  # 30 vs 100 -> mismatch
    r = assess("invoice", cols, lines, has_line_schema=True, missing_required=[])
    assert r.status == "line_sum_mismatch"
    assert r.is_complete is False


def test_within_tolerance_reconciles():
    cols = {"invoice_id": "INV1", "invoice_amount": 100.0}
    lines = [{"line_amount": 98.0}]  # 2% < 5% tolerance
    r = assess("invoice", cols, lines, has_line_schema=True, missing_required=[])
    assert r.is_complete is True


def test_no_header_total_cannot_flag_mismatch():
    cols = {"invoice_id": "INV1"}  # no invoice_amount
    lines = [{"line_amount": 30.0}]
    r = assess("invoice", cols, lines, has_line_schema=True, missing_required=[])
    assert r.status == "complete"  # can't reconcile without a total -> don't flag


def test_doc_without_line_schema_is_complete_on_header():
    cols = {"contract_id": "C1"}
    r = assess("contract", cols, [], has_line_schema=False, missing_required=[])
    assert r.is_complete is True


def test_quote_uses_line_total_column():
    cols = {"quote_id": "Q1", "total_amount": 50.0}
    lines = [{"line_total": 50.0}]
    r = assess("quote", cols, lines, has_line_schema=True, missing_required=[])
    assert r.is_complete is True


def test_line_sum_helper():
    assert line_sum("invoice", [{"line_amount": 1.0}, {"line_amount": 2.5}]) == 3.5
    assert line_sum("quote", [{"line_total": 4.0}]) == 4.0
    assert line_sum("invoice", []) is None


def test_header_subtotal_helper():
    from src.services.extraction.completeness import header_subtotal
    assert header_subtotal("invoice", {"invoice_amount": "1234.50"}) == 1234.5
    assert header_subtotal("quote", {"total_amount": 50.0}) == 50.0
    assert header_subtotal("invoice", {}) is None
    assert header_subtotal("contract", {"foo": 1}) is None
