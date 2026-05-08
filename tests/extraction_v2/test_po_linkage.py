"""Tests for PO linkage cross-document validation."""
from __future__ import annotations

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.services.extraction_v2.invariants import Severity  # noqa: E402
from src.services.extraction_v2.po_linkage import PoLinkage  # noqa: E402


def test_passes_when_invoice_qty_within_authorized():
    fetch = lambda po_id: {"WIDGET-1": 100.0, "WIDGET-2": 50.0}
    v = PoLinkage(fetch_fn=fetch)
    h = {"po_id": "PO-1"}
    lines = [
        {"item_id": "WIDGET-1", "quantity": 80, "line_amount": 800},
        {"item_id": "WIDGET-2", "quantity": 50, "line_amount": 500},
    ]
    r = v.check(h, lines, "Invoice")
    assert r.passed is True


def test_fails_critical_when_over_billed():
    fetch = lambda po_id: {"WIDGET-1": 100.0}
    v = PoLinkage(fetch_fn=fetch)
    h = {"po_id": "PO-1"}
    lines = [{"item_id": "WIDGET-1", "quantity": 150, "line_amount": 1500}]
    r = v.check(h, lines, "Invoice")
    assert r.passed is False
    assert r.severity == Severity.CRITICAL
    assert "over-billing" in r.message


def test_allows_small_over_billing_within_tolerance():
    """1% or 1 unit tolerance for rounding / partial-delivery cases."""
    fetch = lambda po_id: {"WIDGET-1": 100.0}
    v = PoLinkage(fetch_fn=fetch)
    h = {"po_id": "PO-1"}
    lines = [{"item_id": "WIDGET-1", "quantity": 101, "line_amount": 1010}]
    r = v.check(h, lines, "Invoice")
    # 1 unit over an authorized 100 → within tolerance
    assert r.passed is True


def test_fails_when_item_billed_but_not_on_po():
    fetch = lambda po_id: {"WIDGET-1": 100.0}
    v = PoLinkage(fetch_fn=fetch)
    h = {"po_id": "PO-1"}
    lines = [{"item_id": "ROGUE-99", "quantity": 1, "line_amount": 50}]
    r = v.check(h, lines, "Invoice")
    assert r.passed is False
    assert "not_on_PO" in r.message


def test_warns_when_po_id_not_found():
    fetch = lambda po_id: {}  # PO not in DB
    v = PoLinkage(fetch_fn=fetch)
    h = {"po_id": "PO-NOTFOUND"}
    lines = [{"item_id": "WIDGET-1", "quantity": 1, "line_amount": 50}]
    r = v.check(h, lines, "Invoice")
    assert r.passed is False
    assert r.severity == Severity.WARNING


def test_aggregates_quantity_across_multiple_lines_for_same_item():
    fetch = lambda po_id: {"WIDGET-1": 100.0}
    v = PoLinkage(fetch_fn=fetch)
    h = {"po_id": "PO-1"}
    lines = [
        {"item_id": "WIDGET-1", "quantity": 60, "line_amount": 600},
        {"item_id": "WIDGET-1", "quantity": 60, "line_amount": 600},
    ]
    r = v.check(h, lines, "Invoice")
    # Cumulative 120 > 101 (1% of 100 + abs tol) → over-billing
    assert r.passed is False


def test_na_when_no_po_id():
    v = PoLinkage(fetch_fn=lambda po_id: {})
    r = v.check({}, [{"item_id": "X", "quantity": 1}], "Invoice")
    assert r.passed is True
    assert r.message == "not_applicable"


def test_na_for_non_invoice_doc_types():
    v = PoLinkage(fetch_fn=lambda po_id: {"X": 100.0})
    assert v.applicable("Quote") is False
    assert v.applicable("Purchase_Order") is False
    assert v.applicable("Invoice") is True


def test_db_unavailable_skips_with_info():
    """A flaky DB must not fail a clean extraction."""
    def raising_fetch(po_id):
        return None
    v = PoLinkage(fetch_fn=raising_fetch)
    h = {"po_id": "PO-1"}
    lines = [{"item_id": "X", "quantity": 1, "line_amount": 50}]
    r = v.check(h, lines, "Invoice")
    assert r.passed is True
    assert r.severity == Severity.INFO
    assert "po_lookup_skipped" in r.message
