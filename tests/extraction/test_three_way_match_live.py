"""The three-way match against real tables, end to end.

tests/extraction/test_three_way_match.py proves the arithmetic on fixtures.
This proves the SEAM: that the loaders read the real column names, that the
whole set of invoices against an order is considered, that a re-read does not
count the same invoice twice, and that a finding comes out shaped like every
other discrepancy in the queue.

Live-only. Run with:
    set -a && . ./.env && set +a
    PROCWISE_TEST_LIVE_DB=1 ./venv/bin/python -m pytest \
        tests/extraction/test_three_way_match_live.py
"""
from __future__ import annotations

import os
from uuid import uuid4

import pytest

from src.services.db import get_conn
from src.services.extraction import three_way_match as twm3

_LIVE = os.environ.get("PROCWISE_TEST_LIVE_DB", "").strip().lower() in ("1", "true", "yes", "on")
pytestmark = pytest.mark.skipif(not _LIVE, reason="needs PROCWISE_TEST_LIVE_DB=1")


@pytest.fixture()
def order():
    """One purchase order with one line of 10 'each', and nothing else."""
    tag = f"{uuid4().int % 100000000:08d}"
    po = f"47{tag}"
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("INSERT INTO proc.bp_purchase_order_trgt (po_id, supplier_name, deal_id) "
                    "VALUES (%s, 'Northwind Trading Ltd', %s)", (po, f"DEALTEST-{tag}"))
        cur.execute(
            "INSERT INTO proc.bp_po_line_items_trgt "
            "(po_line_id, po_id, line_number, item_description, quantity, unit_of_measure) "
            "VALUES (%s, %s, 1, 'Widget A', 10, 'each')", (f"{po}-L1", po))
    try:
        yield po
    finally:
        with get_conn() as conn, conn.cursor() as cur:
            for table, col in (
                ("proc.bp_goods_receipt_line_items_stg", "po_id"),
                ("proc.bp_goods_receipt_line_items_trgt", "po_id"),
                ("proc.bp_invoice_line_items_stg", "po_id"),
                ("proc.bp_invoice_line_items_trgt", "po_id"),
                ("proc.bp_po_line_items_trgt", "po_id"),
                ("proc.bp_purchase_order_trgt", "po_id"),
            ):
                cur.execute(f"DELETE FROM {table} WHERE {col} = %s", (po,))


def _receipt(cur, po, grn, qty, *, rejected=None, uom="each"):
    cur.execute(
        "INSERT INTO proc.bp_goods_receipt_line_items_stg "
        "(goods_receipt_line_id, grn_id, line_no, item_description, quantity_received, "
        " quantity_rejected, unit_of_measure, po_id) "
        "VALUES (%s, %s, 1, 'Widget A', %s, %s, %s, %s)",
        (f"{grn}-L1", grn, qty, rejected, uom, po))


def _invoice(cur, po, inv, qty, *, uom="each"):
    cur.execute(
        "INSERT INTO proc.bp_invoice_line_items_stg "
        "(invoice_line_id, invoice_id, line_no, item_description, quantity, "
        " unit_of_measure, po_id) "
        "VALUES (%s, %s, 1, 'Widget A', %s, %s, %s)",
        (f"{inv}-L1", inv, qty, uom, po))


def test_an_over_billed_invoice_raises_a_queue_shaped_finding(order):
    with get_conn() as conn, conn.cursor() as cur:
        _receipt(cur, order, "GRN-A", 6)

    found = twm3.check_against_receipts(
        "invoice",
        {"po_id": order, "invoice_id": "INV-NEW"},
        [{"item_description": "Widget A", "quantity": 10, "unit_of_measure": "each"}],
    )
    assert [d.issue_type for d in found] == ["billed_not_received"]
    d = found[0]
    assert d.severity == "critical"
    assert d.blocks_promotion is False
    assert d.field_name == "po_line[1]"
    assert "4 each more than arrived" in d.notes
    assert "GRN-A" in d.notes


def test_the_whole_set_of_invoices_is_considered_not_just_this_one(order):
    """An invoice that fits on its own but not beside one already on file."""
    with get_conn() as conn, conn.cursor() as cur:
        _receipt(cur, order, "GRN-A", 6)
        _invoice(cur, order, "INV-OLD", 4)

    found = twm3.check_against_receipts(
        "invoice",
        {"po_id": order, "invoice_id": "INV-NEW"},
        [{"item_description": "Widget A", "quantity": 4, "unit_of_measure": "each"}],
    )
    assert [d.issue_type for d in found] == ["billed_not_received"]
    assert "8 each billed" in found[0].notes
    assert "INV-OLD" in found[0].notes and "INV-NEW" in found[0].notes


def test_re_reading_the_same_invoice_does_not_double_count_it(order):
    """The persisted copy is SUBSTITUTED, not appended. Appending is how a
    re-read manufactures an over-billing out of nothing."""
    with get_conn() as conn, conn.cursor() as cur:
        _receipt(cur, order, "GRN-A", 10)
        _invoice(cur, order, "INV-SAME", 10)

    found = twm3.check_against_receipts(
        "invoice",
        {"po_id": order, "invoice_id": "INV-SAME"},
        [{"item_description": "Widget A", "quantity": 10, "unit_of_measure": "each"}],
    )
    assert found == [], [d.notes for d in found]


def test_a_receipt_arriving_later_closes_the_gap(order):
    """A goods receipt is extracted after the invoice. The match runs on the
    receipt side too, which is the only moment the gap can close."""
    with get_conn() as conn, conn.cursor() as cur:
        _invoice(cur, order, "INV-OLD", 10)

    short = twm3.check_against_receipts(
        "goods_receipt", {"po_id": order, "grn_id": "GRN-B"},
        [{"item_description": "Widget A", "quantity_received": 6,
          "unit_of_measure": "each"}])
    assert [d.issue_type for d in short] == ["billed_not_received"]

    full = twm3.check_against_receipts(
        "goods_receipt", {"po_id": order, "grn_id": "GRN-B"},
        [{"item_description": "Widget A", "quantity_received": 10,
          "unit_of_measure": "each"}])
    assert full == []


def test_an_order_with_no_receipt_raises_nothing(order):
    """The corpus has no receipts at all. If this said something, the queue
    would fill with findings about paperwork that was never sent."""
    assert twm3.check_against_receipts(
        "invoice", {"po_id": order, "invoice_id": "INV-NEW"},
        [{"item_description": "Widget A", "quantity": 10, "unit_of_measure": "each"}],
    ) == []


def test_a_quote_is_not_three_way_matched(order):
    assert twm3.check_against_receipts(
        "quote", {"po_id": order, "quote_id": "Q-1"},
        [{"item_description": "Widget A", "quantity": 10}]) == []


def test_the_po_is_found_through_the_same_normalisation_an_invoice_uses(order):
    with get_conn() as conn, conn.cursor() as cur:
        _receipt(cur, order, "GRN-A", 6)

    found = twm3.check_against_receipts(
        "invoice",
        {"po_id": f"PO-{order}", "invoice_id": "INV-NEW"},
        [{"item_description": "Widget A", "quantity": 10, "unit_of_measure": "each"}],
    )
    assert [d.issue_type for d in found] == ["billed_not_received"]
