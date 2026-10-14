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
    """An invoice that fits on its own but not beside one already on file.

    The gap belongs to BOTH invoices, so the one being read carries it back to
    its caller and the other is filed directly -- both are cleaned up here,
    because a finding written against a document this test invented must not
    outlive it.
    """
    old_inv, new_inv = f"INV-OLD-{order}", f"INV-NEW-{order}"
    try:
        with get_conn() as conn, conn.cursor() as cur:
            _receipt(cur, order, "GRN-A", 6)
            _invoice(cur, order, old_inv, 4)

        found = twm3.check_against_receipts(
            "invoice",
            {"po_id": order, "invoice_id": new_inv},
            [{"item_description": "Widget A", "quantity": 4, "unit_of_measure": "each"}],
        )
        assert [d.issue_type for d in found] == ["billed_not_received"]
        assert "8 each billed" in found[0].notes
        assert old_inv in found[0].notes and new_inv in found[0].notes

        # The other invoice's copy was filed, under its OWN source_file, so the
        # two cannot collide on the open-row key.
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("SELECT source_file FROM proc.bp_extraction_discrepancy "
                        "WHERE doc_pk_candidate = %s AND status = 'open'", (old_inv,))
            rows = cur.fetchall()
        assert len(rows) == 1 and old_inv in rows[0][0], rows
    finally:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM proc.bp_extraction_discrepancy "
                        "WHERE doc_pk_candidate IN (%s, %s)", (old_inv, new_inv))


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


def _open_gaps(invoice_id):
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT field_name, issue_type, status FROM proc.bp_extraction_discrepancy "
            "WHERE doc_pk_candidate = %s AND issue_type = 'billed_not_received' "
            "ORDER BY discrepancy_id", (invoice_id,))
        return cur.fetchall()


def test_a_gap_is_filed_against_the_invoice_that_over_billed(order):
    """The gap is contained in the INVOICE, not in whichever document was being
    read when it was spotted (design section 8).

    Filing it against the delivery note produced TWO rows for one gap -- one on
    the note, one on the invoice after a re-read -- each needing separate
    resolution, with the deal staying false until both were cleared. The
    receipt's own return value carries nothing, because the receipt did not
    over-bill anything.
    """
    inv = f"INV-OWNER-{order}"
    try:
        with get_conn() as conn, conn.cursor() as cur:
            _invoice(cur, order, inv, 10)

        returned = twm3.check_against_receipts(
            "goods_receipt", {"po_id": order, "grn_id": "GRN-B"},
            [{"item_description": "Widget A", "quantity_received": 6,
              "unit_of_measure": "each"}])
        assert [d.issue_type for d in returned
                if d.issue_type == "billed_not_received"] == [], (
            "the note did not over-bill; the finding is not its to carry")

        gaps = _open_gaps(inv)
        assert gaps == [("po_line[1]", "billed_not_received", "open")], gaps
    finally:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM proc.bp_extraction_discrepancy "
                        "WHERE doc_pk_candidate = %s", (inv,))


def test_a_receipt_arriving_later_closes_the_gap(order):
    """The module's whole claim for running on the receipt side: a note
    arriving after the bill is the moment an over-billing stops being one.

    That was computed and never recorded -- nothing in this codebase resolved
    an extraction discrepancy -- so a closed gap stayed open in the Action
    Centre for ever and the deal stayed false. Resolved, not deleted: the row
    is the record that it was once true.
    """
    inv = f"INV-CLOSES-{order}"
    try:
        with get_conn() as conn, conn.cursor() as cur:
            _invoice(cur, order, inv, 10)

        twm3.check_against_receipts(
            "goods_receipt", {"po_id": order, "grn_id": "GRN-B"},
            [{"item_description": "Widget A", "quantity_received": 6,
              "unit_of_measure": "each"}])
        assert [g[2] for g in _open_gaps(inv)] == ["open"]

        # The rest of the delivery arrives.
        twm3.check_against_receipts(
            "goods_receipt", {"po_id": order, "grn_id": "GRN-B"},
            [{"item_description": "Widget A", "quantity_received": 10,
              "unit_of_measure": "each"}])
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT status, resolved_by FROM proc.bp_extraction_discrepancy "
                "WHERE doc_pk_candidate = %s AND issue_type = 'billed_not_received'",
                (inv,))
            rows = cur.fetchall()
        assert rows == [("resolved", "three_way_match")], rows
    finally:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM proc.bp_extraction_discrepancy "
                        "WHERE doc_pk_candidate = %s", (inv,))


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


def test_a_receipt_whose_every_line_is_refused_leaves_the_deal_not_assessed(order):
    """The review's most serious finding, end to end.

    `check()` computed `unverifiable` correctly and `check_against_receipts`
    threw it away. The deal overview's verdict was "a receipt with lines exists
    AND no open gap" -- so a delivery note counting in `each` against an order
    in `box`, which the design calls the single most likely practical failure,
    raised nothing, had lines, and published as "Goods billed were received:
    100%".

    The receipt now records how many PO lines the match could actually compare,
    and the view requires that to be above zero before it gives a verdict.
    """
    deal = f"DEALTEST-REFUSED-{order}"
    grn, inv = f"GRN-REF-{order}", f"INV-REF-{order}"
    try:
        with get_conn() as conn, conn.cursor() as cur:
            # The order's line is in 'each' (the fixture); the note counts boxes.
            cur.execute("UPDATE proc.bp_purchase_order_trgt SET deal_id=%s, "
                        "deal_name='refusal probe' WHERE po_id=%s", (deal, order))
            cur.execute("INSERT INTO proc.bp_invoice_trgt (invoice_id, deal_id, "
                        "deal_name, invoice_amount, invoice_date) VALUES "
                        "(%s, %s, 'refusal probe', 100, DATE '2026-02-01')", (inv, deal))
            _invoice(cur, order, inv, 10)
            cur.execute("INSERT INTO proc.bp_goods_receipt_trgt (grn_id, po_id, deal_id) "
                        "VALUES (%s, %s, %s)", (grn, order, deal))
            _receipt(cur, order, grn, 10, uom="box")
            cur.execute(
                "INSERT INTO proc.bp_goods_receipt_line_items_trgt "
                "(goods_receipt_line_id, grn_id, line_no, item_description, "
                " quantity_received, unit_of_measure, po_id, deal_id) "
                "VALUES (%s, %s, 1, 'Widget A', 10, 'box', %s, %s)",
                (f"{grn}-L1", grn, order, deal))

        found = twm3.check_against_receipts(
            "goods_receipt", {"po_id": order, "grn_id": grn},
            [{"item_description": "Widget A", "quantity_received": 10,
              "unit_of_measure": "box"}])

        # No accusation, and the refusal is SAID OUT LOUD rather than implied.
        assert [d.issue_type for d in found] == ["receipt_unit_not_comparable"]
        assert found[0].severity == "info"
        assert found[0].blocks_promotion is False

        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("SELECT lines_assessed, lines_unverifiable "
                        "FROM proc.bp_goods_receipt_trgt WHERE grn_id=%s", (grn,))
            assert cur.fetchone() == (0, 1)
            cur.execute("SELECT three_way_matched FROM proc.bp_deal_overview "
                        "WHERE deal_id=%s", (deal,))
            row = cur.fetchone()
            assert row is not None, f"{deal} is not in bp_deal_overview"
            assert row[0] is None, (
                "a receipt whose every line was refused must leave the deal "
                "NOT ASSESSED, never 'goods received'")
    finally:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM proc.bp_goods_receipt_line_items_trgt WHERE grn_id=%s", (grn,))
            cur.execute("DELETE FROM proc.bp_goods_receipt_line_items_stg WHERE grn_id=%s", (grn,))
            cur.execute("DELETE FROM proc.bp_goods_receipt_trgt WHERE grn_id=%s", (grn,))
            cur.execute("DELETE FROM proc.bp_invoice_trgt WHERE invoice_id=%s", (inv,))
            cur.execute("DELETE FROM proc.bp_extraction_discrepancy WHERE doc_pk_candidate=%s", (inv,))


def test_a_receipt_the_match_could_compare_does_give_the_deal_a_verdict(order):
    """The other side of the gate: lines_assessed > 0 and the deal answers."""
    deal = f"DEALTEST-OK-{order}"
    grn, inv = f"GRN-OK-{order}", f"INV-OK-{order}"
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("UPDATE proc.bp_purchase_order_trgt SET deal_id=%s, "
                        "deal_name='verdict probe' WHERE po_id=%s", (deal, order))
            cur.execute("INSERT INTO proc.bp_invoice_trgt (invoice_id, deal_id, "
                        "deal_name, invoice_amount, invoice_date) VALUES "
                        "(%s, %s, 'verdict probe', 100, DATE '2026-02-01')", (inv, deal))
            _invoice(cur, order, inv, 10)
            cur.execute("INSERT INTO proc.bp_goods_receipt_trgt (grn_id, po_id, deal_id) "
                        "VALUES (%s, %s, %s)", (grn, order, deal))
            cur.execute(
                "INSERT INTO proc.bp_goods_receipt_line_items_trgt "
                "(goods_receipt_line_id, grn_id, line_no, item_description, "
                " quantity_received, unit_of_measure, po_id, deal_id) "
                "VALUES (%s, %s, 1, 'Widget A', 10, 'each', %s, %s)",
                (f"{grn}-L1", grn, order, deal))
            _receipt(cur, order, grn, 10)

        twm3.check_against_receipts(
            "goods_receipt", {"po_id": order, "grn_id": grn},
            [{"item_description": "Widget A", "quantity_received": 10,
              "unit_of_measure": "each"}])

        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("SELECT lines_assessed FROM proc.bp_goods_receipt_trgt "
                        "WHERE grn_id=%s", (grn,))
            assert cur.fetchone()[0] == 1
            cur.execute("SELECT three_way_matched FROM proc.bp_deal_overview "
                        "WHERE deal_id=%s", (deal,))
            assert cur.fetchone()[0] is True
    finally:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM proc.bp_goods_receipt_line_items_trgt WHERE grn_id=%s", (grn,))
            cur.execute("DELETE FROM proc.bp_goods_receipt_line_items_stg WHERE grn_id=%s", (grn,))
            cur.execute("DELETE FROM proc.bp_goods_receipt_trgt WHERE grn_id=%s", (grn,))
            cur.execute("DELETE FROM proc.bp_invoice_trgt WHERE invoice_id=%s", (inv,))
            cur.execute("DELETE FROM proc.bp_extraction_discrepancy WHERE doc_pk_candidate=%s", (inv,))


def test_the_same_invoice_spelled_differently_is_not_counted_twice(order):
    """`_with_this_document` compared ids with a plain strip(), so "INV-1"
    persisted and "inv-1" in hand were treated as two invoices -- the document's
    lines counted twice, manufacturing an over-billing out of nothing. Exactly
    the failure the substitution exists to prevent."""
    inv = f"INV-CASE-{order}"
    try:
        with get_conn() as conn, conn.cursor() as cur:
            _receipt(cur, order, "GRN-C", 10)
            _invoice(cur, order, inv, 10)

        for spelling in (inv, inv.lower(), inv.replace("-", "")):
            found = twm3.check_against_receipts(
                "invoice", {"po_id": order, "invoice_id": spelling},
                [{"item_description": "Widget A", "quantity": 10,
                  "unit_of_measure": "each"}])
            gaps = [d.issue_type for d in found
                    if d.issue_type == "billed_not_received"]
            assert gaps == [], f"{spelling!r} double-counted: {found and found[0].notes}"
    finally:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM proc.bp_extraction_discrepancy "
                        "WHERE doc_pk_candidate IN (%s, %s, %s)",
                        (inv, inv.lower(), inv.replace("-", "")))


def test_a_receipt_assessed_at_ingestion_records_its_own_denominator(order):
    """The match must run AFTER the receipt is persisted, or it writes nothing.

    Found on the live re-run: `check_against_receipts` is called from dispatch
    while it is still gathering discrepancies -- BEFORE promote() creates the
    _stg row and before link_receipt_to_po copies it to _trgt. So
    `_record_outcome`'s UPDATE matched zero rows, `lines_assessed` stayed NULL,
    and the verdict gate then read a correctly-assessed deal as NOT ASSESSED.
    The gate turned from a fix into a blindfold.
    """
    from src.services.extraction.goods_receipt_link import run_match_for_receipt

    deal = f"DEALTEST-ORDER-{order}"
    grn, inv = f"GRN-ORD-{order}", f"INV-ORD-{order}"
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("UPDATE proc.bp_purchase_order_trgt SET deal_id=%s, "
                        "deal_name='ordering probe' WHERE po_id=%s", (deal, order))
            cur.execute("INSERT INTO proc.bp_invoice_trgt (invoice_id, deal_id, "
                        "deal_name, invoice_amount, invoice_date) VALUES "
                        "(%s, %s, 'ordering probe', 100, DATE '2026-02-01')", (inv, deal))
            _invoice(cur, order, inv, 10)
            # The receipt as the pipeline leaves it: _stg and _trgt both present,
            # lines_assessed untouched.
            cur.execute("INSERT INTO proc.bp_goods_receipt_stg (grn_id, po_id, deal_id) "
                        "VALUES (%s, %s, %s)", (grn, order, deal))
            cur.execute("INSERT INTO proc.bp_goods_receipt_trgt (grn_id, po_id, deal_id) "
                        "VALUES (%s, %s, %s)", (grn, order, deal))
            _receipt(cur, order, grn, 6)
            cur.execute(
                "INSERT INTO proc.bp_goods_receipt_line_items_trgt "
                "(goods_receipt_line_id, grn_id, line_no, item_description, "
                " quantity_received, unit_of_measure, po_id, deal_id) "
                "VALUES (%s, %s, 1, 'Widget A', 6, 'each', %s, %s)",
                (f"{grn}-L1", grn, order, deal))

        assert run_match_for_receipt(grn)["assessed"] == 1

        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("SELECT lines_assessed, lines_unverifiable "
                        "FROM proc.bp_goods_receipt_trgt WHERE grn_id=%s", (grn,))
            assert cur.fetchone() == (1, 0)
            cur.execute("SELECT three_way_matched FROM proc.bp_deal_overview "
                        "WHERE deal_id=%s", (deal,))
            assert cur.fetchone()[0] is False, "6 received, 10 billed: a gap"
            cur.execute("SELECT count(*) FROM proc.bp_extraction_discrepancy "
                        "WHERE doc_pk_candidate=%s AND status='open' "
                        "AND issue_type='billed_not_received'", (inv,))
            assert cur.fetchone()[0] == 1
    finally:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM proc.bp_extraction_discrepancy WHERE doc_pk_candidate=%s", (inv,))
            cur.execute("DELETE FROM proc.bp_goods_receipt_line_items_trgt WHERE grn_id=%s", (grn,))
            cur.execute("DELETE FROM proc.bp_goods_receipt_line_items_stg WHERE grn_id=%s", (grn,))
            cur.execute("DELETE FROM proc.bp_goods_receipt_trgt WHERE grn_id=%s", (grn,))
            cur.execute("DELETE FROM proc.bp_goods_receipt_stg WHERE grn_id=%s", (grn,))
            cur.execute("DELETE FROM proc.bp_invoice_trgt WHERE invoice_id=%s", (inv,))
