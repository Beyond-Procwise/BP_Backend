"""The three-way match: purchase order, goods receipt, invoice -- on quantity.

The PO line is the spine. Receipt lines and invoice lines both assign to it
through the SAME line matcher the two-way match uses, so the two sides of the
comparison can never disagree about which ordered line they mean.

Fixtures, not corpus: there are zero goods receipts in the corpus (design §13),
so every positive path here is exercised by a fixture and by the live run
recorded in Task 11. The negative paths -- a line with no receipt, a unit that
cannot be received, a pair of units that cannot be converted -- are the ones
most likely to be wrong, and they are each pinned by name below.
"""
from __future__ import annotations

import pytest

from src.services.extraction import three_way_match as twm3


def _po(line_no, desc, qty, uom="each"):
    return {"line_number": line_no, "item_description": desc,
            "quantity": qty, "unit_of_measure": uom}


def _rec(desc, qty, uom="each", rejected=None):
    row = {"item_description": desc, "quantity_received": qty, "unit_of_measure": uom}
    if rejected is not None:
        row["quantity_rejected"] = rejected
    return row


def _inv(desc, qty, uom="each"):
    return {"item_description": desc, "quantity": qty, "unit_of_measure": uom}


def test_receipt_lines_assign_to_the_po_lines_they_describe():
    po = [_po(1, "FORD Focus 1.9TDI, 100 HP", 10),
          _po(2, "Floor mats, set", 10, "set")]
    rec = [_rec("FORD Focus 1.9TDI", 6)]
    assigned = twm3.assign_receipt_lines(rec, po, po_id="4500018832")
    assert assigned[0]["line_number"] == 1


def test_two_receipts_for_two_different_lines_do_not_collide():
    """The set view is the reason to delegate rather than reimplement: matching
    each line on its own lets two receipt lines settle on one PO line while
    another is reported as never delivered."""
    po = [_po(1, "FORD Focus 1.9TDI, 100 HP", 10),
          _po(2, "Floor mats, set", 10, "set")]
    rec = [_rec("Floor mats set", 10, "set"), _rec("FORD Focus 1.9TDI", 10)]
    assigned = twm3.assign_receipt_lines(rec, po, po_id="4500018832")
    assert {i: a["line_number"] for i, a in assigned.items()} == {0: 2, 1: 1}


def test_a_receipt_line_naming_nothing_on_the_order_is_left_unassigned():
    po = [_po(1, "FORD Focus 1.9TDI, 100 HP", 10)]
    rec = [_rec("Reams of A4 paper", 5)]
    assert twm3.assign_receipt_lines(rec, po, po_id="4500018832") == {}


# --- Review Focus #1 ---------------------------------------------------------

def test_a_receipt_in_a_different_unit_is_unverifiable_not_a_finding():
    """box against each is the likeliest real failure and the corpus cannot
    size it. Refusing is correct; guessing a conversion is not, and reporting a
    shortfall that is really a unit difference would be worse than silence."""
    po = [_po(1, "Widget A", 10, "box")]
    rec = [_rec("Widget A", 100, "each")]
    inv = [_inv("Widget A", 100, "each")]
    result = twm3.check(po_lines=po, receipt_lines=rec, invoice_lines=inv)
    assert result.findings == []
    assert result.unverifiable == [{"po_line": 1, "reason": "UNVERIFIABLE_UOM"}]


# --- the arithmetic ----------------------------------------------------------

def test_billed_more_than_received_raises_the_finding_this_exists_for():
    po = [_po(1, "Widget A", 10)]
    rec = [_rec("Widget A", 6)]
    inv = [_inv("Widget A", 10)]
    f = twm3.check(po_lines=po, receipt_lines=rec, invoice_lines=inv).findings
    assert [x["type"] for x in f] == ["BILLED_NOT_RECEIVED"]
    assert f[0]["ordered"] == 10 and f[0]["received"] == 6 and f[0]["billed"] == 10


def test_billed_less_than_received_is_silent():
    """Partial invoicing is normal procurement. Crying wolf on it is how a
    check gets ignored -- the same reasoning two_way_match applies to value."""
    po = [_po(1, "Widget A", 10)]
    rec = [_rec("Widget A", 10)]
    inv = [_inv("Widget A", 4)]
    assert twm3.check(po_lines=po, receipt_lines=rec, invoice_lines=inv).findings == []


def test_billed_exactly_what_was_received_is_silent():
    po = [_po(1, "Widget A", 10)]
    rec = [_rec("Widget A", 10)]
    inv = [_inv("Widget A", 10)]
    r = twm3.check(po_lines=po, receipt_lines=rec, invoice_lines=inv)
    assert r.findings == [] and r.assessed == [1]


def test_more_delivered_than_ordered_warns():
    po = [_po(1, "Widget A", 10)]
    rec = [_rec("Widget A", 12)]
    inv = [_inv("Widget A", 10)]
    f = twm3.check(po_lines=po, receipt_lines=rec, invoice_lines=inv).findings
    assert [x["type"] for x in f] == ["OVER_DELIVERED"]
    assert f[0]["received"] == 12 and f[0]["ordered"] == 10


def test_billed_with_no_receipt_at_all_is_its_own_finding():
    """Distinct from a shortfall: nothing arrived, and something was billed.
    Saying BILLED_NOT_RECEIVED for a received quantity of zero would read as
    'we got some of it'."""
    po = [_po(1, "Widget A", 10)]
    inv = [_inv("Widget A", 10)]
    f = twm3.check(po_lines=po, receipt_lines=[], invoice_lines=inv).findings
    assert [x["type"] for x in f] == ["NOTHING_RECEIVED"]
    assert f[0]["received"] == 0 and f[0]["billed"] == 10


# --- Review Focus #3 ---------------------------------------------------------

def test_two_invoices_each_fitting_alone_raise_once_together():
    """6 received; two invoices of 4 each. Either alone is under. Together they
    bill 8. Checking one document at a time cannot see this, which is exactly
    the hole the value check had to close."""
    po = [_po(1, "Widget A", 10)]
    rec = [_rec("Widget A", 6)]
    inv = [{**_inv("Widget A", 4), "invoice_id": "INV-1"},
           {**_inv("Widget A", 4), "invoice_id": "INV-2"}]
    f = twm3.check(po_lines=po, receipt_lines=rec, invoice_lines=inv).findings
    assert len(f) == 1 and f[0]["type"] == "BILLED_NOT_RECEIVED" and f[0]["billed"] == 8


def test_two_receipts_for_one_line_add_up():
    """Partial deliveries are ordinary. Two notes of 5 against a line of 10
    mean 10 arrived, not two separate shortfalls."""
    po = [_po(1, "Widget A", 10)]
    rec = [_rec("Widget A", 5), _rec("Widget A", 5)]
    inv = [_inv("Widget A", 10)]
    r = twm3.check(po_lines=po, receipt_lines=rec, invoice_lines=inv)
    assert r.findings == [] and r.assessed == [1]


# --- Review Focus #2 ---------------------------------------------------------

def test_a_po_line_with_no_receipt_is_not_assessed_not_failed():
    """Missing paperwork must not manufacture a failure rate. A line nobody
    sent a GRN for has not failed the match; it has not been checked."""
    po = [_po(1, "Widget A", 10)]
    r = twm3.check(po_lines=po, receipt_lines=[], invoice_lines=[])
    assert r.findings == []
    assert r.assessed == []
    assert r.unverifiable == [{"po_line": 1, "reason": "NO_RECEIPT"}]


def test_a_time_based_line_can_never_be_proved_by_a_delivery_note():
    """39.5% of PO lines are like this. Reported as unverifiable, never as a
    pass and never as a failure -- the headline rate is bounded by it and the
    board paper must show it as a denominator."""
    po = [_po(1, "Consultancy", 40, "hour")]
    inv = [_inv("Consultancy", 40, "hour")]
    r = twm3.check(po_lines=po, receipt_lines=[], invoice_lines=inv)
    assert r.findings == []
    assert r.unverifiable == [{"po_line": 1, "reason": "UNVERIFIABLE_BY_RECEIPT"}]


# --- Review Focus #5 ---------------------------------------------------------

def test_goods_delivered_and_refused_were_not_received():
    """quantity_rejected must not be summed into quantity_received, or a
    refused delivery reads as accepted."""
    po = [_po(1, "Widget A", 10)]
    rec = [_rec("Widget A", 10, rejected=4)]
    inv = [_inv("Widget A", 10)]
    f = twm3.check(po_lines=po, receipt_lines=rec, invoice_lines=inv).findings
    assert f[0]["type"] == "BILLED_NOT_RECEIVED" and f[0]["received"] == 6


# --- the tolerances are governed, not hard-coded -----------------------------

def test_a_missing_tolerance_refuses_rather_than_assuming_one():
    """project_governed_limits: an unset limit is not an unlimited one. With no
    policy row the match must raise, not quietly pass everything."""
    from src.services.governed_limits import LimitUnavailable

    po = [_po(1, "Widget A", 10)]
    with pytest.raises((LimitUnavailable, KeyError)):
        twm3.check(po_lines=po, receipt_lines=[_rec("Widget A", 6)],
                   invoice_lines=[_inv("Widget A", 10)], limits={})


def test_a_widened_tolerance_silences_the_finding():
    """The governed number actually governs. Same data, two tolerances."""
    po = [_po(1, "Widget A", 10)]
    rec = [_rec("Widget A", 6)]
    inv = [_inv("Widget A", 10)]
    tight = {"billed_over_received_qty": 0, "over_delivery_pct": 0.0}
    loose = {"billed_over_received_qty": 5, "over_delivery_pct": 0.0}
    assert twm3.check(po_lines=po, receipt_lines=rec, invoice_lines=inv,
                      limits=tight).findings != []
    assert twm3.check(po_lines=po, receipt_lines=rec, invoice_lines=inv,
                      limits=loose).findings == []
