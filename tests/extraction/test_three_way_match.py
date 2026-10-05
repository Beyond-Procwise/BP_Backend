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

@pytest.mark.xfail(
    strict=True,
    reason="three_way_match.check lands in Task 8; watched red here as "
           "AttributeError: module has no attribute 'check'",
)
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
