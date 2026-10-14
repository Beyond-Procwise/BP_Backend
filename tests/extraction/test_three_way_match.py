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

#: The arithmetic is tested WITHOUT the database, deliberately.
#:
#: `receipt_basis()` reads proc.bp_uom_canonical and `governed_tolerances()`
#: reads proc.bp_policy, so the first version of this file needed a live
#: database for every case -- and the suite's documented default mode uses a
#: fake connection whose cursor is not a context manager, so 20 of these 23
#: guards were RED for anyone running `pytest` the ordinary way. Measured in
#: review: `13 failed, 3 passed` by default, `16 passed` with the live flag.
#: A guard that is red in the mode people actually run is a guard nobody reads.
#:
#: So the two pieces of reference data are injected. What they contain is
#: pinned against the live tables by tests/sql/test_uom_receipt_basis.py and
#: tests/governance/test_governed_limits.py, which is what keeps this from
#: becoming a copy that proves a value the product does not use.
_BASIS = {"each": "goods_receipt", "box": "goods_receipt", "pack": "goods_receipt",
          "tonne": "goods_receipt", "metre": "goods_receipt", "case": "goods_receipt",
          "hour": "service_entry", "day": "service_entry", "month": "service_entry",
          "licence": "service_entry", "seat": "service_entry",
          "module": "service_entry"}
_TOLERANCES = {"billed_over_received_qty": 0, "over_delivery_pct": 0.0}


def _basis(unit):
    return _BASIS.get(twm3._norm_uom(unit), "none")


def _check(**kw):
    """twm3.check with the reference data injected unless a case overrides it."""
    kw.setdefault("basis_lookup", _basis)
    kw.setdefault("limits", _TOLERANCES)
    return twm3.check(**kw)


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
    result = _check(po_lines=po, receipt_lines=rec, invoice_lines=inv)
    assert result.findings == []
    assert result.unverifiable == [{"po_line": 1, "reason": "UNVERIFIABLE_UOM"}]


# --- the arithmetic ----------------------------------------------------------

def test_billed_more_than_received_raises_the_finding_this_exists_for():
    po = [_po(1, "Widget A", 10)]
    rec = [_rec("Widget A", 6)]
    inv = [_inv("Widget A", 10)]
    f = _check(po_lines=po, receipt_lines=rec, invoice_lines=inv).findings
    assert [x["type"] for x in f] == ["BILLED_NOT_RECEIVED"]
    assert f[0]["ordered"] == 10 and f[0]["received"] == 6 and f[0]["billed"] == 10


def test_billed_less_than_received_is_silent():
    """Partial invoicing is normal procurement. Crying wolf on it is how a
    check gets ignored -- the same reasoning two_way_match applies to value."""
    po = [_po(1, "Widget A", 10)]
    rec = [_rec("Widget A", 10)]
    inv = [_inv("Widget A", 4)]
    assert _check(po_lines=po, receipt_lines=rec, invoice_lines=inv).findings == []


def test_billed_exactly_what_was_received_is_silent():
    po = [_po(1, "Widget A", 10)]
    rec = [_rec("Widget A", 10)]
    inv = [_inv("Widget A", 10)]
    r = _check(po_lines=po, receipt_lines=rec, invoice_lines=inv)
    assert r.findings == [] and r.assessed == [1]


def test_more_delivered_than_ordered_warns():
    po = [_po(1, "Widget A", 10)]
    rec = [_rec("Widget A", 12)]
    inv = [_inv("Widget A", 10)]
    f = _check(po_lines=po, receipt_lines=rec, invoice_lines=inv).findings
    assert [x["type"] for x in f] == ["OVER_DELIVERED"]
    assert f[0]["received"] == 12 and f[0]["ordered"] == 10


def test_billed_with_no_receipt_at_all_is_its_own_finding():
    """Distinct from a shortfall: nothing arrived, and something was billed.
    Saying BILLED_NOT_RECEIVED for a received quantity of zero would read as
    'we got some of it'."""
    po = [_po(1, "Widget A", 10)]
    inv = [_inv("Widget A", 10)]
    f = _check(po_lines=po, receipt_lines=[], invoice_lines=inv).findings
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
    f = _check(po_lines=po, receipt_lines=rec, invoice_lines=inv).findings
    assert len(f) == 1 and f[0]["type"] == "BILLED_NOT_RECEIVED" and f[0]["billed"] == 8


def test_two_receipts_for_one_line_add_up():
    """Partial deliveries are ordinary. Two notes of 5 against a line of 10
    mean 10 arrived, not two separate shortfalls."""
    po = [_po(1, "Widget A", 10)]
    rec = [_rec("Widget A", 5), _rec("Widget A", 5)]
    inv = [_inv("Widget A", 10)]
    r = _check(po_lines=po, receipt_lines=rec, invoice_lines=inv)
    assert r.findings == [] and r.assessed == [1]


# --- Review Focus #2 ---------------------------------------------------------

def test_a_po_line_with_no_receipt_is_not_assessed_not_failed():
    """Missing paperwork must not manufacture a failure rate. A line nobody
    sent a GRN for has not failed the match; it has not been checked."""
    po = [_po(1, "Widget A", 10)]
    r = _check(po_lines=po, receipt_lines=[], invoice_lines=[])
    assert r.findings == []
    assert r.assessed == []
    assert r.unverifiable == [{"po_line": 1, "reason": "NO_RECEIPT"}]


def test_a_time_based_line_can_never_be_proved_by_a_delivery_note():
    """39.5% of PO lines are like this. Reported as unverifiable, never as a
    pass and never as a failure -- the headline rate is bounded by it and the
    board paper must show it as a denominator."""
    po = [_po(1, "Consultancy", 40, "hour")]
    inv = [_inv("Consultancy", 40, "hour")]
    r = _check(po_lines=po, receipt_lines=[], invoice_lines=inv)
    assert r.findings == []
    assert r.unverifiable == [{"po_line": 1, "reason": "UNVERIFIABLE_BY_RECEIPT"}]


# --- Review Focus #5 ---------------------------------------------------------

def test_goods_delivered_and_refused_were_not_received():
    """quantity_rejected must not be summed into quantity_received, or a
    refused delivery reads as accepted."""
    po = [_po(1, "Widget A", 10)]
    rec = [_rec("Widget A", 10, rejected=4)]
    inv = [_inv("Widget A", 10)]
    f = _check(po_lines=po, receipt_lines=rec, invoice_lines=inv).findings
    assert f[0]["type"] == "BILLED_NOT_RECEIVED" and f[0]["received"] == 6


# --- the tolerances are governed, not hard-coded -----------------------------

def test_a_missing_tolerance_refuses_rather_than_assuming_one():
    """project_governed_limits: an unset limit is not an unlimited one. With no
    policy row the match must raise, not quietly pass everything."""
    from src.services.governed_limits import LimitUnavailable

    po = [_po(1, "Widget A", 10)]
    with pytest.raises((LimitUnavailable, KeyError)):
        twm3.check(po_lines=po, receipt_lines=[_rec("Widget A", 6)],
                   invoice_lines=[_inv("Widget A", 10)], limits={},
                   basis_lookup=_basis)


def test_a_widened_tolerance_silences_the_finding():
    """The governed number actually governs. Same data, two tolerances."""
    po = [_po(1, "Widget A", 10)]
    rec = [_rec("Widget A", 6)]
    inv = [_inv("Widget A", 10)]
    tight = {"billed_over_received_qty": 0, "over_delivery_pct": 0.0}
    loose = {"billed_over_received_qty": 5, "over_delivery_pct": 0.0}
    assert _check(po_lines=po, receipt_lines=rec, invoice_lines=inv,
                  limits=tight).findings != []
    assert _check(po_lines=po, receipt_lines=rec, invoice_lines=inv,
                  limits=loose).findings == []


# ==========================================================================
# What the whole-branch review of 2026-10-05 proved, with real inputs.
#
# Three of these turned "this could not be checked" into either a critical
# accusation against a supplier who did nothing wrong, or a clean pass. They
# are the two failure modes this feature exists to avoid, so they are grouped
# and named here rather than scattered.
# ==========================================================================

def test_an_unreadable_received_quantity_refuses_instead_of_reading_zero():
    """A receipt line with no usable quantity is NOT a delivery of nothing.

    `_f` returns None for an absent value and its own docstring says absence
    must not read as "nothing arrived" -- and then the call site did
    `_f(...) or 0.0`. A delivery note whose quantity column did not parse
    produced "10 each billed against 0 each received -- 10 more than arrived",
    critical, against a supplier who delivered everything. `quantity_received`
    is `required: true` in the schema, but build_line_items does not enforce
    required, so a line with only a description promotes with NULL.
    """
    po = [_po(1, "Widget A", 10)]
    inv = [_inv("Widget A", 10)]
    for broken in ({"item_description": "Widget A", "unit_of_measure": "each"},
                   {"item_description": "Widget A", "quantity_received": None,
                    "unit_of_measure": "each"},
                   {"item_description": "Widget A", "quantity_received": "",
                    "unit_of_measure": "each"}):
        r = _check(po_lines=po, receipt_lines=[broken], invoice_lines=inv)
        assert r.findings == [], f"accused on {broken!r}: {r.findings}"
        assert r.assessed == [], f"claimed to have assessed {broken!r}"
        assert r.unverifiable == [
            {"po_line": 1, "reason": twm3.UNVERIFIABLE_QUANTITY}], broken


def test_an_unreadable_billed_quantity_is_not_a_clean_pass():
    """The mirror image. An invoice line with no quantity billed 0, which is
    always <= what arrived, so the line counted as VERIFIED while the billed
    side was unreadable. Services and lump-sum lines have no quantity by
    design, so this is the common case, and it inflated the only rate this
    feature can report."""
    po = [_po(1, "Widget A", 10)]
    rec = [_rec("Widget A", 10)]
    r = _check(po_lines=po, receipt_lines=rec,
                   invoice_lines=[{"item_description": "Widget A",
                                   "unit_of_measure": "each"}])
    assert r.findings == []
    assert r.assessed == []
    assert r.unverifiable == [{"po_line": 1, "reason": twm3.UNVERIFIABLE_QUANTITY}]


def test_a_receipt_line_is_placed_by_the_po_line_it_names():
    """A delivery note that NAMES the PO line must be believed over a fuzzy
    description match.

    `po_line_ref` was extracted, stored and selected, and then never read:
    check() matched on description similarity alone. A note printing a
    shortened description or a bare item code -- which delivery notes routinely
    do -- went unplaced, and the PO line then read NOTHING_RECEIVED: "10 billed
    and no delivery recorded at all", for a line whose delivery note is on file
    and names it.
    """
    po = [_po(1, "Widget A, blue, 10mm", 10), _po(2, "Gasket set", 4)]
    rec = [{"item_description": "10mm blue widgets", "quantity_received": 10,
            "unit_of_measure": "each", "po_line_ref": "1"}]
    inv = [_inv("Widget A, blue, 10mm", 10)]
    r = _check(po_lines=po, receipt_lines=rec, invoice_lines=inv)
    assert [f["type"] for f in r.findings] == [], r.findings
    assert 1 in r.assessed


def test_a_receipt_line_that_cannot_be_placed_is_reported_not_dropped():
    """An unplaceable receipt line is a quantity that EXISTS. Dropping it in
    silence and then reporting the PO line as never delivered counts a real
    delivery as nonexistent."""
    po = [_po(1, "Widget A", 10)]
    rec = [{"item_description": "Reams of A4 paper", "quantity_received": 5,
            "unit_of_measure": "each"}]
    inv = [_inv("Widget A", 10)]
    r = _check(po_lines=po, receipt_lines=rec, invoice_lines=inv)
    assert [f["type"] for f in r.findings] == [], (
        "a note with an unplaced line must not support NOTHING_RECEIVED")
    assert {"po_line": None, "reason": twm3.RECEIPT_LINE_UNPLACED} in r.unverifiable


def test_a_unit_less_delivery_note_cannot_support_an_accusation():
    """Trusting the order's unit is safe for SILENCE, not for RAISING.

    PO 40 each; the note prints 5 with no unit (five boxes of eight); the
    invoice bills 40. The old rule read "5 received" and raised a critical
    over-billing of 35.
    """
    po = [_po(1, "Widget A", 40)]
    rec = [{"item_description": "Widget A", "quantity_received": 5}]
    inv = [_inv("Widget A", 40)]
    r = _check(po_lines=po, receipt_lines=rec, invoice_lines=inv)
    assert r.findings == []
    assert r.unverifiable == [{"po_line": 1, "reason": twm3.UNVERIFIABLE_UOM}]


def test_a_unit_less_note_that_agrees_is_still_silent():
    """The same missing unit must not become a finding in the other direction
    either. 40 billed, 40 counted, no unit printed: nothing to say, and
    nothing claimed as verified."""
    po = [_po(1, "Widget A", 40)]
    rec = [{"item_description": "Widget A", "quantity_received": 40}]
    inv = [_inv("Widget A", 40)]
    r = _check(po_lines=po, receipt_lines=rec, invoice_lines=inv)
    assert r.findings == []
    assert r.assessed == [], "a unit nobody printed is not a verified unit"


def test_more_rejected_than_received_refuses_rather_than_going_negative():
    """received 1, rejected 9 is a document that does not make sense. It
    produced "against -8 each received -- 9 each more than arrived"."""
    po = [_po(1, "Widget A", 10)]
    rec = [_rec("Widget A", 1, rejected=9)]
    inv = [_inv("Widget A", 1)]
    r = _check(po_lines=po, receipt_lines=rec, invoice_lines=inv)
    assert r.findings == []
    assert r.unverifiable == [{"po_line": 1, "reason": twm3.UNVERIFIABLE_QUANTITY}]


def test_the_injected_reference_data_matches_what_the_database_says():
    """The copy above is only safe because something compares it to the source.

    tests/conftest.py's governed-limit seed carries the same rule and
    tests/governance/test_governed_limits.py holds it to the live rows; this
    does the same job for the unit basis. Live-only, because that is the point.
    """
    import os

    if os.environ.get("PROCWISE_TEST_LIVE_DB", "").strip().lower() not in (
            "1", "true", "yes", "on"):
        pytest.skip("needs PROCWISE_TEST_LIVE_DB=1 -- it reads the live unit table")

    twm3.reset_unit_cache()
    drifted = {u: (expected, twm3.receipt_basis(u))
               for u, expected in _BASIS.items()
               if twm3.receipt_basis(u) != expected}
    assert drifted == {}, (
        "the injected unit basis has drifted from proc.bp_uom_canonical: "
        f"{drifted}")

    # The tolerances are compared against the SEED, not the live rows: the
    # suite's autouse fixture patches governed_limits._engine, so a live read is
    # not available here. The seed is itself held to the live rows by
    # tests/governance/test_governed_limits.py
    # ::test_the_in_memory_seed_matches_the_live_rows, so the chain closes --
    # this link just makes sure THIS file's copy is in it.
    from src.services import governed_limits
    governed_limits.reset_cache()
    seeded = twm3.governed_tolerances()
    assert {k: seeded[k] for k in _TOLERANCES} == _TOLERANCES, (
        f"the injected tolerances have drifted from the governed seed: {seeded}")
