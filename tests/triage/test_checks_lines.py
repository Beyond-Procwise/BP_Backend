from datetime import date
from decimal import Decimal as D

from src.services.triage.checks import (
    check_description, check_line_arithmetic, check_quantity, check_unit_price,
    check_unlinked_lines)
from src.services.triage.link import link
from src.services.triage.model import Outcome
from tests.triage.helpers import deal, inv, line, make_cfg, po, quote

CFG = make_cfg()


def _run(check, ds):
    return check(ds, link(ds, CFG), CFG)


def _one(results):
    assert len(results) == 1, results
    return results[0]


# --- unit price -------------------------------------------------------------

def test_price_above_po_beyond_tolerance_is_conflict_with_exposure():
    ds = deal(po(lines=[line(1, qty="300", price="12.00")]),
              inv(lines=[line(1, qty="300", price="13.50")]))
    r = _one(_run(check_unit_price, ds))
    assert r.outcome == Outcome.CONFLICT
    assert r.exposure == D("450")
    assert r.delta == D("1.50")
    assert (r.claim_doc, r.auth_doc, r.po_id, r.claim_line) == ("INV-1", "PO-1", "PO-1", "1")


def test_price_within_one_percent_is_within_tolerance():
    ds = deal(po(lines=[line(1, price="12.00")]), inv(lines=[line(1, price="12.10")]))
    assert _one(_run(check_unit_price, ds)).outcome == Outcome.WITHIN_TOL


def test_price_just_beyond_one_percent_conflicts():
    ds = deal(po(lines=[line(1, price="12.00")]), inv(lines=[line(1, price="12.13")]))
    assert _one(_run(check_unit_price, ds)).outcome == Outcome.CONFLICT


def test_underpricing_has_a_looser_tolerance():
    ok = deal(po(lines=[line(1, price="12.00")]), inv(lines=[line(1, price="11.50")]))
    bad = deal(po(lines=[line(1, price="12.00")]), inv(lines=[line(1, price="11.00")]))
    assert _one(_run(check_unit_price, ok)).outcome == Outcome.WITHIN_TOL
    assert _one(_run(check_unit_price, bad)).outcome == Outcome.CONFLICT


def test_price_falls_back_to_quote_when_po_line_has_none():
    ds = deal(quote(lines=[line(1, price="12.00")]),
              po(quote_ref="Q-1", lines=[line(1, price=None, amount="120")]),
              inv(lines=[line(1, price="13.50")]))
    r = _one(_run(check_unit_price, ds))
    assert r.auth_doc == "Q-1" and r.outcome == Outcome.CONFLICT


def test_low_extraction_confidence_makes_conflict_unverifiable():
    ds = deal(po(lines=[line(1, price="12.00")]),
              inv(lines=[line(1, price="13.50")], confidence=0.5))
    assert _one(_run(check_unit_price, ds)).outcome == Outcome.UNVERIFIABLE


def test_credit_note_lines_are_not_price_checked():
    ds = deal(po(), inv("CN-1", lines=[line(1, price="99", amount="-990")], net="-990"))
    assert _run(check_unit_price, ds) == []


def test_line_without_price_or_quantity_is_skipped_not_crashed():
    ds = deal(po(), inv(lines=[line(1, qty=None, price=None, amount="120")]))
    assert _one(_run(check_unit_price, ds)).outcome == Outcome.ABSENT_SUBORDINATE
    assert _run(check_quantity, ds) == []
    assert _run(check_line_arithmetic, ds) == []


# --- quantity ---------------------------------------------------------------

def test_partial_invoicing_is_explained():
    ds = deal(po(lines=[line(1, qty="10")]), inv(lines=[line(1, qty="6")]))
    r = _one(_run(check_quantity, ds))
    assert r.outcome == Outcome.EXPLAINED and "partially invoiced" in r.note


def test_cumulative_quantity_over_po_conflicts_on_the_later_invoice():
    ds = deal(po(lines=[line(1, qty="10")]),
              inv("INV-1", lines=[line(1, qty="6")], inv_date=date(2026, 2, 1)),
              inv("INV-2", lines=[line(1, qty="6")], inv_date=date(2026, 3, 1)))
    r = _one(_run(check_quantity, ds))
    assert r.outcome == Outcome.CONFLICT
    assert r.claim_doc == "INV-2" and r.exposure == D("24")
    assert "INV-1" in r.note and "INV-2" in r.note


def test_small_over_delivery_is_within_allowance():
    ds = deal(po(lines=[line(1, qty="10")]), inv(lines=[line(1, qty="10.4")]))
    assert _one(_run(check_quantity, ds)).outcome == Outcome.WITHIN_TOL


def test_credit_note_quantity_is_subtracted():
    ds = deal(po(lines=[line(1, qty="10")]),
              inv("INV-1", lines=[line(1, qty="10")]),
              inv("INV-2", lines=[line(1, qty="10")]),
              inv("CN-1", lines=[line(1, qty="10", amount="-120")], net="-120"))
    assert _one(_run(check_quantity, ds)).outcome == Outcome.EXPLAINED


def test_low_confidence_early_invoice_makes_over_delivery_unverifiable():
    ds = deal(po(lines=[line(1, qty="10")]),
              inv("INV-1", lines=[line(1, qty="6")], inv_date=date(2026, 2, 1), confidence=0.5),
              inv("INV-2", lines=[line(1, qty="6")], inv_date=date(2026, 3, 1)))
    r = _one(_run(check_quantity, ds))
    assert r.outcome == Outcome.UNVERIFIABLE


# --- line arithmetic ---------------------------------------------------------

def test_line_amount_not_equal_to_qty_times_price_conflicts():
    ds = deal(po(), inv(lines=[line(1, amount="125.00")]))
    r = _one(_run(check_line_arithmetic, ds))
    assert r.outcome == Outcome.CONFLICT and r.exposure == D("5.00")


def test_rounding_difference_is_within_tolerance():
    ds = deal(po(), inv(lines=[line(1, amount="120.01")]))
    assert _one(_run(check_line_arithmetic, ds)).outcome == Outcome.WITHIN_TOL


def test_credit_note_line_arithmetic_uses_magnitudes():
    ds = deal(po(), inv("CN-1", lines=[line(1, amount="-120")], net="-120"))
    assert _one(_run(check_line_arithmetic, ds)).outcome == Outcome.MATCH


# --- description and unlinked lines ------------------------------------------

def test_description_conflict_when_item_codes_match_but_text_differs():
    ds = deal(po(lines=[line(1, desc="Steel bolts M8")]), inv(lines=[line(1, desc="Office chair")]))
    r = _one(_run(check_description, ds))
    assert r.outcome == Outcome.CONFLICT and r.field_class == "description"


def test_invoice_line_with_no_po_line_is_absent_authoritative():
    ds = deal(po(), inv(lines=[line(1), line(2, item="FRT", desc="Expedited freight",
                                           qty="1", price="120.00")]))
    r = _one(_run(check_unlinked_lines, ds))
    assert r.outcome == Outcome.ABSENT_AUTHORITATIVE
    assert r.exposure == D("120.00") and r.claim_line == "2"


def test_rolled_up_lines_are_explained_notes():
    ds = deal(po(lines=[line(1), line(2, item=None, desc="Installation", qty="1", price="5000.00")]),
              inv(lines=[line(1)] + [line(n, item=None, desc=f"Day {n}", qty="1", price="1250.00")
                                     for n in (2, 3, 4, 5)]))
    rs = _run(check_unlinked_lines, ds)
    assert [r.outcome for r in rs] == [Outcome.EXPLAINED] * 4
    assert {r.rule_id for r in rs} == {"rollup"}


# --- final review F3: credit notes never raise an unlinked-line finding -------

def test_credit_note_line_with_no_po_line_is_a_note_not_a_finding():
    ds = deal(po(), inv("CN-1", net="-975",
                        lines=[line(1, item="CRD", desc="Credit note — Q3 on-call overcharge",
                                    qty="1", price="975.00", amount="-975.00")]))
    r = _one(_run(check_unlinked_lines, ds))
    assert r.rule_id == "unlinked_line"
    assert r.outcome == Outcome.ABSENT_SUBORDINATE
    assert r.exposure == D("0") and r.note == "credit line with no PO line"


def test_negative_line_on_an_ordinary_invoice_is_a_note_not_a_finding():
    ds = deal(po(), inv(lines=[line(1), line(2, item="DSC", desc="Loyalty discount",
                                             qty="1", price="5.00", amount="-5.00")]))
    r = _one(_run(check_unlinked_lines, ds))
    assert r.outcome == Outcome.ABSENT_SUBORDINATE and r.exposure == D("0")
