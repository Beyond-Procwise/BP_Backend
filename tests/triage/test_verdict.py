from datetime import date

from tests.triage.helpers import deal, inv, line, pipeline, po, quote


def test_clean_deal_is_matched():
    v = pipeline(deal(po(), inv())).verdict
    assert v.verdict == "Matched" and v.summary == "Matched"


def test_partial_invoicing_is_matched_with_notes():
    v = pipeline(deal(po(lines=[line(1, qty="20")]), inv(lines=[line(1, qty="10")]))).verdict
    assert v.verdict == "Matched with notes" and v.notes >= 1


def test_price_error_is_blocked_with_a_summary():
    v = pipeline(deal(po(lines=[line(1, qty="300", price="12.00")]),
                      inv(lines=[line(1, qty="300", price="13.50")]))).verdict
    assert v.verdict == "Blocked" and v.s1 == 1
    assert v.summary.startswith("Blocked · 1 finding needs action")
    assert "exposure £450.00" in v.summary


def test_payment_terms_difference_needs_review():
    v = pipeline(deal(po(), inv(terms="60 days"))).verdict
    assert v.verdict == "Needs review" and v.s2 == 1


def test_quote_only_deal_is_incomplete():
    v = pipeline(deal(quote())).verdict
    assert v.verdict == "Incomplete" and v.s1 == v.s2 == 0


def test_uninvoiced_po_is_incomplete():
    assert pipeline(deal(po())).verdict.verdict == "Incomplete"


def test_invoice_without_po_is_incomplete():
    assert pipeline(deal(po(), inv(), inv("INV-2", po_id=None))).verdict.verdict == "Incomplete"
