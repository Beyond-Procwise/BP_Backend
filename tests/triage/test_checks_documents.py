from datetime import date
from decimal import Decimal as D

from src.services.triage.checks import (
    check_cumulative_total, check_currency, check_duplicates, check_invoice_date,
    check_invoice_totals, check_payment_terms, check_po_links, check_supplier,
    check_tax_rate, run_checks)
from src.services.triage.link import link
from src.services.triage.model import DuplicateFlag, Outcome
from tests.triage.helpers import deal, inv, line, make_cfg, po

CFG = make_cfg()


def _run(check, ds):
    return check(ds, link(ds, CFG), CFG)


def _one(results):
    assert len(results) == 1, results
    return results[0]


def test_invoice_lines_not_summing_to_net_conflict():
    rs = _run(check_invoice_totals, deal(po(), inv(net="130")))
    net = next(r for r in rs if r.field_name == "net")
    assert net.outcome == Outcome.CONFLICT and net.exposure == D("10")


def test_gross_not_equal_net_plus_tax_conflicts():
    rs = _run(check_invoice_totals, deal(po(), inv(gross="150")))
    gross = next(r for r in rs if r.field_name == "gross")
    assert gross.outcome == Outcome.CONFLICT and gross.exposure == D("6")


def test_invoices_beyond_po_total_conflict_on_the_po():
    ds = deal(po(), inv("INV-1"), inv("INV-2"))
    r = _one(_run(check_cumulative_total, ds))
    assert r.outcome == Outcome.CONFLICT and r.exposure == D("120")
    assert r.claim_doc == "PO-1" and r.po_id == "PO-1"
    assert D(r.tolerance["allowance"]) == D("0.6")


def test_credit_note_brings_running_total_back_within_po():
    ds = deal(po(), inv("INV-1"), inv("INV-2"), inv("CN-1", net="-120"))
    assert _one(_run(check_cumulative_total, ds)).outcome == Outcome.MATCH


def test_partly_invoiced_po_is_explained():
    ds = deal(po(net="500", lines=[line(1, qty="50")]), inv())
    assert _one(_run(check_cumulative_total, ds)).outcome == Outcome.EXPLAINED


def test_tax_at_an_allowed_rate_matches_and_off_rate_conflicts():
    assert _one(_run(check_tax_rate, deal(po(), inv()))).outcome == Outcome.MATCH
    assert _one(_run(check_tax_rate, deal(po(), inv(tax="6")))).outcome == Outcome.MATCH
    r = _one(_run(check_tax_rate, deal(po(), inv(tax="25"))))
    assert r.outcome == Outcome.CONFLICT and r.exposure == D("1.00")


def test_currency_mismatch_conflicts():
    r = _one(_run(check_currency, deal(po(), inv(currency="EUR"))))
    assert r.outcome == Outcome.CONFLICT and r.exposure == D("120")


def test_supplier_mismatch_conflicts():
    r = _one(_run(check_supplier, deal(po(), inv(supplier="SUP-2"))))
    assert r.outcome == Outcome.CONFLICT and r.field_class == "party"


def test_invoice_before_order_conflicts():
    r = _one(_run(check_invoice_date, deal(po(), inv(inv_date=date(2026, 1, 1)))))
    assert r.outcome == Outcome.CONFLICT


def test_payment_terms():
    assert _one(_run(check_payment_terms, deal(po(), inv()))).outcome == Outcome.MATCH
    assert _one(_run(check_payment_terms, deal(po(), inv(terms="60 days")))).outcome == Outcome.CONFLICT
    assert _one(_run(check_payment_terms, deal(po(), inv(terms=None)))).outcome == Outcome.ABSENT_SUBORDINATE
    assert _one(_run(check_payment_terms, deal(po(), inv(terms="on receipt")))).outcome == Outcome.UNVERIFIABLE


def test_flagged_duplicate_conflicts_with_invoice_net_as_exposure():
    ds = deal(po(), inv("INV-1"), inv("INV-2"),
              duplicates=[DuplicateFlag("INV-2", "INV-1", D("144"))])
    r = _one(_run(check_duplicates, ds))
    assert (r.claim_doc, r.auth_doc, r.po_id) == ("INV-2", "INV-1", "PO-1")
    assert r.exposure == D("120")


def test_bad_and_missing_po_references():
    ds = deal(po(), inv("INV-1", po_id="PO-404"), inv("INV-2", po_id=None))
    rs = {r.rule_id: r for r in _run(check_po_links, ds)}
    assert rs["bad_po_ref"].claim_doc == "INV-1" and rs["bad_po_ref"].claim_value == "PO-404"
    assert rs["no_po"].claim_doc == "INV-2"
    assert {r.outcome for r in rs.values()} == {Outcome.ABSENT_AUTHORITATIVE}


def test_clean_deal_produces_only_matches():
    rs = run_checks(deal(po(), inv()), link(deal(po(), inv()), CFG), CFG)
    assert rs and {r.outcome for r in rs} == {Outcome.MATCH}


# --- final review F6c: a blank header value is a note (invoice) or nothing (PO) -----

def test_blank_invoice_currency_is_a_note():
    r = _one(_run(check_currency, deal(po(), inv(currency=" "))))
    assert r.outcome == Outcome.ABSENT_SUBORDINATE and r.exposure == D("0")


def test_blank_po_currency_is_skipped():
    assert _run(check_currency, deal(po(currency=None), inv())) == []


def test_blank_invoice_supplier_is_a_note():
    r = _one(_run(check_supplier, deal(po(), inv(supplier=None))))
    assert r.outcome == Outcome.ABSENT_SUBORDINATE and r.exposure == D("0")


def test_blank_po_supplier_is_skipped():
    assert _run(check_supplier, deal(po(supplier=""), inv())) == []


# --- the money behind each comparison (Action Centre "at stake") ---------------

def test_invoice_totals_amounts():
    rs = _run(check_invoice_totals, deal(po(), inv(net="130")))
    net = next(r for r in rs if r.field_name == "net")
    assert (net.claim_amount, net.auth_amount) == (D("130"), D("120"))
    rs = _run(check_invoice_totals, deal(po(), inv(gross="150")))
    gross = next(r for r in rs if r.field_name == "gross")
    assert (gross.claim_amount, gross.auth_amount) == (D("150"), D("144.00"))


def test_invoice_totals_net_amounts_are_magnitudes_on_a_credit_note():
    rs = _run(check_invoice_totals,
              deal(po(), inv("CN-1", lines=[line(1, amount="-120")], net="-130")))
    net = next(r for r in rs if r.field_name == "net")
    assert (net.claim_amount, net.auth_amount) == (D("130"), D("120"))


def test_cumulative_total_amounts_are_invoiced_total_and_po_net():
    r = _one(_run(check_cumulative_total, deal(po(), inv("INV-1"), inv("INV-2"))))
    assert (r.claim_amount, r.auth_amount) == (D("240"), D("120"))


def test_tax_rate_amounts_are_tax_and_tax_at_the_nearest_allowed_rate():
    r = _one(_run(check_tax_rate, deal(po(), inv(tax="25"))))
    assert (r.claim_amount, r.auth_amount) == (D("25"), D("24.00"))


def test_duplicate_amounts_are_exposure_against_zero():
    ds = deal(po(), inv("INV-1"), inv("INV-2"),
              duplicates=[DuplicateFlag("INV-2", "INV-1", D("144"))])
    r = _one(_run(check_duplicates, ds))
    assert (r.claim_amount, r.auth_amount) == (D("120"), D("0"))


def test_bad_po_ref_amounts_are_invoice_net_against_zero_and_no_po_has_none():
    ds = deal(po(), inv("INV-1", po_id="PO-404"), inv("INV-2", po_id=None))
    rs = {r.rule_id: r for r in _run(check_po_links, ds)}
    assert (rs["bad_po_ref"].claim_amount, rs["bad_po_ref"].auth_amount) == (D("120"), D("0"))
    assert rs["no_po"].claim_amount is None and rs["no_po"].auth_amount is None


def test_header_checks_carry_no_amounts():
    ds = deal(po(), inv(currency="EUR", supplier="SUP-2", inv_date=date(2026, 1, 1),
                        terms="60 days"))
    for check in (check_currency, check_supplier, check_invoice_date, check_payment_terms):
        r = _one(_run(check, ds))
        assert r.claim_amount is None and r.auth_amount is None, check.__name__
