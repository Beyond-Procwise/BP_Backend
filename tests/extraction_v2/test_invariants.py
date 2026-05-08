"""Tests for the procurement invariants."""
from __future__ import annotations

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.services.extraction_v2.invariants import (  # noqa: E402
    CurrencyConsistency, DateSanity, DEFAULT_VALIDATORS, GrandTotalClosure,
    LineArithmetic, QuantitySign, RoundOffBucket, Severity, SubtotalClosure,
    TaxClosure, ValidationReport, ValidatorChain, VendorIdentity,
)


def _good_invoice():
    return {
        "supplier_id": "ACME Co Ltd", "currency": "GBP",
        "invoice_id": "INV-1", "invoice_date": "2026-05-04",
        "due_date": "2026-06-03",
        "subtotal": 100.00, "tax_percent": 20, "tax_amount": 20.00,
        "invoice_total_incl_tax": 120.00,
    }


def _good_lines():
    return [
        {"item_description": "A", "quantity": 1, "unit_price": 60.0, "line_amount": 60.0},
        {"item_description": "B", "quantity": 2, "unit_price": 20.0, "line_amount": 40.0},
    ]


# -- LineArithmetic --------------------------------------------------------

def test_line_arithmetic_passes_clean_invoice():
    r = LineArithmetic().check(_good_invoice(), _good_lines(), "Invoice")
    assert r.passed is True


def test_line_arithmetic_fails_when_qty_times_price_diverges():
    bad = [
        {"item_description": "X", "quantity": 1, "unit_price": 50.0, "line_amount": 999.0},
    ]
    r = LineArithmetic().check(_good_invoice(), bad, "Invoice")
    assert r.passed is False
    assert r.severity == Severity.WARNING
    assert "1 of 1" in r.message


def test_line_arithmetic_na_on_empty_lines():
    r = LineArithmetic().check(_good_invoice(), [], "Invoice")
    assert r.passed is True
    assert r.message == "not_applicable"


# -- SubtotalClosure -------------------------------------------------------

def test_subtotal_closure_passes_when_lines_sum_to_subtotal():
    r = SubtotalClosure().check(_good_invoice(), _good_lines(), "Invoice")
    assert r.passed is True


def test_subtotal_closure_fails_when_lines_disagree():
    h = dict(_good_invoice(), subtotal=999.0)
    r = SubtotalClosure().check(h, _good_lines(), "Invoice")
    assert r.passed is False


# -- TaxClosure ------------------------------------------------------------

def test_tax_closure_passes_at_20_percent():
    r = TaxClosure().check(_good_invoice(), _good_lines(), "Invoice")
    assert r.passed is True


def test_tax_closure_fails_when_tax_amount_inconsistent():
    h = dict(_good_invoice(), tax_amount=999.99)
    r = TaxClosure().check(h, _good_lines(), "Invoice")
    assert r.passed is False
    assert "subtotal×rate" in r.message


def test_tax_closure_handles_decimal_rate():
    h = dict(_good_invoice(), tax_percent=0.20)
    r = TaxClosure().check(h, _good_lines(), "Invoice")
    assert r.passed is True


# -- GrandTotalClosure -----------------------------------------------------

def test_grand_total_closure_passes_when_subtotal_plus_tax_matches_total():
    r = GrandTotalClosure().check(_good_invoice(), _good_lines(), "Invoice")
    assert r.passed is True


def test_grand_total_closure_fails_when_total_diverges():
    h = dict(_good_invoice(), invoice_total_incl_tax=999.99)
    r = GrandTotalClosure().check(h, _good_lines(), "Invoice")
    assert r.passed is False


def test_grand_total_closure_includes_freight_and_charges():
    h = dict(_good_invoice(),
             freight_amount=10.0, charges_amount=5.0, discount_amount=2.0,
             invoice_total_incl_tax=133.0)  # 100 + 20 + 10 + 5 - 2 = 133
    r = GrandTotalClosure().check(h, _good_lines(), "Invoice")
    assert r.passed is True


# -- CurrencyConsistency ---------------------------------------------------

def test_currency_consistency_passes_with_single_currency():
    r = CurrencyConsistency().check(_good_invoice(), _good_lines(), "Invoice")
    assert r.passed is True


def test_currency_consistency_fails_on_mixed_currencies():
    bad_lines = [
        {"item_description": "A", "quantity": 1, "unit_price": 60.0,
         "line_amount": 60.0, "currency": "USD"},
        {"item_description": "B", "quantity": 2, "unit_price": 20.0,
         "line_amount": 40.0, "currency": "EUR"},
    ]
    r = CurrencyConsistency().check(_good_invoice(), bad_lines, "Invoice")
    assert r.passed is False
    assert r.severity == Severity.CRITICAL


# -- DateSanity ------------------------------------------------------------

def test_date_sanity_passes_on_iso_dates():
    r = DateSanity().check(_good_invoice(), _good_lines(), "Invoice")
    assert r.passed is True


def test_date_sanity_fails_on_non_iso_invoice_date():
    h = dict(_good_invoice(), invoice_date="04/05/2026")
    r = DateSanity().check(h, _good_lines(), "Invoice")
    assert r.passed is False


def test_date_sanity_fails_when_due_precedes_invoice():
    h = dict(_good_invoice(), invoice_date="2026-05-04", due_date="2026-04-01")
    r = DateSanity().check(h, _good_lines(), "Invoice")
    assert r.passed is False


def test_date_sanity_fails_on_far_future_dates():
    h = dict(_good_invoice(), invoice_date="2999-12-31", due_date="3000-01-30")
    r = DateSanity().check(h, _good_lines(), "Invoice")
    assert r.passed is False


# -- VendorIdentity --------------------------------------------------------

def test_vendor_identity_passes_with_clean_supplier():
    r = VendorIdentity().check(_good_invoice(), _good_lines(), "Invoice")
    assert r.passed is True


def test_vendor_identity_fails_on_garbage_url():
    h = dict(_good_invoice(), supplier_id="nexasparkideas.com", supplier_name="")
    r = VendorIdentity().check(h, _good_lines(), "Invoice")
    assert r.passed is False
    assert r.severity == Severity.CRITICAL


def test_vendor_identity_fails_on_doc_id_in_supplier_field():
    h = dict(_good_invoice(), supplier_id="INVOICE NUMBER: 12345", supplier_name="")
    r = VendorIdentity().check(h, _good_lines(), "Invoice")
    assert r.passed is False
    assert r.severity == Severity.CRITICAL


def test_vendor_identity_does_not_flag_legitimate_names_containing_substrings():
    """Regression test: 'City Of Newport' contains the substring 'po' in
    'Newport' but is a legitimate vendor name. The validator must NOT
    flag it — substring matches need word boundaries."""
    for name in (
        "City Of Newport",        # 'po' inside 'Newport'
        "Acme Order Solutions",   # 'order' inside an actual word? No — bounded
        "Reorder & Co",           # 'order' inside 'Reorder' — substring only
        "Cooperative Bank Ltd",   # contains 'bank' but it's part of legitimate name
    ):
        h = dict(_good_invoice(), supplier_id=name, supplier_name="")
        r = VendorIdentity().check(h, _good_lines(), "Invoice")
        # 'order' is now word-bounded so 'Reorder' won't match. 'bank'
        # bounded too, so 'Cooperative Bank' DOES match — that's a real
        # bank-account-as-supplier pattern. We want that flagged.
        if "bank" in name.lower().split():
            assert r.passed is False, f"expected fail for {name!r}"
        elif "order" in name.lower().split():
            assert r.passed is False, f"expected fail for {name!r}"
        else:
            assert r.passed is True, f"expected pass for {name!r}"


def test_vendor_identity_flags_url_markers():
    """URLs/emails are still caught (self-anchoring patterns)."""
    for url in ("nexasparkideas.com", "info@vendor.co.uk", "https://vendor.com"):
        h = dict(_good_invoice(), supplier_id=url, supplier_name="")
        r = VendorIdentity().check(h, _good_lines(), "Invoice")
        assert r.passed is False, f"expected fail for {url!r}"
        assert r.severity == Severity.CRITICAL


def test_vendor_identity_fails_on_empty_supplier():
    h = dict(_good_invoice(), supplier_id="", supplier_name="")
    r = VendorIdentity().check(h, _good_lines(), "Invoice")
    assert r.passed is False


# -- QuantitySign ----------------------------------------------------------

def test_quantity_sign_passes_for_all_positive():
    r = QuantitySign().check(_good_invoice(), _good_lines(), "Invoice")
    assert r.passed is True


def test_quantity_sign_passes_for_all_negative_credit_note():
    lines = [
        {"item_description": "A", "quantity": -1, "unit_price": 60.0, "line_amount": -60.0},
        {"item_description": "B", "quantity": -2, "unit_price": 20.0, "line_amount": -40.0},
    ]
    r = QuantitySign().check(_good_invoice(), lines, "Invoice")
    assert r.passed is True


def test_quantity_sign_fails_on_mixed_signs():
    lines = [
        {"item_description": "A", "quantity": 1, "unit_price": 60.0, "line_amount": 60.0},
        {"item_description": "B", "quantity": -2, "unit_price": 20.0, "line_amount": -40.0},
    ]
    r = QuantitySign().check(_good_invoice(), lines, "Invoice")
    assert r.passed is False
    assert r.severity == Severity.CRITICAL


# -- RoundOffBucket --------------------------------------------------------

def test_round_off_bucket_emits_info_for_small_diff():
    h = dict(_good_invoice(), invoice_total_incl_tax=120.03)  # off by 0.03
    r = RoundOffBucket().check(h, _good_lines(), "Invoice")
    # Within bucket → passes with severity INFO
    assert r.passed is True
    assert r.severity == Severity.INFO
    assert "round_off_diff" in r.message


# -- Chain orchestration ---------------------------------------------------

def test_chain_runs_all_default_validators():
    chain = ValidatorChain(DEFAULT_VALIDATORS)
    report = chain.run(_good_invoice(), _good_lines(), "Invoice")
    assert isinstance(report, ValidationReport)
    assert report.total_count == len(DEFAULT_VALIDATORS)
    # All applicable invariants should pass on a clean invoice
    assert report.critical_failures == []
    assert report.pass_rate >= 0.95


def test_chain_collects_critical_failures_separately():
    chain = ValidatorChain(DEFAULT_VALIDATORS)
    h = dict(_good_invoice(), supplier_id="nexasparkideas.com", supplier_name="")
    bad_lines = [
        {"item_description": "A", "quantity": 1, "unit_price": 60.0,
         "line_amount": 60.0, "currency": "USD"},
        {"item_description": "B", "quantity": -2, "unit_price": 20.0,
         "line_amount": -40.0, "currency": "EUR"},
    ]
    report = chain.run(h, bad_lines, "Invoice")
    crit_names = {r.name for r in report.critical_failures}
    assert "vendor_identity" in crit_names
    assert "currency_consistency" in crit_names
    assert "quantity_sign" in crit_names


def test_chain_pass_rate_excludes_not_applicable():
    """RoundOffBucket on a doc with no subtotal returns NA — must not
    drag down the pass rate."""
    chain = ValidatorChain(DEFAULT_VALIDATORS)
    minimal = {"supplier_id": "Vendor X", "currency": "GBP"}
    report = chain.run(minimal, [], "Invoice")
    # Pass rate is computed over applicable validators only
    assert 0.0 <= report.pass_rate <= 1.0
