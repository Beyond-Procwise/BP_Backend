from decimal import Decimal as D

from src.services.triage.model import (
    Outcome, Result, Severity, fingerprint, money, pct_change)
from tests.triage.helpers import inv, line


def test_severity_orders_worst_highest():
    assert max(Severity.S3, Severity.S1, Severity.S2) == Severity.S1
    assert min(Severity.S1, Severity.S2) == Severity.S2


def test_fingerprint_is_stable_and_distinguishes_causes():
    a = fingerprint("D1", "unit_price", "INV-1|3")
    assert a == fingerprint("D1", "unit_price", "INV-1|3")
    assert a != fingerprint("D1", "unit_price", "INV-1|4")


def test_result_exposure_gbp_uses_fx_and_magnitude():
    r = Result("D1", "unit_price", "money", Outcome.CONFLICT, "INV-1", "unit_price",
               exposure=D("-100"), fx_to_gbp=D("0.85"))
    assert r.exposure_gbp == D("85.00")
    assert Result("D1", "x", "money", Outcome.CONFLICT, "INV-1", "x",
                  exposure=D("1")).exposure_gbp is None


def test_money_formats_gbp_and_others():
    assert money(D("1234.5"), "GBP") == "£1,234.50"
    assert money(D("1234.5"), "EUR") == "1,234.50 EUR"
    assert money(None, "GBP") == "n/a"


def test_pct_change():
    r = Result("D1", "unit_price", "money", Outcome.CONFLICT, "INV-1", "unit_price",
               auth_value="100", delta=D("3.5"))
    assert pct_change(r) == D("3.5")


def test_invoice_po_ref_falls_back_to_lines_and_credit_note_flag():
    doc = inv(po_id=None, lines=[line(1, po_id="PO-9")])
    assert doc.po_ref == "PO-9"
    assert inv(net="-10").is_credit_note
