from decimal import Decimal as D

from tests.triage.helpers import deal, inv, line, pipeline, po


def test_price_finding_text_follows_the_writing_rules():
    out = pipeline(deal(po(lines=[line(3, qty="300", price="12.00")]),
                        inv(lines=[line(3, qty="300", price="13.50")])))
    (f,) = out.findings
    assert f.headline == "Unit price above PO on line 3"
    assert "£450.00" in f.text
    assert "PO-1 12.00" in f.text and "INV-1 13.50" in f.text
    assert "Also changes: PO running total" in f.text


def test_non_gbp_exposure_shows_both_currencies():
    out = pipeline(deal(po(currency="EUR", fx="0.5", lines=[line(1, qty="300", price="12.00")]),
                        inv(currency="EUR", fx="0.5", lines=[line(1, qty="300", price="13.50")])))
    (f,) = out.findings
    assert "£225.00 (450.00 EUR)" in f.text


def test_text_without_fx_rate():
    out = pipeline(deal(po(currency="XXX", fx=None, lines=[line(1, qty="300", price="12.00")]),
                        inv(currency="XXX", fx=None, lines=[line(1, qty="300", price="13.50")])))
    assert any("(no FX rate)" in f.text for f in out.findings)


def test_uplift_headline():
    items = "ABCD"
    out = pipeline(deal(po(lines=[line(i, item=items[i], qty="1", price="100.00") for i in range(4)]),
                        inv(lines=[line(i, item=items[i], qty="1", price="103.50") for i in range(4)])))
    (f,) = out.findings
    assert f.headline == "Prices 3.5% above PO on 4 lines"


def test_quantity_headline_names_the_po():
    from datetime import date
    out = pipeline(deal(po(), inv("INV-1", inv_date=date(2026, 2, 1)),
                        inv("INV-2", inv_date=date(2026, 2, 2))))
    (f,) = [f for f in out.findings if f.rule_id == "quantity"]
    assert f.headline == "Quantity above PO PO-1 on 1 line"
