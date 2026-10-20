"""Purchase price variance: what was paid above (+) or below (-) the PO price, on the quantity billed."""
import pytest

from src.services.supplier_performance.ppv import purchase_price_variance as ppv


def line(po, inv, qty=10.0, ccy="GBP"):
    return {"po_unit_price": po, "invoice_unit_price": inv, "quantity": qty, "currency": ccy}


def test_overpaying_is_positive_and_underpaying_negative():
    r = ppv(lines=[line(10.0, 11.0, 100), line(5.0, 4.5, 40)])
    assert r["total_ppv"] == pytest.approx(100.0 - 20.0)
    assert r["unfavourable"] == pytest.approx(100.0)
    assert r["favourable"] == pytest.approx(-20.0)


def test_percentage_is_of_the_standard_spend_the_po_price_times_quantity():
    r = ppv(lines=[line(10.0, 11.0, 100), line(5.0, 5.0, 40)])
    assert r["standard_spend"] == pytest.approx(1000.0 + 200.0)
    assert r["ppv_pct"] == pytest.approx(100.0 / 1200.0 * 100)


def test_no_variance_is_a_measured_zero_not_unassessed():
    r = ppv(lines=[line(10.0, 10.0, 5)])
    assert r["total_ppv"] == 0 and r["ppv_pct"] == 0
    assert r["lines_assessed"] == 1


def test_a_line_that_cannot_be_assessed_is_counted_never_zeroed():
    r = ppv(lines=[line(10.0, 12.0, 10), line(None, 12.0, 10), line(10.0, 12.0, None),
                   line(10.0, 12.0, 0), line(-1.0, 2.0, 5)])
    assert r["lines_assessed"] == 1
    assert r["lines_unassessed"] == 4
    assert r["total_ppv"] == pytest.approx(20.0)


def test_nothing_assessable_is_unassessed_not_zero():
    assert ppv(lines=[line(None, 1.0)]) is None   # the registered formula turns None into UNASSESSED
    assert ppv(lines=[]) is None


def test_a_zero_standard_spend_has_no_percentage():
    r = ppv(lines=[line(0.0, 2.0, 5)])
    assert r["total_ppv"] == pytest.approx(10.0)
    assert r["ppv_pct"] is None


# --- currency -----------------------------------------------------------------------------

def test_one_currency_needs_no_fx_table():
    r = ppv(lines=[line(10.0, 11.0, 10, "EUR")])
    assert r["currency"] == "EUR" and r["total_ppv"] == pytest.approx(10.0)


def test_mixed_currencies_are_not_added_without_a_table():
    r = ppv(lines=[line(10.0, 11.0, 10, "EUR"), line(10.0, 11.0, 10, "USD")])
    assert r["total_ppv"] is None and r["ppv_pct"] is None and r["currency"] is None
    assert r["by_currency"]["EUR"]["ppv"] == pytest.approx(10.0)
    assert r["by_currency"]["USD"]["ppv"] == pytest.approx(10.0)


RATES = {"GBP": 0.75, "USD": 1.0, "EUR": 0.9}


def test_mixed_currencies_convert_with_a_dated_table():
    r = ppv(lines=[line(10.0, 11.0, 10, "EUR"), line(10.0, 11.0, 10, "USD")],
            buyer_currency="GBP", fx_rates=RATES, fx_table_date="2026-10-20")
    assert r["currency"] == "GBP"
    assert r["total_ppv"] == pytest.approx(10.0 / 0.9 * 0.75 + 10.0 * 0.75)
    assert r["fx_table_date"] == "2026-10-20"


def test_a_currency_with_no_rate_is_left_out_and_named_never_taken_one_to_one():
    r = ppv(lines=[line(10.0, 11.0, 10, "GBP"), line(10.0, 99.0, 10, "NZD")],
            buyer_currency="GBP", fx_rates=RATES, fx_table_date="2026-10-20")
    assert r["total_ppv"] == pytest.approx(10.0)
    assert r["excluded_currencies"] == ["NZD"]
    assert r["lines_unassessed"] == 1


def test_an_undated_table_is_refused():
    with pytest.raises(ValueError, match="dated"):
        ppv(lines=[line(10.0, 11.0, 10, "EUR")], buyer_currency="GBP", fx_rates=RATES)


def test_a_line_with_no_currency_is_unassessed():
    r = ppv(lines=[line(10.0, 11.0, 10, None), line(10.0, 11.0, 10, "GBP")])
    assert r["lines_unassessed"] == 1 and r["total_ppv"] == pytest.approx(10.0)
