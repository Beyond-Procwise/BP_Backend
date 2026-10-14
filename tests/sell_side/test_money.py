from decimal import Decimal as D

import pytest

from src.services.sell_side import money


def test_a_line_prices_to_two_places_half_up():
    p = money.price_line(D("3"), D("10.005"), D("7.0000"), D("12.0000"))
    assert p.line_total == D("30.02")        # 30.015 -> 30.02
    assert p.line_cost == D("21.00")
    assert p.line_margin == D("9.02")
    assert p.line_margin_pct == D("0.3005")
    assert p.discount_pct == D("0.1663")     # 1 - 10.005/12


def test_an_unknown_cost_gives_no_margin_not_a_zero_one():
    p = money.price_line(D("2"), D("5"), None, None)
    assert p.line_total == D("10.00")
    assert (p.line_cost, p.line_margin, p.line_margin_pct, p.discount_pct) == (None,) * 4


def test_a_free_line_has_no_margin_percentage():
    p = money.price_line(D("1"), D("0"), D("3"), D("5"))
    assert p.line_margin == D("-3.00")
    assert p.line_margin_pct is None


def test_quote_totals_sum_lines():
    a = money.price_line(D("1"), D("10"), D("6"), None)
    b = money.price_line(D("2"), D("5"), D("4"), None)
    t = money.total_quote([a, b])
    assert (t.total_ex_tax, t.total_cost, t.total_margin, t.margin_pct) == \
        (D("20.00"), D("14.00"), D("6.00"), D("0.3000"))


def test_one_uncosted_line_makes_the_quote_margin_unknown():
    """A margin over some lines is not the quote's margin."""
    a = money.price_line(D("1"), D("10"), D("6"), None)
    b = money.price_line(D("1"), D("10"), None, None)
    t = money.total_quote([a, b])
    assert t.total_ex_tax == D("20.00")
    assert (t.total_cost, t.total_margin, t.margin_pct) == (None, None, None)


@pytest.mark.parametrize("raw,ok", [("gbp", "GBP"), (" EUR ", "EUR")])
def test_iso_currency_normalises(raw, ok):
    assert money.iso_currency(raw) == ok


@pytest.mark.parametrize("raw", ["", None, "£", "Pounds", "GB"])
def test_iso_currency_refuses(raw):
    with pytest.raises(ValueError):
        money.iso_currency(raw)
