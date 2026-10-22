from datetime import date
from decimal import Decimal

from src.services.analytics import spend_cube as SC
from src.services.analytics.currency import DisplayCurrency

# USD-quoted: 1 USD = 0.8 GBP = 0.9 EUR
GBP = DisplayCurrency(target="GBP", rates={"GBP": "0.8", "EUR": "0.9"})
TODAY = date(2026, 10, 21)


def _fetch(rows):
    return lambda by, period: rows


def test_total_is_the_converted_sum_and_shares_add_up():
    rows = [("Acme", "GBP", Decimal("600"), 3), ("Beta", "GBP", Decimal("400"), 2)]
    out = SC.build("supplier", fetch=_fetch(rows), display=GBP, today=TODAY)
    assert out["total"] == 1000.0 and out["invoices"] == 5
    assert [(c["supplier"], c["share_pct"]) for c in out["cells"]] == [("Acme", 60.0), ("Beta", 40.0)]


def test_each_currency_is_converted_once_from_its_own_amount():
    out = SC.build("supplier", fetch=_fetch([("Acme", "EUR", Decimal("900"), 1), ("Acme", "GBP", Decimal("100"), 1)]),
                   display=GBP, today=TODAY)
    # 900 EUR -> 1000 USD -> 800 GBP; the GBP amount stays exactly 100
    assert out["total"] == 900.0
    assert out["cells"][0]["as_billed"] == {"EUR": 900.0, "GBP": 100.0}


def test_currency_without_a_rate_is_left_out_and_named_never_one_to_one():
    out = SC.build("supplier", fetch=_fetch([("Acme", "GBP", Decimal("100"), 1), ("Zed", "JPY", Decimal("5000"), 1)]),
                   display=GBP, today=TODAY)
    assert out["total"] == 100.0 and out["excluded_currencies"] == ["JPY"]
    zed = [c for c in out["cells"] if c["supplier"] == "Zed"][0]
    assert zed["amount"] is None and zed["as_billed"] == {"JPY": 5000.0}


def test_missing_dimension_value_is_labelled_not_dropped():
    out = SC.build("country", fetch=_fetch([(None, "GBP", Decimal("50"), 1), ("UK", "GBP", Decimal("50"), 1)]),
                   display=GBP, today=TODAY)
    assert {c["country"] for c in out["cells"]} == {SC.NOT_RECORDED, "UK"} and out["total"] == 100.0


def test_only_a_known_dimension_reaches_the_query():
    bad = SC.build("1; DROP TABLE x", fetch=_fetch([]), display=GBP, today=TODAY)
    assert bad["found"] is False and "category" in bad["note"]
    assert all("{" not in SC.sql_for(d) for d in SC.DIMENSIONS)


def test_top_n_cuts_the_table_but_not_the_total():
    rows = [(f"S{i}", "GBP", Decimal(i + 1), 1) for i in range(5)]
    out = SC.build("supplier", top_n=2, fetch=_fetch(rows), display=GBP, today=TODAY)
    assert len(out["cells"]) == 2 and out["total"] == 15.0 and out["cell_count"] == 5


def test_unreadable_store_is_not_zero_spend():
    def boom(by, period):
        raise RuntimeError("db")
    out = SC.build("supplier", fetch=boom, display=GBP, today=TODAY)
    assert out["found"] is False and out["available"] is False and "does not mean" in out["note"]


def test_empty_period_says_so_and_bad_dates_are_refused():
    assert SC.build("month", fetch=_fetch([]), display=GBP, today=TODAY)["invoices"] == 0
    assert SC.build("month", start="soon", fetch=_fetch([]), display=GBP, today=TODAY)["found"] is False
    assert SC.build("month", start="2026-05-01", end="2026-01-01", fetch=_fetch([]), display=GBP,
                    today=TODAY)["found"] is False


def test_default_period_is_the_fiscal_year_to_date():
    seen = {}
    SC.build("month", fetch=lambda by, p: seen.setdefault("p", p) and [], display=GBP, today=TODAY)
    assert (seen["p"].start, seen["p"].end) == (date(2026, 4, 1), TODAY)


def test_order_bottom_lists_the_smallest_cells_first_and_says_so():
    rows = [("Acme", "GBP", Decimal("600"), 3), ("Beta", "GBP", Decimal("400"), 2), ("Tiny", "GBP", Decimal("5"), 1)]
    out = SC.build("supplier", fetch=_fetch(rows), display=GBP, top_n=2, order="bottom", today=TODAY)
    assert [c["supplier"] for c in out["cells"]] == ["Tiny", "Beta"]
    assert out["order"] == "bottom" and out["total"] == 1005.0 and out["cell_count"] == 3
    assert "Showing the bottom 2 of 3" in out["note"]


def test_order_defaults_to_top_and_an_unknown_order_is_refused_by_name():
    rows = [("Acme", "GBP", Decimal("600"), 3), ("Tiny", "GBP", Decimal("5"), 1)]
    assert SC.build("supplier", fetch=_fetch(rows), display=GBP, today=TODAY)["order"] == "top"
    out = SC.build("supplier", fetch=_fetch(rows), display=GBP, order="sideways", today=TODAY)
    assert out["found"] is False and "order must be one of" in out["note"]
