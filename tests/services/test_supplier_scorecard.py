import datetime as dt
from decimal import Decimal
from types import SimpleNamespace

import pytest

from src.services.contract_tools import store
from src.services.supplier_performance import scorecard as SC

TODAY = dt.date(2026, 10, 22)
RULES = {"otif_late_tolerance_days": 0, "otif_in_full_tolerance": 0.0, "otif_early_tolerance_days": None}


class _Display:
    """Adds amounts in one currency and refuses the rest, like the dashboards' converter for GBP only."""
    target = "GBP"

    def total(self, pairs):
        value, excluded = Decimal(0), []
        for amount, ccy in pairs:
            if ccy == "GBP":
                value += amount
            else:
                excluded.append(ccy)
        return SimpleNamespace(value=value if pairs and len(excluded) < len(pairs) else None,
                               excluded_currencies=excluded)


def _rows(suppliers=(("S1", "Acme Ltd"),), spend=(("GBP", Decimal("1200.50"), 3),),
          prices=((10.0, 11.0, 100.0, "GBP", "GBP"), (5.0, 4.5, 40.0, "GBP", "GBP")),
          deliveries=(("PO1", dt.date(2026, 3, 10), dt.date(2026, 3, 10)),
                      ("PO2", dt.date(2026, 3, 10), dt.date(2026, 3, 12)),
                      ("PO3", dt.date(2026, 4, 1), None)),
          findings=((2,),)):
    seen = []

    def rows(sql, params):
        seen.append((sql, list(params)))
        if "bp_supplier s" in sql:
            return list(suppliers)
        if "bp_invoice_trgt i" in sql:
            return list(spend)
        if "WITH pol" in sql:
            return list(prices)
        if "bp_goods_receipt_trgt" in sql:
            return list(deliveries)
        if "bp_detection_finding" in sql:
            return list(findings)
        raise AssertionError(sql)
    rows.seen = seen
    return rows


def _build(**kw):
    kw.setdefault("rows", _rows())
    kw.setdefault("limits", lambda: dict(RULES))
    kw.setdefault("display", _Display())
    kw.setdefault("today", TODAY)
    return SC.build(kw.pop("supplier", "acme"), kw.pop("start", None), kw.pop("end", None), kw.pop("currency", None), **kw)


def test_needs_a_supplier_and_refuses_bad_dates():
    assert _build(supplier="")["found"] is False
    assert "date like" in _build(start="soon")["note"]
    assert _build(start="2026-05-01", end="2026-04-01")["note"] == "start is after end."


def test_dimensions_rest_on_the_records_and_say_how_many():
    out = _build()
    assert out["found"] and out["supplier"] == {"supplier_id": "S1", "name": "Acme Ltd"}
    d = out["dimensions"]
    assert d["spend"] == {"assessed": True, "invoices": 3, "currency": "GBP", "amount": 1200.5,
                          "as_billed": {"GBP": 1200.5}, "excluded_currencies": []}
    assert d["price_variance"]["assessed"] and d["price_variance"]["total_ppv"] == 80.0
    assert d["price_variance"]["lines_assessed"] == 2 and d["price_variance"]["lines_matched"] == 2
    ot = d["on_time_delivery"]
    assert ot["assessed"] and ot["on_time_rate"] == 0.5 and ot["on_time_assessed"] == 2
    assert ot["purchase_orders"] == 3 and ot["unassessed"] == 1 and "not assessed" in ot["in_full"]
    assert ot["tolerances"]["source"] == "policy supplier_scorecard"
    assert d["open_findings"]["count"] == 2
    assert out["composite"] is None and out["not_assessed"] == []


def test_default_period_is_fiscal_year_to_date_and_custom_dates_reach_the_queries():
    rows = _rows()
    out = _build(rows=rows, start="2026-01-01", end="2026-06-30")
    assert out["period"] == {"label": "custom", "start": "2026-01-01", "end": "2026-06-30"}
    spend_sql, spend_params = rows.seen[1]
    assert spend_params == ["S1", dt.date(2026, 1, 1), dt.date(2026, 6, 30)]
    assert _build()["period"]["label"] != "custom"


def test_without_the_governed_policy_on_time_is_not_assessed_not_defaulted():
    def no_policy():
        raise RuntimeError("no active supplier_scorecard policy")
    ot = _build(limits=no_policy)["dimensions"]["on_time_delivery"]
    assert ot["assessed"] is False and "on_time_rate" not in ot
    assert "supplier_scorecard" in ot["reason"] and "not a default" in ot["reason"]


def test_a_malformed_tolerance_is_not_assessed():
    ot = _build(limits=lambda: {"otif_late_tolerance_days": -1, "otif_in_full_tolerance": 0.0})["dimensions"]["on_time_delivery"]
    assert ot["assessed"] is False and "not usable" in ot["reason"]
    ot = _build(limits=lambda: {"otif_in_full_tolerance": 0.0})["dimensions"]["on_time_delivery"]
    assert ot["assessed"] is False


def test_a_po_without_a_receipt_is_unassessed_not_late():
    rows = _rows(deliveries=(("PO3", dt.date(2026, 4, 1), None),))
    ot = _build(rows=rows)["dimensions"]["on_time_delivery"]
    assert ot["assessed"] is False and "goods receipt" in ot["reason"] and ot["purchase_orders"] == 1


def test_a_line_billed_in_another_currency_is_left_out_and_named():
    rows = _rows(prices=((10.0, 11.0, 100.0, "USD", "GBP"), (5.0, 4.5, 40.0, "GBP", "GBP")))
    pv = _build(rows=rows)["dimensions"]["price_variance"]
    assert pv["assessed"] and pv["lines_assessed"] == 1 and pv["mixed_currency_lines"] == 1 and pv["total_ppv"] == -20.0
    rows = _rows(prices=((10.0, 11.0, 100.0, "USD", "GBP"),))
    pv = _build(rows=rows)["dimensions"]["price_variance"]
    assert pv["assessed"] is False and "different currency" in pv["reason"]


def test_nothing_in_the_period_is_said_per_dimension():
    out = _build(rows=_rows(spend=(), prices=(), deliveries=(), findings=((0,),)))
    d = out["dimensions"]
    assert d["spend"]["assessed"] is False and d["spend"]["invoices"] == 0 and "no invoice" in d["spend"]["reason"]
    assert d["price_variance"]["assessed"] is False and d["on_time_delivery"]["assessed"] is False
    assert d["open_findings"]["count"] == 0
    assert sorted(out["not_assessed"]) == ["on_time_delivery", "price_variance", "spend"]
    assert "Not assessed: spend, price_variance, on_time_delivery" in out["note"]


def test_spend_in_a_currency_without_a_rate_is_shown_as_billed_not_converted():
    out = _build(rows=_rows(spend=(("GBP", Decimal("100"), 1), ("XOF", Decimal("5000"), 2))))
    s = out["dimensions"]["spend"]
    assert s["amount"] == 100.0 and s["excluded_currencies"] == ["XOF"] and s["as_billed"]["XOF"] == 5000.0


def test_unknown_or_ambiguous_supplier():
    assert "not on record" in _build(rows=_rows(suppliers=()))["note"] or _build(rows=_rows(suppliers=()))["found"] is False
    out = _build(rows=_rows(suppliers=(("S1", "Acme Ltd"), ("S2", "Acme Logistics"))))
    assert out["found"] is False and out["ambiguous"] and [c["supplier_id"] for c in out["candidates"]] == ["S1", "S2"]


def test_outage_is_not_reported_as_no_history():
    def down(sql, params):
        raise store.StoreUnavailable("db down")
    out = _build(rows=down)
    assert out["found"] is False and out["available"] is False and "does not mean" in out["note"]

    def later_down(sql, params):
        if "bp_supplier s" in sql:
            return [("S1", "Acme Ltd")]
        raise store.StoreUnavailable("db down")
    assert _build(rows=later_down)["available"] is False


def test_governed_tolerances_are_read_from_the_supplier_scorecard_policy(monkeypatch):
    from src.services import governed_limits
    monkeypatch.setattr(governed_limits, "_rules", lambda policy, fresh=False: dict(RULES) if policy == "supplier_scorecard" else {})
    assert SC._governed_tolerances() == RULES


@pytest.mark.parametrize("supplier", ["S1", "acme"])
def test_supplier_is_matched_by_id_or_name(supplier):
    rows = _rows()
    _build(rows=rows, supplier=supplier)
    assert rows.seen[0][1] == [supplier, f"%{supplier}%", f"%{supplier}%"]
