"""The Advisor sorts a deal in pounds, on suppliers that are really on record (audit rec. 4).

Measured 2026-10-15. bp_testdb: 29 of 300 sampled deals were "high spend" only
because a rupee, dollar, dirham or euro amount was compared with the £98,175 bar
as if it were pounds (8 are, converted). Every deal had at least 234 suppliers
quoting its items against a bar of 93, so none could read as a thin market.
bp_sqldb: 14 of 15 deals had only their own supplier on record, and that absence
of data was read as a thin market.
"""
from __future__ import annotations

import pytest

from src.services import value_summary_service
from src.services.negotiation_advice import classification as cl
from src.services.negotiation_advice import signals as sg
from tests.services.negotiation_advice.test_signals import _DEAL_COLS, _Cur

_RATES = {"GBP": 1.0, "INR": 110.0, "USD": 1.25, "_fetched_at": "2026-10-15"}


def _sig(gbp, alt, *, native=None, currency="GBP"):
    return {"deal_value": native if native is not None else gbp, "currency": currency,
            "deal_value_gbp": gbp, "alternative_supplier_count": alt}


# ---- spend is compared in pounds --------------------------------------------

def test_a_large_foreign_amount_is_compared_in_pounds():
    # ₹22.1M is about £200,000 -> high spend; ₹2.2M is about £20,000 -> not.
    assert cl.classify(_sig(200_909.0, 150, native=22_100_000.0, currency="INR"))["quadrant"] == "Leverage"
    assert cl.classify(_sig(20_000.0, 150, native=2_200_000.0, currency="INR"))["quadrant"] == "Transactional"


def test_no_rate_to_pounds_is_no_verdict_not_a_native_comparison():
    out = cl.classify(_sig(None, 150, native=22_100_000.0, currency="INR"))
    assert out["indeterminate"] is True and out["quadrant"] is None
    assert any("exchange rate" in r for r in out["quadrant_reasons"])


def test_the_reason_states_pounds_and_the_original_amount():
    reasons = " ".join(cl.classify(_sig(200_909.0, 150, native=22_100_000.0,
                                        currency="INR"))["quadrant_reasons"])
    assert "£200,909" in reasons and "INR 22,100,000" in reasons


def test_a_caller_without_a_pounds_figure_still_classifies_on_deal_value():
    assert cl.classify({"deal_value": 250_000.0, "alternative_supplier_count": 150})["quadrant"] == "Leverage"


# ---- the market bar is a supplier count, governed --------------------------------

def test_three_suppliers_on_record_is_a_contested_market():
    assert cl.default_thresholds()["many_alternatives"] == 3
    assert cl.classify(_sig(4_000.0, 3))["quadrant"] == "Transactional"
    assert cl.classify(_sig(4_000.0, 2))["quadrant"] == "Bottleneck"


def test_no_other_supplier_on_record_is_no_verdict():
    out = cl.classify(_sig(4_000.0, None))
    assert out["indeterminate"] is True
    assert any("no other supplier on record" in r for r in out["quadrant_reasons"])


# ---- gather_signals ------------------------------------------------------------

def _cur(alt, currency="GBP", invoice=105_000.0):
    return _Cur({
        "from proc.bp_deal_overview": (_DEAL_COLS, [("D-1", "SUP-1", "Orbis", currency, 1,
                                                     None, None, invoice, None)]),
        "bp_quote_line_items_trgt": (["n"], [(alt,)]),
    })


@pytest.fixture
def rates(monkeypatch):
    calls = []

    def fake():
        calls.append(1)
        return dict(_RATES)

    monkeypatch.setattr(value_summary_service, "_get_rates", fake)
    return calls


def test_only_the_deals_own_supplier_on_record_is_unknown_not_one(rates):
    s = sg.gather_signals(_cur(1), "D-1")
    assert s["alternative_supplier_count"] is None
    assert s["suppliers_on_record"] == 1
    s = sg.gather_signals(_cur(2), "D-1")
    assert s["alternative_supplier_count"] == 2


def test_a_foreign_deal_value_is_converted_to_pounds(rates):
    s = sg.gather_signals(_cur(5, currency="INR", invoice=22_100_000.0), "D-1")
    assert s["deal_value"] == 22_100_000.0
    assert s["deal_value_gbp"] == pytest.approx(200_909.09, abs=0.01)


def test_a_pounds_deal_needs_no_rate(rates):
    s = sg.gather_signals(_cur(5), "D-1")
    assert s["deal_value_gbp"] == 105_000.0
    assert rates == []


def test_no_currency_or_no_rate_leaves_pounds_unknown(rates):
    assert sg.gather_signals(_cur(5, currency=None), "D-1")["deal_value_gbp"] is None
    assert sg.gather_signals(_cur(5, currency="XYZ"), "D-1")["deal_value_gbp"] is None


# ---- the market-risk nudge reads the same governed bar ----------------------------

def test_the_thin_market_nudge_uses_the_governed_bar():
    bars = {"many_alternatives": 3}
    assert sg.market_context_dict({"alternative_supplier_count": 2}, bars) == {"supply_risk": "elevated"}
    assert sg.market_context_dict({"alternative_supplier_count": 150}, bars) == {}
    # A buyer's stated count arrives as text.
    assert sg.market_context_dict({"alternative_supplier_count": "2"}, bars) == {"supply_risk": "elevated"}
    assert sg.market_context_dict({"alternative_supplier_count": None}, bars) == {}
