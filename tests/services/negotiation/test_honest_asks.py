"""The agent asks only for what the deal supports (lever audit recs. 5 and 7).

The audit's two scenarios -- a 3-month haulage lane and a 5-year services
renewal -- both got "250/500 unit tiers", "alternative specs", "packaging,
warranty", "split shipment (20-30%)" and an assumed 2% early-payment discount.
None of it came from either deal. From round 2 every email also said "We're
evaluating multiple proposals", whether or not anyone else was bidding.

An ask now carries a figure only when the deal stated it or a governed setting
works it out; with neither, the ask is left out.
"""
from __future__ import annotations

import pytest

import agents.negotiation_agent as na
import src.agents.negotiation_agent as src_na
from agents.email_drafting_agent import (
    _cta_guidance,
    _determine_recommended_tactics,
    _generate_strategic_elements,
)
from agents.negotiation.models import NegotiationContext, SupplierSignals
from agents.negotiation_agent import (
    NegotiationAgent,
    _merge_optimised,
    early_pay_discount,
    plan_counter,
)

STOCK = ("250/500", "alternative specs", "packaging", "warranty", "split shipment",
         "net 30", "net 45", "net45", "net15", "1–2%", "2.0%", "stepped pricing",
         "expedited production")


def _stock_in(asks):
    text = " ".join(asks).lower()
    return [s for s in STOCK if s.lower() in text]


@pytest.fixture(autouse=True)
def _coc(monkeypatch):
    for mod in (na, src_na):
        monkeypatch.setattr(mod, "COST_OF_CAPITAL_APR", lambda: 0.12)


def _plan(round_index=1, offer=1200.0, target=1000.0, message="", **kw):
    return plan_counter(
        NegotiationContext(current_offer=offer, target_price=target,
                           round_index=round_index, **kw),
        SupplierSignals(message_text=message))


# ---- the agent's own asks --------------------------------------------------

@pytest.mark.parametrize("round_index", [1, 2, 3])
def test_no_round_carries_a_stock_ask(round_index):
    assert _stock_in(_plan(round_index)["asks"]) == []


def test_a_narrow_round_one_gap_carries_no_stock_ask():
    assert _stock_in(_plan(1, offer=1050.0)["asks"]) == []


def test_final_offer_and_at_target_branches_carry_no_stock_ask():
    final = _plan(2, offer=990.0, message="This is our final offer.")
    at_target = _plan(2, offer=950.0)
    assert final["decision"] == at_target["decision"] == "accept"
    assert _stock_in(final["asks"] + at_target["asks"]) == []


def test_the_volume_ask_names_the_deals_own_quantity():
    asks = " ".join(_plan(1, volume_units=400)["asks"])
    assert "400 units" in asks


def test_no_quantity_no_volume_ask():
    assert "units" not in " ".join(_plan(1)["asks"])


# ---- early payment ---------------------------------------------------------

def test_the_discount_is_cost_of_capital_over_the_payment_days():
    assert early_pay_discount(45, 0.12) == pytest.approx(0.12 * 45 / 365)
    assert early_pay_discount(None, 0.12) is None
    assert early_pay_discount(0, 0.12) is None
    assert early_pay_discount(45, None) is None


def test_the_early_pay_ask_is_worked_out_from_the_suppliers_days():
    plan = na.compute_decision({"current_offer": 1200.0, "target_price": 1000.0,
                                "round": 1, "term_days": 45})
    asks = " ".join(plan["asks"])
    assert "1.5%" in asks and "45 days" in asks


def test_no_payment_days_no_early_pay_ask():
    plan = na.compute_decision({"current_offer": 1200.0, "target_price": 1000.0, "round": 1})
    assert "early" not in " ".join(plan["asks"]).lower()


def test_a_persons_own_discount_figure_is_used():
    plan = na.compute_decision({"current_offer": 1200.0, "target_price": 1000.0,
                                "round": 1, "term_days": 45, "ask_early_pay_disc": 0.03})
    assert "3.0%" in " ".join(plan["asks"])


def test_the_registered_counter_plan_takes_the_days_and_quantity():
    plan = na.decide_strategy({"current_offer": 1200.0, "target_price": 1000.0,
                               "round": 1, "term_days": 60, "volume_units": 250})
    asks = " ".join(plan["asks"])
    assert "2.0%" in asks and "60 days" in asks and "250 units" in asks


# ---- lead time ---------------------------------------------------------------

def test_a_long_lead_time_is_asked_about_in_its_own_weeks():
    plan = na.decide_strategy({"current_offer": 1200.0, "target_price": 1000.0,
                               "round": 1, "ask_lead_time_keep": False}, lead_weeks=6)
    assert "6-week" in plan["lead_time_request"]
    assert _stock_in(plan["asks"] + [plan["lead_time_request"]]) == []


# ---- the adaptive step and the optimiser ------------------------------------

def _agent():
    return NegotiationAgent.__new__(NegotiationAgent)


def test_a_final_offer_package_trade_adds_no_stock_ask():
    out = _agent()._adaptive_strategy(
        base_decision={"strategy": "counter", "counter_price": 90.0, "asks": []},
        zopa={}, signals={"finality_hint": True}, round_hint=2, lead_weeks=6.0,
        target_price=80.0, price=100.0)
    assert _stock_in(out["asks"] + [out.get("lead_time_request") or ""]) == []


def test_tradeable_terms_and_delivery_flex_add_no_figures():
    out = _agent()._adaptive_strategy(
        base_decision={"strategy": "counter", "counter_price": 90.0, "asks": []},
        zopa={}, signals={"payment_terms_hint": "tradeable", "delivery_flex": "possible"},
        round_hint=2, lead_weeks=6.0, target_price=80.0, price=100.0)
    assert _stock_in(out["asks"]) == []


def test_the_optimiser_offers_no_stock_terms_or_tiers(monkeypatch):
    for mod in (na, src_na):
        monkeypatch.setattr(mod, "LEAD_TIME_VALUE_PCT_PER_WEEK", lambda: 0.01)
    out = _agent()._optimize_multi_issue(
        price=100.0, currency="GBP", target=80.0, lead_weeks=6.0, weights={},
        policy={}, constraints={}, zopa={}, signals={}, round_no=1)
    asks = out["decision_overrides"].get("asks") or []
    assert _stock_in(asks) == []
    for option in out["counter_options"]:
        assert option["terms"] is None
        assert option["bundle"].get("volume_tier") is None


def test_a_locked_price_holds_across_every_counter_option():
    """Audit scenario B: the option said 220,000 while the locked counter was 240,000."""
    decision = {"counter_price": 240000.0, "price_plan_locked": True,
                "asks": ["Hold quoted price for the full project timeline."]}
    optimised = {"decision_overrides": {"counter_price": 220000.0,
                                        "asks": ["Can the quoted 6-week lead time be shortened?"]},
                 "counter_options": [{"price": 220000.0, "terms": None, "bundle": {}}]}
    options = _merge_optimised(decision, optimised)
    assert decision["counter_price"] == 240000.0
    assert [o["price"] for o in options] == [240000.0]
    # The plan's own asks survive the optimiser's; they used to be replaced.
    assert decision["asks"] == ["Hold quoted price for the full project timeline.",
                                "Can the quoted 6-week lead time be shortened?"]


def test_an_unlocked_price_takes_the_optimisers_counter():
    decision = {"counter_price": 240000.0, "price_plan_locked": False, "asks": []}
    options = _merge_optimised(decision, {
        "decision_overrides": {"counter_price": 220000.0},
        "counter_options": [{"price": 220000.0, "terms": None, "bundle": {}}]})
    assert decision["counter_price"] == 220000.0
    assert [o["price"] for o in options] == [220000.0]


# ---- what the email claims about competition --------------------------------

@pytest.mark.parametrize("bidders", [None, 0, 1])
def test_no_second_bidder_no_claim_of_other_proposals(bidders):
    data = {} if bidders is None else {"bidder_count": bidders}
    for round_no in (2, 3):
        text = " ".join(_generate_strategic_elements(round_no, data)).lower()
        assert "proposals" not in text
        assert "competitive" not in _determine_recommended_tactics(round_no, 10.0, {}, bidders).lower()
    assert "alternatives" not in _cta_guidance(3, bidders)


def test_a_real_second_bidder_may_be_mentioned():
    assert any("multiple proposals" in e for e in _generate_strategic_elements(2, {"bidder_count": 3}))
    assert "competitive" in _determine_recommended_tactics(2, 10.0, {}, 3).lower()
    assert "alternatives" in _cta_guidance(3, 3)


def test_database_edit_script_matches_the_code_prompt():
    import importlib.util
    from pathlib import Path

    from agents import email_drafting_agent as eda

    path = Path(__file__).resolve().parents[3] / "scripts" / "negotiation_playbook_prompt_bidders.py"
    spec = importlib.util.spec_from_file_location("np_bidders", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for old, new in mod.EDITS:
        assert old not in eda.NEGOTIATION_PLAYBOOK_SYSTEM
        assert new in eda.NEGOTIATION_PLAYBOOK_SYSTEM
    assert "multiple proposals" not in eda.NEGOTIATION_PLAYBOOK_SYSTEM


def test_a_long_lead_time_is_asked_about_not_held():
    plan = na.decide_strategy({"current_offer": 1200.0, "target_price": 1000.0, "round": 1},
                              lead_weeks=6)
    assert plan["lead_time_request"] == "Can the quoted 6-week lead time be shortened?"


def test_the_optimisers_lead_ask_is_not_repeated_as_an_ask():
    decision = {"counter_price": 90.0, "price_plan_locked": True, "asks": [],
                "lead_time_request": "Can the quoted 6-week lead time be shortened?"}
    _merge_optimised(decision, {"decision_overrides": {
        "asks": ["Can the quoted 6-week lead time be shortened?"]}, "counter_options": []})
    assert decision["asks"] == []


def test_a_worked_out_early_pay_ask_is_not_followed_by_an_open_one():
    out = _agent()._adaptive_strategy(
        base_decision={"strategy": "counter", "counter_price": 90.0,
                       "asks": ["Offer to pay on receipt instead of at 45 days for a 1.5% discount."]},
        zopa={}, signals={"payment_terms_hint": "tradeable"}, round_hint=2, lead_weeks=None,
        target_price=80.0, price=100.0)
    assert len(out["asks"]) == 1
