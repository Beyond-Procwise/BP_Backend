"""A decision to accept or decline is final. Nothing downstream may reopen it.

`plan_counter` already resolves finality itself: when the supplier says "best and
final", it compares the offer against the walk-away (or the target) and returns
`accept` or `decline`. That is a terminal decision -- the negotiation is over.

`_adaptive_strategy` then ran its own finality branch over the top and overwrote
`decision["strategy"]` with `"package-trade"` unconditionally, including on those
two terminal decisions. `_execute_negotiation_round` closes a supplier only when
the strategy is `accept` or `decline` (negotiation_agent.py, the
`strategy_lower in {"accept", "decline"}` test), so the supplier was never marked
ACCEPTED, never added to `completed_suppliers`, and the loop opened another round
against an offer we had already decided to take.

`price_plan_locked` guards the counter *price* against exactly this class of
overwrite. Nothing guarded the *decision*. These tests are that guard.
"""
import pytest

from agents.negotiation_agent import NegotiationAgent, decide_strategy
from tests.test_negotiation_agent import DummyNick

#: The strategies `_execute_negotiation_round` treats as closing a supplier.
#: Duplicated deliberately from the production literal: if the two drift, these
#: tests stop describing the loop they exist to protect.
CLOSES_THE_NEGOTIATION = {"accept", "decline"}

FINAL_OFFER = "this is our best and final offer"


@pytest.fixture
def agent():
    return NegotiationAgent(DummyNick())


@pytest.fixture
def firm_signals():
    """What `_extract_negotiation_signals` yields for final-offer language.

    Built literally rather than by calling the extractor: that method makes a
    live Ollama call, and the only field this behaviour turns on is the hint.
    """
    return {"finality_hint": True, "tone": "firm", "capacity_tight": False,
            "payment_terms_hint": None, "delivery_flex": None,
            "performance": {}, "market": {}}


def _plan(offer, target, **kw):
    return decide_strategy(
        {"current_offer": offer, "target_price": target, "round": 2, **kw},
        supplier_message=FINAL_OFFER,
    )


class TestAnAcceptableFinalOfferCloses:
    """Supplier's final offer is 79.00 against our 80.00 target -- take it."""

    def test_the_plan_says_accept(self):
        """Guards the premise: if this breaks, the rest of the file is moot."""
        assert _plan(79.0, 80.0)["strategy"] == "accept"

    def test_accept_survives_adaptive_strategy(self, agent, firm_signals):
        decision = agent._adaptive_strategy(
            base_decision=_plan(79.0, 80.0), zopa={}, signals=firm_signals,
            round_hint=2, lead_weeks=None, target_price=80.0, price=79.0,
        )
        assert decision["strategy"] == "accept"

    def test_the_round_loop_would_close_the_supplier(self, agent, firm_signals):
        """The harm, stated as the loop states it."""
        decision = agent._adaptive_strategy(
            base_decision=_plan(79.0, 80.0), zopa={}, signals=firm_signals,
            round_hint=2, lead_weeks=None, target_price=80.0, price=79.0,
        )
        assert decision["strategy"].lower() in CLOSES_THE_NEGOTIATION

    def test_the_agreed_price_is_not_disturbed(self, agent, firm_signals):
        decision = agent._adaptive_strategy(
            base_decision=_plan(79.0, 80.0), zopa={}, signals=firm_signals,
            round_hint=2, lead_weeks=None, target_price=80.0, price=79.0,
        )
        assert decision["counter_price"] == 79.0


class TestAnUnacceptableFinalOfferAlsoCloses:
    """120.00 against an 80.00 target and an 85.00 walk-away -- decline it."""

    def test_the_plan_says_decline(self):
        assert _plan(120.0, 80.0, walkaway_price=85.0)["strategy"] == "decline"

    def test_decline_survives_adaptive_strategy(self, agent, firm_signals):
        decision = agent._adaptive_strategy(
            base_decision=_plan(120.0, 80.0, walkaway_price=85.0), zopa={},
            signals=firm_signals, round_hint=2, lead_weeks=None,
            target_price=80.0, price=120.0,
        )
        assert decision["strategy"] == "decline"

    def test_no_counter_price_is_invented_for_a_declined_offer(
        self, agent, firm_signals
    ):
        """`plan_counter` returns None here. package-trade must not fill it in."""
        decision = agent._adaptive_strategy(
            base_decision=_plan(120.0, 80.0, walkaway_price=85.0), zopa={},
            signals=firm_signals, round_hint=2, lead_weeks=None,
            target_price=80.0, price=120.0,
        )
        assert decision["counter_price"] is None


class TestTheTradeStillHappensWhenWeAreStillNegotiating:
    """The fix must not cost us package-trade on a live negotiation.

    A firm supplier we are still countering is exactly who the tactic is for.
    """

    def test_a_counter_becomes_a_package_trade(self, agent, firm_signals):
        plan = decide_strategy(
            {"current_offer": 100.0, "target_price": 80.0, "round": 2},
            supplier_message="we are holding at this price for now",
        )
        assert plan["strategy"] == "counter", "premise: still negotiating"

        decision = agent._adaptive_strategy(
            base_decision=plan, zopa={}, signals=firm_signals, round_hint=2,
            lead_weeks=None, target_price=80.0, price=100.0,
        )
        assert decision["strategy"] == "package-trade"

    def test_the_trade_asks_are_added(self, agent, firm_signals):
        plan = decide_strategy(
            {"current_offer": 100.0, "target_price": 80.0, "round": 2},
            supplier_message="we are holding at this price for now",
        )
        decision = agent._adaptive_strategy(
            base_decision=plan, zopa={}, signals=firm_signals, round_hint=2,
            lead_weeks=None, target_price=80.0, price=100.0,
        )
        assert any("payment-terms trade" in ask for ask in decision["asks"])
