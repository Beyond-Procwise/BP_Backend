"""The rules of the negotiation are read from policy, not written into the agent.

What ends a negotiation (the final-offer phrases), how long it runs (the round
limit) and what each round offers (the round tactics) were constants in
negotiation_agent.py and its config -- the formula registry's own notes called
them "not governed and not configurable" (gap report F22). They are now rules of
NegotiationBoundsPolicy, seeded with the values the code ran on, so nothing
changes until someone edits the row. These tests change the row and watch the
agent follow, and take a rule away and watch it refuse.
"""
import pytest

from agents.negotiation import config
from agents.negotiation.models import NegotiationContext, SupplierSignals
from agents.negotiation_agent import _detect_finality, compute_decision, plan_counter
from src.services import governed_limits
from tests.conftest import GOVERNED_LIMIT_SEED, _SeededPolicyEngine


class _EditedBounds:
    """The seeded policy store with NegotiationBoundsPolicy's rules replaced."""

    def __init__(self, rules):
        self.rules = rules
        self.seed = _SeededPolicyEngine()

    def get_policy(self, slug):
        if slug != "negotiation_bounds":
            return self.seed.get_policy(slug)
        return {"policyName": slug, "details": {"policy_identifier": slug, "rules": dict(self.rules)}}


@pytest.fixture
def bounds(monkeypatch):
    """Edit NegotiationBoundsPolicy for one test: ``bounds(max_rounds=2)`` or ``bounds(drop="max_rounds")``."""

    def _edit(drop=None, **changes):
        rules = {**GOVERNED_LIMIT_SEED["negotiation_bounds"], **changes}
        if drop:
            rules.pop(drop)
        governed_limits.reset_cache()
        monkeypatch.setattr(governed_limits, "_engine", lambda: _EditedBounds(rules))

    yield _edit
    governed_limits.reset_cache()


def _plan(offer, target, round_no, **payload):
    return compute_decision({"current_offer": offer, "target_price": target, "round": round_no, **payload})


class TestUnchangedUntilSomeoneEditsTheRow:
    """The seeded values reproduce the figures that were literals."""

    def test_round_1_anchor_and_midpoint(self):
        assert _plan(100.0, 80.0, 1)["counter_price"] == 88.0
        assert _plan(100.0, 95.0, 1)["counter_price"] == 97.5

    def test_round_2_push(self):
        assert _plan(100.0, 80.0, 2)["counter_price"] == 88.0

    def test_round_3_buffer(self):
        assert _plan(100.0, 80.0, 3)["counter_price"] == 84.8

    def test_past_the_round_limit_it_holds(self):
        assert _plan(100.0, 80.0, 4)["decision"] == "hold"

    def test_the_final_offer_phrases(self):
        assert _detect_finality("This is our FINAL PRICE.") is True
        assert _detect_finality("Happy to keep talking.") is False
        assert len(config.FINAL_OFFER_PHRASES()) == 14


class TestTheAgentFollowsTheRow:
    def test_a_new_phrase_ends_a_negotiation(self, bounds):
        assert _detect_finality("That is our closing figure.") is False
        bounds(final_offer_phrases=["closing figure"])
        assert _detect_finality("That is our closing figure.") is True
        assert _detect_finality("This is our final price.") is False, "the row is the whole list"

    def test_a_shorter_round_limit_holds_sooner(self, bounds):
        assert _plan(100.0, 80.0, 3)["decision"] == "counter"
        bounds(max_rounds=2)
        assert _plan(100.0, 80.0, 3)["decision"] == "hold"

    def test_a_caller_may_ask_for_fewer_rounds_than_policy(self):
        assert _plan(100.0, 80.0, 2, max_rounds=1)["decision"] == "hold"

    def test_the_first_counter_follows_the_governed_reduction(self, bounds):
        bounds(first_counter_aggr_pct=0.15)
        assert _plan(100.0, 80.0, 1)["counter_price"] == 85.0

    def test_the_anchor_threshold_follows_the_row(self, bounds):
        """A 25% gap is above the 10% threshold (anchor) and under a 30% one (midpoint)."""
        bounds(round1_anchor_gap_pct=0.30)
        assert _plan(100.0, 80.0, 1)["counter_price"] == 90.0

    def test_the_round_2_share_follows_the_row(self, bounds):
        bounds(round2_gap_share=0.5)
        assert _plan(100.0, 80.0, 2)["counter_price"] == 90.0

    def test_the_closing_buffer_follows_the_row(self, bounds):
        bounds(closing_risk_buffer_pct=0.10)
        assert _plan(100.0, 80.0, 3)["counter_price"] == 88.0


@pytest.mark.parametrize("rule", [
    "max_rounds", "round1_anchor_gap_pct", "round2_gap_share", "closing_risk_buffer_pct",
    "closing_min_abs_buffer", "closing_step_pct_of_gap", "first_counter_aggr_pct",
])
def test_a_missing_tactic_refuses_rather_than_assuming(bounds, rule):
    bounds(drop=rule)
    with pytest.raises(governed_limits.LimitUnavailable):
        _plan(100.0, 80.0, 1)


def test_missing_phrases_refuse_rather_than_treating_nothing_as_final(bounds):
    bounds(drop="final_offer_phrases")
    with pytest.raises(governed_limits.LimitUnavailable):
        _detect_finality("This is our final price.")


def test_phrases_that_are_not_a_list_are_refused(bounds):
    bounds(final_offer_phrases="final price")
    with pytest.raises((ValueError, TypeError)):
        config.FINAL_OFFER_PHRASES()


def test_the_planner_itself_takes_its_tactics_from_the_context():
    """plan_counter stays a pure function of what it is handed."""
    ctx = NegotiationContext(current_offer=100.0, target_price=80.0, round_index=1,
                             anchor_gap_pct=0.10, anchor_reduction_pct=0.20)
    assert plan_counter(ctx, SupplierSignals(message_text=""))["counter_price"] == 80.0
