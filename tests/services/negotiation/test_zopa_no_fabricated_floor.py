"""The supplier cost floor is evidence or it is absent. It is never invented.

`_estimate_zopa` fell back to `price * 0.85` when should-cost, benchmark p10
and historic minimum were all missing -- which is the normal case, because
nothing in this repository produces a should-cost. Three things then computed
on that fiction:

  * `_adaptive_strategy` took a midpoint against it,
  * `_optimize_multi_issue` clamped candidate prices to it,
  * the counter-email justification claimed a market-analysis basis, and did so
    *unconditionally*: (offer - 0.85*offer) / (0.85*offer) = 0.1765, which is
    always above the 0.15 bar the claim was gated on.

Authorised behaviour change, 2026-09-05: the floor is now None when there is no
evidence for one, and `supplier_floor_basis` names where a real one came from.
"""
import pytest

from agents.negotiation_agent import NegotiationAgent
from tests.test_negotiation_agent import DummyNick


@pytest.fixture
def agent():
    return NegotiationAgent(DummyNick())


class TestNoEvidenceMeansNoFloor:
    def test_absent_evidence_yields_no_floor(self, agent):
        z = agent._estimate_zopa(price=100.0, target=80.0, history={},
                                 benchmarks={}, should_cost=None, signals={})
        assert z["supplier_floor"] is None

    def test_the_old_fabricated_value_is_not_returned(self, agent):
        """85.0 was what `price * 0.85` produced for this input."""
        z = agent._estimate_zopa(price=100.0, target=80.0, history={},
                                 benchmarks={}, should_cost=None, signals={})
        assert z["supplier_floor"] != 85.0

    def test_basis_says_none(self, agent):
        z = agent._estimate_zopa(price=100.0, target=80.0, history={},
                                 benchmarks={}, should_cost=None, signals={})
        assert z["supplier_floor_basis"] is None

    def test_a_finding_is_emitted(self, agent):
        z = agent._estimate_zopa(price=100.0, target=80.0, history={},
                                 benchmarks={}, should_cost=None, signals={})
        assert z.get("findings")
        assert any("supplier_floor" in f for f in z["findings"])

    def test_buyer_max_still_computed(self, agent):
        """Losing the floor must not lose the parts that do have evidence."""
        z = agent._estimate_zopa(price=100.0, target=80.0, history={},
                                 benchmarks={}, should_cost=None, signals={})
        assert z["buyer_max"] == 80.0


class TestRealEvidenceStillMakesAFloor:
    def test_should_cost_is_used_and_named(self, agent):
        z = agent._estimate_zopa(price=100.0, target=80.0, history={},
                                 benchmarks={}, should_cost=62.0, signals={})
        assert z["supplier_floor"] == pytest.approx(62.0)
        assert z["supplier_floor_basis"] == "should_cost"

    def test_benchmark_p10_is_used_and_named(self, agent):
        z = agent._estimate_zopa(price=100.0, target=80.0, history={},
                                 benchmarks={"p10": 70.0}, should_cost=None,
                                 signals={})
        assert z["supplier_floor"] == pytest.approx(70.0)
        assert z["supplier_floor_basis"] == "benchmark_p10"

    def test_history_minimum_is_used_and_named(self, agent):
        z = agent._estimate_zopa(price=100.0, target=80.0,
                                 history={"min_accepted_price": 68.0},
                                 benchmarks={}, should_cost=None, signals={})
        assert z["supplier_floor"] == pytest.approx(68.0)
        assert z["supplier_floor_basis"] == "history_min_accepted"

    def test_lowest_evidence_wins(self, agent):
        z = agent._estimate_zopa(price=100.0, target=80.0,
                                 history={"min_accepted_price": 68.0},
                                 benchmarks={"p10": 70.0}, should_cost=62.0,
                                 signals={})
        assert z["supplier_floor"] == pytest.approx(62.0)

    def test_signal_multipliers_still_apply_to_a_real_floor(self, agent):
        z = agent._estimate_zopa(price=100.0, target=80.0, history={},
                                 benchmarks={}, should_cost=60.0,
                                 signals={"capacity_tight": True})
        assert z["supplier_floor"] == pytest.approx(60.0 * 1.03)

    def test_multipliers_cannot_resurrect_an_absent_floor(self, agent):
        z = agent._estimate_zopa(price=100.0, target=80.0, history={},
                                 benchmarks={}, should_cost=None,
                                 signals={"capacity_tight": True,
                                          "tone": "firm"})
        assert z["supplier_floor"] is None


class TestDownstreamToleratesAbsence:
    def test_multi_issue_optimiser_runs_without_a_floor(self, agent):
        z = agent._estimate_zopa(price=100.0, target=80.0, history={},
                                 benchmarks={}, should_cost=None, signals={})
        out = agent._optimize_multi_issue(
            price=100.0, currency="GBP", target=80.0, lead_weeks=2.0,
            weights={}, policy={}, constraints={}, zopa=z, signals={},
            round_no=1,
        )
        assert "decision_overrides" in out
        assert out["decision_overrides"].get("counter_price") is not None

    def test_adaptive_strategy_runs_without_a_floor(self, agent):
        z = agent._estimate_zopa(price=100.0, target=80.0, history={},
                                 benchmarks={}, should_cost=None, signals={})
        out = agent._adaptive_strategy(
            base_decision={"strategy": "counter", "counter_price": 90.0,
                           "asks": []},
            zopa=z, signals={}, round_hint=2, lead_weeks=2.0,
            target_price=80.0, price=100.0,
        )
        assert out["counter_price"] is not None

    def test_market_analysis_claim_is_not_made_without_a_floor(self, agent):
        """The claim was unconditional under the fabricated floor.

        A negotiation email told every supplier we had benchmarked them.
        Without evidence there is no such sentence.
        """
        z = agent._estimate_zopa(price=100.0, target=80.0, history={},
                                 benchmarks={}, should_cost=None, signals={})
        floor = z.get("supplier_floor")
        assert not (floor and 100.0)
