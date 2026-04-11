"""Tests for NegotiationStrategyEngine.

Covers:
- All six strategy selection paths
- Strategy adaptation across rounds
- Position generation (target price, arguments, BATNA, tone)
- Policy rails (should_continue: accept, escalate, continue)
"""

import sys

import pytest

sys.path.insert(0, "src")

from engines.negotiation_strategy_engine import (
    ContinueDecision,
    NegotiationContext,
    NegotiationPosition,
    NegotiationStrategyEngine,
    Strategy,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ctx(
    supplier_name: str = "AcmeCorp",
    order_value: float = 20_000.0,
    category: str = "IT Hardware",
    supplier_history_count: int = 0,
    alternative_quotes: int = 0,
    round_number: int = 1,
    urgency: str = "normal",
    quote_price: float = 0.0,
) -> NegotiationContext:
    return NegotiationContext(
        supplier_name=supplier_name,
        order_value=order_value,
        category=category,
        supplier_history_count=supplier_history_count,
        alternative_quotes=alternative_quotes,
        round_number=round_number,
        urgency=urgency,
        quote_price=quote_price,
    )


def _engine(**kwargs) -> NegotiationStrategyEngine:
    defaults = dict(max_rounds=3, auto_approve_threshold=5_000.0, escalation_threshold=50_000.0)
    defaults.update(kwargs)
    return NegotiationStrategyEngine(**defaults)


# ---------------------------------------------------------------------------
# 1. Dataclass defaults
# ---------------------------------------------------------------------------

class TestDataclassDefaults:
    def test_negotiation_context_defaults(self):
        ctx = NegotiationContext(supplier_name="X", order_value=1000.0, category="MRO")
        assert ctx.supplier_history_count == 0
        assert ctx.alternative_quotes == 0
        assert ctx.round_number == 1
        assert ctx.urgency == "normal"
        assert ctx.quote_price == 0.0

    def test_strategy_fields_accessible(self):
        engine = _engine()
        s = engine.STRATEGY_COOPERATIVE
        assert isinstance(s.name, str)
        assert isinstance(s.target_discount, float)
        assert isinstance(s.fallback_name, str)
        assert isinstance(s.description, str)
        assert isinstance(s.tone, str)

    def test_continue_decision_fields(self):
        d = ContinueDecision(action="continue", reason="within range")
        assert d.action == "continue"
        assert d.reason == "within range"

    def test_negotiation_position_fields(self):
        pos = NegotiationPosition(
            target_price=9500.0,
            arguments=["arg1"],
            tone="assertive",
            batna_analysis="BATNA: ...",
        )
        assert pos.target_price == 9500.0
        assert pos.arguments == ["arg1"]


# ---------------------------------------------------------------------------
# 2. Strategy selection paths
# ---------------------------------------------------------------------------

class TestStrategySelection:
    def test_cooperative_when_high_history(self):
        engine = _engine()
        ctx = _ctx(supplier_history_count=5)
        strategy = engine.select_strategy(ctx)
        assert strategy.name == "cooperative"

    def test_cooperative_when_history_exceeds_threshold(self):
        engine = _engine()
        ctx = _ctx(supplier_history_count=10)
        strategy = engine.select_strategy(ctx)
        assert strategy.name == "cooperative"

    def test_competitive_when_alternatives_available(self):
        engine = _engine()
        ctx = _ctx(supplier_history_count=0, alternative_quotes=2)
        strategy = engine.select_strategy(ctx)
        assert strategy.name == "competitive"

    def test_competitive_when_many_alternatives(self):
        engine = _engine()
        ctx = _ctx(supplier_history_count=1, alternative_quotes=5)
        strategy = engine.select_strategy(ctx)
        assert strategy.name == "competitive"

    def test_anchoring_when_no_history_no_alternatives(self):
        engine = _engine()
        ctx = _ctx(supplier_history_count=0, alternative_quotes=0)
        strategy = engine.select_strategy(ctx)
        assert strategy.name == "anchoring"

    def test_bundling_when_multiple_categories(self):
        engine = _engine()
        ctx = _ctx(
            supplier_history_count=1,
            alternative_quotes=1,
            category="IT Hardware, Office Supplies",
        )
        strategy = engine.select_strategy(ctx)
        assert strategy.name == "bundling"

    def test_time_leverage_when_urgency_high(self):
        engine = _engine()
        ctx = _ctx(
            supplier_history_count=1,
            alternative_quotes=1,
            urgency="high",
            category="MRO",
        )
        strategy = engine.select_strategy(ctx)
        assert strategy.name == "time_leverage"

    def test_default_competitive(self):
        """Supplier with 1 history, 1 alternative, normal urgency, single category."""
        engine = _engine()
        ctx = _ctx(supplier_history_count=1, alternative_quotes=1, urgency="normal", category="MRO")
        strategy = engine.select_strategy(ctx)
        assert strategy.name == "competitive"

    def test_history_takes_priority_over_alternatives(self):
        """High history should win even if alternatives >= 2."""
        engine = _engine()
        ctx = _ctx(supplier_history_count=5, alternative_quotes=3)
        strategy = engine.select_strategy(ctx)
        assert strategy.name == "cooperative"

    def test_alternatives_takes_priority_over_anchoring(self):
        """alternatives >= 2 should beat zero-history/zero-alternative anchoring rule."""
        engine = _engine()
        ctx = _ctx(supplier_history_count=0, alternative_quotes=2)
        strategy = engine.select_strategy(ctx)
        assert strategy.name == "competitive"


# ---------------------------------------------------------------------------
# 3. Strategy adaptation
# ---------------------------------------------------------------------------

class TestStrategyAdaptation:
    def test_adapt_to_collaborative_when_no_movement_round3(self):
        engine = _engine()
        current = engine.STRATEGY_COMPETITIVE
        result = engine.adapt_strategy(current, {"supplier_movement": 0, "round_number": 3})
        assert result.name == "collaborative_problem_solving"

    def test_adapt_to_collaborative_when_no_movement_beyond_round3(self):
        engine = _engine()
        current = engine.STRATEGY_ANCHORING
        result = engine.adapt_strategy(current, {"supplier_movement": 0, "round_number": 5})
        assert result.name == "collaborative_problem_solving"

    def test_keep_strategy_when_supplier_moved(self):
        engine = _engine()
        current = engine.STRATEGY_COMPETITIVE
        result = engine.adapt_strategy(current, {"supplier_movement": 500, "round_number": 2})
        assert result.name == "competitive"

    def test_keep_strategy_when_supplier_moved_small_amount(self):
        engine = _engine()
        current = engine.STRATEGY_COOPERATIVE
        result = engine.adapt_strategy(current, {"supplier_movement": 1, "round_number": 3})
        assert result.name == "cooperative"

    def test_keep_strategy_no_movement_round1(self):
        """No movement at round 1 should NOT trigger collaborative (threshold is 3)."""
        engine = _engine()
        current = engine.STRATEGY_ANCHORING
        result = engine.adapt_strategy(current, {"supplier_movement": 0, "round_number": 1})
        assert result.name == "anchoring"

    def test_keep_strategy_no_movement_round2(self):
        """No movement at round 2 should NOT trigger collaborative."""
        engine = _engine()
        current = engine.STRATEGY_COMPETITIVE
        result = engine.adapt_strategy(current, {"supplier_movement": 0, "round_number": 2})
        assert result.name == "competitive"

    def test_adapt_missing_keys_defaults_gracefully(self):
        """Missing keys in round_result should not raise."""
        engine = _engine()
        current = engine.STRATEGY_COMPETITIVE
        result = engine.adapt_strategy(current, {})
        assert result.name == "competitive"


# ---------------------------------------------------------------------------
# 4. Position generation
# ---------------------------------------------------------------------------

class TestPositionGeneration:
    def test_target_price_calculated_from_discount(self):
        engine = _engine()
        strategy = engine.STRATEGY_COOPERATIVE  # 5% discount
        ctx = _ctx(order_value=10_000.0)
        pos = engine.generate_position(strategy, ctx)
        assert pos.target_price == pytest.approx(9_500.0, rel=1e-6)

    def test_target_price_anchoring(self):
        engine = _engine()
        strategy = engine.STRATEGY_ANCHORING  # 15% discount
        ctx = _ctx(order_value=10_000.0)
        pos = engine.generate_position(strategy, ctx)
        assert pos.target_price == pytest.approx(8_500.0, rel=1e-6)

    def test_target_price_competitive(self):
        engine = _engine()
        strategy = engine.STRATEGY_COMPETITIVE  # 10% discount
        ctx = _ctx(order_value=20_000.0)
        pos = engine.generate_position(strategy, ctx)
        assert pos.target_price == pytest.approx(18_000.0, rel=1e-6)

    def test_arguments_list_not_empty(self):
        engine = _engine()
        for strategy in engine._strategy_map.values():
            ctx = _ctx(supplier_history_count=5, alternative_quotes=3)
            pos = engine.generate_position(strategy, ctx)
            assert len(pos.arguments) > 0, f"No arguments for {strategy.name}"

    def test_tone_matches_strategy(self):
        engine = _engine()
        for strategy in engine._strategy_map.values():
            ctx = _ctx()
            pos = engine.generate_position(strategy, ctx)
            assert pos.tone == strategy.tone

    def test_batna_analysis_mentions_alternatives(self):
        engine = _engine()
        strategy = engine.STRATEGY_COMPETITIVE
        ctx = _ctx(alternative_quotes=3)
        pos = engine.generate_position(strategy, ctx)
        assert "3" in pos.batna_analysis or "alternative" in pos.batna_analysis.lower()

    def test_batna_analysis_no_alternatives(self):
        engine = _engine()
        strategy = engine.STRATEGY_ANCHORING
        ctx = _ctx(supplier_history_count=0, alternative_quotes=0)
        pos = engine.generate_position(strategy, ctx)
        assert "batna" in pos.batna_analysis.lower()

    def test_cooperative_arguments_mention_history(self):
        engine = _engine()
        strategy = engine.STRATEGY_COOPERATIVE
        ctx = _ctx(supplier_history_count=8, supplier_name="BestCo")
        pos = engine.generate_position(strategy, ctx)
        args_text = " ".join(pos.arguments).lower()
        assert "8" in " ".join(pos.arguments) or "bestco" in args_text

    def test_bundling_arguments_mention_category(self):
        engine = _engine()
        strategy = engine.STRATEGY_BUNDLING
        ctx = _ctx(category="IT Hardware, Office Supplies")
        pos = engine.generate_position(strategy, ctx)
        args_text = " ".join(pos.arguments)
        assert "IT Hardware" in args_text or "Office Supplies" in args_text

    def test_position_is_negotiation_position_instance(self):
        engine = _engine()
        ctx = _ctx()
        pos = engine.generate_position(engine.STRATEGY_COMPETITIVE, ctx)
        assert isinstance(pos, NegotiationPosition)


# ---------------------------------------------------------------------------
# 5. Policy rails – should_continue
# ---------------------------------------------------------------------------

class TestShouldContinue:
    def test_escalate_when_max_rounds_exceeded(self):
        engine = _engine(max_rounds=3)
        ctx = _ctx(round_number=4, order_value=20_000.0)
        decision = engine.should_continue(ctx)
        assert decision.action == "escalate"
        assert "round" in decision.reason.lower()

    def test_escalate_when_far_beyond_max_rounds(self):
        engine = _engine(max_rounds=3)
        ctx = _ctx(round_number=10, order_value=20_000.0)
        decision = engine.should_continue(ctx)
        assert decision.action == "escalate"

    def test_accept_when_below_auto_approve_threshold(self):
        engine = _engine(auto_approve_threshold=5_000.0)
        ctx = _ctx(order_value=4_999.99, round_number=1)
        decision = engine.should_continue(ctx)
        assert decision.action == "accept"
        assert "auto" in decision.reason.lower() or "threshold" in decision.reason.lower()

    def test_accept_when_well_below_auto_approve_threshold(self):
        engine = _engine(auto_approve_threshold=5_000.0)
        ctx = _ctx(order_value=100.0, round_number=1)
        decision = engine.should_continue(ctx)
        assert decision.action == "accept"

    def test_escalate_when_above_escalation_threshold(self):
        engine = _engine(escalation_threshold=50_000.0)
        ctx = _ctx(order_value=50_001.0, round_number=1)
        decision = engine.should_continue(ctx)
        assert decision.action == "escalate"
        assert "escalation" in decision.reason.lower() or "senior" in decision.reason.lower()

    def test_escalate_when_far_above_escalation_threshold(self):
        engine = _engine(escalation_threshold=50_000.0)
        ctx = _ctx(order_value=500_000.0, round_number=2)
        decision = engine.should_continue(ctx)
        assert decision.action == "escalate"

    def test_continue_within_normal_range(self):
        engine = _engine(max_rounds=3, auto_approve_threshold=5_000.0, escalation_threshold=50_000.0)
        ctx = _ctx(order_value=20_000.0, round_number=2)
        decision = engine.should_continue(ctx)
        assert decision.action == "continue"

    def test_continue_on_first_round(self):
        engine = _engine()
        ctx = _ctx(order_value=25_000.0, round_number=1)
        decision = engine.should_continue(ctx)
        assert decision.action == "continue"

    def test_continue_on_max_round(self):
        engine = _engine(max_rounds=3)
        ctx = _ctx(order_value=20_000.0, round_number=3)
        decision = engine.should_continue(ctx)
        assert decision.action == "continue"

    def test_max_rounds_priority_over_auto_approve(self):
        """Exhausted rounds should escalate even if order value is low."""
        engine = _engine(max_rounds=3, auto_approve_threshold=5_000.0)
        ctx = _ctx(order_value=3_000.0, round_number=4)
        decision = engine.should_continue(ctx)
        assert decision.action == "escalate"

    def test_decision_is_continue_decision_instance(self):
        engine = _engine()
        ctx = _ctx(order_value=20_000.0, round_number=1)
        decision = engine.should_continue(ctx)
        assert isinstance(decision, ContinueDecision)
        assert decision.action in ("continue", "escalate", "accept", "walk_away")

    def test_custom_thresholds_respected(self):
        engine = NegotiationStrategyEngine(
            max_rounds=5,
            auto_approve_threshold=2_000.0,
            escalation_threshold=100_000.0,
        )
        # Below custom auto_approve
        ctx = _ctx(order_value=1_500.0, round_number=1)
        assert engine.should_continue(ctx).action == "accept"

        # Above custom escalation
        ctx = _ctx(order_value=100_001.0, round_number=1)
        assert engine.should_continue(ctx).action == "escalate"

        # Within custom range
        ctx = _ctx(order_value=50_000.0, round_number=1)
        assert engine.should_continue(ctx).action == "continue"


# ---------------------------------------------------------------------------
# 6. Six strategies exist with correct attributes
# ---------------------------------------------------------------------------

class TestSixStrategies:
    def test_all_six_strategies_defined(self):
        engine = _engine()
        names = set(engine._strategy_map.keys())
        expected = {
            "cooperative",
            "competitive",
            "anchoring",
            "bundling",
            "time_leverage",
            "collaborative_problem_solving",
        }
        assert names == expected

    @pytest.mark.parametrize("strategy_name,expected_tone", [
        ("cooperative", "collaborative"),
        ("competitive", "assertive"),
        ("anchoring", "firm"),
        ("bundling", "analytical"),
        ("time_leverage", "urgent"),
        ("collaborative_problem_solving", "collaborative"),
    ])
    def test_strategy_tones(self, strategy_name, expected_tone):
        engine = _engine()
        strategy = engine._strategy_map[strategy_name]
        assert strategy.tone == expected_tone

    @pytest.mark.parametrize("strategy_name", [
        "cooperative",
        "competitive",
        "anchoring",
        "bundling",
        "time_leverage",
        "collaborative_problem_solving",
    ])
    def test_strategy_has_positive_discount(self, strategy_name):
        engine = _engine()
        strategy = engine._strategy_map[strategy_name]
        assert 0.0 < strategy.target_discount < 1.0

    @pytest.mark.parametrize("strategy_name", [
        "cooperative",
        "competitive",
        "anchoring",
        "bundling",
        "time_leverage",
        "collaborative_problem_solving",
    ])
    def test_strategy_has_fallback(self, strategy_name):
        engine = _engine()
        strategy = engine._strategy_map[strategy_name]
        assert strategy.fallback_name in engine._strategy_map

    @pytest.mark.parametrize("strategy_name", [
        "cooperative",
        "competitive",
        "anchoring",
        "bundling",
        "time_leverage",
        "collaborative_problem_solving",
    ])
    def test_strategy_has_non_empty_description(self, strategy_name):
        engine = _engine()
        strategy = engine._strategy_map[strategy_name]
        assert len(strategy.description) > 10
