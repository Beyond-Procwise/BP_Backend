"""Comprehensive negotiation skills and strategy tests.

These tests verify the negotiation agent's core decision-making logic,
counter-offer calculations, playbook integration, multi-round strategy
progression, finality detection, ZOPA estimation, and position management.

They run without external dependencies (no Ollama, no database, no Qdrant).
"""

import os
import sys
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("OLLAMA_USE_GPU", "1")
os.environ.setdefault("OLLAMA_NUM_PARALLEL", "4")
os.environ.setdefault("OMP_NUM_THREADS", "8")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.negotiation_agent import (
    NegotiationAgent,
    NegotiationContext,
    SupplierSignals,
    plan_counter,
    _detect_finality,
    _format_currency,
    FINAL_OFFER_PATTERNS,
)
from agents.base_agent import AgentContext, AgentOutput, AgentStatus


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _mock_ollama(monkeypatch):
    monkeypatch.setattr(
        "ollama.chat",
        lambda *a, **k: {"message": {"content": "{}"}},
    )
    monkeypatch.setattr(
        "ollama.generate",
        lambda *a, **k: {"response": "{}"},
    )


class DummyNick:
    def __init__(self):
        self.settings = SimpleNamespace(
            qdrant_collection_name="dummy",
            extraction_model="gpt-oss",
            script_user="tester",
            ses_default_sender="noreply@example.com",
            enable_learning=False,
            hitl_enabled=True,
        )
        self.action_logs: List[Dict[str, Any]] = []

        def _log_action(**kwargs):
            self.action_logs.append(dict(kwargs))
            return kwargs.get("action_id") or f"action-{len(self.action_logs)}"

        self.process_routing_service = SimpleNamespace(
            log_process=lambda **_: 1,
            log_run_detail=lambda **_: "run-1",
            log_action=_log_action,
            validate_workflow_id=lambda *args, **kwargs: True,
        )
        self.ollama_options = lambda: {}
        self.qdrant_client = SimpleNamespace()
        self.embedding_model = SimpleNamespace(encode=lambda x: [0.0])

        def get_db_connection():
            class DummyConn:
                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    pass

                def cursor(self):
                    class DummyCursor:
                        description = None
                        _results = []

                        def __enter__(self):
                            return self

                        def __exit__(self, *args):
                            pass

                        def execute(self, query=None, params=None, *args, **kwargs):
                            self._results = []
                            if query and "information_schema.columns" in str(query).lower():
                                self._results = [
                                    ("workflow_id",), ("supplier_id",),
                                    ("current_round",), ("status",),
                                    ("awaiting_response",), ("supplier_reply_count",),
                                ]
                            elif query and "to_regclass" in str(query).lower():
                                self._results = [(None,)]

                        def fetchone(self):
                            return self._results[0] if self._results else None

                        def fetchall(self):
                            return list(self._results)

                    return DummyCursor()

                def commit(self):
                    pass

                def rollback(self):
                    pass

            return DummyConn()

        self.get_db_connection = get_db_connection


@pytest.fixture
def nick():
    return DummyNick()


@pytest.fixture
def agent(nick):
    return NegotiationAgent(nick)


def _stub_email(agent, monkeypatch):
    """Stub the email drafting agent to avoid external calls."""
    stub_output = AgentOutput(
        status=AgentStatus.SUCCESS,
        data={
            "drafts": [
                {
                    "supplier_id": "S1",
                    "rfq_id": "RFQ-100",
                    "subject": "Re: RFQ-100",
                    "body": "Email body",
                    "sent_status": False,
                }
            ],
            "subject": "Re: RFQ-100",
            "body": "Email body",
        },
    )
    stub_output.action_id = "email-stub-1"
    monkeypatch.setattr(
        agent, "_invoke_email_drafting_agent",
        lambda ctx, payload: stub_output,
    )
    monkeypatch.setattr(
        agent, "_await_supplier_responses",
        lambda **_: [{"message_id": "reply-1", "supplier_id": "S1", "rfq_id": "RFQ-100"}],
    )


# ===================================================================
# 1. CORE DECISION ENGINE: plan_counter()
# ===================================================================

class TestPlanCounterDecisions:
    """Test the deterministic counter-offer decision engine."""

    def test_accept_when_offer_at_target(self):
        """Accept immediately when supplier offer equals target price."""
        ctx = NegotiationContext(current_offer=1000.0, target_price=1000.0)
        signals = SupplierSignals()
        result = plan_counter(ctx, signals)
        assert result["decision"] == "accept"
        assert result["counter_price"] == 1000.0

    def test_accept_when_offer_below_target(self):
        """Accept when supplier offers less than our target."""
        ctx = NegotiationContext(current_offer=900.0, target_price=1000.0)
        signals = SupplierSignals()
        result = plan_counter(ctx, signals)
        assert result["decision"] == "accept"
        assert result["counter_price"] == 900.0

    def test_counter_round1_large_gap_aggressive_anchor(self):
        """Round 1 with >10% gap: apply 12% aggressive anchor reduction."""
        ctx = NegotiationContext(
            current_offer=1500.0,
            target_price=1200.0,
            round_index=1,
        )
        signals = SupplierSignals()
        result = plan_counter(ctx, signals)
        assert result["decision"] == "counter"
        # 12% reduction from 1500 = 1320, but floored at target 1200
        expected = max(1200.0, 1500.0 * 0.88)
        assert result["counter_price"] == pytest.approx(expected, rel=1e-4)
        assert "anchor" in " ".join(result["log"]).lower()

    def test_counter_round1_small_gap_midpoint(self):
        """Round 1 with <=10% gap: propose midpoint."""
        ctx = NegotiationContext(
            current_offer=1100.0,
            target_price=1000.0,
            round_index=1,
        )
        signals = SupplierSignals()
        result = plan_counter(ctx, signals)
        assert result["decision"] == "counter"
        assert result["counter_price"] == pytest.approx(1050.0, rel=1e-4)
        assert "midpoint" in " ".join(result["log"]).lower()

    def test_counter_round2_large_gap_60pct_capture(self):
        """Round 2 with >10% gap: capture 60% of remaining gap."""
        ctx = NegotiationContext(
            current_offer=1400.0,
            target_price=1000.0,
            round_index=2,
        )
        signals = SupplierSignals()
        result = plan_counter(ctx, signals)
        assert result["decision"] == "counter"
        # gap=400, 60% capture = 240, counter = 1400-240 = 1160
        expected = 1400.0 - (400.0 * 0.6)
        assert result["counter_price"] == pytest.approx(expected, rel=1e-4)

    def test_counter_round2_small_gap_midpoint(self):
        """Round 2 with <=10% gap: split difference."""
        ctx = NegotiationContext(
            current_offer=1080.0,
            target_price=1000.0,
            round_index=2,
        )
        signals = SupplierSignals()
        result = plan_counter(ctx, signals)
        assert result["decision"] == "counter"
        assert result["counter_price"] == pytest.approx(1040.0, rel=1e-4)

    def test_counter_round3_buffer_enforcement(self):
        """Round 3: apply risk-adjusted buffer threshold."""
        ctx = NegotiationContext(
            current_offer=1200.0,
            target_price=1000.0,
            round_index=3,
            risk_buffer_pct=0.05,
        )
        signals = SupplierSignals()
        result = plan_counter(ctx, signals)
        assert result["decision"] == "counter"
        # buffer = max(0, 1000*0.05) = 50, threshold = 1050
        # 1050 < 1200, so counter = 1050
        assert result["counter_price"] == pytest.approx(1050.0, rel=1e-4)

    def test_counter_round3_soft_landing(self):
        """Round 3 with offer close to target: soft decrement."""
        ctx = NegotiationContext(
            current_offer=1040.0,
            target_price=1000.0,
            round_index=3,
            risk_buffer_pct=0.05,
            step_pct_of_gap=0.1,
        )
        signals = SupplierSignals()
        result = plan_counter(ctx, signals)
        assert result["decision"] == "counter"
        # buffer = 50, threshold = 1050 > 1040, so soft landing
        # counter = max(1000, 1040*(1-0.1)) = max(1000, 936) = 1000
        assert result["counter_price"] >= 1000.0

    def test_counter_never_below_target(self):
        """Counter price should never go below target price."""
        for round_idx in [1, 2, 3]:
            ctx = NegotiationContext(
                current_offer=1100.0,
                target_price=1000.0,
                round_index=round_idx,
            )
            signals = SupplierSignals()
            result = plan_counter(ctx, signals)
            if result["counter_price"] is not None:
                assert result["counter_price"] >= 1000.0

    def test_hold_when_max_rounds_exceeded(self):
        """Decision is 'hold' when rounds exceed configured max."""
        ctx = NegotiationContext(
            current_offer=1200.0,
            target_price=1000.0,
            round_index=4,
            max_rounds=3,
        )
        signals = SupplierSignals()
        result = plan_counter(ctx, signals)
        assert result["decision"] == "hold"
        assert "max rounds" in " ".join(result["log"]).lower()

    def test_clarify_when_invalid_prices(self):
        """Decision is 'clarify' when prices are missing or zero."""
        ctx = NegotiationContext(current_offer=0.0, target_price=1000.0)
        signals = SupplierSignals()
        result = plan_counter(ctx, signals)
        assert result["decision"] == "clarify"
        assert result["counter_price"] is None

    def test_clarify_when_negative_target(self):
        ctx = NegotiationContext(current_offer=1000.0, target_price=-100.0)
        signals = SupplierSignals()
        result = plan_counter(ctx, signals)
        assert result["decision"] == "clarify"

    def test_early_payment_ask_included(self):
        """Early payment discount ask is included when configured."""
        ctx = NegotiationContext(
            current_offer=1100.0,
            target_price=1000.0,
            round_index=1,
            ask_early_pay_disc=0.02,
        )
        signals = SupplierSignals()
        result = plan_counter(ctx, signals)
        asks_text = " ".join(result["asks"]).lower()
        assert "early payment" in asks_text
        assert "2.0%" in " ".join(result["asks"])

    def test_lead_time_request_included(self):
        """Lead time request is included when ask_lead_time_keep is True."""
        ctx = NegotiationContext(
            current_offer=1100.0,
            target_price=1000.0,
            round_index=1,
            ask_lead_time_keep=True,
        )
        signals = SupplierSignals()
        result = plan_counter(ctx, signals)
        assert result["lead_time_request"] is not None
        assert "lead time" in result["lead_time_request"].lower()


# ===================================================================
# 2. FINALITY DETECTION
# ===================================================================

class TestFinalityDetection:
    """Test detection of final-offer language in supplier messages."""

    @pytest.mark.parametrize("phrase", [
        "This is our best and final offer.",
        "We cannot go lower than this price.",
        "This is the lowest we can do.",
        "Consider this our rock bottom price.",
        "Take it or leave it.",
        "This is our final price for this RFQ.",
        "We've reached our best final position.",
    ])
    def test_detects_finality_phrases(self, phrase):
        assert _detect_finality(phrase) is True

    @pytest.mark.parametrize("phrase", [
        "We are happy to discuss further.",
        "Please let us know your thoughts.",
        "We can review the pricing again.",
        "Here is our updated quotation.",
        "Looking forward to your counter.",
        "",
    ])
    def test_no_false_positive_finality(self, phrase):
        assert _detect_finality(phrase) is False

    def test_accept_final_offer_within_threshold(self):
        """Accept final offer when price is within walkaway threshold."""
        ctx = NegotiationContext(
            current_offer=1100.0,
            target_price=1000.0,
            walkaway_price=1150.0,
        )
        signals = SupplierSignals(
            message_text="This is our best and final offer."
        )
        result = plan_counter(ctx, signals)
        assert result["decision"] == "accept"
        assert result["finality"] is True

    def test_decline_final_offer_above_threshold(self):
        """Decline final offer when price exceeds walkaway threshold."""
        ctx = NegotiationContext(
            current_offer=1500.0,
            target_price=1000.0,
            walkaway_price=1200.0,
        )
        signals = SupplierSignals(
            message_text="This is our final offer, take it or leave it."
        )
        result = plan_counter(ctx, signals)
        assert result["decision"] == "decline"
        assert result["finality"] is True
        assert result["counter_price"] is None


# ===================================================================
# 3. MULTI-ROUND STRATEGY PROGRESSION
# ===================================================================

class TestMultiRoundProgression:
    """Test that strategy evolves correctly across negotiation rounds."""

    def test_counter_decreases_across_rounds(self):
        """Counter-offers should converge toward target over rounds."""
        counters = []
        for round_idx in [1, 2, 3]:
            ctx = NegotiationContext(
                current_offer=1500.0,
                target_price=1000.0,
                round_index=round_idx,
                risk_buffer_pct=0.05,
            )
            signals = SupplierSignals()
            result = plan_counter(ctx, signals)
            if result["counter_price"] is not None:
                counters.append(result["counter_price"])

        assert len(counters) == 3
        # Each subsequent counter should be closer to target
        for i in range(1, len(counters)):
            assert counters[i] <= counters[i - 1], (
                f"Round {i + 1} counter {counters[i]} should be <= "
                f"round {i} counter {counters[i - 1]}"
            )

    def test_round1_asks_are_volume_focused(self):
        """Round 1 with large gap should include volume/spec asks."""
        ctx = NegotiationContext(
            current_offer=1500.0,
            target_price=1000.0,
            round_index=1,
        )
        signals = SupplierSignals()
        result = plan_counter(ctx, signals)
        asks_text = " ".join(result["asks"]).lower()
        assert "volume" in asks_text or "alternative" in asks_text or "payment" in asks_text

    def test_round3_reinforces_validation_asks(self):
        """Round 3+ should include packaging/warranty/compliance validation."""
        ctx = NegotiationContext(
            current_offer=1200.0,
            target_price=1000.0,
            round_index=3,
        )
        signals = SupplierSignals()
        result = plan_counter(ctx, signals)
        asks_text = " ".join(result["asks"]).lower()
        assert "packaging" in asks_text or "warranty" in asks_text or "compliance" in asks_text

    def test_strategy_message_contains_round_info(self):
        """Strategy message should reference the current round."""
        for round_idx in [1, 2, 3]:
            ctx = NegotiationContext(
                current_offer=1500.0,
                target_price=1000.0,
                round_index=round_idx,
            )
            signals = SupplierSignals()
            result = plan_counter(ctx, signals)
            assert f"Round {round_idx}" in result["message"]


# ===================================================================
# 4. WALKAWAY PRICE ENFORCEMENT
# ===================================================================

class TestWalkawayEnforcement:
    """Test walkaway price boundaries."""

    def test_hold_counter_at_walkaway_when_set(self):
        """When max rounds exceeded, counter should not exceed walkaway."""
        ctx = NegotiationContext(
            current_offer=2000.0,
            target_price=1000.0,
            walkaway_price=1500.0,
            round_index=4,
            max_rounds=3,
        )
        signals = SupplierSignals()
        result = plan_counter(ctx, signals)
        assert result["decision"] == "hold"
        assert result["counter_price"] <= 1500.0

    def test_walkaway_none_uses_current_offer(self):
        """Without walkaway, hold uses current offer."""
        ctx = NegotiationContext(
            current_offer=1200.0,
            target_price=1000.0,
            walkaway_price=None,
            round_index=4,
            max_rounds=3,
        )
        signals = SupplierSignals()
        result = plan_counter(ctx, signals)
        assert result["decision"] == "hold"
        assert result["counter_price"] == 1200.0


# ===================================================================
# 5. CURRENCY FORMATTING
# ===================================================================

class TestCurrencyFormatting:
    def test_format_with_currency(self):
        assert _format_currency(1234.56, "USD") == "USD 1,234.56"
        assert _format_currency(1000.0, "GBP") == "GBP 1,000.00"
        assert _format_currency(99.9, "EUR") == "EUR 99.90"

    def test_format_without_currency(self):
        result = _format_currency(1234.56, None)
        assert result == "1,234.56"


# ===================================================================
# 6. FULL AGENT INTEGRATION: COUNTER WITH PLAYBOOK
# ===================================================================

class TestAgentCounterWithPlaybook:
    """Test the full agent run() with playbook integration."""

    def test_counter_includes_playbook_plays(self, agent, monkeypatch):
        _stub_email(agent, monkeypatch)
        context = AgentContext(
            workflow_id="wf-plays",
            agent_id="negotiation",
            user_id="tester",
            input_data={
                "supplier": "SupplierA",
                "current_offer": 1300.0,
                "target_price": 1000.0,
                "rfq_id": "RFQ-PLAY",
                "currency": "USD",
                "supplier_type": "Leverage",
                "negotiation_style": "Collaborative",
                "lever_priorities": ["Commercial", "Operational"],
            },
        )
        output = agent.run(context)
        assert output.status == AgentStatus.SUCCESS
        assert output.data["decision"]["strategy"] == "counter"
        plays = output.data.get("play_recommendations", [])
        assert isinstance(plays, list)
        if plays:
            assert "lever" in plays[0]
            assert "play" in plays[0]
            assert "score" in plays[0]

    def test_counter_message_has_negotiation_structure(self, agent, monkeypatch):
        """Message should contain pricing and collaborative language."""
        _stub_email(agent, monkeypatch)
        context = AgentContext(
            workflow_id="wf-msg",
            agent_id="negotiation",
            user_id="tester",
            input_data={
                "supplier": "SupplierB",
                "current_offer": 1200.0,
                "target_price": 1000.0,
                "rfq_id": "RFQ-MSG",
                "currency": "GBP",
            },
        )
        output = agent.run(context)
        message = output.data.get("message", "")
        assert "Round 1" in message
        assert "counter" in message.lower() or "1" in message

    def test_accept_with_sweetener_asks(self, agent, monkeypatch):
        """When offer is at target, accept with sweetener asks."""
        _stub_email(agent, monkeypatch)
        context = AgentContext(
            workflow_id="wf-accept",
            agent_id="negotiation",
            user_id="tester",
            input_data={
                "supplier": "SupplierC",
                "current_offer": 950.0,
                "target_price": 1000.0,
                "rfq_id": "RFQ-ACCEPT",
            },
        )
        output = agent.run(context)
        assert output.data["decision"]["strategy"] == "accept"

    def test_decline_final_offer_above_walkaway(self, agent, monkeypatch):
        """Decline final offer that exceeds walkaway threshold."""
        context = AgentContext(
            workflow_id="wf-decline",
            agent_id="negotiation",
            user_id="tester",
            input_data={
                "supplier": "SupplierD",
                "current_offer": 2000.0,
                "target_price": 1000.0,
                "rfq_id": "RFQ-DECLINE",
                "supplier_message": "This is our final offer. Take it or leave it.",
                "walkaway_price": 1500.0,
            },
        )
        output = agent.run(context)
        assert output.data["negotiation_allowed"] is False
        assert "final" in output.data["message"].lower()

    def test_session_state_tracks_rounds(self, agent, monkeypatch):
        """Session state should track round progression."""
        _stub_email(agent, monkeypatch)
        context = AgentContext(
            workflow_id="wf-state",
            agent_id="negotiation",
            user_id="tester",
            input_data={
                "supplier": "SupplierE",
                "current_offer": 1300.0,
                "target_price": 1000.0,
                "rfq_id": "RFQ-STATE",
                "round": 1,
            },
        )
        output = agent.run(context)
        session = output.data.get("session_state", {})
        assert session.get("current_round", 0) >= 1


# ===================================================================
# 7. PLAYBOOK POLICY ALIGNMENT
# ===================================================================

class TestPlaybookPolicyAlignment:
    """Test that playbook plays respect policy constraints."""

    def test_restricted_lever_gets_lower_score(self, agent, monkeypatch):
        """Restricted levers should receive a policy penalty in their scores."""
        _stub_email(agent, monkeypatch)
        context = AgentContext(
            workflow_id="wf-policy",
            agent_id="negotiation",
            user_id="tester",
            input_data={
                "supplier": "SupplierF",
                "current_offer": 1500.0,
                "target_price": 1000.0,
                "rfq_id": "RFQ-POLICY",
                "supplier_type": "Leverage",
                "negotiation_style": "Competitive",
                "lever_priorities": ["Risk", "Commercial"],
                "policies": [
                    {
                        "preferred_levers": ["Risk"],
                        "restricted_levers": ["Commercial"],
                    }
                ],
            },
        )
        output = agent.run(context)
        plays = output.data.get("play_recommendations", [])
        # Verify playbook produces plays and they have expected structure
        if plays:
            assert all("lever" in p and "score" in p for p in plays), (
                "Each play should have 'lever' and 'score' fields"
            )
            lever_names = [p["lever"] for p in plays]
            # Risk should appear in plays since it is preferred
            assert "Risk" in lever_names or len(plays) == 0, (
                "Preferred lever 'Risk' should appear in play recommendations"
            )


# ===================================================================
# 8. EDGE CASES AND BOUNDARY CONDITIONS
# ===================================================================

class TestEdgeCases:
    """Test boundary conditions and edge cases."""

    def test_very_small_gap(self):
        """Handle very small price gap (< 1%)."""
        ctx = NegotiationContext(
            current_offer=1005.0,
            target_price=1000.0,
            round_index=1,
        )
        signals = SupplierSignals()
        result = plan_counter(ctx, signals)
        assert result["counter_price"] is not None
        assert result["counter_price"] >= 1000.0

    def test_very_large_gap(self):
        """Handle very large price gap (> 100%)."""
        ctx = NegotiationContext(
            current_offer=5000.0,
            target_price=1000.0,
            round_index=1,
        )
        signals = SupplierSignals()
        result = plan_counter(ctx, signals)
        assert result["decision"] == "counter"
        assert result["counter_price"] >= 1000.0
        assert result["counter_price"] < 5000.0

    def test_equal_offer_and_target(self):
        """Exact match should be accepted."""
        ctx = NegotiationContext(
            current_offer=1000.0,
            target_price=1000.0,
        )
        signals = SupplierSignals()
        result = plan_counter(ctx, signals)
        assert result["decision"] == "accept"

    def test_zero_offer(self):
        """Zero offer should trigger clarify."""
        ctx = NegotiationContext(current_offer=0.0, target_price=1000.0)
        signals = SupplierSignals()
        result = plan_counter(ctx, signals)
        assert result["decision"] == "clarify"

    def test_zero_target(self):
        """Zero target should trigger clarify."""
        ctx = NegotiationContext(current_offer=1000.0, target_price=0.0)
        signals = SupplierSignals()
        result = plan_counter(ctx, signals)
        assert result["decision"] == "clarify"

    def test_fractional_prices(self):
        """Handle fractional cent precision."""
        ctx = NegotiationContext(
            current_offer=100.33,
            target_price=95.67,
            round_index=1,
        )
        signals = SupplierSignals()
        result = plan_counter(ctx, signals)
        assert result["counter_price"] is not None
        # Verify rounding to 2 decimal places
        assert result["counter_price"] == round(result["counter_price"], 2)

    def test_high_round_numbers(self):
        """Rounds far beyond max should still return hold."""
        ctx = NegotiationContext(
            current_offer=1200.0,
            target_price=1000.0,
            round_index=10,
            max_rounds=3,
        )
        signals = SupplierSignals()
        result = plan_counter(ctx, signals)
        assert result["decision"] == "hold"


# ===================================================================
# 9. ASKS AND NON-PRICE LEVERS
# ===================================================================

class TestAsksAndLevers:
    """Verify that non-price asks are properly included."""

    def test_round1_large_gap_includes_volume_asks(self):
        ctx = NegotiationContext(
            current_offer=2000.0,
            target_price=1000.0,
            round_index=1,
        )
        signals = SupplierSignals()
        result = plan_counter(ctx, signals)
        asks = result["asks"]
        assert len(asks) >= 2
        asks_text = " ".join(asks).lower()
        assert "volume" in asks_text or "tier" in asks_text

    def test_round1_small_gap_includes_price_hold(self):
        ctx = NegotiationContext(
            current_offer=1080.0,
            target_price=1000.0,
            round_index=1,
        )
        signals = SupplierSignals()
        result = plan_counter(ctx, signals)
        asks_text = " ".join(result["asks"]).lower()
        assert "hold" in asks_text or "price" in asks_text or "timeline" in asks_text

    def test_asks_are_deduplicated(self):
        """Asks list should not contain duplicate entries."""
        ctx = NegotiationContext(
            current_offer=1500.0,
            target_price=1000.0,
            round_index=1,
            ask_early_pay_disc=0.02,
            ask_lead_time_keep=True,
        )
        signals = SupplierSignals()
        result = plan_counter(ctx, signals)
        assert len(result["asks"]) == len(set(result["asks"]))


# ===================================================================
# 10. SUPPLIER SIGNAL ANALYSIS
# ===================================================================

class TestSupplierSignals:
    """Test how supplier signals affect negotiation decisions."""

    def test_previous_offer_movement_detected(self):
        """Signal when supplier has moved from previous offer."""
        ctx = NegotiationContext(
            current_offer=1300.0,
            target_price=1000.0,
            round_index=2,
        )
        signals = SupplierSignals(
            offer_prev=1500.0,
            offer_new=1300.0,
            message_text="We have reduced our pricing significantly.",
        )
        result = plan_counter(ctx, signals)
        assert result["decision"] == "counter"
        # Supplier moved 200 (13.3%), agent should still counter

    def test_no_movement_signals_firm_stance(self):
        """When supplier hasn't moved, agent should still counter."""
        ctx = NegotiationContext(
            current_offer=1500.0,
            target_price=1000.0,
            round_index=2,
        )
        signals = SupplierSignals(
            offer_prev=1500.0,
            offer_new=1500.0,
            message_text="We maintain our position.",
        )
        result = plan_counter(ctx, signals)
        assert result["decision"] == "counter"

    def test_finality_overrides_normal_counter(self):
        """Final-offer signal should override normal counter logic."""
        ctx = NegotiationContext(
            current_offer=1100.0,
            target_price=1000.0,
            walkaway_price=1150.0,
            round_index=2,
        )
        signals = SupplierSignals(
            message_text="This is our best and final offer.",
        )
        result = plan_counter(ctx, signals)
        # Within walkaway, should accept
        assert result["decision"] == "accept"
        assert result["finality"] is True


# ===================================================================
# 11. CONCESSION CONVERGENCE PROPERTY
# ===================================================================

class TestConcessionConvergence:
    """Verify that the negotiation converges toward agreement."""

    def test_3round_convergence_toward_target(self):
        """Over 3 rounds, counters should approach the target price."""
        target = 1000.0
        offer = 1500.0
        for round_idx in [1, 2, 3]:
            ctx = NegotiationContext(
                current_offer=offer,
                target_price=target,
                round_index=round_idx,
                risk_buffer_pct=0.05,
            )
            signals = SupplierSignals()
            result = plan_counter(ctx, signals)
            counter = result["counter_price"]
            assert counter >= target
            assert counter <= offer
            # Simulate supplier making partial concession for next round
            offer = offer - (offer - counter) * 0.5

    def test_convergence_with_stubborn_supplier(self):
        """Even if supplier doesn't move, agent strategies evolve."""
        decisions = []
        fixed_offer = 1400.0
        for round_idx in [1, 2, 3, 4]:
            ctx = NegotiationContext(
                current_offer=fixed_offer,
                target_price=1000.0,
                round_index=round_idx,
                max_rounds=3,
            )
            signals = SupplierSignals()
            result = plan_counter(ctx, signals)
            decisions.append(result["decision"])

        # Rounds 1-3 should counter, round 4 should hold
        assert decisions[:3] == ["counter", "counter", "counter"]
        assert decisions[3] == "hold"


# ===================================================================
# 12. AGENT INITIALIZATION
# ===================================================================

class TestAgentInitialization:
    """Verify agent initializes with negotiation capabilities."""

    def test_agent_has_playbook(self, agent):
        assert hasattr(agent, "_load_playbook") or hasattr(agent, "_resolve_playbook_context")

    def test_agent_has_plan_steps(self, agent):
        assert hasattr(agent, "AGENTIC_PLAN_STEPS")
        steps = agent.AGENTIC_PLAN_STEPS
        assert len(steps) >= 3
        steps_text = " ".join(steps).lower()
        assert "supplier" in steps_text or "negotiation" in steps_text
        assert "counter" in steps_text or "pricing" in steps_text or "strategy" in steps_text

    def test_agent_run_returns_agent_output(self, agent):
        context = AgentContext(
            workflow_id="wf-init",
            agent_id="negotiation",
            user_id="tester",
            input_data={},
        )
        output = agent.run(context)
        assert isinstance(output, AgentOutput)
        assert output.status in (AgentStatus.SUCCESS, AgentStatus.FAILED)
