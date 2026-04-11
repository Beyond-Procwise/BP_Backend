"""Comprehensive tests for WorkflowContext shared blackboard.

Run from the src directory:
    cd /home/muthu/PycharmProjects/BP_Backend/src
    python -m pytest ../tests/test_workflow_context.py -v
"""
import sys
import uuid
from collections import OrderedDict
from datetime import datetime, timezone

sys.path.insert(0, "src")

import pytest

from orchestration.workflow_context import AgentSignal, SignalType, WorkflowContext


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def ctx():
    """Return a fresh WorkflowContext with a deterministic workflow_id."""
    return WorkflowContext(
        goal="Find best supplier for 1000 units of steel pipe",
        workflow_id="wf-test-001",
        escalation_policy={"auto_escalate_on": ["RECOMMEND_ESCALATION"], "threshold": 2},
    )


@pytest.fixture()
def minimal_ctx():
    """Return a WorkflowContext created with only the required argument."""
    return WorkflowContext(goal="Minimal goal")


# ── Construction & Defaults ───────────────────────────────────────────────────


class TestConstruction:
    def test_explicit_workflow_id_preserved(self, ctx):
        assert ctx.workflow_id == "wf-test-001"

    def test_auto_generated_workflow_id_is_uuid(self, minimal_ctx):
        # Should not raise and should be a valid UUID string
        parsed = uuid.UUID(minimal_ctx.workflow_id)
        assert str(parsed) == minimal_ctx.workflow_id

    def test_goal_stored(self, ctx):
        assert ctx.goal == "Find best supplier for 1000 units of steel pipe"

    def test_escalation_policy_stored(self, ctx):
        assert ctx.escalation_policy["threshold"] == 2

    def test_escalation_policy_defaults_to_empty_dict(self, minimal_ctx):
        assert minimal_ctx.escalation_policy == {}

    def test_agent_results_is_ordered_dict(self, ctx):
        assert isinstance(ctx.agent_results, OrderedDict)

    def test_agent_results_starts_empty(self, ctx):
        assert len(ctx.agent_results) == 0

    def test_shared_data_starts_empty(self, ctx):
        assert ctx.shared_data == {}

    def test_signals_start_empty(self, ctx):
        assert ctx.get_signals() == []


# ── record_result / get_prior_result ─────────────────────────────────────────


class TestRecordResult:
    def test_record_and_retrieve(self, ctx):
        ctx.record_result("opportunity_agent", {"opportunities": [1, 2, 3]})
        result = ctx.get_prior_result("opportunity_agent")
        assert result == {"opportunities": [1, 2, 3]}

    def test_missing_agent_returns_none(self, ctx):
        assert ctx.get_prior_result("nonexistent_agent") is None

    def test_insertion_order_preserved(self, ctx):
        ctx.record_result("agent_a", {"score": 0.9})
        ctx.record_result("agent_b", {"score": 0.7})
        ctx.record_result("agent_c", {"score": 0.5})
        assert list(ctx.agent_results.keys()) == ["agent_a", "agent_b", "agent_c"]

    def test_overwrite_existing_agent_result(self, ctx):
        ctx.record_result("agent_a", {"v": 1})
        ctx.record_result("agent_a", {"v": 2})
        assert ctx.get_prior_result("agent_a") == {"v": 2}

    def test_all_prior_results_visible_from_agent_results(self, ctx):
        ctx.record_result("parser", {"parsed": True})
        ctx.record_result("ranker", {"rank": 1})
        assert "parser" in ctx.agent_results
        assert "ranker" in ctx.agent_results

    def test_empty_output_dict_accepted(self, ctx):
        ctx.record_result("noop_agent", {})
        assert ctx.get_prior_result("noop_agent") == {}


# ── update_shared ─────────────────────────────────────────────────────────────


class TestUpdateShared:
    def test_set_and_read_key(self, ctx):
        ctx.update_shared("supplier_country", "Germany")
        assert ctx.shared_data["supplier_country"] == "Germany"

    def test_overwrite_key(self, ctx):
        ctx.update_shared("qty", 100)
        ctx.update_shared("qty", 200)
        assert ctx.shared_data["qty"] == 200

    def test_multiple_keys_coexist(self, ctx):
        ctx.update_shared("a", 1)
        ctx.update_shared("b", 2)
        assert ctx.shared_data == {"a": 1, "b": 2}

    def test_nested_value_stored(self, ctx):
        ctx.update_shared("meta", {"source": "email", "confidence": 0.95})
        assert ctx.shared_data["meta"]["confidence"] == 0.95

    def test_list_value_stored(self, ctx):
        ctx.update_shared("suppliers", ["A", "B", "C"])
        assert ctx.shared_data["suppliers"] == ["A", "B", "C"]


# ── set_procurement_brief / get_procurement_brief ─────────────────────────────


class TestProcurementBrief:
    def test_set_and_get(self, ctx):
        brief = {"item": "steel pipe", "qty": 1000, "currency": "USD"}
        ctx.set_procurement_brief(brief)
        assert ctx.get_procurement_brief() == brief

    def test_stored_under_reserved_key(self, ctx):
        brief = {"item": "widget"}
        ctx.set_procurement_brief(brief)
        assert ctx.shared_data["procurement_brief"] == brief

    def test_get_returns_none_when_not_set(self, ctx):
        assert ctx.get_procurement_brief() is None

    def test_overwrite_brief(self, ctx):
        ctx.set_procurement_brief({"v": 1})
        ctx.set_procurement_brief({"v": 2})
        assert ctx.get_procurement_brief() == {"v": 2}


# ── emit_signal ───────────────────────────────────────────────────────────────


class TestEmitSignal:
    def test_returns_agent_signal_instance(self, ctx):
        sig = ctx.emit_signal(
            "ranker_agent",
            SignalType.CONFIDENCE_LOW,
            "Confidence below threshold",
        )
        assert isinstance(sig, AgentSignal)

    def test_signal_agent_stored(self, ctx):
        ctx.emit_signal("agent_x", SignalType.NEEDS_ATTENTION, "Check this")
        signals = ctx.get_signals()
        assert signals[0].agent == "agent_x"

    def test_signal_type_stored(self, ctx):
        ctx.emit_signal("agent_x", SignalType.RECOMMEND_ESCALATION, "Escalate now")
        assert ctx.get_signals()[0].signal_type == SignalType.RECOMMEND_ESCALATION

    def test_signal_message_stored(self, ctx):
        ctx.emit_signal("agent_x", SignalType.SUGGEST_AGENT, "Use supplier agent")
        assert ctx.get_signals()[0].message == "Use supplier agent"

    def test_signal_data_default_empty_dict(self, ctx):
        ctx.emit_signal("agent_x", SignalType.NEEDS_ATTENTION, "msg")
        assert ctx.get_signals()[0].data == {}

    def test_signal_data_stored(self, ctx):
        ctx.emit_signal(
            "agent_x",
            SignalType.SUGGEST_AGENT,
            "Suggest supplier_agent",
            data={"suggested_agent": "supplier_agent"},
        )
        assert ctx.get_signals()[0].data["suggested_agent"] == "supplier_agent"

    def test_signal_timestamp_is_utc_datetime(self, ctx):
        before = datetime.now(timezone.utc)
        ctx.emit_signal("agent_x", SignalType.NEEDS_ATTENTION, "msg")
        after = datetime.now(timezone.utc)
        ts = ctx.get_signals()[0].timestamp
        assert before <= ts <= after

    def test_multiple_signals_ordered(self, ctx):
        ctx.emit_signal("a1", SignalType.NEEDS_ATTENTION, "first")
        ctx.emit_signal("a2", SignalType.CONFIDENCE_LOW, "second")
        ctx.emit_signal("a3", SignalType.RECOMMEND_ESCALATION, "third")
        agents = [s.agent for s in ctx.get_signals()]
        assert agents == ["a1", "a2", "a3"]


# ── get_signals (filtering) ────────────────────────────────────────────────────


class TestGetSignals:
    def _populate(self, ctx):
        ctx.emit_signal("agent_a", SignalType.CONFIDENCE_LOW, "low conf")
        ctx.emit_signal("agent_a", SignalType.NEEDS_ATTENTION, "attention")
        ctx.emit_signal("agent_b", SignalType.CONFIDENCE_LOW, "also low")
        ctx.emit_signal("agent_b", SignalType.RECOMMEND_ESCALATION, "escalate")

    def test_no_filter_returns_all(self, ctx):
        self._populate(ctx)
        assert len(ctx.get_signals()) == 4

    def test_filter_by_signal_type(self, ctx):
        self._populate(ctx)
        low = ctx.get_signals(signal_type=SignalType.CONFIDENCE_LOW)
        assert len(low) == 2
        assert all(s.signal_type == SignalType.CONFIDENCE_LOW for s in low)

    def test_filter_by_agent(self, ctx):
        self._populate(ctx)
        a_sigs = ctx.get_signals(agent="agent_a")
        assert len(a_sigs) == 2
        assert all(s.agent == "agent_a" for s in a_sigs)

    def test_filter_by_both_type_and_agent(self, ctx):
        self._populate(ctx)
        sigs = ctx.get_signals(signal_type=SignalType.CONFIDENCE_LOW, agent="agent_b")
        assert len(sigs) == 1
        assert sigs[0].agent == "agent_b"
        assert sigs[0].signal_type == SignalType.CONFIDENCE_LOW

    def test_filter_returns_empty_when_no_match(self, ctx):
        self._populate(ctx)
        sigs = ctx.get_signals(signal_type=SignalType.SUGGEST_AGENT)
        assert sigs == []

    def test_empty_context_returns_empty_list(self, ctx):
        assert ctx.get_signals(signal_type=SignalType.NEEDS_ATTENTION) == []


# ── has_signal ────────────────────────────────────────────────────────────────


class TestHasSignal:
    def test_true_when_signal_exists(self, ctx):
        ctx.emit_signal("a", SignalType.NEEDS_ATTENTION, "msg")
        assert ctx.has_signal(signal_type=SignalType.NEEDS_ATTENTION) is True

    def test_false_when_signal_absent(self, ctx):
        ctx.emit_signal("a", SignalType.CONFIDENCE_LOW, "msg")
        assert ctx.has_signal(signal_type=SignalType.NEEDS_ATTENTION) is False

    def test_true_with_agent_filter(self, ctx):
        ctx.emit_signal("agent_x", SignalType.RECOMMEND_ESCALATION, "msg")
        assert ctx.has_signal(agent="agent_x") is True

    def test_false_with_wrong_agent_filter(self, ctx):
        ctx.emit_signal("agent_x", SignalType.RECOMMEND_ESCALATION, "msg")
        assert ctx.has_signal(agent="agent_y") is False

    def test_no_filters_returns_true_when_any_signal(self, ctx):
        ctx.emit_signal("any", SignalType.NEEDS_ATTENTION, "msg")
        assert ctx.has_signal() is True

    def test_no_filters_returns_false_when_empty(self, ctx):
        assert ctx.has_signal() is False


# ── to_dict ───────────────────────────────────────────────────────────────────


class TestToDict:
    def test_returns_dict(self, ctx):
        assert isinstance(ctx.to_dict(), dict)

    def test_workflow_id_present(self, ctx):
        assert ctx.to_dict()["workflow_id"] == "wf-test-001"

    def test_goal_present(self, ctx):
        assert ctx.to_dict()["goal"] == "Find best supplier for 1000 units of steel pipe"

    def test_escalation_policy_present(self, ctx):
        d = ctx.to_dict()
        assert d["escalation_policy"]["threshold"] == 2

    def test_agent_results_present_and_plain_dict(self, ctx):
        ctx.record_result("agent_a", {"score": 1})
        d = ctx.to_dict()
        assert isinstance(d["agent_results"], dict)
        assert d["agent_results"]["agent_a"] == {"score": 1}

    def test_shared_data_present(self, ctx):
        ctx.update_shared("key", "val")
        assert ctx.to_dict()["shared_data"]["key"] == "val"

    def test_signals_serialised(self, ctx):
        ctx.emit_signal("a", SignalType.CONFIDENCE_LOW, "low", data={"score": 0.2})
        sigs = ctx.to_dict()["signals"]
        assert len(sigs) == 1
        assert sigs[0]["signal_type"] == "CONFIDENCE_LOW"
        assert sigs[0]["agent"] == "a"
        assert sigs[0]["message"] == "low"
        assert sigs[0]["data"]["score"] == 0.2

    def test_signal_timestamp_is_iso_string(self, ctx):
        ctx.emit_signal("a", SignalType.NEEDS_ATTENTION, "msg")
        ts_str = ctx.to_dict()["signals"][0]["timestamp"]
        # Should be parseable as ISO 8601
        parsed = datetime.fromisoformat(ts_str)
        assert isinstance(parsed, datetime)

    def test_empty_signals_list_when_none_emitted(self, ctx):
        assert ctx.to_dict()["signals"] == []

    def test_agent_results_order_preserved_in_dict(self, ctx):
        for i in range(5):
            ctx.record_result(f"agent_{i}", {"i": i})
        keys = list(ctx.to_dict()["agent_results"].keys())
        assert keys == [f"agent_{i}" for i in range(5)]

    def test_full_round_trip_completeness(self, ctx):
        ctx.record_result("extractor", {"fields": ["qty", "price"]})
        ctx.update_shared("currency", "EUR")
        ctx.set_procurement_brief({"item": "bolts"})
        ctx.emit_signal("extractor", SignalType.SUGGEST_AGENT, "Try ranker", data={"agent": "ranker"})
        d = ctx.to_dict()
        assert d["agent_results"]["extractor"]["fields"] == ["qty", "price"]
        assert d["shared_data"]["currency"] == "EUR"
        assert d["shared_data"]["procurement_brief"]["item"] == "bolts"
        assert d["signals"][0]["data"]["agent"] == "ranker"


# ── AgentSignal dataclass ─────────────────────────────────────────────────────


class TestAgentSignalDataclass:
    def test_to_dict_keys(self):
        sig = AgentSignal(
            agent="test_agent",
            signal_type=SignalType.NEEDS_ATTENTION,
            message="Watch out",
            data={"ref": "PO-123"},
        )
        d = sig.to_dict()
        assert set(d.keys()) == {"agent", "signal_type", "message", "data", "timestamp"}

    def test_signal_type_serialised_as_string(self):
        sig = AgentSignal(
            agent="a",
            signal_type=SignalType.RECOMMEND_ESCALATION,
            message="Escalate",
        )
        assert sig.to_dict()["signal_type"] == "RECOMMEND_ESCALATION"

    def test_default_data_is_empty_dict(self):
        sig = AgentSignal(agent="a", signal_type=SignalType.CONFIDENCE_LOW, message="m")
        assert sig.data == {}

    def test_custom_timestamp_preserved(self):
        ts = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        sig = AgentSignal(
            agent="a",
            signal_type=SignalType.NEEDS_ATTENTION,
            message="m",
            timestamp=ts,
        )
        assert sig.timestamp == ts
        assert "2025-01-15" in sig.to_dict()["timestamp"]


# ── SignalType enum ───────────────────────────────────────────────────────────


class TestSignalTypeEnum:
    def test_all_four_types_present(self):
        names = {m.name for m in SignalType}
        assert names == {
            "NEEDS_ATTENTION",
            "CONFIDENCE_LOW",
            "RECOMMEND_ESCALATION",
            "SUGGEST_AGENT",
        }

    def test_string_value_matches_name(self):
        for member in SignalType:
            assert member.value == member.name

    def test_comparison_with_string(self):
        assert SignalType.NEEDS_ATTENTION == "NEEDS_ATTENTION"


# ── Integration scenario ──────────────────────────────────────────────────────


class TestIntegrationScenario:
    """End-to-end scenario mimicking a 3-agent procurement pipeline."""

    def test_full_pipeline_scenario(self):
        ctx = WorkflowContext(
            goal="Source 500 laptops for Q3 deployment",
            workflow_id="wf-laptops-q3",
            escalation_policy={"auto_escalate_on": ["RECOMMEND_ESCALATION"]},
        )

        # Opportunity Miner Agent runs first
        ctx.record_result(
            "opportunity_miner",
            {
                "opportunities": ["Vendor A", "Vendor B"],
                "confidence": 0.88,
            },
        )
        ctx.update_shared("category", "IT Hardware")
        ctx.update_shared("budget_usd", 250_000)

        # Supplier Ranking Agent reads prior output and emits a signal
        prior = ctx.get_prior_result("opportunity_miner")
        assert prior is not None
        assert len(prior["opportunities"]) == 2

        ctx.record_result(
            "supplier_ranker",
            {
                "ranked": ["Vendor B", "Vendor A"],
                "top_score": 0.72,
            },
        )
        ctx.emit_signal(
            "supplier_ranker",
            SignalType.CONFIDENCE_LOW,
            "Top score 0.72 is below acceptable threshold",
            data={"top_score": 0.72, "threshold": 0.80},
        )

        # Negotiation Agent checks signal and escalates
        assert ctx.has_signal(signal_type=SignalType.CONFIDENCE_LOW)
        ctx.emit_signal(
            "negotiation_agent",
            SignalType.RECOMMEND_ESCALATION,
            "Confidence too low; human review required",
        )
        ctx.record_result(
            "negotiation_agent",
            {"action": "escalated", "reason": "confidence_low"},
        )

        # Verify final context serialisation
        d = ctx.to_dict()
        assert d["workflow_id"] == "wf-laptops-q3"
        assert list(d["agent_results"].keys()) == [
            "opportunity_miner",
            "supplier_ranker",
            "negotiation_agent",
        ]
        assert d["shared_data"]["category"] == "IT Hardware"
        assert len(d["signals"]) == 2
        assert d["signals"][1]["signal_type"] == "RECOMMEND_ESCALATION"

        # Escalation policy check
        escalation_triggers = ctx.escalation_policy.get("auto_escalate_on", [])
        assert "RECOMMEND_ESCALATION" in escalation_triggers
        assert ctx.has_signal(signal_type=SignalType.RECOMMEND_ESCALATION)
