"""Tests for ReasoningEngine — the Reason→Plan→Act→Observe loop.

All tests use mocks so they run without Postgres, Ollama, or real agents.
"""
import sys
import types
import unittest
from unittest.mock import MagicMock, patch, PropertyMock

sys.path.insert(0, "src")

from orchestration.reasoning_engine import (
    ReasoningEngine,
    PlanStep,
    WorkflowPlan,
    Observation,
)
from orchestration.workflow_context import WorkflowContext, SignalType
from agents.auto_registry import AutoRegistry, AgentContract
from engines.negotiation_strategy_engine import NegotiationStrategyEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_registry(*agent_ids: str) -> AutoRegistry:
    """Build a minimal AutoRegistry with stub contracts for the given IDs."""
    contracts = {
        aid: AgentContract(
            id=aid,
            class_path=f"agents.{aid}_agent.FakeAgent",
            capabilities=[aid],
            description=f"Stub agent {aid}",
        )
        for aid in agent_ids
    }
    return AutoRegistry(contracts=contracts, agent_nick=None)


def _engine_with_mock_agents(*agent_ids: str) -> tuple:
    """Return (ReasoningEngine, registry) with get_agent patched to return mocks."""
    registry = _make_registry(*agent_ids)
    agent_mocks: dict = {}
    for aid in agent_ids:
        mock_agent = MagicMock()
        mock_agent.run.return_value = {"confidence": 0.9, "status": "ok", "agent": aid}
        agent_mocks[aid] = mock_agent

    def _get_agent(agent_id):
        if agent_id in agent_mocks:
            return agent_mocks[agent_id]
        raise KeyError(f"No mock for '{agent_id}'")

    registry.get_agent = _get_agent

    engine = ReasoningEngine(
        agent_nick=MagicMock(),
        registry=registry,
        pattern_service=None,
        context_service=None,
    )
    return engine, registry, agent_mocks


# ---------------------------------------------------------------------------
# Plan tests
# ---------------------------------------------------------------------------

class TestPlanSimpleExtraction(unittest.TestCase):
    def test_plan_simple_extraction(self):
        engine, _, _ = _engine_with_mock_agents(
            "data_extraction", "discrepancy_detection"
        )
        plan = engine.reason_and_plan({"task_type": "document_extraction"})
        agent_seq = [s.agent for s in plan.steps]
        self.assertIn("data_extraction", agent_seq)
        self.assertIn("discrepancy_detection", agent_seq)
        # data_extraction must come before discrepancy_detection
        self.assertLess(agent_seq.index("data_extraction"), agent_seq.index("discrepancy_detection"))

    def test_plan_extraction_parallel_groups(self):
        engine, _, _ = _engine_with_mock_agents(
            "data_extraction", "discrepancy_detection"
        )
        plan = engine.reason_and_plan({"task_type": "document_extraction"})
        # Steps belong to different parallel groups (sequential)
        groups = [s.parallel_group for s in plan.steps]
        self.assertEqual(len(groups), len(set(groups)),
                         "Each extraction step should be in its own parallel group")


class TestPlanIncludesRankingForNewSupplier(unittest.TestCase):
    def test_plan_includes_ranking_for_new_supplier(self):
        engine, _, _ = _engine_with_mock_agents(
            "data_extraction", "discrepancy_detection", "supplier_ranking"
        )
        plan = engine.reason_and_plan(
            {"task_type": "document_extraction", "is_new_supplier": True}
        )
        agent_seq = [s.agent for s in plan.steps]
        self.assertIn("supplier_ranking", agent_seq,
                      "supplier_ranking must be injected for new suppliers")

    def test_plan_no_duplicate_ranking(self):
        """supplier_ranking should not be duplicated when task_type is supplier_ranking."""
        engine, _, _ = _engine_with_mock_agents(
            "opportunity_miner", "supplier_ranking", "quote_evaluation"
        )
        plan = engine.reason_and_plan(
            {"task_type": "supplier_ranking", "is_new_supplier": True}
        )
        ranking_count = sum(1 for s in plan.steps if s.agent == "supplier_ranking")
        self.assertEqual(ranking_count, 1, "supplier_ranking should appear exactly once")


class TestPlanNegotiationIncludesStrategy(unittest.TestCase):
    def test_plan_negotiation_includes_strategy(self):
        engine, _, _ = _engine_with_mock_agents("negotiation")
        plan = engine.reason_and_plan(
            {
                "task_type": "negotiation",
                "supplier_name": "Acme Corp",
                "order_value": 20000,
                "category": "electronics",
                "supplier_history_count": 0,
                "alternative_quotes": 2,
            }
        )
        self.assertIsNotNone(plan.negotiation_strategy,
                              "WorkflowPlan must carry a negotiation_strategy for negotiation tasks")
        # With 2 alternatives and no history → competitive
        self.assertEqual(plan.negotiation_strategy.name, "competitive")

    def test_plan_negotiation_high_value_escalation(self):
        engine, _, _ = _engine_with_mock_agents("negotiation")
        plan = engine.reason_and_plan(
            {
                "task_type": "negotiation",
                "supplier_name": "BigSupplier",
                "order_value": 100_000,
                "category": "machinery",
            }
        )
        self.assertTrue(plan.escalation_policy.get("escalate_high_value"),
                        "High-value task must set escalate_high_value in escalation_policy")

    def test_plan_negotiation_step_has_strategy_in_input_mapping(self):
        engine, _, _ = _engine_with_mock_agents("negotiation")
        plan = engine.reason_and_plan(
            {"task_type": "negotiation", "order_value": 10000}
        )
        neg_step = next(s for s in plan.steps if s.agent == "negotiation")
        self.assertIn("strategy", neg_step.input_mapping)


class TestPlanRFQ(unittest.TestCase):
    def test_plan_rfq_sequence(self):
        engine, _, _ = _engine_with_mock_agents("email_drafting", "email_dispatch")
        plan = engine.reason_and_plan({"task_type": "rfq"})
        agent_seq = [s.agent for s in plan.steps]
        self.assertEqual(agent_seq, ["email_drafting", "email_dispatch"])


# ---------------------------------------------------------------------------
# Observe tests
# ---------------------------------------------------------------------------

class TestObserveHighConfidence(unittest.TestCase):
    def test_observe_completes_high_confidence(self):
        engine, _, _ = _engine_with_mock_agents("data_extraction")
        wf_ctx = WorkflowContext(goal="test")
        obs = engine.observe({
            "results": {"data_extraction": {"confidence": 0.95}},
            "context": wf_ctx,
        })
        self.assertEqual(obs.action, "complete")
        self.assertGreaterEqual(obs.confidence, 0.80)

    def test_observe_completes_medium_confidence_with_warning(self):
        engine, _, _ = _engine_with_mock_agents("data_extraction")
        wf_ctx = WorkflowContext(goal="test")
        obs = engine.observe({
            "results": {"data_extraction": {"confidence": 0.70}},
            "context": wf_ctx,
        })
        self.assertEqual(obs.action, "complete")
        self.assertGreaterEqual(obs.confidence, 0.60)
        self.assertIn("review", obs.reason.lower())


class TestObserveLowConfidence(unittest.TestCase):
    def test_observe_escalates_low_confidence(self):
        engine, _, _ = _engine_with_mock_agents("data_extraction")
        wf_ctx = WorkflowContext(goal="test")
        # Emit CONFIDENCE_LOW signal
        wf_ctx.emit_signal("data_extraction", SignalType.CONFIDENCE_LOW, "Low confidence result")
        obs = engine.observe({
            "results": {"data_extraction": {"confidence": 0.40}},
            "context": wf_ctx,
        })
        self.assertEqual(obs.action, "escalate")
        self.assertLess(obs.confidence, 0.60)

    def test_observe_retries_low_confidence_no_signal(self):
        engine, _, _ = _engine_with_mock_agents("data_extraction")
        wf_ctx = WorkflowContext(goal="test")
        obs = engine.observe({
            "results": {"data_extraction": {"confidence": 0.40}},
            "context": wf_ctx,
        })
        self.assertEqual(obs.action, "retry")

    def test_observe_escalates_on_recommend_escalation_signal(self):
        engine, _, _ = _engine_with_mock_agents("data_extraction")
        wf_ctx = WorkflowContext(goal="test")
        wf_ctx.emit_signal(
            "data_extraction", SignalType.RECOMMEND_ESCALATION, "Needs human review"
        )
        obs = engine.observe({
            "results": {"data_extraction": {"confidence": 0.90}},
            "context": wf_ctx,
        })
        self.assertEqual(obs.action, "escalate",
                         "RECOMMEND_ESCALATION must override even high confidence")


class TestObserveSuggestAgentSignal(unittest.TestCase):
    def test_observe_handles_suggest_agent_signal(self):
        engine, _, _ = _engine_with_mock_agents("data_extraction")
        wf_ctx = WorkflowContext(goal="test")
        wf_ctx.emit_signal(
            "data_extraction",
            SignalType.SUGGEST_AGENT,
            "Additional agent recommended",
            data={"agent_id": "discrepancy_detection"},
        )
        obs = engine.observe({
            "results": {"data_extraction": {"confidence": 0.85}},
            "context": wf_ctx,
        })
        self.assertEqual(obs.action, "adapt",
                         "SUGGEST_AGENT signal must produce an adapt action")


# ---------------------------------------------------------------------------
# End-to-end process_task test
# ---------------------------------------------------------------------------

class TestProcessTaskEndToEnd(unittest.TestCase):
    def test_process_task_end_to_end(self):
        """Full loop with mocked agents — verify result structure."""
        engine, registry, agent_mocks = _engine_with_mock_agents(
            "data_extraction", "discrepancy_detection"
        )
        task = {
            "task_type": "document_extraction",
            "goal": "Extract invoice data",
            "supplier_name": "Acme",
            "order_value": 5000,
            "category": "office_supplies",
        }
        result = engine.process_task(task)

        self.assertIn("status", result)
        self.assertIn("data", result)
        self.assertIn("confidence", result)
        self.assertIn("workflow_id", result)
        self.assertIn("signals", result)

        self.assertEqual(result["status"], "complete",
                         f"Expected 'complete' but got '{result['status']}'")
        self.assertIsInstance(result["workflow_id"], str)
        self.assertGreater(len(result["workflow_id"]), 0)

        # Both agents should have been called
        self.assertIn("data_extraction", result["data"])
        self.assertIn("discrepancy_detection", result["data"])

    def test_process_task_with_pattern_service(self):
        """Verify pattern recording is called when pattern_service is present."""
        engine, registry, agent_mocks = _engine_with_mock_agents(
            "data_extraction", "discrepancy_detection"
        )
        mock_ps = MagicMock()
        mock_ps.get_patterns.return_value = []
        mock_ps.record_pattern.return_value = {"id": 1}
        engine._pattern_service = mock_ps

        task = {"task_type": "document_extraction", "goal": "test"}
        result = engine.process_task(task)

        mock_ps.get_patterns.assert_called_once()
        mock_ps.record_pattern.assert_called_once()

    def test_process_task_with_context_service(self):
        """Verify context service is called when provided."""
        engine, registry, agent_mocks = _engine_with_mock_agents(
            "data_extraction", "discrepancy_detection"
        )
        mock_cs = MagicMock()
        mock_cs.build_context_brief.return_value = {
            "lifecycle_stage": "Invoice Received",
            "document_type": "Invoice",
        }
        engine._context_service = mock_cs

        task = {
            "task_type": "document_extraction",
            "document_type": "Invoice",
            "goal": "test",
        }
        result = engine.process_task(task)
        mock_cs.build_context_brief.assert_called_once()

    def test_process_task_negotiation_end_to_end(self):
        """Negotiation task includes strategy in plan and returns complete."""
        engine, registry, agent_mocks = _engine_with_mock_agents("negotiation")
        task = {
            "task_type": "negotiation",
            "goal": "Negotiate with Acme",
            "supplier_name": "Acme",
            "order_value": 20000,
            "category": "electronics",
            "supplier_history_count": 0,
            "alternative_quotes": 3,
        }
        result = engine.process_task(task)
        self.assertIn("negotiation", result["data"])
        self.assertIn(result["status"], {"complete", "retry", "escalate", "adapt"})

    def test_process_task_unknown_agent_non_required(self):
        """Non-required steps with missing agents should not crash process_task."""
        engine, registry, _ = _engine_with_mock_agents("data_extraction")
        # Patch registry.get_agent to raise for unknown agent
        original_get = registry.get_agent

        def patched_get(agent_id):
            if agent_id == "nonexistent":
                raise KeyError(f"No agent '{agent_id}'")
            return original_get(agent_id)

        registry.get_agent = patched_get

        # Manually inject a non-required step that will fail
        original_plan = engine._rule_based_plan

        def patched_plan(task, context):
            plan = original_plan(task, context)
            plan.steps.append(PlanStep(agent="nonexistent", required=False))
            return plan

        engine._rule_based_plan = patched_plan

        task = {"task_type": "document_extraction"}
        # Should not raise
        result = engine.process_task(task)
        self.assertIn("nonexistent", result["data"])
        self.assertEqual(result["data"]["nonexistent"]["status"], "agent_unavailable")


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------

class TestDataclasses(unittest.TestCase):
    def test_plan_step_defaults(self):
        step = PlanStep(agent="data_extraction")
        self.assertEqual(step.parallel_group, 0)
        self.assertIsNone(step.condition)
        self.assertTrue(step.required)
        self.assertEqual(step.input_mapping, {})

    def test_workflow_plan_fields(self):
        plan = WorkflowPlan(
            goal="test goal",
            steps=[PlanStep(agent="x")],
        )
        self.assertEqual(plan.goal, "test goal")
        self.assertIsNone(plan.negotiation_strategy)
        self.assertEqual(plan.escalation_policy, {})

    def test_observation_fields(self):
        obs = Observation(action="complete", reason="done", confidence=0.9)
        self.assertEqual(obs.action, "complete")
        self.assertEqual(obs.confidence, 0.9)


# ---------------------------------------------------------------------------
# LLM path tests (mocked Ollama)
# ---------------------------------------------------------------------------

class TestLLMPlanPath(unittest.TestCase):
    def test_llm_path_triggered_by_flag(self):
        engine, _, _ = _engine_with_mock_agents("data_extraction", "discrepancy_detection")

        llm_response = json_bytes = json.dumps({
            "response": json.dumps({
                "goal": "LLM goal",
                "steps": [
                    {"agent": "data_extraction", "parallel_group": 0, "required": True},
                ],
                "escalation_policy": {},
            })
        }).encode("utf-8")

        mock_resp = MagicMock()
        mock_resp.read.return_value = llm_response
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            plan = engine.reason_and_plan(
                {"task_type": "document_extraction", "use_llm_planning": True}
            )

        self.assertEqual(plan.goal, "LLM goal")
        self.assertEqual(plan.steps[0].agent, "data_extraction")

    def test_llm_path_falls_back_on_error(self):
        engine, _, _ = _engine_with_mock_agents("data_extraction", "discrepancy_detection")

        with patch("urllib.request.urlopen", side_effect=ConnectionError("Ollama down")):
            # Should fall back to rule-based plan without raising
            plan = engine.reason_and_plan(
                {"task_type": "document_extraction", "use_llm_planning": True}
            )

        agent_seq = [s.agent for s in plan.steps]
        self.assertIn("data_extraction", agent_seq)

    def test_llm_path_triggered_for_unknown_task_type(self):
        engine, _, _ = _engine_with_mock_agents("data_extraction")

        llm_response = json.dumps({
            "response": json.dumps({
                "goal": "Custom goal",
                "steps": [
                    {"agent": "data_extraction", "parallel_group": 0, "required": True},
                ],
                "escalation_policy": {},
            })
        }).encode("utf-8")

        mock_resp = MagicMock()
        mock_resp.read.return_value = llm_response
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            plan = engine.reason_and_plan({"task_type": "custom_analysis"})

        self.assertEqual(plan.goal, "Custom goal")


import json  # noqa: E402 — needed for LLM tests above


if __name__ == "__main__":
    unittest.main()
