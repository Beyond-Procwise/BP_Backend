"""ReasoningEngine — the Reason→Plan→Act→Observe loop for AgentNick.

Replaces the monolithic orchestrator with a structured cognitive loop:
  1. Reason: build context from task + procurement intelligence
  2. Plan: compose an ordered/parallel workflow of agent steps
  3. Act: execute the plan, passing WorkflowContext through each agent
  4. Observe: evaluate results and decide complete/retry/escalate/adapt

Spec reference: Task 6 of the agentic re-engineering plan.
"""
from __future__ import annotations

import json
import logging
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from orchestration.workflow_context import WorkflowContext, SignalType
from agents.auto_registry import AutoRegistry
from engines.negotiation_strategy_engine import (
    NegotiationStrategyEngine,
    NegotiationContext,
    Strategy,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PlanStep:
    """A single step in a workflow plan."""

    agent: str
    input_mapping: Dict[str, Any] = field(default_factory=dict)
    parallel_group: int = 0
    condition: Optional[str] = None
    required: bool = True


@dataclass
class WorkflowPlan:
    """A composed workflow plan for a given goal."""

    goal: str
    steps: List[PlanStep]
    negotiation_strategy: Optional[Strategy] = None
    escalation_policy: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Observation:
    """The conclusion reached after observing workflow results."""

    action: str   # complete | retry | escalate | adapt
    reason: str
    confidence: float


# ---------------------------------------------------------------------------
# ReasoningEngine
# ---------------------------------------------------------------------------

_OLLAMA_URL = "http://localhost:11434/api/generate"
_OLLAMA_MODEL = "BeyondProcwise/AgentNick:latest"

# High-value task threshold (mirrors NegotiationStrategyEngine.escalation_threshold)
_HIGH_VALUE_THRESHOLD = 50_000.0


class ReasoningEngine:
    """Core cognitive loop for the AgentNick agentic system.

    Parameters
    ----------
    agent_nick:
        The central dependency/service-locator object forwarded to agents.
    registry:
        :class:`~agents.auto_registry.AutoRegistry` that knows all available agents.
    pattern_service:
        Optional :class:`~services.pattern_service.PatternService` for pattern
        retrieval and recording.  When ``None``, pattern intelligence is skipped.
    context_service:
        Optional :class:`~services.procurement_context_service.ProcurementContextService`
        for building procurement context briefs.  When ``None``, context building
        is skipped.
    """

    def __init__(
        self,
        agent_nick: Any,
        registry: AutoRegistry,
        pattern_service: Any = None,
        context_service: Any = None,
    ) -> None:
        self._agent_nick = agent_nick
        self._registry = registry
        self._pattern_service = pattern_service
        self._context_service = context_service
        self._negotiation_engine = NegotiationStrategyEngine()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_task(self, task: dict) -> dict:
        """Execute the full Reason→Plan→Act→Observe loop.

        Parameters
        ----------
        task:
            Task dictionary.  Expected keys (all optional):
            ``task_type``, ``goal``, ``supplier_name``, ``order_value``,
            ``category``, ``supplier_history_count``, ``document_type``,
            ``is_new_supplier``, ``use_llm_planning``.

        Returns
        -------
        dict with keys:
            ``status`` (str), ``data`` (dict), ``confidence`` (float),
            ``workflow_id`` (str), ``signals`` (list).
        """
        goal = task.get("goal") or f"Process {task.get('task_type', 'unknown')} task"

        # --- Step 1: Build procurement context ---
        context: Dict[str, Any] = {}
        if self._context_service is not None:
            try:
                doc_type = task.get("document_type", "Invoice")
                header = {k: task.get(k) for k in (
                    "supplier_id", "supplier_name", "total_amount", "currency", "category",
                    "po_id", "invoice_id", "quote_id",
                )}
                context = self._context_service.build_context_brief(
                    doc_type=doc_type,
                    header={k: v for k, v in header.items() if v is not None},
                )
            except Exception:
                logger.exception("Failed to build procurement context; continuing without it")

        # --- Step 2: Retrieve relevant patterns ---
        patterns: List[Dict[str, Any]] = []
        if self._pattern_service is not None:
            try:
                category = task.get("category", "")
                patterns = self._pattern_service.get_patterns(category=category or None)
            except Exception:
                logger.exception("Failed to retrieve patterns; continuing without them")

        # --- Step 3: Compose workflow plan ---
        plan = self.reason_and_plan(task, context={"context": context, "patterns": patterns})

        # --- Step 4: Create WorkflowContext ---
        wf_ctx = WorkflowContext(
            goal=plan.goal,
            escalation_policy=plan.escalation_policy,
        )
        if context:
            wf_ctx.set_procurement_brief(context)
        wf_ctx.update_shared("task", task)
        wf_ctx.update_shared("patterns", patterns)

        # --- Step 5: Execute plan steps ---
        results: Dict[str, Any] = {}
        adapted_steps: List[PlanStep] = []
        for step in plan.steps:
            step_result = self._execute_step(step, wf_ctx)
            results[step.agent] = step_result

            # Check signals after each step
            suggest_signals = wf_ctx.get_signals(signal_type=SignalType.SUGGEST_AGENT)
            for sig in suggest_signals:
                suggested = sig.data.get("agent_id")
                if suggested and suggested not in [s.agent for s in adapted_steps]:
                    logger.info(
                        "SUGGEST_AGENT signal: adding agent '%s' to plan", suggested
                    )
                    adapted_steps.append(
                        PlanStep(agent=suggested, required=False)
                    )

        # Execute any dynamically added (adapted) steps
        for step in adapted_steps:
            step_result = self._execute_step(step, wf_ctx)
            results[step.agent] = step_result

        # --- Step 6: Observe and decide ---
        observation = self.observe({"results": results, "context": wf_ctx})

        # --- Step 7: Learn from outcome ---
        if self._pattern_service is not None:
            try:
                task_type = task.get("task_type", "unknown")
                strategy_name = plan.negotiation_strategy.name if plan.negotiation_strategy else "none"
                self._pattern_service.record_pattern(
                    pattern_type="workflow_outcome",
                    pattern_text=f"{task_type}:{observation.action}:confidence={observation.confidence:.2f}",
                    category=task.get("category", ""),
                    confidence=observation.confidence,
                )
            except Exception:
                logger.exception("Failed to record outcome pattern; continuing")

        # --- Step 8: Build result dict ---
        status = "complete" if observation.action in ("complete",) else observation.action
        return {
            "status": status,
            "data": results,
            "confidence": observation.confidence,
            "workflow_id": wf_ctx.workflow_id,
            "signals": [s.to_dict() for s in wf_ctx.get_signals()],
            "observation": {
                "action": observation.action,
                "reason": observation.reason,
                "confidence": observation.confidence,
            },
        }

    def reason_and_plan(
        self, task: dict, context: Optional[dict] = None
    ) -> WorkflowPlan:
        """Compose a :class:`WorkflowPlan` for the given task.

        Uses LLM-based planning when ``task_type`` is unknown or
        ``use_llm_planning`` is ``True``; falls back to rule-based planning
        otherwise.

        Parameters
        ----------
        task:
            Task dictionary.
        context:
            Optional enriched context dict (patterns, procurement brief, etc.).

        Returns
        -------
        :class:`WorkflowPlan`
        """
        task_type = task.get("task_type", "")
        use_llm = task.get("use_llm_planning", False)

        known_types = {
            "document_extraction", "supplier_ranking", "negotiation", "rfq",
        }

        if use_llm or (task_type and task_type not in known_types):
            try:
                return self._llm_compose_plan(task, context or {})
            except Exception:
                logger.exception(
                    "LLM planning failed for task_type='%s'; falling back to rule-based", task_type
                )

        return self._rule_based_plan(task, context or {})

    def observe(self, results: dict) -> Observation:
        """Evaluate workflow results and return an :class:`Observation`.

        Parameters
        ----------
        results:
            Dict with keys ``"results"`` (dict of agent→output) and
            ``"context"`` (:class:`WorkflowContext`).

        Returns
        -------
        :class:`Observation` with ``action`` ∈ {complete, retry, escalate, adapt}.
        """
        wf_ctx: Optional[WorkflowContext] = results.get("context")
        agent_results: Dict[str, Any] = results.get("results", {})

        # Check for escalation signals
        if wf_ctx is not None and wf_ctx.has_signal(signal_type=SignalType.RECOMMEND_ESCALATION):
            return Observation(
                action="escalate",
                reason="RECOMMEND_ESCALATION signal emitted by one or more agents",
                confidence=0.5,
            )

        # Check for SUGGEST_AGENT signals — if any still pending, action = adapt
        if wf_ctx is not None and wf_ctx.has_signal(signal_type=SignalType.SUGGEST_AGENT):
            return Observation(
                action="adapt",
                reason="SUGGEST_AGENT signal received; plan was extended with suggested agent",
                confidence=0.65,
            )

        # Compute aggregate confidence from agent results
        confidence = self._aggregate_confidence(agent_results)

        if confidence >= 0.80:
            return Observation(
                action="complete",
                reason=f"All agents completed with high aggregate confidence ({confidence:.2f})",
                confidence=confidence,
            )

        if confidence >= 0.60:
            return Observation(
                action="complete",
                reason=(
                    f"Agents completed with acceptable confidence ({confidence:.2f}); "
                    "review recommended"
                ),
                confidence=confidence,
            )

        # confidence < 0.60
        # Escalate if any agent flagged low confidence, otherwise retry
        if wf_ctx is not None and wf_ctx.has_signal(signal_type=SignalType.CONFIDENCE_LOW):
            return Observation(
                action="escalate",
                reason=f"Low confidence ({confidence:.2f}) with CONFIDENCE_LOW signal; escalating",
                confidence=confidence,
            )

        return Observation(
            action="retry",
            reason=f"Low aggregate confidence ({confidence:.2f}); retrying workflow",
            confidence=confidence,
        )

    # ------------------------------------------------------------------
    # Private: Rule-based planning
    # ------------------------------------------------------------------

    def _rule_based_plan(self, task: dict, context: dict) -> WorkflowPlan:
        """Compose a :class:`WorkflowPlan` using deterministic rules."""
        task_type = task.get("task_type", "")
        goal = task.get("goal") or f"Execute {task_type} workflow"
        order_value = float(task.get("order_value", 0) or 0)
        is_new_supplier = bool(task.get("is_new_supplier", False))
        escalation_policy: Dict[str, Any] = {}

        # High-value tasks require escalation flag
        if order_value > _HIGH_VALUE_THRESHOLD:
            escalation_policy["escalate_high_value"] = True
            escalation_policy["threshold"] = _HIGH_VALUE_THRESHOLD

        steps: List[PlanStep] = []
        negotiation_strategy: Optional[Strategy] = None

        if task_type == "document_extraction":
            steps = [
                PlanStep(agent="data_extraction", parallel_group=0),
                PlanStep(agent="discrepancy_detection", parallel_group=1),
            ]

        elif task_type == "supplier_ranking":
            steps = [
                PlanStep(agent="opportunity_miner", parallel_group=0),
                PlanStep(agent="supplier_ranking", parallel_group=1),
                PlanStep(agent="quote_evaluation", parallel_group=2),
            ]

        elif task_type == "negotiation":
            # Select negotiation strategy
            neg_ctx = NegotiationContext(
                supplier_name=str(task.get("supplier_name", "")),
                order_value=order_value,
                category=str(task.get("category", "")),
                supplier_history_count=int(task.get("supplier_history_count", 0) or 0),
                alternative_quotes=int(task.get("alternative_quotes", 0) or 0),
                urgency=str(task.get("urgency", "normal")),
            )
            negotiation_strategy = self._negotiation_engine.select_strategy(neg_ctx)
            steps = [
                PlanStep(
                    agent="negotiation",
                    parallel_group=0,
                    input_mapping={"strategy": negotiation_strategy.name},
                ),
            ]

        elif task_type == "rfq":
            steps = [
                PlanStep(agent="email_drafting", parallel_group=0),
                PlanStep(agent="email_dispatch", parallel_group=1),
            ]

        else:
            # Unknown task type — single best-effort pass with data_extraction
            logger.warning(
                "rule_based_plan: unknown task_type='%s'; using default extraction plan",
                task_type,
            )
            steps = [PlanStep(agent="data_extraction", parallel_group=0)]

        # For new suppliers, inject supplier_ranking if not already present
        if is_new_supplier:
            existing_agents = {s.agent for s in steps}
            if "supplier_ranking" not in existing_agents:
                # Insert before the last step (or at the start if only one step)
                insert_at = max(0, len(steps) - 1)
                max_group = max((s.parallel_group for s in steps), default=0)
                steps.insert(
                    insert_at,
                    PlanStep(agent="supplier_ranking", parallel_group=max_group),
                )

        return WorkflowPlan(
            goal=goal,
            steps=steps,
            negotiation_strategy=negotiation_strategy,
            escalation_policy=escalation_policy,
        )

    # ------------------------------------------------------------------
    # Private: LLM-based planning
    # ------------------------------------------------------------------

    def _llm_compose_plan(self, task: dict, context: dict) -> WorkflowPlan:
        """Ask the local Ollama LLM to compose a workflow plan as JSON.

        Sends the available agent catalogue, patterns, and task to
        ``BeyondProcwise/AgentNick:latest`` running at
        ``http://localhost:11434/api/generate`` and parses the JSON response
        into a :class:`WorkflowPlan`.

        Falls back to rule-based planning on any network / parse error.
        """
        agents_desc = self._registry.describe_for_llm()
        patterns = context.get("patterns", [])
        pattern_texts = [p.get("pattern_text", "") for p in patterns if p.get("pattern_text")]

        prompt = (
            "You are a procurement workflow planner. "
            "Given the available agents and the task below, compose a workflow plan as JSON.\n\n"
            f"{agents_desc}\n\n"
            "## Historical Patterns\n"
            + ("\n".join(f"- {t}" for t in pattern_texts) if pattern_texts else "(none)")
            + "\n\n"
            "## Task\n"
            + json.dumps(task, indent=2, default=str)
            + "\n\n"
            "Return ONLY a JSON object in this exact format:\n"
            "{\n"
            '  "goal": "<string>",\n'
            '  "steps": [\n'
            '    {"agent": "<agent_id>", "parallel_group": <int>, "required": <bool>}\n'
            "  ],\n"
            '  "escalation_policy": {}\n'
            "}\n"
            "Use only agent IDs from the catalogue above."
        )

        payload = json.dumps(
            {"model": _OLLAMA_MODEL, "prompt": prompt, "stream": False}
        ).encode("utf-8")

        req = urllib.request.Request(
            _OLLAMA_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = json.loads(resp.read().decode("utf-8"))

        response_text = raw.get("response", "")
        # Extract JSON block (the model may wrap it in markdown code fences)
        json_start = response_text.find("{")
        json_end = response_text.rfind("}") + 1
        if json_start == -1 or json_end == 0:
            raise ValueError("LLM response contains no JSON object")

        plan_dict = json.loads(response_text[json_start:json_end])

        steps = [
            PlanStep(
                agent=s["agent"],
                parallel_group=int(s.get("parallel_group", 0)),
                required=bool(s.get("required", True)),
                input_mapping=s.get("input_mapping", {}),
                condition=s.get("condition"),
            )
            for s in plan_dict.get("steps", [])
        ]

        return WorkflowPlan(
            goal=plan_dict.get("goal", task.get("goal", "LLM-planned workflow")),
            steps=steps,
            escalation_policy=plan_dict.get("escalation_policy", {}),
        )

    # ------------------------------------------------------------------
    # Private: Step execution
    # ------------------------------------------------------------------

    def _execute_step(self, step: PlanStep, wf_ctx: WorkflowContext) -> Dict[str, Any]:
        """Execute a single plan step and record the result into *wf_ctx*.

        Parameters
        ----------
        step:
            :class:`PlanStep` to execute.
        wf_ctx:
            The shared :class:`WorkflowContext` for this workflow run.

        Returns
        -------
        The output dict produced by the agent (or an error dict).
        """
        agent_id = step.agent

        # Build the input payload by merging task data with any step-level mapping
        task_input: Dict[str, Any] = dict(wf_ctx.shared_data.get("task") or {})
        task_input.update(step.input_mapping)
        # Enrich with prior results for downstream agents
        task_input["prior_results"] = dict(wf_ctx.agent_results)

        try:
            agent = self._registry.get_agent(agent_id)
        except (KeyError, ValueError, ImportError, RuntimeError) as exc:
            logger.warning("Cannot load agent '%s': %s", agent_id, exc)
            if step.required:
                result = {"error": str(exc), "agent": agent_id, "status": "agent_unavailable"}
            else:
                result = {"skipped": True, "agent": agent_id, "status": "agent_unavailable"}
            wf_ctx.record_result(agent_id, result)
            return result

        try:
            # Prefer run(task, context=wf_ctx); fall back to run(task)
            if hasattr(agent, "run"):
                try:
                    result = agent.run(task_input, context=wf_ctx)
                except TypeError:
                    result = agent.run(task_input)
            else:
                result = {"error": f"Agent '{agent_id}' has no run() method"}

            if not isinstance(result, dict):
                result = {"output": result}

        except Exception as exc:
            logger.exception("Agent '%s' raised an exception", agent_id)
            result = {"error": str(exc), "agent": agent_id, "status": "agent_error"}

        wf_ctx.record_result(agent_id, result)
        return result

    # ------------------------------------------------------------------
    # Private: Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate_confidence(agent_results: Dict[str, Any]) -> float:
        """Compute average confidence across agent results.

        Looks for a ``confidence`` key in each result dict.  If no agent
        reports confidence, defaults to ``0.75`` (optimistic baseline).
        """
        confidences: List[float] = []
        for result in agent_results.values():
            if isinstance(result, dict):
                conf = result.get("confidence")
                if conf is not None:
                    try:
                        confidences.append(float(conf))
                    except (TypeError, ValueError):
                        pass

        if not confidences:
            return 0.75  # optimistic default when no agent reports confidence

        return sum(confidences) / len(confidences)
