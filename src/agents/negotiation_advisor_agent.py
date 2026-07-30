"""Buyer-facing negotiation advice, as an agent.

Thin by design: classification, ranking, grounding and persistence all live in
src/services/negotiation_advice/. This exists so the advisor appears in the Agent
Workspace alongside the other agents and can be dispatched generically.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from agents.base_agent import AgentContext, AgentOutput, AgentStatus, BaseAgent
from src.services.negotiation_advice import apply_turn, build_advice

logger = logging.getLogger(__name__)


class NegotiationAdvisorAgent(BaseAgent):
    """Ranked, grounded negotiation advice for one deal."""

    AGENTIC_PLAN_STEPS = (
        "Gather the deal's spend, supply-market and variance signals.",
        "Suggest a supplier quadrant and negotiation style, with reasons.",
        "Rank plays and mark each ready or groundwork against the evidence.",
    )

    def run(self, context: AgentContext) -> AgentOutput:
        try:
            data: Dict[str, Any] = context.input_data or {}
            deal_id = str(data.get("deal_id") or "").strip()
            if not deal_id:
                return AgentOutput(
                    status=AgentStatus.SUCCESS, data={},
                    error="deal_id is required for negotiation advice",
                )

            created_by = str(data.get("created_by") or context.user_id or "")
            action = str(data.get("action") or "").strip()
            if action:
                result = apply_turn(deal_id, data, created_by=created_by)
            else:
                result = build_advice(deal_id, created_by=created_by)

            if result is None:
                return AgentOutput(status=AgentStatus.SUCCESS, data={},
                                   error=f"No deal {deal_id}")

            return self._with_plan(context, AgentOutput(
                status=AgentStatus.SUCCESS,
                data=dict(result),
                next_agents=[],
                confidence=float(result.get("quadrant_confidence") or 0.0),
            ))
        except Exception as exc:  # pragma: no cover - top-level guard
            logger.exception("NegotiationAdvisorAgent.run failed")
            return AgentOutput(status=AgentStatus.FAILED, data={}, error=str(exc))
