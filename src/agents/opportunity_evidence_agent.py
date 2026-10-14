"""Assemble the evidence one opportunity rests on. Judge nothing.

A subagent the Opportunity Critic calls as a tool. The split is deliberate: an
agent that both gathers and judges can quietly gather what supports the verdict
it already reached. This one has no verdict vocabulary at all.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from agents.base_agent import AgentContext, AgentOutput, AgentStatus, BaseAgent
from src.services.opportunity_critic.assemble import assemble_candidate

logger = logging.getLogger(__name__)


class OpportunityEvidenceAgent(BaseAgent):
    """Build the critic's input envelope for one finding."""

    AGENTIC_PLAN_STEPS = (
        "Read the finding and the source records it names.",
        "Resolve the anchor, and date it from the invoices behind it where possible.",
        "Resolve contract context by supplier, and report plainly when it cannot be resolved.",
        "Tag every fact with the confidence its provenance earns; upgrade nothing.",
    )

    def run(self, context: AgentContext) -> AgentOutput:
        finding = (context.input_data or {}).get("finding") or {}
        if not finding.get("opportunity_ref_id"):
            return AgentOutput(
                status=AgentStatus.FAILED,
                data={},
                error="no finding supplied: evidence assembly needs opportunity_ref_id",
            )

        conn = None
        try:
            from src.services.db import get_conn

            with get_conn() as conn:
                envelope = assemble_candidate(finding, conn)
        except Exception as exc:  # noqa: BLE001
            logger.error("evidence assembly failed for %s: %s",
                         finding.get("opportunity_ref_id"), exc)
            envelope = assemble_candidate(finding, None)

        return AgentOutput(status=AgentStatus.SUCCESS, data={"candidate": envelope})
