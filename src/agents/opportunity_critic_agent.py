"""Decide whether a detected opportunity would survive a negotiator's scrutiny.

This agent holds no thresholds, no arithmetic and no prose of its own. Its
prompt is a bp_prompt row, its rules are bp_policy rows, and every number in
its verdict comes from a registered formula. What it contributes is judgement:
composing test results into a verdict a negotiator could act on, or declining
to.

Three things it will not do:

  * Run without its governed prompt. A code default that silently wins is how
    governance stops being governance.
  * Persist a critique that breaks an invariant. Code refuses; it does not
    repair. A critique quietly corrected is one nobody knows was wrong.
  * Guess at unparseable model output. "I could not read the answer" and "the
    finding is fine" are different sentences.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Optional

from agents.base_agent import AgentContext, AgentOutput, AgentStatus, BaseAgent
from src.services.opportunity_critic import governed
from src.services.opportunity_critic.assemble import assemble_candidate
from src.services.opportunity_critic.invariants import check_invariants
from src.services.opportunity_critic.shadow import may_suppress
from src.services.opportunity_critic.store import record_critique

logger = logging.getLogger(__name__)

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


class OpportunityCriticAgent(BaseAgent):
    """Critique one detected opportunity."""

    AGENTIC_PLAN_STEPS = (
        "Assemble the candidate's evidence through the evidence subagent.",
        "Resolve the governed thresholds and the governed system prompt.",
        "Run the seven tests, taking every number from the formula registry.",
        "Compose a verdict, and a gap register naming what would decide it.",
        "Refuse to persist anything that breaks the critic's invariants.",
    )

    def __init__(self, agent_nick):
        super().__init__(agent_nick)
        self.policy_engine = getattr(agent_nick, "policy_engine", None)
        self.prompt_engine = getattr(agent_nick, "prompt_engine", None)

    def _parse(self, answer: Any) -> Optional[Dict[str, Any]]:
        """Pull the JSON critique out of the model's answer, or None."""
        if isinstance(answer, dict):
            return answer
        text = str(answer or "")
        match = _JSON_BLOCK.search(text)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
        except (ValueError, TypeError):
            return None
        return parsed if isinstance(parsed, dict) else None

    def run(self, context: AgentContext) -> AgentOutput:
        finding = (context.input_data or {}).get("finding") or {}
        ref_id = finding.get("opportunity_ref_id")
        if not ref_id:
            return AgentOutput(status=AgentStatus.FAILED, data={},
                               error="no finding supplied: need opportunity_ref_id")

        prompt_text, prompt_version = governed.load_system_prompt(self.prompt_engine)
        if not prompt_text:
            return AgentOutput(
                status=AgentStatus.FAILED, data={},
                error=("no governed prompt: bp_prompt row 'opportunity_critic_system' "
                       "could not be resolved, and this agent has no code default"),
            )

        thresholds = governed.load_thresholds(self.policy_engine)

        conn = None
        try:
            from src.services.db import get_conn

            with get_conn() as conn:
                candidate = assemble_candidate(finding, conn)
        except Exception as exc:  # noqa: BLE001
            logger.error("evidence assembly failed for %s: %s", ref_id, exc)
            candidate = assemble_candidate(finding, None)

        task = json.dumps({
            "candidate": candidate,
            "thresholds": {
                "index_band_pp": thresholds.index_band_pp,
                "materiality_floor_gbp": thresholds.materiality_floor_gbp,
                "relative_gap_floor": thresholds.relative_gap_floor,
                "anchor_stale_days": thresholds.anchor_stale_days,
                "friction_bands": thresholds.friction_bands,
            },
        }, default=str)

        # reason() returns a ToolRunResult dataclass (services/tool_runtime.py:96),
        # NOT a dict. Read .answer first; the dict branch exists only for tests and
        # for any caller that hands back a plain mapping.
        result = self.reason(task, extra_system=prompt_text, require_tool_use=False)
        answer = getattr(result, "answer", None)
        if answer is None and isinstance(result, dict):
            answer = result.get("answer")
        reason_error = getattr(result, "error", None)
        if answer is None:
            answer = result
        critique = self._parse(answer)
        if critique is None:
            detail = f" (reason error: {reason_error})" if reason_error else ""
            return AgentOutput(
                status=AgentStatus.FAILED, data={},
                error=("critique could not be parsed as JSON; refusing to guess a "
                       f"verdict{detail}"),
            )

        critique.setdefault("opportunity_ref_id", ref_id)
        critique.setdefault("detector_type", finding.get("detector_type"))
        critique.setdefault("original_claim", finding.get("claim"))
        critique["prompt_version"] = prompt_version
        critique["policy_versions"] = (
            {governed.POLICY_SLUG: (thresholds.source or {}).get("version")}
            if thresholds.source else {}
        )
        critique["formula_versions"] = self._formula_versions()
        critique["run_id"] = context.workflow_id

        violations = check_invariants(critique)
        if violations:
            logger.error("critique for %s broke invariants: %s", ref_id, violations)
            return AgentOutput(status=AgentStatus.FAILED, data={"violations": violations},
                               error="; ".join(violations))

        allowed, shadow_reason = may_suppress(finding, thresholds)
        critique_id = record_critique(critique, shadowed=not allowed)
        if critique_id is None:
            return AgentOutput(
                status=AgentStatus.FAILED, data={},
                error="critique could not be recorded; nothing is suppressed on a failed write",
            )

        return AgentOutput(status=AgentStatus.SUCCESS,
                           data={"critique_id": critique_id,
                                 "verdict": critique.get("verdict"),
                                 "suppression": shadow_reason,
                                 "critique": critique})

    @staticmethod
    def _formula_versions() -> Dict[str, str]:
        """Qualified versions of every critic formula, so a verdict can be dated."""
        try:
            from src.services.formulas import ensure_registered
            from src.services.formulas.registry import REGISTRY

            ensure_registered()
            return {name: spec.qualified_version
                    for name, spec in REGISTRY.items() if name.startswith("critic.")}
        except Exception as exc:  # noqa: BLE001
            logger.error("could not read formula versions: %s", exc)
            return {}
