"""Decide whether a spend request sits within delegated authority.

This agent was previously unreachable AND wrong, in three separate ways:

1. It read thresholds from `approval_policies` and wrote decisions to
   `proc.approvals`. **Neither table exists.** Both statements sat inside bare
   `except: logger.exception(...)` blocks, so the agent returned SUCCESS while
   every approval it "stored" was silently discarded and every threshold
   silently fell back to a hardcoded `1000`.
2. It built "supporting evidence" by embedding `str(amount)` and querying Qdrant
   with the resulting vector — i.e. it searched the document corpus for the
   string "5000" and presented whatever came back as `references`. That is
   fabricated grounding, and it is gone.
3. No workflow graph contained an `approvals` node, so the orchestrator never
   dispatched it.

The rebuild: the threshold is a **governed, human-authored business rule** read
from `proc.bp_policy` (`policy_type='approval'`), never a constant in code. If no
governed threshold can be resolved the agent does not invent one — it escalates
to a human and says why. Every decision is persisted to `proc.bp_approval`
alongside the facts it was computed from, so any verdict can be traced back to
its inputs.
"""

from __future__ import annotations

import json
import logging
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Optional, Tuple

from agents.base_agent import BaseAgent, AgentContext, AgentOutput, AgentStatus
from utils.gpu import configure_gpu

logger = logging.getLogger(__name__)

# The governed policy carrying the spend-authority gate.
_POLICY_SLUG = "approval_threshold"

DECISION_APPROVE = "approve"
DECISION_ESCALATE = "escalate"


class ApprovalsAgent(BaseAgent):
    """Gate a spend amount against the governed approval threshold."""

    AGENTIC_PLAN_STEPS = (
        "Read the requested amount and its currency from the approval request.",
        "Resolve the spend-authority threshold from the governed approval policy.",
        "Compare amount against threshold; escalate to a human when the threshold "
        "is unavailable or the amount exceeds it.",
        "Persist the decision with the facts it was computed from, so it can be traced.",
    )

    def __init__(self, agent_nick):
        super().__init__(agent_nick)
        self.device = configure_gpu()
        self.policy_engine = getattr(agent_nick, "policy_engine", None)

    # ------------------------------------------------------------------
    # Grounding
    # ------------------------------------------------------------------
    def _governed_threshold(self) -> Tuple[Optional[Decimal], Dict[str, Any]]:
        """Return ``(threshold, provenance)`` from the governed approval policy.

        Returns ``(None, ...)`` when no governed threshold exists. The caller MUST
        escalate in that case rather than substitute a default — a made-up
        spend-authority limit could auto-approve real money.
        """
        provenance: Dict[str, Any] = {"threshold_source": "unavailable"}
        if self.policy_engine is None:
            return None, provenance

        try:
            policy = self.policy_engine.get_policy(_POLICY_SLUG)
        except Exception:  # pragma: no cover - defensive
            logger.exception("approval policy lookup failed")
            return None, provenance

        if not policy:
            return None, provenance

        # PolicyEngine._normalise_policy_row returns its own shape: the rule body
        # lives under "details"->"rules", the display name under "policyName", and
        # the numeric bp_policy.policy_id only survives on "raw_row". ("policyId" is
        # the policy_identifier string, not the DB key.) Read that shape, but stay
        # tolerant of a plain row being passed in by a test.
        raw_row = policy.get("raw_row") or {}
        details = policy.get("details") or policy.get("policy_details") or {}
        rules = details.get("rules") or policy.get("rules") or {}
        if not isinstance(rules, dict):
            return None, provenance

        raw = rules.get("default_threshold_gbp")
        if raw is None:
            return None, provenance

        try:
            threshold = Decimal(str(raw))
        except (InvalidOperation, TypeError, ValueError):
            logger.warning("approval policy threshold is not a number: %r", raw)
            return None, provenance

        return threshold, {
            "threshold_source": "governed_policy",
            "policy_id": raw_row.get("policy_id") or policy.get("policy_id"),
            "policy_name": (
                raw_row.get("policy_name")
                or policy.get("policyName")
                or policy.get("policy_name")
            ),
            "policy_currency": rules.get("currency"),
        }

    @staticmethod
    def _as_decimal(value: Any) -> Optional[Decimal]:
        if value is None:
            return None
        try:
            return Decimal(str(value))
        except (InvalidOperation, TypeError, ValueError):
            return None

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    def run(self, context: AgentContext) -> AgentOutput:
        payload = dict(context.input_data or {})

        # A quote-award approval carries the amount on the winning quote.
        best = payload.get("best_quote") or {}
        amount = self._as_decimal(
            payload.get("amount")
            if payload.get("amount") is not None
            else best.get("price")
        )
        if amount is None:
            return self._with_plan(
                context,
                AgentOutput(
                    status=AgentStatus.FAILED, data={}, error="amount not provided"
                ),
            )

        # Currency is never invented. If the caller did not state one it stays NULL.
        currency = payload.get("currency") or best.get("currency")

        # An explicit threshold on the request is a deliberate caller override, and
        # is recorded as such so the trace shows the gate did not come from policy.
        override = self._as_decimal(payload.get("threshold"))
        if override is not None:
            threshold: Optional[Decimal] = override
            provenance: Dict[str, Any] = {"threshold_source": "request_override"}
        else:
            threshold, provenance = self._governed_threshold()

        comparison: Optional[str] = None
        if threshold is None:
            decision = DECISION_ESCALATE
            reason = (
                "No governed approval threshold is available (policy "
                "'approval_threshold' not found), so this cannot be auto-approved "
                "and needs a human decision."
            )
        else:
            within = amount <= threshold
            decision = DECISION_APPROVE if within else DECISION_ESCALATE
            reason = (
                f"Amount {amount} is {'within' if within else 'above'} the "
                f"approval threshold {threshold}."
            )
            # State the relation that actually holds. Rendering this as a fixed
            # "{amount} <= {threshold}" wrote a FALSE statement into the audit
            # trail on every escalation ("25000 <= 10000"). The trace exists to
            # make a decision checkable; it cannot itself assert something untrue.
            comparison = f"{amount} {'<=' if within else '>'} {threshold}"

        grounding: Dict[str, Any] = {
            "amount": str(amount),
            "currency": currency,
            "threshold": str(threshold) if threshold is not None else None,
            "comparison": comparison,
            **provenance,
        }

        approval_id = self._store_approval(
            context=context,
            payload=payload,
            amount=amount,
            currency=currency,
            threshold=threshold,
            decision=decision,
            reason=reason,
            grounding=grounding,
        )

        data: Dict[str, Any] = {
            "decision": decision,
            "approved": decision == DECISION_APPROVE,
            "amount": float(amount),
            "currency": currency,
            "threshold": float(threshold) if threshold is not None else None,
            "decision_reason": reason,
            # The facts the verdict was computed from — what makes the decision
            # checkable rather than merely plausible.
            "grounding": grounding,
            "approval_id": approval_id,
        }
        if best:
            data["best_quote"] = best

        return self._with_plan(
            context,
            AgentOutput(status=AgentStatus.SUCCESS, data=data, next_agents=[]),
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _store_approval(
        self,
        *,
        context: AgentContext,
        payload: Dict[str, Any],
        amount: Decimal,
        currency: Optional[str],
        threshold: Optional[Decimal],
        decision: str,
        reason: str,
        grounding: Dict[str, Any],
    ) -> Optional[int]:
        """Write the decision to ``proc.bp_approval`` and return its approval_id.

        A failure here yields a NULL approval_id that the caller can see, rather
        than being swallowed: the previous version hid a missing table behind a
        bare except and reported success regardless.
        """
        best = payload.get("best_quote") or {}
        try:
            with self.agent_nick.get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO proc.bp_approval (
                            deal_id, rfq_id, finding_id, supplier_id,
                            amount, currency, threshold,
                            decision, decision_reason,
                            policy_id, policy_name, grounding,
                            workflow_id, created_by
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING approval_id
                        """,
                        (
                            payload.get("deal_id"),
                            payload.get("rfq_id"),
                            payload.get("finding_id"),
                            payload.get("supplier_id") or best.get("supplier_id"),
                            amount,
                            currency,
                            threshold,
                            decision,
                            reason,
                            grounding.get("policy_id"),
                            grounding.get("policy_name"),
                            json.dumps(grounding),
                            getattr(context, "workflow_id", None),
                            getattr(context, "user_id", None) or "system",
                        ),
                    )
                    row = cur.fetchone()
                conn.commit()
            return int(row[0]) if row else None
        except Exception:
            logger.exception("failed to persist approval to proc.bp_approval")
            return None
