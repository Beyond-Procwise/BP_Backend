"""The decision engine: which option do we choose, and on what evidence.

Phase 3 of the conformance design (2026-06-28). The Rule Book asks "does this
condition hold"; the policy engine asks "is this allowed"; this asks "so what do
we DO about it" — and, critically, records why.

Two rules govern everything here:

**1. A decision may not rest on anything the engine did not look up.**
Facts are gathered first, from the database, and the decision is computed from
those facts. The engine never asks a model "should we approve this?" — a model's
recollection is not evidence. Where an LLM is involved at all it is downstream, to
narrate a decision that has already been made deterministically.

**2. When it cannot resolve cleanly, it escalates rather than guessing.**
No governed rule, missing facts, or a genuine conflict all produce `escalated`,
not a coin-flip dressed up as a verdict. ApprovalsAgent used to default a missing
spend-authority threshold to a hardcoded 1000 — a fabricated limit that could
auto-approve real money. An escalation is a useful answer; an invented one is a
liability.

Every decision is written to proc.bp_decision with `facts` (what was true) and
`evidence` (where each fact came from), so any verdict can be re-derived from
source rather than taken on trust.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

RESOLVED = "resolved"
ESCALATED = "escalated"


@dataclass
class Evidence:
    """Where one fact came from, precisely enough to go and check it."""

    fact: str
    value: Any
    source: str            # e.g. "proc.bp_invoice_trgt.total_amount"
    reference: Optional[str] = None  # the row/document key, when there is one

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fact": self.fact,
            "value": self.value,
            "source": self.source,
            "reference": self.reference,
        }


@dataclass
class Decision:
    subject_type: str
    subject_id: Optional[str]
    decision: str
    resolution: str = RESOLVED
    rationale: str = ""
    policy_id: Optional[int] = None
    policy_name: Optional[str] = None
    facts: Dict[str, Any] = field(default_factory=dict)
    evidence: List[Evidence] = field(default_factory=list)
    deal_id: Optional[str] = None
    supplier_id: Optional[str] = None
    decision_id: Optional[int] = None

    @property
    def escalated(self) -> bool:
        return self.resolution == ESCALATED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "subject_type": self.subject_type,
            "subject_id": self.subject_id,
            "decision": self.decision,
            "resolution": self.resolution,
            "rationale": self.rationale,
            "policy_id": self.policy_id,
            "policy_name": self.policy_name,
            "facts": self.facts,
            "evidence": [e.to_dict() for e in self.evidence],
            "deal_id": self.deal_id,
            "supplier_id": self.supplier_id,
        }


class DecisionEngine:
    """Decide what to do about a finding, an approval, or a quote award."""

    def __init__(self, agent_nick: Any) -> None:
        self.agent_nick = agent_nick
        self.policy_engine = getattr(agent_nick, "policy_engine", None)

    # ------------------------------------------------------------------
    # Grounding helpers
    # ------------------------------------------------------------------
    def _policy(self, slug: str) -> Optional[Dict[str, Any]]:
        if self.policy_engine is None:
            return None
        try:
            return self.policy_engine.get_policy(slug)
        except Exception:  # pragma: no cover - defensive
            logger.exception("policy lookup failed for %s", slug)
            return None

    @staticmethod
    def _rules(policy: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Pull the rule body out of PolicyEngine's normalised shape."""
        if not policy:
            return {}
        details = policy.get("details") or policy.get("policy_details") or {}
        rules = details.get("rules") or policy.get("rules") or {}
        return rules if isinstance(rules, dict) else {}

    @staticmethod
    def _policy_ids(policy: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not policy:
            return {"policy_id": None, "policy_name": None}
        raw = policy.get("raw_row") or {}
        return {
            "policy_id": raw.get("policy_id"),
            "policy_name": raw.get("policy_name") or policy.get("policyName"),
        }

    @staticmethod
    def _num(value: Any) -> Optional[Decimal]:
        if value is None:
            return None
        try:
            return Decimal(str(value))
        except (InvalidOperation, TypeError, ValueError):
            return None

    def _fetch_finding(self, finding_id: str) -> Optional[Dict[str, Any]]:
        """Read the finding from the table the Action Centre actually reads.

        Column names here are the REAL ones (checked against bp_sqldb, not assumed):
        the key is `discrepancy_id`, there is no `variance_amount` — the variance is
        derived from expected_value vs computed_value — and the row already carries
        document provenance in source_file / evidence_page / evidence_text.
        """
        sql = """
            SELECT discrepancy_id, doc_type, source_file, doc_pk_candidate,
                   field_name, raw_value, expected_value, computed_value,
                   issue_type, severity, status, notes, blocks_promotion,
                   evidence_page, evidence_text
            FROM proc.bp_extraction_discrepancy
            WHERE discrepancy_id::text = %s
        """
        try:
            with self.agent_nick.get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, (str(finding_id),))
                    row = cur.fetchone()
                    if not row:
                        return None
                    cols = [d[0] for d in cur.description]
                    return dict(zip(cols, row))
        except Exception:
            logger.exception("failed to read finding %s", finding_id)
            return None

    # ------------------------------------------------------------------
    # The decision
    # ------------------------------------------------------------------
    def decide_finding(
        self, finding_id: str, *, requested: Optional[str] = None
    ) -> Decision:
        """Decide what to do about an extraction/3-way-match finding.

        ``requested`` is what a human clicked (approve / reject / hold / …). The
        engine still gathers the facts and forms its own view: the point is not to
        rubber-stamp the click but to record whether the evidence supports it, so a
        wrong call is visible afterwards.
        """
        facts: Dict[str, Any] = {}
        evidence: List[Evidence] = []

        row = self._fetch_finding(finding_id)
        if not row:
            # We cannot decide about something we cannot read. Say so; do not invent
            # a subject and approve it.
            return Decision(
                subject_type="finding",
                subject_id=finding_id,
                decision="escalate",
                resolution=ESCALATED,
                rationale=(
                    f"Finding {finding_id} was not found in "
                    "proc.bp_extraction_discrepancy, so there are no facts to decide on."
                ),
            )

        ref = str(row.get("discrepancy_id"))
        for key in (
            "doc_type",
            "issue_type",
            "severity",
            "field_name",
            "raw_value",
            "expected_value",
            "computed_value",
            "status",
            "blocks_promotion",
        ):
            value = row.get(key)
            if value is None:
                continue
            facts[key] = str(value) if isinstance(value, Decimal) else value
            evidence.append(
                Evidence(
                    fact=key,
                    value=facts[key],
                    source=f"proc.bp_extraction_discrepancy.{key}",
                    reference=ref,
                )
            )

        # Cite the source document itself, not just the derived row. This is what lets
        # a human open the PDF at the right page and see the number for themselves,
        # rather than trusting that the extraction got it right.
        if row.get("source_file"):
            page = row.get("evidence_page")
            evidence.append(
                Evidence(
                    fact="source_document",
                    value=row["source_file"],
                    source="proc.bp_extraction_discrepancy.source_file",
                    reference=f"page {page}" if page else None,
                )
            )
        if row.get("evidence_text"):
            evidence.append(
                Evidence(
                    fact="source_text",
                    value=str(row["evidence_text"])[:500],
                    source="proc.bp_extraction_discrepancy.evidence_text",
                    reference=ref,
                )
            )

        # There is no variance_amount column, so the money at stake has to be derived —
        # and the table uses TWO conventions for computed_value, which must not be
        # conflated:
        #
        #   sum_mismatch / tax_percent_mismatch  -> computed_value is an ABSOLUTE value
        #        (expected 440.00, computed 460.0)      variance = |computed - expected|
        #
        #   amount_over_po / line_amount_over_po -> computed_value is ALREADY THE DELTA,
        #        written with an explicit sign          variance = |computed|
        #        (expected 28610.00, computed +950.00)
        #
        # Treating the signed delta as an absolute gives |950 - 28610| = 27,660 for a
        # finding whose real over-billing is 950 — wrong by a factor of 29, and wrong in
        # the direction that makes a trivial exception look like a catastrophe. The sign
        # is the tell, and it is present on 100% of the delta-style rows.
        raw_computed = str(row.get("computed_value") or "").strip()
        expected = self._num(row.get("expected_value"))
        computed = self._num(raw_computed)
        variance: Optional[Decimal] = None
        derivation: Optional[str] = None

        if computed is not None and raw_computed[:1] in ("+", "-"):
            variance = abs(computed)
            derivation = "computed_value is already the delta (explicitly signed)"
        elif expected is not None and computed is not None:
            variance = abs(computed - expected)
            derivation = "derived: abs(computed_value - expected_value)"

        if variance is not None:
            facts["variance"] = str(variance)
            evidence.append(
                Evidence(
                    fact="variance",
                    value=str(variance),
                    source=derivation or "derived",
                    reference=ref,
                )
            )

        severity = str(row.get("severity") or "").lower()
        blocks = bool(row.get("blocks_promotion"))

        # The governed rule for what a human may wave through unaided.
        policy = self._policy("approval_threshold")
        rules = self._rules(policy)
        ids = self._policy_ids(policy)
        threshold = self._num(rules.get("default_threshold_gbp"))
        if threshold is not None:
            facts["approval_threshold_gbp"] = str(threshold)
            evidence.append(
                Evidence(
                    fact="approval_threshold_gbp",
                    value=str(threshold),
                    source="proc.bp_policy(policy_type='approval').rules.default_threshold_gbp",
                    reference=str(ids.get("policy_id")),
                )
            )

        base = Decision(
            subject_type="finding",
            subject_id=ref,
            decision="escalate",
            resolution=ESCALATED,
            facts=facts,
            evidence=evidence,
            policy_id=ids.get("policy_id"),
            policy_name=ids.get("policy_name"),
        )

        # --- the rules, applied to the facts just gathered -----------------------
        # Order matters: the things that must never be auto-cleared are tested first,
        # so no amount of "the number is small" can wave through a blocking finding.

        if blocks:
            base.rationale = (
                "This finding blocks promotion of its document, so it cannot be "
                "auto-resolved on value alone — clearing it releases the document "
                "downstream. A human must decide."
            )
            return base

        if severity == "critical":
            base.rationale = (
                "Severity is 'critical', which is never auto-resolved regardless of "
                f"amount{f' (variance {variance})' if variance is not None else ''}."
            )
            return base

        if threshold is None:
            base.rationale = (
                "No governed approval threshold exists (policy 'approval_threshold'), "
                "so this finding cannot be tested against delegated authority. A human "
                "must decide. A default limit is NOT assumed."
            )
            return base

        if variance is None:
            # We do not know what it is worth, so we cannot say it is small enough to
            # ignore. Common for missing-field and reference issues.
            base.decision = "investigate"
            base.rationale = (
                "Its expected and computed values are not both numeric, so the amount "
                "at stake cannot be derived. It cannot be cleared on value; someone "
                "needs to look at it."
            )
            return base

        if variance <= threshold:
            base.decision = "approve"
            base.resolution = RESOLVED
            base.rationale = (
                f"The variance is {variance} ({row.get('field_name')}: expected "
                f"{expected}, computed {computed}), within the governed approval "
                f"threshold of {threshold}, severity is '{severity or 'unset'}', and it "
                "does not block promotion — so it is within delegated authority."
            )
            return base

        base.rationale = (
            f"The variance is {variance} ({row.get('field_name')}: expected {expected}, "
            f"computed {computed}), which exceeds the governed approval threshold of "
            f"{threshold}. It is above delegated authority and needs sign-off."
        )
        return base

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def record(
        self,
        decision: Decision,
        *,
        workflow_id: Optional[str] = None,
        agent: Optional[str] = None,
        created_by: str = "system",
    ) -> Optional[int]:
        """Persist the decision and its evidence. Returns the new decision_id."""
        try:
            with self.agent_nick.get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO proc.bp_decision (
                            subject_type, subject_id, deal_id, supplier_id,
                            decision, resolution, rationale,
                            policy_id, policy_name, facts, evidence,
                            workflow_id, agent, created_by
                        )
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                        RETURNING decision_id
                        """,
                        (
                            decision.subject_type,
                            decision.subject_id,
                            decision.deal_id,
                            decision.supplier_id,
                            decision.decision,
                            decision.resolution,
                            decision.rationale,
                            decision.policy_id,
                            decision.policy_name,
                            json.dumps(decision.facts, default=str),
                            json.dumps(
                                [e.to_dict() for e in decision.evidence], default=str
                            ),
                            workflow_id,
                            agent,
                            created_by,
                        ),
                    )
                    row = cur.fetchone()
                conn.commit()
            decision.decision_id = int(row[0]) if row else None
            return decision.decision_id
        except Exception:
            logger.exception("failed to persist decision to proc.bp_decision")
            return None

    # ------------------------------------------------------------------
    # Execution — actually do the thing
    # ------------------------------------------------------------------
    #
    # Every one of these used to be a toast. In particular "Apply value" never
    # applied a value: the UI posted {id, action} and dropped the corrected figure on
    # the floor, so the finding was closed and the correction was lost.
    #
    # Note what is NOT done here: the extracted source data is never overwritten.
    # bp_extraction_discrepancy carries resolved_value / resolution_action /
    # resolved_by for exactly this purpose — the correction is RECORDED against the
    # finding, leaving what the document actually said intact. Destroying the
    # extraction to make a number look right would defeat the point of extracting it.

    # Verbs that close a finding. `flag` is deliberately absent: flagging something is
    # how you ask for attention, not how you make it go away.
    CLOSING_ACTIONS = {"apply_value", "confirm", "approve", "reject", "dismiss"}
    KNOWN_ACTIONS = CLOSING_ACTIONS | {"flag", "escalate", "hold", "assign", "investigate", "query"}

    def execute(
        self,
        finding_id: str,
        action: str,
        *,
        user_id: str = "api",
        value: Optional[str] = None,
        override_reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Carry out the HUMAN's decision, with the engine advising.

        These are human-in-the-loop actions, so the human is the authority. The engine
        does NOT get a veto: refusing to act would not be human-in-the-loop, it would
        be automation overruling the person accountable for the call.

        What the engine does instead:
          * it states what the evidence supports, BEFORE anything happens;
          * if the human's action contradicts that, it asks for a reason and will not
            proceed on a bare click — an override must be deliberate;
          * it records who acted, what the engine advised, and why they went the other
            way, so the override is answerable afterwards.

        A confirmation is not friction for its own sake. Closing a GBP 27,660 critical
        over-billing should take one more second and leave a name against it.
        """
        action = (action or "").strip().lower()
        if action not in self.KNOWN_ACTIONS:
            return {"applied": False, "error": f"unknown action '{action}'"}

        recommendation = self.decide_finding(finding_id, requested=action)

        row = self._fetch_finding(finding_id)
        if not row:
            return {
                "applied": False,
                "error": f"finding {finding_id} not found",
                "recommendation": recommendation.to_dict(),
            }

        # Does the human's action contradict the evidence? Closing a finding the engine
        # says must escalate is the case that matters.
        conflicts = action in self.CLOSING_ACTIONS and recommendation.escalated

        if conflicts and not override_reason:
            # Not a refusal — a confirmation step. The human may absolutely do this;
            # they just have to mean it, and say why. Everything needed to make that
            # call is returned: the verdict, the reasoning, and the evidence behind it.
            return {
                "applied": False,
                "requires_override": True,
                "recommendation": recommendation.to_dict(),
                "prompt": (
                    f"The evidence does not support '{action}' here: "
                    f"{recommendation.rationale} You can still proceed, but the reason "
                    "will be recorded against your name."
                ),
            }

        # Work out the new state of the finding.
        if action == "apply_value":
            # THE fix: actually carry the expected value across. Closing this without
            # writing resolved_value is what "Apply value" has always done.
            resolved_value = value or row.get("expected_value")
            if resolved_value is None:
                return {
                    "applied": False,
                    "error": (
                        "There is no expected value to apply on this finding, and none "
                        "was supplied. Supply one explicitly, or use a different action."
                    ),
                    "recommendation": recommendation.to_dict(),
                }
            new_status, resolved = "resolved", str(resolved_value)
        elif action == "confirm":
            # Confirming says "what we extracted was right" — so the resolved value is
            # the extracted one, not the expected one.
            new_status, resolved = "resolved", str(
                row.get("computed_value") or row.get("raw_value") or ""
            ) or None
        elif action in ("dismiss", "reject"):
            new_status, resolved = "ignored", None
        elif action == "approve":
            new_status, resolved = "resolved", value
        elif action == "flag":
            # Stays OPEN. A flag is a request for a human, not a resolution.
            new_status, resolved = "flagged", None
        elif action == "hold":
            new_status, resolved = "on_hold", None
        else:  # escalate | assign | investigate | query
            new_status, resolved = "escalated", None

        try:
            with self.agent_nick.get_db_connection() as conn:
                with conn.cursor() as cur:
                    if new_status in ("resolved", "ignored"):
                        cur.execute(
                            """
                            UPDATE proc.bp_extraction_discrepancy
                               SET status = %s,
                                   resolution_action = %s,
                                   resolved_value = %s,
                                   resolved_by = %s,
                                   resolved_at = NOW()
                             WHERE discrepancy_id::text = %s
                            """,
                            (new_status, action, resolved, user_id, str(finding_id)),
                        )
                    else:
                        # Open states keep resolved_* NULL — they are not resolved.
                        cur.execute(
                            """
                            UPDATE proc.bp_extraction_discrepancy
                               SET status = %s,
                                   resolution_action = %s
                             WHERE discrepancy_id::text = %s
                            """,
                            (new_status, action, str(finding_id)),
                        )
                conn.commit()
        except Exception:
            logger.exception("failed to apply action %s to finding %s", action, finding_id)
            return {
                "applied": False,
                "error": "could not update the finding",
                "recommendation": recommendation.to_dict(),
            }

        # Record what the ENGINE advised alongside what the HUMAN actually did. Storing
        # only the outcome would lose the most interesting fact in the row: that someone
        # was told the evidence said otherwise and went ahead anyway.
        decision_id = self._record_human_action(
            recommendation,
            human_action=action,
            actor=user_id,
            override_reason=override_reason if conflicts else None,
        )

        return {
            "applied": True,
            "action": action,
            "finding_id": str(finding_id),
            "new_status": new_status,
            # What was actually written. For apply_value this is the number carried
            # across — the thing the button has always claimed to do and never did.
            "resolved_value": resolved,
            "overridden": bool(conflicts),
            "override_reason": override_reason if conflicts else None,
            "actioned_by": user_id,
            "decision_id": decision_id,
            "recommendation": recommendation.to_dict(),
        }

    def _record_human_action(
        self,
        recommendation: "Decision",
        *,
        human_action: str,
        actor: str,
        override_reason: Optional[str],
    ) -> Optional[int]:
        """Persist the engine's advice, the human's action, and any override."""
        try:
            with self.agent_nick.get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO proc.bp_decision (
                            subject_type, subject_id, deal_id, supplier_id,
                            decision, resolution, rationale,
                            policy_id, policy_name, facts, evidence,
                            status, actioned_by, actioned_at, override_reason,
                            agent, created_by
                        )
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,NOW(),%s,%s,%s)
                        RETURNING decision_id
                        """,
                        (
                            recommendation.subject_type,
                            recommendation.subject_id,
                            recommendation.deal_id,
                            recommendation.supplier_id,
                            # `decision` is what actually happened — the human's call.
                            human_action,
                            recommendation.resolution,
                            # The rationale keeps the ENGINE's reasoning, so the row shows
                            # what the human was told at the moment they decided.
                            recommendation.rationale,
                            recommendation.policy_id,
                            recommendation.policy_name,
                            json.dumps(recommendation.facts, default=str),
                            json.dumps(
                                [e.to_dict() for e in recommendation.evidence], default=str
                            ),
                            "overridden" if override_reason else "actioned",
                            actor,
                            override_reason,
                            "decision_engine",
                            actor,
                        ),
                    )
                    row = cur.fetchone()
                conn.commit()
            return int(row[0]) if row else None
        except Exception:
            logger.exception("failed to record human action on proc.bp_decision")
            return None

    def trace(self, decision_id: int) -> Optional[Dict[str, Any]]:
        """Return a decision with the full evidence that produced it."""
        try:
            with self.agent_nick.get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT decision_id, subject_type, subject_id, deal_id,
                               decision, resolution, rationale, policy_id, policy_name,
                               facts, evidence, status, created_by, created_at
                        FROM proc.bp_decision WHERE decision_id = %s
                        """,
                        (decision_id,),
                    )
                    row = cur.fetchone()
                    if not row:
                        return None
                    cols = [d[0] for d in cur.description]
                    return dict(zip(cols, row))
        except Exception:
            logger.exception("failed to read decision %s", decision_id)
            return None
