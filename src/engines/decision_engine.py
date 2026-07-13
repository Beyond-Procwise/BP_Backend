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

        # There is no variance_amount column. Derive the money at stake from the two
        # figures that disagree, and only when BOTH are numeric — a mismatch between
        # two strings has no financial value we can assert.
        expected = self._num(row.get("expected_value"))
        computed = self._num(row.get("computed_value"))
        variance: Optional[Decimal] = None
        if expected is not None and computed is not None:
            variance = abs(computed - expected)
            facts["variance"] = str(variance)
            evidence.append(
                Evidence(
                    fact="variance",
                    value=str(variance),
                    source="derived: abs(computed_value - expected_value)",
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
