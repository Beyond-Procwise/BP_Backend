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
    # Email replies
    # ------------------------------------------------------------------
    _EMAIL_SUBJECT_TYPE = "email_reply"

    # The ONE structured location the prior offer is read from. Verified against all
    # 18 live rows of proc.draft_rfq_emails (bp_sqldb, 2026-07-28): the negotiation
    # drafting path records its counter here, e.g. 96000.0 on draft 17, corroborated
    # by "Our target positioning: GBP 96,000.00" in that draft's own body. The
    # brief's `target_price` / `offer_price` do NOT exist in any row -- a scan of
    # jsonb_object_keys(payload) for price|target|amount|value|offer|cost returns
    # nothing -- so they are not read; a lookup that can never hit is worse than no
    # lookup, because it reads as though a prior offer were being checked for.
    _PRIOR_OFFER_PATH = ("metadata", "counter_price")

    @property
    def _prior_offer_source(self) -> str:
        return "proc.draft_rfq_emails.payload." + ".".join(self._PRIOR_OFFER_PATH)

    def _prior_offer(self, payload: Any) -> tuple[Optional[Decimal], Optional[str]]:
        """The price WE last put to this supplier, and where it came from.

        Read only from `payload->'metadata'->>'counter_price'`. Deliberately narrow:

        * Never the supplier's own number. Using their figure as "ours" would make
          every reply show nothing at stake -- the one wrong answer that turns this
          gate off silently.
        * Never parsed out of the body prose. The £96,000 is also written in the
          draft's HTML, which is how the structured key was corroborated, but prose
          is not a source of record and a regex over it would invent precision.
        * Missing, non-numeric or unparseable is ABSENT, not zero. It returns
          (None, None) and the caller escalates on the missing-prior gate, exactly as
          it did before this was wired.

        Returns (value, source) so the caller can cite the real path rather than a
        hardcoded guess about where the number came from.
        """
        if not isinstance(payload, dict):
            return None, None
        cursor: Any = payload
        for key in self._PRIOR_OFFER_PATH:
            if not isinstance(cursor, dict):
                return None, None
            cursor = cursor.get(key)
        value = self._num(cursor)
        if value is None:
            return None, None
        return value, self._prior_offer_source

    def _fetch_email_reply(self, response_id: str) -> Optional[Dict[str, Any]]:
        """The supplier's reply, plus the offer it is replying to.

        Column names are the REAL ones on proc.supplier_response (checked against
        bp_sqldb 2026-07-28): the body is `response_text`, the key is `id`, and the
        link back to what we sent is `unique_id`.

        The join is on `unique_id`, and it was verified against the live row rather
        than assumed: supplier_response row 1
        (`089580f2-...-PeopleFirst HR Solutions Ltd`) matches draft_rfq_emails row 17
        on that column, so no `payload->>'message_id'` fallback is needed. The unique
        index on drafts is (workflow_id, unique_id), so `unique_id` alone is not
        formally unique -- hence ORDER BY d.id DESC LIMIT 1, which makes the single
        fetched row the most recent draft on the thread rather than an arbitrary one.

        Neither table has a `deal_id` column, so none is read. deal_id in this system
        is assigned by a database stored procedure and is never set in app code.

        `auto_replies_on_thread` is derived, not stored -- it counts what the agent
        has already sent unattended on this thread, which is what the per-thread cap
        governs. It is None, not 0, if that count could not be taken: an unknown
        history must not read as an empty one.

        `prior_price` / `prior_price_source` come from the draft payload via
        `_prior_offer` -- see that method for where, and why only there.
        """
        sql = """
            SELECT sr.id, sr.workflow_id, sr.unique_id, sr.supplier_id,
                   sr.response_subject, sr.response_text, sr.response_from,
                   sr.round_number, sr.match_confidence,
                   sr.price, sr.currency, sr.payment_terms, sr.lead_time,
                   d.subject           AS draft_subject,
                   d.recipient_email   AS draft_recipient,
                   d.payload           AS draft_payload
              FROM proc.supplier_response sr
              LEFT JOIN proc.draft_rfq_emails d
                     ON d.unique_id = sr.unique_id
             WHERE sr.id::text = %s
             ORDER BY d.id DESC
             LIMIT 1
        """
        try:
            with self.agent_nick.get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, (str(response_id),))
                    row = cur.fetchone()
                    if not row:
                        return None
                    cols = [d[0] for d in cur.description]
                    record = dict(zip(cols, row))
                    prior, prior_source = self._prior_offer(record.get("draft_payload"))
                    record["prior_price"] = prior
                    record["prior_price_source"] = prior_source
                    try:
                        cur.execute(
                            """
                            SELECT count(*) FROM proc.bp_decision
                             WHERE subject_type = %s AND subject_id = %s
                               AND resolution = %s AND decision = 'send'
                            """,
                            (self._EMAIL_SUBJECT_TYPE, record.get("unique_id"), RESOLVED),
                        )
                        record["auto_replies_on_thread"] = int((cur.fetchone() or [0])[0] or 0)
                    except Exception:
                        # Not fatal to reading the reply, but it IS fatal to enforcing
                        # the cap. None says "unknown", and the cap gate escalates on
                        # unknown rather than treating it as none-so-far.
                        logger.exception(
                            "failed to count prior unattended replies on thread %s",
                            record.get("unique_id"),
                        )
                        record["auto_replies_on_thread"] = None
                    return record
        except Exception:
            logger.exception("failed to read supplier reply %s", response_id)
            return None

    def _reply_caller(self) -> Optional[Any]:
        """An object exposing ``call_ollama``, for the intent classifier.

        ``classify_reply`` needs a one-shot LLM call. That method is ``call_ollama``
        and it lives on ``BaseAgent``; ``AgentNick`` does NOT inherit from BaseAgent
        and has no such method, so ``self.agent_nick`` cannot be passed. Any already
        registered agent instance will do -- they are all BaseAgent subclasses sharing
        one AgentNick, so the model, settings and connection pool are identical
        whichever is picked. Failing that, construct a bare BaseAgent.

        Returns None if neither is possible. It never raises, and it never reaches for
        a different model: AgentNick is the only model in this system.
        """
        agents = getattr(self.agent_nick, "agents", None)
        if isinstance(agents, dict):
            for instance in agents.values():
                if callable(getattr(instance, "call_ollama", None)):
                    return instance
        try:
            from src.agents.base_agent import BaseAgent

            return BaseAgent(self.agent_nick)
        except Exception:
            logger.exception("no LLM caller could be obtained for reply classification")
            return None

    def _classify(self, body: str):
        """Seam for tests; production path goes to the grounded classifier."""
        from src.services.email_intent import UNCLASSIFIED, ReplyIntent, classify_reply

        caller = self._reply_caller()
        if caller is None:
            # Unusable, not unclassified-but-fine: decide_email_reply escalates on
            # `grounded is False`, which is the right outcome when we could not read
            # the reply at all.
            return ReplyIntent(
                intent=UNCLASSIFIED,
                confidence=0.0,
                quote="",
                grounded=False,
                reason="no LLM caller was available to classify the reply",
            )
        try:
            return classify_reply(body, caller=caller)
        except Exception:  # pragma: no cover - classify_reply is documented not to raise
            logger.exception("reply classification failed")
            return ReplyIntent(
                intent=UNCLASSIFIED,
                confidence=0.0,
                quote="",
                grounded=False,
                reason="the classifier raised",
            )

    def decide_email_reply(
        self,
        response_id: str,
        *,
        authority: Optional[Dict[str, Any]] = None,
        requested: Optional[str] = None,
    ) -> Decision:
        """Answer it ourselves, or put it in front of a human -- and say which, and why.

        Deterministic in the part that matters: the model contributes an intent label
        and a quoted sentence, and every gate below is arithmetic and set membership
        over governed values. No model is asked whether to send.
        """
        facts: Dict[str, Any] = {}
        evidence: List[Evidence] = []

        row = self._fetch_email_reply(response_id)
        if not row:
            return Decision(
                subject_type=self._EMAIL_SUBJECT_TYPE,
                subject_id=str(response_id),
                decision="escalate",
                resolution=ESCALATED,
                rationale=(
                    f"Supplier reply {response_id} was not found in "
                    "proc.supplier_response, so there are no facts to decide on."
                ),
            )

        ref = str(row.get("id"))
        subject_id = str(row.get("unique_id") or response_id)
        for key in ("supplier_id", "response_subject", "response_from", "round_number",
                    "match_confidence", "price", "currency", "payment_terms", "lead_time"):
            value = row.get(key)
            if value is None:
                continue
            facts[key] = str(value) if isinstance(value, Decimal) else value
            evidence.append(Evidence(fact=key, value=facts[key],
                                     source=f"proc.supplier_response.{key}", reference=ref))

        policy_name = (authority or {}).get("policy_name")
        policy_id = (authority or {}).get("policy_id")

        def _escalate(rationale: str) -> Decision:
            return Decision(
                subject_type=self._EMAIL_SUBJECT_TYPE, subject_id=subject_id,
                decision="escalate", resolution=ESCALATED, rationale=rationale,
                policy_id=policy_id, policy_name=policy_name,
                facts=facts, evidence=evidence,
                # deal_id is NOT set: neither proc.supplier_response nor
                # proc.draft_rfq_emails has that column, and deal_id is assigned by a
                # DB stored procedure. A None here is honest; a guess would not be.
                supplier_id=row.get("supplier_id"),
            )

        # 1. Authority. No governed limit means no unattended send -- the same rule
        #    that stops a missing approval threshold from auto-approving money.
        if not authority or not authority.get("governed"):
            reason = (authority or {}).get("reason") or (
                "no send authority was resolved for policy 'email_reply_autonomy'"
            )
            facts["authority"] = "ungoverned"
            evidence.append(Evidence(fact="authority", value="ungoverned",
                                     source="proc.bp_policy(email_reply_autonomy)",
                                     reference=reason))
            return _escalate(
                f"This reply needs a human because {reason}. Nothing is sent on an "
                "unknown limit."
            )

        # 2. Classification, with its quote checked against the supplier's own words.
        intent = self._classify(str(row.get("response_text") or ""))
        facts["intent"] = intent.intent
        facts["intent_confidence"] = intent.confidence
        evidence.append(Evidence(fact="intent", value=intent.intent,
                                 source="AgentNick classification of supplier_response.response_text",
                                 reference=ref))
        if intent.quote:
            evidence.append(Evidence(fact="supporting_sentence", value=intent.quote,
                                     source="proc.supplier_response.response_text",
                                     reference="verbatim" if intent.grounded else "NOT FOUND in source"))
        if not intent.grounded:
            return _escalate(
                f"The classification '{intent.intent}' could not be grounded: "
                f"{intent.reason}. An ungrounded reading of a supplier's message is not "
                "a basis for replying unattended."
            )

        # A governed minimum we do not have is not a minimum of zero. Same rule as the
        # value limit below and the thread cap after it: resolve_authority can return
        # governed=True with any of the three as None (it only populates what the
        # policy carries), and shipping one of them fail-closed and the others
        # permissive would be a trap for whoever reads this next.
        min_conf = authority.get("min_intent_confidence")
        if min_conf is None:
            facts["min_intent_confidence"] = None
            evidence.append(Evidence(
                fact="min_intent_confidence", value=None,
                source="proc.bp_policy(email_reply_autonomy).rules.min_intent_confidence",
                reference=authority.get("reason")))
            return _escalate(
                f"No governed minimum confidence was resolved under "
                f"{policy_name or 'the autonomy policy'} (min_intent_confidence is "
                f"missing), so the classifier's {intent.confidence:.2f} cannot be tested "
                "against anything. An absent minimum is not a minimum of zero."
            )
        if intent.confidence < float(min_conf):
            return _escalate(
                f"Confidence in '{intent.intent}' is {intent.confidence:.2f}, below the "
                f"governed minimum of {float(min_conf):.2f}."
            )

        # 3. Governed intent lists.
        if intent.intent in (authority.get("escalate_intents") or []):
            return _escalate(
                f"'{intent.intent}' is a governed escalate-only intent under "
                f"{policy_name or 'the autonomy policy'}: a human decides this one."
            )
        if intent.intent not in (authority.get("auto_intents") or []):
            return _escalate(
                f"'{intent.intent}' is not on the governed auto-reply list, so it goes "
                "to a human. Widen auto_reply_intents in the policy to change that."
            )

        # 4. Value at stake against the governed spend limit. Derived, and the
        #    derivation is stated: two numbers, both cited above.
        #
        #    Order matters, and this is the order:
        #      4.  is there a price at all?      no  -> no money moves, skip the block
        #      4.  a governed limit?             no  -> escalate (absent != unlimited)
        #      4a. the same denomination?        no  -> escalate (never convert)
        #      4b. a prior offer to move from?   no  -> escalate (absent != unchanged)
        #      4c. abs(price - prior) > limit?   yes -> escalate
        #    The currency gate sits at 4a, after the limit is in hand and before any
        #    subtraction or comparison, so no arithmetic here ever crosses currencies.
        #    It cannot go earlier: with no limit resolved there is no limit_currency to
        #    compare against, and "no limit" is the more fundamental authority failure
        #    to report. It must not go later: a mismatch invalidates the comparison
        #    whether or not a prior offer exists, so reporting a missing prior first
        #    would name the smaller problem.
        #
        #    resolve_authority() returns governed=True with limit_gbp=None whenever the
        #    autonomy policy omits `defer_value_limit_to`, so "no limit" is a reachable
        #    state -- and an absent limit is never an unlimited one.
        price = self._num(row.get("price"))
        prior = self._num(row.get("prior_price"))
        limit = self._num(authority.get("limit_gbp"))
        reply_currency = str(row.get("currency") or "").strip()
        if price is not None:
            priced = f"{price} {reply_currency}".strip()
            if limit is None:
                facts["value_limit_gbp"] = None
                evidence.append(Evidence(
                    fact="value_limit_gbp", value=None,
                    source="proc.bp_policy(email_reply_autonomy).rules.defer_value_limit_to",
                    reference=authority.get("reason")))
                return _escalate(
                    f"The reply carries a price of {priced} but no governed "
                    f"value limit was resolved under "
                    f"{policy_name or 'the autonomy policy'} (limit_gbp is missing, so "
                    "the policy defers to no approval threshold). There is nothing to "
                    "test the amount against, and an absent limit is not an unlimited "
                    "one, so a human decides."
                )

            facts["value_limit_gbp"] = str(limit)
            evidence.append(Evidence(
                fact="value_limit_gbp", value=str(limit),
                source="proc.bp_policy(email_reply_autonomy) -> approval threshold "
                       "(resolve_authority.limit_gbp)",
                reference=str(policy_id) if policy_id is not None else None))

            # 4a. Same denomination, or no comparison. This fires BEFORE any
            #     subtraction or comparison, so no arithmetic in this method ever
            #     crosses currencies. It is a GATE, not a conversion: there is an FX
            #     facility in this codebase, but a rate nobody chose -- fabricated or
            #     stale -- underneath a spend decision is worse than a human looking at
            #     it. Amounts in different currencies are never combined here without an
            #     explicit conversion basis, and none is chosen.
            limit_currency = str(authority.get("limit_currency") or "").strip()
            facts["value_limit_currency"] = limit_currency or None
            evidence.append(Evidence(
                fact="value_limit_currency", value=limit_currency or None,
                source="proc.bp_policy(email_reply_autonomy) -> approval threshold "
                       "(resolve_authority.limit_currency)",
                reference=str(policy_id) if policy_id is not None else None))
            if not reply_currency:
                evidence.append(Evidence(
                    fact="currency", value=None,
                    source="proc.supplier_response.currency", reference=ref))
                return _escalate(
                    f"The reply carries a price of {price} but no currency is recorded "
                    f"against it, while the governed limit of {limit} is denominated in "
                    f"{limit_currency or 'an unstated currency'}. No comparison was "
                    "attempted, because an amount whose denomination is unknown cannot "
                    "be tested against a limit in a specific one. This is a currency "
                    "problem, not a pricing dispute."
                )
            if not limit_currency:
                return _escalate(
                    f"The reply is priced in {reply_currency} but the governed limit of "
                    f"{limit} carries no currency of its own under "
                    f"{policy_name or 'the autonomy policy'}, so there is nothing to "
                    "confirm the two are the same denomination. No comparison was "
                    "attempted. This is a currency problem, not a pricing dispute."
                )
            if reply_currency.upper() != limit_currency.upper():
                return _escalate(
                    f"The reply is priced in {reply_currency} but the governed limit of "
                    f"{limit} is denominated in {limit_currency}. No comparison was "
                    "attempted: amounts in different currencies are never combined "
                    "without an explicit conversion basis, and none is chosen here. "
                    "This is a currency problem, not a pricing dispute."
                )

            if prior is None:
                # We can see their number but not ours. That is not "nothing at stake".
                return _escalate(
                    f"The supplier quotes {priced} but no prior offer is "
                    "recorded on the draft, so the amount at stake cannot be computed. "
                    "A human should compare these."
                )

            # Both numbers are cited individually, so the subtraction can be checked
            # rather than believed.
            prior_source = row.get("prior_price_source") or self._prior_offer_source
            facts["prior_offer"] = str(prior)
            evidence.append(Evidence(fact="prior_offer", value=str(prior),
                                     source=prior_source, reference=ref))

            at_stake = abs(price - prior)
            facts["value_at_stake"] = str(at_stake)
            evidence.append(Evidence(
                fact="value_at_stake", value=str(at_stake),
                source=f"derived: abs(proc.supplier_response.price - {prior_source})",
                reference=ref))
            if at_stake > limit:
                # Both sides are in reply_currency, which gate 4a has already confirmed
                # equals limit_currency -- so no default denomination is assumed here.
                return _escalate(
                    f"The reply moves {at_stake} {reply_currency} "
                    f"(supplier {price} against our {prior}), above the governed limit "
                    f"of {limit} {limit_currency}."
                )

        # 5. Per-thread cap on unattended replies. Fail-closed on BOTH unknowns: an
        #    absent cap is not an unlimited one, and an uncounted history is not an
        #    empty one.
        cap = authority.get("max_auto_replies_per_thread")
        already = row.get("auto_replies_on_thread")
        if cap is None:
            facts["max_auto_replies_per_thread"] = None
            evidence.append(Evidence(
                fact="max_auto_replies_per_thread", value=None,
                source="proc.bp_policy(email_reply_autonomy).rules.max_auto_replies_per_thread",
                reference=authority.get("reason")))
            return _escalate(
                f"No governed cap on unattended replies per thread was resolved under "
                f"{policy_name or 'the autonomy policy'} "
                "(max_auto_replies_per_thread is missing), so there is nothing to stop "
                "the agent answering this thread indefinitely. An absent cap is not an "
                "unlimited one."
            )

        facts["auto_replies_on_thread"] = already
        evidence.append(Evidence(fact="auto_replies_on_thread", value=already,
                                 source="proc.bp_decision (prior sends on this thread)",
                                 reference=subject_id))
        if already is None:
            return _escalate(
                "How many times the agent has already answered this thread "
                f"unattended could not be counted, so the governed cap of {cap} "
                "cannot be enforced. An unknown history is not an empty one."
            )
        if int(already) >= int(cap):
            return _escalate(
                f"The agent has already answered this thread {int(already)} time(s) "
                f"unattended, at the governed cap of {cap}. A human takes it from here."
            )

        return Decision(
            subject_type=self._EMAIL_SUBJECT_TYPE, subject_id=subject_id,
            decision="send", resolution=RESOLVED,
            rationale=(
                f"'{intent.intent}' is on the governed auto-reply list under "
                f"{policy_name or 'the autonomy policy'}, the supporting sentence is "
                f"verbatim from the supplier's reply, confidence is "
                f"{intent.confidence:.2f}, and nothing exceeds the governed limit."
            ),
            policy_id=policy_id, policy_name=policy_name,
            facts=facts, evidence=evidence,
            supplier_id=row.get("supplier_id"),
        )

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
