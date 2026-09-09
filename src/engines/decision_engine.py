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
import uuid
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

    # Where the draft records which negotiation round its counter belongs to. Verified
    # live on all 18 rows: `metadata.round` is present (and equals the top-level
    # `round`). metadata is preferred because it is the same object as counter_price, so
    # it describes THAT counter rather than the draft in general.
    _PRIOR_ROUND_PATHS = (("metadata", "round"), ("round",))

    def _prior_offer_round(self, payload: Any) -> Optional[Decimal]:
        """Which round the prior offer belongs to, or None if the draft does not say."""
        if not isinstance(payload, dict):
            return None
        for path in self._PRIOR_ROUND_PATHS:
            cursor: Any = payload
            for key in path:
                if not isinstance(cursor, dict):
                    cursor = None
                    break
                cursor = cursor.get(key)
            value = self._num(cursor)
            if value is not None and value.is_finite():
                return value
        return None

    def _prior_offer(self, payload: Any) -> tuple[Optional[Decimal], Optional[str]]:
        """The price WE last put to this supplier, and where it came from.

        Read only from `payload->'metadata'->>'counter_price'`. Deliberately narrow:

        * Never the supplier's own number. Using their figure as "ours" would make
          every reply show nothing at stake -- the one wrong answer that turns this
          gate off silently.
        * Never parsed out of the body prose. The £96,000 is also written in the
          draft's HTML, which is how the structured key was corroborated, but prose
          is not a source of record and a regex over it would invent precision.
        * Missing, non-numeric, non-finite or unparseable is ABSENT, not zero. It
          returns (None, None) and the caller escalates on the missing-prior gate,
          exactly as it did before this was wired. Non-finite matters as much as
          non-numeric: a Decimal('NaN') prior would survive to `at_stake > limit` and
          raise InvalidOperation there instead of escalating.

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
        if value is None or not value.is_finite():
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
                    record["prior_price_round"] = self._prior_offer_round(
                        record.get("draft_payload")
                    )
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

        Never raises, which is the same property `_fetch_email_reply` and `_classify`
        already have. The gates coerce values out of the authority block
        (`float(min_intent_confidence)`, `int(max_auto_replies_per_thread)`), and while
        `resolve_authority` cannot produce a block that breaks them, Task 7 hands this
        method an authority dict over HTTP. A malformed one must produce an escalation
        that says so, not a 500 -- an exception on the send path is the one outcome that
        is neither a send nor a decision anybody can act on.

        `facts`/`evidence` are owned HERE and handed down, not built inside
        `_decide_email_reply`, so an internal failure still persists whatever was
        gathered before it. They used to be locals of the inner method, which made the
        guard's escalation the one kind a human could not re-derive: `trace()` showed a
        rationale and nothing else. The inner method mutates these in place.
        """
        facts: Dict[str, Any] = {}
        evidence: List[Evidence] = []
        try:
            return self._decide_email_reply(
                response_id, authority=authority, requested=requested,
                facts=facts, evidence=evidence,
            )
        except Exception as exc:  # noqa: BLE001 - fail CLOSED, and say what broke
            # The exception type, its message and the traceback go to the LOG, under a
            # short reference that also appears in the rationale -- so the failure is
            # fully diagnosable without a database driver's text (which routinely names
            # tables and columns) rendering on a buyer's review screen. The reference is
            # what connects the two.
            ref = uuid.uuid4().hex[:8]
            logger.exception(
                "email reply decision failed for %s [ref %s]: %s: %s",
                response_id, ref, type(exc).__name__, exc,
            )
            return Decision(
                subject_type=self._EMAIL_SUBJECT_TYPE,
                subject_id=str(response_id),
                decision="escalate",
                resolution=ESCALATED,
                rationale=(
                    f"Deciding supplier reply {response_id} did not finish, so nothing "
                    "was sent and it needs a person. The technical details were recorded "
                    f"for support under reference {ref}."
                ),
                policy_id=(authority or {}).get("policy_id"),
                policy_name=(authority or {}).get("policy_name"),
                # Whatever was gathered before the failure, so this escalation is
                # re-derivable like every other one. Empty when the failure happened
                # before the first fact was read -- which is itself a true statement.
                facts=facts, evidence=evidence,
                # From the row if it was read at all; never guessed. deal_id stays None
                # for the reason given in `_escalate` below.
                supplier_id=facts.get("supplier_id"),
            )

    def _decide_email_reply(
        self,
        response_id: str,
        *,
        authority: Optional[Dict[str, Any]] = None,
        requested: Optional[str] = None,
        facts: Optional[Dict[str, Any]] = None,
        evidence: Optional[List[Evidence]] = None,
    ) -> Decision:
        """The decision itself. See `decide_email_reply` for the no-raise guarantee.

        Deterministic in the part that matters: the model contributes an intent label
        and a quoted sentence, and every gate below is arithmetic and set membership
        over governed values. No model is asked whether to send.

        `facts`/`evidence` are supplied by `decide_email_reply` and mutated in place, so
        its never-raises guard can persist what was gathered before a failure. Defaulted
        here so calling this method directly still works.
        """
        if facts is None:
            facts = {}
        if evidence is None:
            evidence = []

        row = self._fetch_email_reply(response_id)
        if not row:
            return Decision(
                subject_type=self._EMAIL_SUBJECT_TYPE,
                subject_id=str(response_id),
                decision="escalate",
                resolution=ESCALATED,
                rationale=(
                    # Plain English on purpose: this string is rendered verbatim on the
                    # Action Centre card, and the name of a storage table means nothing
                    # to the person reading it (while telling anyone else more about our
                    # internals than a review screen should).
                    f"Supplier reply {response_id} could not be found in our records, "
                    "so there are no facts to decide on."
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
            # Plain English, no slug: this string is interpolated into the rationale
            # below and rendered verbatim on the buyer's card. The same rule
            # `resolve_authority`'s own reasons follow.
            reason = (authority or {}).get("reason") or (
                "no authority to answer replies unattended was resolved from governed "
                "policy"
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
        evidence.append(Evidence(
            fact="intent_confidence", value=intent.confidence,
            source="AgentNick classification of supplier_response.response_text "
                   "(self-reported confidence)",
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
        # Recorded whether or not it fires, like value_limit_gbp below: a send that
        # cleared a governed minimum is only re-derivable if the record says what the
        # minimum was, not merely what the model claimed.
        min_conf = authority.get("min_intent_confidence")
        facts["min_intent_confidence"] = None if min_conf is None else float(min_conf)
        evidence.append(Evidence(
            fact="min_intent_confidence",
            value=facts["min_intent_confidence"],
            source="proc.bp_policy(email_reply_autonomy).rules.min_intent_confidence",
            reference=(authority.get("reason") if min_conf is None
                       else (str(policy_id) if policy_id is not None else None))))
        if min_conf is None:
            # Plain English, and still re-derivable: it names the policy and says which
            # of its settings is missing, in words rather than as a config key. The
            # person reading this screen is not the person who edits the policy file.
            return _escalate(
                f"{policy_name or 'The autonomy policy'} sets no minimum confidence for "
                f"answering a reply unattended, so how sure the agent is about this one "
                f"({intent.confidence:.2f}) cannot be tested against anything. A missing "
                "minimum is not a minimum of zero."
            )
        if intent.confidence < float(min_conf):
            return _escalate(
                f"How sure the agent is that this reply is about '{intent.intent}' is "
                f"{intent.confidence:.2f}, below the minimum of {float(min_conf):.2f} "
                f"that {policy_name or 'the autonomy policy'} requires before a reply "
                "may be answered unattended."
            )

        # 3. Governed intent lists.
        if intent.intent in (authority.get("escalate_intents") or []):
            return _escalate(
                f"'{intent.intent}' is a kind of reply that "
                f"{policy_name or 'the autonomy policy'} always puts in front of a "
                "person, whatever it says: a human decides this one."
            )
        if intent.intent not in (authority.get("auto_intents") or []):
            return _escalate(
                f"'{intent.intent}' is not one of the kinds of reply the agent is allowed "
                f"to answer on its own under {policy_name or 'the autonomy policy'}, so "
                "it goes to a human. Adding it to that policy's list of replies the agent "
                "may answer unattended would change that."
            )

        # 4. Value at stake against the governed spend limit. Derived, and the
        #    derivation is stated: two numbers, both cited above.
        #
        #    Order matters, and this is the order:
        #      4.  a price that is not a number? yes -> escalate (junk != no money)
        #      4.  no price, but WE offered one? yes -> escalate (uncomputable)
        #      4.  is there a price at all?      no  -> no money moves, skip the block
        #      4.  a governed limit?             no  -> escalate (absent != unlimited)
        #      4a. the same denomination?        no  -> escalate (never convert)
        #      4b. a prior offer to move from?   no  -> escalate (absent != unchanged)
        #      4c. the round it answers?         differs -> escalate (uncomputable)
        #      4d. abs(price - prior) > limit?   yes -> escalate
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
        raw_price = row.get("price")
        price = self._num(raw_price)
        prior = self._num(row.get("prior_price"))
        limit = self._num(authority.get("limit_gbp"))
        reply_currency = str(row.get("currency") or "").strip()
        limit_tested: Optional[str] = None   # set when an amount really was compared

        # A price we cannot read is not an absence of price. `_num` returns None for
        # junk ("TBC", "circa 90k") and accepts Decimal('NaN'), which would then raise
        # InvalidOperation on the comparison below instead of escalating. Both are money
        # we cannot reason about, so both stop here rather than falling through to the
        # `price is None` path, which means "nothing priced".
        if raw_price is not None and (price is None or not price.is_finite()):
            facts["price_unreadable"] = str(raw_price)[:200]
            evidence.append(Evidence(
                fact="price_unreadable", value=str(raw_price)[:200],
                source="proc.supplier_response.price", reference=ref))
            return _escalate(
                f"The reply records a price of '{str(raw_price)[:80]}', which is not a "
                "finite number, so no amount can be derived from it. An unreadable price "
                "is not the same as no price: a human should read this one."
            )

        # No price extracted, but WE put a number to them. `price` is an upstream
        # extraction outcome, not proof the email mentions no money -- and a reply to a
        # priced offer is about that price whether or not extraction caught it. Same
        # reasoning as the price-without-prior gate, in the other direction.
        if price is None and prior is not None:
            facts["prior_offer"] = str(prior)
            evidence.append(Evidence(
                fact="prior_offer", value=str(prior),
                source=row.get("prior_price_source") or self._prior_offer_source,
                reference=ref))
            return _escalate(
                f"No price was extracted from this reply, but our own last offer on the "
                f"thread was {prior}, so the reply may well answer it and the amount at "
                "stake cannot be computed. A human should read what they actually said."
            )

        if price is not None:
            priced = f"{price} {reply_currency}".strip()
            if limit is None:
                facts["value_limit_gbp"] = None
                evidence.append(Evidence(
                    fact="value_limit_gbp", value=None,
                    source="proc.bp_policy(email_reply_autonomy).rules.defer_value_limit_to",
                    reference=authority.get("reason")))
                return _escalate(
                    f"The reply carries a price of {priced} but "
                    f"{policy_name or 'the autonomy policy'} points at no approval "
                    "threshold for amounts, so there is no limit to test it against. An "
                    "absent limit is not an unlimited one, so a human decides."
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
                    f"against it, while the limit of {limit} set by "
                    f"{policy_name or 'the autonomy policy'} is denominated in "
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
                    f"The reply is priced in {reply_currency} but the limit of {limit} "
                    f"set by {policy_name or 'the autonomy policy'} is denominated in "
                    f"{limit_currency}. No comparison was "
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

            # 4c. Is that our offer to THIS reply? The draft is matched on `unique_id`,
            #     which is workflow+supplier and not round-specific, so a multi-round
            #     thread has several drafts sharing it and the newest is not necessarily
            #     the one this reply answers. If round N+1 was dispatched before the
            #     round-N reply landed, subtracting the newer offer computes a move the
            #     supplier never saw -- and that error can run permissive.
            #
            #     Match when both sides state a round; escalate as uncomputable when they
            #     state different ones; DISCLOSE when either side states none. Not a
            #     blanket escalate: that would put the limit gate back to never
            #     executing, which is the outcome rejected in round 3.
            #
            #     Live 2026-07-28: all 18 draft rows record metadata.round (and a
            #     top-level `round`), and supplier_response.round_number is populated, so
            #     the MATCHING branch is the live path -- for reply 1 both state round 1.
            reply_round = self._num(row.get("round_number"))
            draft_round = self._num(row.get("prior_price_round"))
            if reply_round is not None and draft_round is not None:
                facts["prior_offer_round"] = str(draft_round)
                evidence.append(Evidence(
                    fact="prior_offer_round", value=str(draft_round),
                    source="proc.draft_rfq_emails.payload.metadata.round",
                    reference=ref))
                if reply_round != draft_round:
                    return _escalate(
                        f"The reply is round {reply_round}, but the last offer recorded "
                        f"on the thread ({prior}) belongs to round {draft_round}, so it "
                        "is not the offer this reply answers and the amount at stake "
                        "cannot be computed against it. A human should compare the "
                        "reply with the offer it actually replies to."
                    )
            else:
                missing = ("the reply" if reply_round is None else "the draft")
                basis = f"unverified; {missing} states no round"
                facts["prior_offer_round_basis"] = basis
                evidence.append(Evidence(
                    fact="prior_offer_round_basis", value=basis,
                    source=(
                        "ASSUMPTION: the prior offer is the most recent draft on this "
                        "thread (proc.draft_rfq_emails matched on unique_id, ORDER BY "
                        "id DESC), because the round it belongs to could not be "
                        f"confirmed -- {missing} records none. It may not be the offer "
                        "this reply answers."
                    ),
                    reference=ref))

            at_stake = abs(price - prior)
            # DENOMINATED, not bare. `facts["currency"]` sits right beside this one, but
            # nothing forces a reader (or the Action Centre card, which renders this fact
            # directly) to put the two together -- and a bare "36500.0000 at stake" is a
            # sum of money with no unit. Gate 4a above has already proved the reply's
            # currency equals the limit's, so this is the denomination the comparison was
            # actually made in, not an assumed default.
            at_stake_stated = f"{at_stake} {reply_currency}"
            facts["value_at_stake"] = at_stake_stated
            evidence.append(Evidence(
                fact="value_at_stake", value=at_stake_stated,
                source=f"derived: abs(proc.supplier_response.price - {prior_source})",
                reference=ref))

            # DISCLOSED ASSUMPTION, not a gate. Gate 4a proved the REPLY's currency
            # matches the limit's. It cannot prove the same of the prior offer, because
            # `metadata` carries no currency key at all -- so one operand of the
            # subtraction above has an unstated denomination, and this subtraction
            # assumes it matches the reply's.
            #
            # This is recorded rather than gated deliberately: a gate here would fire on
            # every row (the currency is unstated on all of them), which would return
            # the limit gate to never executing. But "every fact carries a source" means
            # an assumed operand must be visible as assumed, or the decision is not
            # re-derivable from its evidence. It lands in proc.bp_decision.evidence and
            # shows up in trace(). The real fix is upstream -- the drafting agent
            # recording a currency alongside counter_price. See the Task 6 report.
            currency_basis = (
                f"assumed {reply_currency}; unstated in source"
            )
            facts["prior_offer_currency_basis"] = currency_basis
            evidence.append(Evidence(
                fact="prior_offer_currency_basis", value=currency_basis,
                source=(
                    f"ASSUMPTION: {prior_source} carries no currency key, so the prior "
                    f"offer's denomination is unstated in the source. It was assumed to "
                    f"match the reply's {reply_currency} "
                    f"(proc.supplier_response.currency). No conversion was applied."
                ),
                reference=ref))

            if at_stake > limit:
                # Both sides are in reply_currency, which gate 4a has already confirmed
                # equals limit_currency -- so no default denomination is assumed here.
                return _escalate(
                    f"The reply moves {at_stake} {reply_currency} "
                    f"(supplier {price} against our {prior}, whose currency is unstated "
                    f"in the source and assumed to be {reply_currency}), above the "
                    f"limit of {limit} {limit_currency} set by "
                    f"{policy_name or 'the autonomy policy'}."
                )
            # An amount really was compared. Only now may a send claim so.
            limit_tested = (
                f"the amount at stake is {at_stake} {reply_currency}, within the governed "
                f"limit of {limit} {limit_currency}"
            )

        # 5. Per-thread cap on unattended replies. Fail-closed on BOTH unknowns: an
        #    absent cap is not an unlimited one, and an uncounted history is not an
        #    empty one. Recorded whether or not it fires, like the limit and the
        #    confidence minimum: a send that stayed under a cap is only re-derivable if
        #    the record says what the cap was.
        cap = authority.get("max_auto_replies_per_thread")
        already = row.get("auto_replies_on_thread")
        facts["max_auto_replies_per_thread"] = None if cap is None else int(cap)
        evidence.append(Evidence(
            fact="max_auto_replies_per_thread",
            value=facts["max_auto_replies_per_thread"],
            source="proc.bp_policy(email_reply_autonomy).rules.max_auto_replies_per_thread",
            reference=(authority.get("reason") if cap is None
                       else (str(policy_id) if policy_id is not None else None))))
        if cap is None:
            return _escalate(
                f"{policy_name or 'The autonomy policy'} sets no limit on how many times "
                "the agent may answer one thread unattended, so there is nothing to stop "
                "it answering this one indefinitely. A missing cap is not an unlimited "
                "one."
            )

        facts["auto_replies_on_thread"] = already
        evidence.append(Evidence(fact="auto_replies_on_thread", value=already,
                                 source="proc.bp_decision (prior sends on this thread)",
                                 reference=subject_id))
        if already is None:
            return _escalate(
                "How many times the agent has already answered this thread "
                f"unattended could not be counted, so the cap of {cap} set by "
                f"{policy_name or 'the autonomy policy'} cannot be enforced. An unknown "
                "history is not an empty one."
            )
        if int(already) >= int(cap):
            return _escalate(
                f"The agent has already answered this thread {int(already)} time(s) "
                f"unattended, at the cap of {cap} set by "
                f"{policy_name or 'the autonomy policy'}. A human takes it from here."
            )

        return Decision(
            subject_type=self._EMAIL_SUBJECT_TYPE, subject_id=subject_id,
            decision="send", resolution=RESOLVED,
            rationale=(
                f"'{intent.intent}' is on the governed auto-reply list under "
                f"{policy_name or 'the autonomy policy'}, the supporting sentence is "
                f"verbatim from the supplier's reply, confidence is "
                f"{intent.confidence:.2f} against a governed minimum of "
                f"{float(min_conf):.2f}, and "
                # Only claim a limit was cleared if one was actually applied to an
                # amount. On the unpriced path no limit was consulted -- and there may
                # not even be one -- so saying "nothing exceeds the governed limit"
                # would assert a conclusion no evidence in this record supports.
                + (limit_tested if limit_tested else
                   "no price was recorded on the reply, so no amount was tested")
                # `already` counts the sends BEFORE this one, so the reply this record
                # describes is the next one: already + 1. The GATE above is correct and
                # caps sends at exactly `cap`; the sentence used to say "reply 0 of a
                # permitted 2", which understated an audit trail whose whole purpose is
                # re-derivability.
                + f". This is reply {int(already) + 1} of a permitted {int(cap)} on the thread."
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

    # ------------------------------------------------------------------
    # Email replies -- human action on an already-recorded decision
    # ------------------------------------------------------------------
    def _fetch_email_decision(self, decision_id: int) -> Optional[Dict[str, Any]]:
        """The previously recorded email-reply decision, exactly as persisted.

        Reads proc.bp_decision ONLY -- never proc.bp_extraction_discrepancy. That
        table (and `_fetch_finding` / `decide_finding` / `execute`, which read and
        write it) belongs to the extraction-findings path; an email decision_id has
        no row there at all, which is the whole reason this sibling method exists.

        Scoped to subject_type = 'email_reply' in SQL, so a finding's decision_id
        cannot be actioned through the email path by accident.
        """
        sql = """
            SELECT decision_id, subject_type, subject_id, deal_id, supplier_id,
                   decision, resolution, rationale, policy_id, policy_name,
                   facts, evidence
              FROM proc.bp_decision
             WHERE decision_id = %s AND subject_type = %s
        """
        try:
            with self.agent_nick.get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, (decision_id, self._EMAIL_SUBJECT_TYPE))
                    row = cur.fetchone()
                    if not row:
                        return None
                    cols = [d[0] for d in cur.description]
                    return dict(zip(cols, row))
        except Exception:
            logger.exception("failed to read email decision %s", decision_id)
            return None

    def _close_original_email_decision(self, decision_id: int, *, status: str) -> bool:
        """Mark the ORIGINAL escalated decision's queue entry closed.

        `_record_human_action` (unmodified) always INSERTs a new audit row; it
        never updates the row it is auditing. Left alone, that original row keeps
        `status = 'open'` forever, which is exactly what `GET /decisions` filters
        on by default -- so a decision a human has already sent or rejected would
        keep showing up in the Todo queue, looking untouched.

        Only `status` is written here. `rationale` / `facts` / `evidence` on the
        original row are left exactly as recorded: they describe what the AGENT
        decided and why, and that does not change because a human later acted on
        it. What the human did, when, and any override reason live on the separate
        row `_record_human_action` inserts -- this call only flips the lifecycle
        column that determines whether the ORIGINAL row still matches the queue.

        Uses the SAME status vocabulary `_record_human_action` already writes on
        that new row ('actioned' / 'overridden') rather than inventing a third
        value -- the caller passes whichever one applies.

        Scoped to `decision_id` AND `subject_type = 'email_reply'`, same as
        `_fetch_email_decision`, so this can never write to a findings row -- the
        findings screen's queue does not read this column and must not be touched.

        Returns True on success, False on failure (never raises) -- the caller
        must be able to tell a closed queue entry from one that silently stayed
        open.
        """
        try:
            with self.agent_nick.get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE proc.bp_decision
                           SET status = %s
                         WHERE decision_id = %s AND subject_type = %s
                        """,
                        (status, decision_id, self._EMAIL_SUBJECT_TYPE),
                    )
                conn.commit()
            return True
        except Exception:
            logger.exception(
                "failed to close the queue entry for email decision %s", decision_id
            )
            return False

    def act_on_email_reply(
        self,
        decision_id: int,
        action: str,
        *,
        user_id: str = "api",
        override_reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Record the human's send/reject on an already-decided email reply.

        This is the email-reply sibling of `execute()`, for a path `execute()`
        cannot serve: `execute()` and `decide_finding()` key off `finding_id` and
        read/write proc.bp_extraction_discrepancy, and an email decision has no row
        there. This method never touches that table -- it reads the decision back
        from proc.bp_decision via `_fetch_email_decision` (keyed by `decision_id`,
        which is all the queue in `list_decisions` hands the caller) and persists
        the human's action with the shared `_record_human_action`, the exact
        mechanism `execute()` uses for findings.

        Same human-in-the-loop convention as `execute()`, not a different one:
          * the stored decision already states what the evidence supported;
          * sending against a recommendation that was ESCALATED contradicts that,
            so it comes back with requires_override=True and the reasoning rather
            than proceeding on a bare click;
          * rejecting one does not conflict -- "do not send" is exactly what an
            escalation asks a human to weigh, so no override is required for it;
          * supplying override_reason proceeds and is recorded against the actor.
        The human is never blocked; they are asked to mean it.

        After the audit row is recorded, the ORIGINAL escalated decision's
        `status` is also closed (see `_close_original_email_decision`), so it
        stops matching `GET /decisions`'s queue filter. These are two separate
        writes on proc.bp_decision (through the same, unmodified
        `_record_human_action`, plus a new narrowly-scoped UPDATE) rather than one
        atomic transaction -- `_record_human_action` manages and commits its own
        connection internally, and touching that is out of scope here. The audit
        INSERT is done FIRST: losing the record of who acted and why would be
        worse than a queue entry that stays open one call longer. If the INSERT
        succeeds but the closing UPDATE fails, the response says so explicitly
        via `queue_closed: False` and a `warning` -- it never reports a plain
        success while the queue is left stale, silently.
        """
        action = (action or "").strip().lower()
        if action not in ("send", "reject"):
            return {
                "applied": False,
                "error": f"unknown action '{action}' (expected 'send' or 'reject')",
            }

        row = self._fetch_email_decision(decision_id)
        if not row:
            return {
                "applied": False,
                "error": f"email decision {decision_id} not found",
            }

        facts = row.get("facts") or {}
        if isinstance(facts, str):
            try:
                facts = json.loads(facts)
            except Exception:
                facts = {}

        raw_evidence = row.get("evidence") or []
        if isinstance(raw_evidence, str):
            try:
                raw_evidence = json.loads(raw_evidence)
            except Exception:
                raw_evidence = []
        evidence = [
            Evidence(
                fact=e.get("fact"),
                value=e.get("value"),
                source=e.get("source"),
                reference=e.get("reference"),
            )
            for e in raw_evidence
            if isinstance(e, dict)
        ]

        recommendation = Decision(
            subject_type=row.get("subject_type") or self._EMAIL_SUBJECT_TYPE,
            subject_id=row.get("subject_id"),
            decision=row.get("decision"),
            resolution=row.get("resolution") or ESCALATED,
            rationale=row.get("rationale") or "",
            policy_id=row.get("policy_id"),
            policy_name=row.get("policy_name"),
            facts=facts,
            evidence=evidence,
            deal_id=row.get("deal_id"),
            supplier_id=row.get("supplier_id"),
            decision_id=row.get("decision_id"),
        )

        # Same test as execute(): does the human's action contradict the evidence?
        # An escalated recommendation says "a human must decide"; rejecting (not
        # sending) is a human deciding not to, which agrees with it. Sending is the
        # one action that goes the other way.
        conflicts = action == "send" and recommendation.escalated

        if conflicts and not override_reason:
            return {
                "applied": False,
                "requires_override": True,
                "recommendation": recommendation.to_dict(),
                "prompt": (
                    "The evidence does not support sending here: "
                    f"{recommendation.rationale} You can still proceed, but the "
                    "reason will be recorded against your name."
                ),
            }

        new_decision_id = self._record_human_action(
            recommendation,
            human_action=action,
            actor=user_id,
            override_reason=override_reason if conflicts else None,
        )

        # Fix round 2, CRITICAL: the close must be GATED on the audit write
        # having actually succeeded. `_record_human_action` catches its own
        # exceptions and returns None on failure (never raises) -- if it failed,
        # there is nothing worth closing: closing the original row anyway would
        # be the worst available outcome -- the record of who acted is gone,
        # the task vanishes from the human's queue, and the caller is told it
        # worked. Leaving the original row open is correct here, because the
        # decision genuinely has not been dealt with.
        if new_decision_id is None:
            return {
                "applied": False,
                # Same discipline as the missing-reply rationale above: this reaches a
                # person, so it says what happened and what it means for them, without
                # naming where the record would have been written.
                "error": (
                    "The action could not be saved, so nothing was closed. The "
                    "decision remains in the queue."
                ),
                "recommendation": recommendation.to_dict(),
            }

        # Same status word `_record_human_action` just wrote on the NEW row --
        # the original row's queue entry is closed with the identical vocabulary,
        # not a third value.
        status_word = "overridden" if conflicts else "actioned"
        queue_closed = self._close_original_email_decision(decision_id, status=status_word)

        result: Dict[str, Any] = {
            "applied": True,
            "action": action,
            # Fix round 2: two DIFFERENT decision_ids were being returned under
            # the same ambiguous key at different response depths --
            # `decision_id` here (the NEW audit row) vs. `recommendation.
            # decision_id` (the ORIGINAL row `Decision.to_dict()` already emits,
            # unchanged -- that shape is shared with the findings path and is
            # not touched here). These two explicit names are now authoritative;
            # `decision_id` is kept ONLY as a backwards-compatible alias for
            # `audit_decision_id` and should not be read as "the" decision id.
            "audit_decision_id": new_decision_id,
            "original_decision_id": decision_id,
            "decision_id": new_decision_id,
            "overridden": bool(conflicts),
            "override_reason": override_reason if conflicts else None,
            "actioned_by": user_id,
            "recommendation": recommendation.to_dict(),
            "queue_closed": queue_closed,
        }
        if not queue_closed:
            # The action WAS recorded (new_decision_id is real audit trail) -- but
            # the caller must not read `applied: True` as "the queue is up to
            # date". Say plainly that it may not be.
            #
            # NOTE for callers: this is NOT safely retryable by re-POSTing the
            # same action. Re-POSTing re-runs `_record_human_action` and inserts
            # a SECOND audit row rather than retrying only the close. There is
            # currently no endpoint that retries just the close step.
            result["warning"] = (
                "Your action was recorded, but decision "
                f"{decision_id} could not be marked '{status_word}', so it may "
                "still appear in the escalation queue. Sending this action "
                "again is NOT a safe retry -- it records a second, separate "
                "audit entry rather than retrying only the part that failed."
            )
        return result


# ---------------------------------------------------------------------------
# Authorization deferrals
#
# The gate answers three ways: allow, deny, and "nobody's rule said". The third
# is not a verdict, it is a question -- and a question belongs in front of a
# person rather than being resolved by a default nobody chose.
#
# This is where those questions become visible. It writes the same
# proc.bp_decision row every other escalation uses, so the Action Centre lists
# them without a new endpoint: list_decisions already filters on subject_type
# and status='open'.
#
# It is a module-level function rather than a DecisionEngine method because the
# gate has no agent_nick and must not acquire one. What matters is that
# escalations are created in one place, and this is that place.
# ---------------------------------------------------------------------------

AUTHORIZATION_SUBJECT_TYPE = "authorization"


def escalate_authorization(
    *,
    action: str,
    action_class: Optional[str] = None,
    principal_subject: Optional[str] = None,
    role: Optional[str] = None,
    reason: str = "",
    policy_id: Optional[Any] = None,
    policy_name: Optional[str] = None,
    evidence: Optional[Dict[str, Any]] = None,
) -> Optional[int]:
    """Put an unresolved authorization in front of a person. Returns the id.

    Raised against the person who was stopped, so they can see why; resolving it
    is a policy judgement and belongs to whoever can make one.

    Never raises. A question that cannot be asked must not become permission --
    the caller keeps refusing either way, and the failure is logged loudly
    because an unasked question is a governance gap, not a hiccup.
    """

    try:
        from src.services.db import get_conn

        facts = {
            "action": action,
            "action_class": action_class,
            "role": role,
            "requested_by": principal_subject,
        }
        with get_conn() as conn:
            conn.autocommit = False
            cur = conn.cursor()
            try:
                cur.execute(
                    """
                    INSERT INTO proc.bp_decision (
                        subject_type, subject_id, decision, resolution,
                        rationale, policy_id, policy_name, facts, evidence,
                        status, agent, created_by
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    RETURNING decision_id
                    """,
                    (
                        AUTHORIZATION_SUBJECT_TYPE,
                        action,
                        "clarify_policy",
                        ESCALATED,
                        reason,
                        None,
                        policy_name,
                        json.dumps(facts, default=str),
                        json.dumps(evidence or {}, default=str),
                        "open",
                        "guardrail",
                        principal_subject or "system",
                    ),
                )
                row = cur.fetchone()
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        return int(row[0]) if row else None
    except Exception as exc:  # noqa: BLE001 - an unasked question is not consent
        logger.error(
            "could not raise an authorization question for %s: %s -- the action "
            "stays refused, but nobody has been asked about it",
            action,
            exc,
        )
        return None
