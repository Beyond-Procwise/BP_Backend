"""The five checks every outbound message passes before it reaches SES.

Order matters and is deliberate: cheapest and most decisive first. An
unapproved draft is refused before anything is classified, and no message is
classified for a recipient that is not on the supplier master.

The stored draft is authoritative for recipients. A caller may narrow that
list; it may never add to it. Accepting caller-supplied addresses is the
specific hole this closes.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Iterable, List, Optional

from src.services import approval_store, email_sensitivity, guardrail, rbac

logger = logging.getLogger(__name__)


class DispatchDenied(PermissionError):
    """Raised by the send path when a guard refuses the message."""

    def __init__(self, decision: guardrail.Decision) -> None:
        super().__init__(decision.reason)
        self.decision = decision


def _supplier_emails(conn: Any, supplier_id: Optional[str]) -> List[str]:
    """Addresses on the supplier master for this supplier.

    A test connection may answer directly; a real one is queried.
    """

    if hasattr(conn, "lookup_supplier_emails"):
        return list(conn.lookup_supplier_emails(supplier_id) or [])
    if not supplier_id:
        return []
    cur = conn.cursor()
    cur.execute(
        "SELECT contact_email_1, contact_email_2 FROM proc.bp_supplier "
        "WHERE supplier_id = %s",
        (supplier_id,),
    )
    out: List[str] = []
    for row in cur.fetchall():
        for value in row:
            text = str(value or "").strip()
            if text:
                out.append(text)
    return out


def _supplier_clearance(conn: Any, supplier_id: Optional[str]) -> Optional[str]:
    """This supplier's clearance level, or None to use the policy default."""

    if hasattr(conn, "lookup_supplier_clearance"):
        return conn.lookup_supplier_clearance(supplier_id)
    if not supplier_id:
        return None
    cur = conn.cursor()
    cur.execute(
        "SELECT clearance_level FROM proc.bp_supplier WHERE supplier_id = %s",
        (supplier_id,),
    )
    row = cur.fetchone()
    return row[0] if row else None


def _peer_prices(
    conn: Any, deal_id: Optional[str], supplier_id: Optional[str]
) -> Optional[List[Dict[str, Any]]]:
    """Quote totals on this deal belonging to suppliers other than the recipient.

    Scoped to the deal so the comparison is against genuine competitors on the
    same requirement. ``[]`` means "no competing quote on file" -- a real,
    proceed-able answer. ``None`` means the lookup itself could not be
    performed, and the caller must deny rather than silently read that the
    same way: an empty peer list because ``bp_quote_trgt`` is unreachable is
    not the same fact as an empty peer list because there genuinely is no
    competing quote, and collapsing them together downgraded a broken query
    into a pass on the highest-value leak this detector exists to catch.

    The draft itself carries no deal_id (``proc.draft_rfq_emails`` has no such
    column); the caller must resolve the deal from the approval row instead
    and pass it in here. No deal_id is treated as a genuine "nothing to
    compare against" rather than a failure -- there is nothing to look up.
    """

    if hasattr(conn, "lookup_peer_prices"):
        try:
            result = conn.lookup_peer_prices(deal_id, supplier_id)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "peer price lookup failed for deal %s: %s -- denying rather "
                "than treating this deal as having no competing quotes",
                deal_id,
                exc,
            )
            return None
        return list(result or [])
    if not deal_id:
        return []
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT supplier_id, total_amount FROM proc.bp_quote_trgt "
            "WHERE deal_id = %s AND supplier_id IS DISTINCT FROM %s "
            "AND total_amount IS NOT NULL",
            (deal_id, supplier_id),
        )
        return [{"supplier_id": r[0], "amount": r[1]} for r in cur.fetchall()]
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "peer price lookup failed for deal %s: %s -- denying rather than "
            "treating this deal as having no competing quotes",
            deal_id,
            exc,
        )
        return None


def _daily_send_count(conn: Any, principal_subject: Optional[str]) -> Optional[int]:
    """Sends already allowed for this principal in the last 24 hours.

    ``None`` means the count could not be determined -- no principal to
    count against, or the lookup itself failed -- and the caller must treat
    that as a denial. An unenforceable cap is not an absent cap.
    """

    if not principal_subject:
        return None
    if hasattr(conn, "lookup_daily_send_count"):
        try:
            count = conn.lookup_daily_send_count(principal_subject)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "daily send count lookup failed for %s: %s", principal_subject, exc
            )
            return None
        return int(count) if count is not None else None
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT COUNT(*) FROM proc.bp_agent_actions "
            "WHERE action_type = 'email.send' AND status = 'allowed' "
            "AND details ->> 'principal' = %s "
            "AND created_at >= NOW() - INTERVAL '24 hours'",
            (principal_subject,),
        )
        row = cur.fetchone()
        return int(row[0]) if row else 0
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "daily send count lookup failed for %s: %s", principal_subject, exc
        )
        return None


def _normalise(values: Optional[Iterable[Any]]) -> List[str]:
    out: List[str] = []
    for value in values or []:
        text = str(value or "").strip()
        if text and text not in out:
            out.append(text)
    return out


def resolve_recipients(
    draft: Dict[str, Any], requested: Optional[Iterable[Any]]
) -> List[str]:
    """Recipients for this send: the stored draft's list, optionally narrowed."""

    stored = _normalise(draft.get("recipients"))
    if not stored and draft.get("receiver"):
        stored = _normalise([draft.get("receiver")])
    if requested is None:
        return stored
    asked = {r.casefold() for r in _normalise(requested)}
    return [r for r in stored if r.casefold() in asked]


def _policy(policy_engine: Optional[Any], slug: str) -> Optional[Dict[str, Any]]:
    """The named policy's full ``PolicyEngine.get_policy`` row, or ``None``.

    Used to attribute a denial's ``policy_id``/``policy_version`` (spec G8),
    not just the hardcoded display name every check used to carry alone.
    """

    engine = policy_engine if policy_engine is not None else rbac.policy_engine()
    if engine is None:
        return None
    try:
        policy = engine.get_policy(slug)
    except Exception:  # noqa: BLE001
        return None
    return policy if isinstance(policy, dict) else None


def _rules(policy_engine: Optional[Any], slug: str) -> Dict[str, Any]:
    # None means "resolve the real thing", not "no rules exist" -- exactly
    # like rbac.authorize and email_sensitivity.classify already do for the
    # same argument. `check_dispatch` never receives a policy_engine from any
    # production caller, so treating None as empty here made the volume cap
    # (both max_per_run and max_per_user_per_day) permanently unenforceable:
    # every send_draft call would read {} and skip the cap regardless of what
    # bp_policy actually declares.
    engine = policy_engine if policy_engine is not None else rbac.policy_engine()
    if engine is None:
        return {}
    try:
        policy = engine.get_policy(slug)
    except Exception:  # noqa: BLE001
        return {}
    if not isinstance(policy, dict):
        return {}
    details = policy.get("details")
    if not isinstance(details, dict):
        return {}
    rules = details.get("rules")
    return rules if isinstance(rules, dict) else {}


def check_recipient_and_sensitivity(
    *,
    conn: Any,
    supplier_id: Optional[str],
    recipients: Optional[Iterable[str]],
    subject: Optional[str],
    body: Optional[str],
    attachments: Optional[Iterable[Any]] = None,
    sender: Optional[str] = None,
    deal_id: Optional[str] = None,
    internal_domains: Optional[Iterable[str]] = None,
    peer_prices: Optional[Iterable[Dict[str, Any]]] = None,
    policy_engine: Optional[Any] = None,
) -> guardrail.Decision:
    """Checks 2 and 3 of the five: recipient allow-list, then content
    sensitivity versus supplier clearance.

    Factored out so a caller with no approval workflow of its own -- the
    value-summary supplier-query send path, which is not a governed RFQ
    dispatch and so has no ``bp_approval`` row to check -- can still run the
    same allow-list and sensitivity gate ``check_dispatch`` runs, rather than
    a second, drifting implementation of the same two checks. On allow,
    ``evidence["content_class"]`` carries the resolved classification for
    the caller's own audit row.
    """

    try:
        recipient_list = _normalise(recipients)
        if not recipient_list:
            return guardrail.deny_from_policy(
                "no recipient survived allow-list resolution",
                _policy(policy_engine, "email_recipient_allowlist"),
                policy_name="EmailRecipientAllowlistPolicy",
            )
        known = {
            str(a).casefold() for a in (_supplier_emails(conn, supplier_id) or [])
        }
        unknown = [r for r in recipient_list if r.casefold() not in known]
        if unknown:
            return guardrail.deny_from_policy(
                f"recipient not on the supplier allow-list: {unknown[0]}",
                _policy(policy_engine, "email_recipient_allowlist"),
                policy_name="EmailRecipientAllowlistPolicy",
                unknown_recipients=unknown,
            )

        clearance = _supplier_clearance(conn, supplier_id)
        resolved_peers = (
            list(peer_prices)
            if peer_prices is not None
            else _peer_prices(conn, deal_id, supplier_id)
        )
        if resolved_peers is None:
            # The lookup itself failed -- not "no competing quotes". Reading
            # a broken bp_quote_trgt query as an empty peer list would
            # silently downgrade the sensitivity check to a pass on the
            # highest-value leak this detector exists to catch.
            return guardrail.deny_from_policy(
                "could not determine competing quotes for this deal; denying "
                "rather than classifying content unmetered",
                _policy(policy_engine, "email_sensitivity"),
                policy_name="EmailSensitivityPolicy",
                deal_id=deal_id,
            )
        classification = email_sensitivity.classify(
            subject=subject,
            body=body,
            attachments=attachments,
            recipient_supplier_id=supplier_id,
            peer_prices=resolved_peers,
            internal_domains=internal_domains or [],
            sender=sender,
            policy_engine=policy_engine,
        )
        if not email_sensitivity.clearance_permits(
            classification.content_class, clearance, policy_engine=policy_engine
        ):
            return guardrail.deny_from_policy(
                f"content is {classification.content_class}; supplier clearance "
                f"is {clearance or 'default'}",
                _policy(policy_engine, "email_sensitivity"),
                policy_name="EmailSensitivityPolicy",
                detectors_fired=classification.detectors_fired,
                detector_evidence=classification.evidence,
            )

        return guardrail.Decision(
            allowed=True,
            reason="recipient allow-list and content sensitivity checks passed",
            evidence={
                "recipients": recipient_list,
                "content_class": classification.content_class,
            },
        )
    except Exception as exc:  # noqa: BLE001 - a broken guard is a closed guard
        logger.error(
            "email_dispatch_guard.check_recipient_and_sensitivity failed: %s", exc
        )
        return guardrail.Decision(
            allowed=False,
            reason="dispatch guard failed; denying",
            evidence={"error": str(exc)},
        )


def check_dispatch(
    *,
    conn: Any,
    draft: Dict[str, Any],
    recipients: Optional[Iterable[str]],
    subject: Optional[str],
    body: Optional[str],
    attachments: Optional[Iterable[Any]],
    principal: Optional[Any],
    sender: Optional[str] = None,
    run_count: int = 0,
    policy_engine: Optional[Any] = None,
    approval_lookup: Optional[Callable[..., Optional[Dict[str, Any]]]] = None,
    internal_domains: Optional[Iterable[str]] = None,
    peer_prices: Optional[Iterable[Dict[str, Any]]] = None,
    agent_name: Optional[str] = None,
    intent: Optional[str] = None,
    authority_lookup: Optional[Callable[[str], Dict[str, Any]]] = None,
) -> guardrail.Decision:
    """Run the five checks. Returns a Decision; never raises."""

    try:
        supplier_id = draft.get("supplier_id")
        recipient_list = _normalise(recipients)

        # --- 1. Approval, verified against the store ---------------------
        lookup = approval_lookup or approval_store.find_dispatch_approval
        approval = lookup(
            rfq_id=draft.get("rfq_id"),
            workflow_id=draft.get("workflow_id"),
            unique_id=draft.get("unique_id"),
            conn=conn,
        )
        if not approval:
            # No human approved this, so it is agent-initiated. That question
            # is already governed by EmailReplyAutonomyPolicy via
            # resolve_authority, which the orchestrator and decision engine
            # consult -- use it rather than adding a third mechanism.
            if not agent_name:
                return guardrail.deny_from_policy(
                    "no recorded human approval for this draft",
                    _policy(policy_engine, "email_dispatch_approval"),
                    policy_name="EmailDispatchApprovalPolicy",
                    unique_id=draft.get("unique_id"),
                )
            try:
                if authority_lookup is not None:
                    verdict = authority_lookup(agent_name)
                else:
                    from src.services.governance_tools.authority import (
                        resolve_authority,
                    )

                    engine = policy_engine or rbac.policy_engine()
                    verdict = (resolve_authority(engine, [agent_name]) or {}).get(
                        agent_name
                    ) or {}
            except Exception as exc:  # noqa: BLE001 - unresolvable authority denies
                logger.error("authority lookup failed for %s: %s", agent_name, exc)
                return guardrail.Decision(
                    allowed=False,
                    reason="agent send authority could not be resolved; denying",
                    evidence={"error": str(exc)},
                )

            # governed=False means escalate. Otherwise autonomy exists only
            # for a named intent in auto_intents -- which is empty on the
            # live policy, so nothing is autonomous today. A plain dispatch
            # carries no intent and therefore never matches, which is the
            # wanted outcome: an agent must not send unprompted outbound
            # mail.
            auto_intents = {str(i) for i in (verdict.get("auto_intents") or [])}
            permitted = bool(
                verdict.get("governed") and intent and str(intent) in auto_intents
            )
            if not permitted:
                return guardrail.Decision(
                    allowed=False,
                    reason=(
                        "no human approval, and policy does not grant this "
                        "agent autonomy to send"
                    ),
                    policy_name="EmailReplyAutonomyPolicy",
                    evidence={
                        "agent": agent_name,
                        "intent": intent,
                        "governed": verdict.get("governed"),
                        "auto_intents": sorted(auto_intents),
                        "reason": verdict.get("reason"),
                    },
                )
            # Autonomy granted: there is no approval to bind to, so the
            # content-hash check below must be skipped rather than compared
            # against nothing.
            approval = {"autonomous": True, "agent": agent_name}

        # --- 1b. The approval covers the email that was approved -----------
        # Without this an approval is standing permission on a mutable object:
        # approve a routine RFQ, edit the body to carry a competitor's price,
        # and the original approval still releases it.
        #
        # Skipped when autonomy was granted above: there is no approval row,
        # and therefore no approved hash to compare the current draft against.
        if not approval.get("autonomous"):
            from src.services.approval_content import content_hash

            grounding = approval.get("grounding")
            approved_hash = (
                grounding.get("content_hash") if isinstance(grounding, dict) else None
            )
            mismatch_mode = str(
                (_rules(policy_engine, "email_dispatch_approval") or {}).get(
                    "on_content_mismatch"
                )
                or "deny"
            ).lower()
            # Hash what is actually about to be transmitted -- recipient_list,
            # subject, body and attachments are this call's own resolved
            # values, already reflecting any subject_override/body_override
            # the caller supplied. Hashing `draft` (the stored row) instead,
            # as this used to, let an override sail through unchecked: the
            # stored row never changed, so its approved hash still matched.
            # That was verbatim the attack this check exists to prevent.
            current_hash = content_hash(
                {
                    "recipients": recipient_list,
                    "subject": subject,
                    "body": body,
                    "attachments": attachments,
                }
            )
            if approved_hash != current_hash and mismatch_mode != "warn":
                return guardrail.Decision(
                    allowed=False,
                    reason=(
                        "the draft changed since it was approved; it must be "
                        "approved again"
                    ),
                    policy_name="EmailDispatchApprovalPolicy",
                    evidence={
                        "approved_content_hash": approved_hash,
                        "current_content_hash": current_hash,
                    },
                )

        # The draft has no deal_id; the approval does. Check 1 has already
        # fetched it by the time the classifier needs it.
        deal_id = approval.get("deal_id")

        # --- 2 & 3. Recipient allow-list, then content sensitivity -------
        sensitivity_decision = check_recipient_and_sensitivity(
            conn=conn,
            supplier_id=supplier_id,
            recipients=recipient_list,
            subject=subject,
            body=body,
            attachments=attachments,
            sender=sender,
            deal_id=deal_id,
            internal_domains=internal_domains,
            peer_prices=peer_prices,
            policy_engine=policy_engine,
        )
        if not sensitivity_decision.allowed:
            return sensitivity_decision
        content_class = sensitivity_decision.evidence.get("content_class")

        # --- 4. Policy ----------------------------------------------------
        decision = guardrail.authorize(
            "email.send",
            "communicate",
            principal,
            {
                "unique_id": draft.get("unique_id"),
                "supplier_id": supplier_id,
                "recipients": recipient_list,
            },
            policy_engine=policy_engine,
        )
        if not decision.allowed:
            return decision

        # --- 5. Volume ----------------------------------------------------
        volume = _rules(policy_engine, "email_volume")
        try:
            max_per_run = int(volume.get("max_per_run"))
        except (TypeError, ValueError):
            max_per_run = None
        if max_per_run is not None and int(run_count) >= max_per_run:
            return guardrail.deny_from_policy(
                f"volume cap reached: {run_count}/{max_per_run} for this run",
                _policy(policy_engine, "email_volume"),
                policy_name="EmailVolumePolicy",
            )

        try:
            max_per_day = int(volume.get("max_per_user_per_day"))
        except (TypeError, ValueError):
            max_per_day = None
        if max_per_day is not None:
            principal_subject = getattr(principal, "subject", None)
            daily_count = _daily_send_count(conn, principal_subject)
            if daily_count is None:
                return guardrail.deny_from_policy(
                    "could not determine this user's daily send count; "
                    "denying rather than sending unmetered",
                    _policy(policy_engine, "email_volume"),
                    policy_name="EmailVolumePolicy",
                )
            if daily_count >= max_per_day:
                return guardrail.deny_from_policy(
                    f"daily volume cap reached: {daily_count}/{max_per_day} "
                    "for this user",
                    _policy(policy_engine, "email_volume"),
                    policy_name="EmailVolumePolicy",
                    principal=principal_subject,
                )

        return guardrail.Decision(
            allowed=True,
            reason="all dispatch checks passed",
            policy_id=decision.policy_id,
            policy_name=decision.policy_name,
            policy_version=decision.policy_version,
            evidence={
                "approval_id": approval.get("approval_id"),
                "approved_by": approval.get("actioned_by"),
                "content_class": content_class,
                "recipients": recipient_list,
            },
        )

    except Exception as exc:  # noqa: BLE001 - a broken guard is a closed guard
        logger.error("email_dispatch_guard.check_dispatch failed: %s", exc)
        return guardrail.Decision(
            allowed=False,
            reason="dispatch guard failed; denying",
            evidence={"error": str(exc)},
        )
