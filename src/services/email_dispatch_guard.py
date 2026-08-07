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

from src.services import approval_store, email_sensitivity, guardrail

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
) -> List[Dict[str, Any]]:
    """Quote totals on this deal belonging to suppliers other than the recipient.

    Scoped to the deal so the comparison is against genuine competitors on the
    same requirement. An empty result means "no competing quote on file", which
    is a real answer -- but it is also what a broken lookup returns, so failures
    are logged loudly rather than swallowed.

    The draft itself carries no deal_id (``proc.draft_rfq_emails`` has no such
    column); the caller must resolve the deal from the approval row instead
    and pass it in here.
    """

    if hasattr(conn, "lookup_peer_prices"):
        return list(conn.lookup_peer_prices(deal_id, supplier_id) or [])
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
            "peer price lookup failed for deal %s: %s -- the third_party_price "
            "detector cannot fire for this message",
            deal_id,
            exc,
        )
        return []


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


def _rules(policy_engine: Optional[Any], slug: str) -> Dict[str, Any]:
    if policy_engine is None:
        return {}
    try:
        policy = policy_engine.get_policy(slug)
    except Exception:  # noqa: BLE001
        return {}
    if not isinstance(policy, dict):
        return {}
    details = policy.get("details")
    if not isinstance(details, dict):
        return {}
    rules = details.get("rules")
    return rules if isinstance(rules, dict) else {}


def check_dispatch(
    *,
    conn: Any,
    draft: Dict[str, Any],
    recipients: Optional[Iterable[str]],
    subject: Optional[str],
    body: Optional[str],
    attachments: Optional[Iterable[Any]],
    principal: Optional[Any],
    run_count: int = 0,
    policy_engine: Optional[Any] = None,
    approval_lookup: Optional[Callable[..., Optional[Dict[str, Any]]]] = None,
    internal_domains: Optional[Iterable[str]] = None,
    peer_prices: Optional[Iterable[Dict[str, Any]]] = None,
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
            return guardrail.Decision(
                allowed=False,
                reason="no recorded human approval for this draft",
                policy_name="EmailDispatchApprovalPolicy",
                evidence={"unique_id": draft.get("unique_id")},
            )

        # The draft has no deal_id; the approval does. Check 1 has already
        # fetched it by the time the classifier needs it.
        deal_id = approval.get("deal_id")

        # --- 2. Recipient allow-list -------------------------------------
        if not recipient_list:
            return guardrail.Decision(
                allowed=False,
                reason="no recipient survived allow-list resolution",
                policy_name="EmailRecipientAllowlistPolicy",
            )
        known = {
            str(a).casefold() for a in (_supplier_emails(conn, supplier_id) or [])
        }
        unknown = [r for r in recipient_list if r.casefold() not in known]
        if unknown:
            return guardrail.Decision(
                allowed=False,
                reason=f"recipient not on the supplier allow-list: {unknown[0]}",
                policy_name="EmailRecipientAllowlistPolicy",
                evidence={"unknown_recipients": unknown},
            )

        # --- 3. Sensitivity versus supplier clearance --------------------
        clearance = _supplier_clearance(conn, supplier_id)
        resolved_peers = (
            list(peer_prices)
            if peer_prices is not None
            else _peer_prices(conn, deal_id, supplier_id)
        )
        classification = email_sensitivity.classify(
            subject=subject,
            body=body,
            attachments=attachments,
            recipient_supplier_id=supplier_id,
            peer_prices=resolved_peers,
            internal_domains=internal_domains or [],
            policy_engine=policy_engine,
        )
        if not email_sensitivity.clearance_permits(
            classification.content_class, clearance, policy_engine=policy_engine
        ):
            return guardrail.Decision(
                allowed=False,
                reason=(
                    f"content is {classification.content_class}; supplier clearance "
                    f"is {clearance or 'default'}"
                ),
                policy_name="EmailSensitivityPolicy",
                evidence={
                    "detectors_fired": classification.detectors_fired,
                    "detector_evidence": classification.evidence,
                },
            )

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
            return guardrail.Decision(
                allowed=False,
                reason=f"volume cap reached: {run_count}/{max_per_run} for this run",
                policy_name="EmailVolumePolicy",
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
                "content_class": classification.content_class,
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
