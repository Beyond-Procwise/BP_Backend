"""Recording that a human approved something.

The approver is ``principal.subject`` and nothing else. A previous version of
this surface accepted the approver's name from the request body over an
unauthenticated route, which let anyone forge a human approval -- so the body
here carries only WHAT is being approved, never WHO approved it. Do not add a
body field for the actor, and do not add a fallback when the principal is
absent: a fallback is the forgery, taken conditionally.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from api.auth import require_user
from api.endpoint_gate import require as gate
from src.services import approval_store, guardrail, rbac
from src.services.agent_actions import record_action_or_fail
from src.services.approval_content import content_hash
from src.services.draft_hydration import DRAFT_COLUMNS, hydrate_draft, resolve_effective_content

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/approvals", tags=["Approvals"])

_ACTION = "approval.email"
_ACTION_CLASS = "approve_email"
_CAPABILITY_POLICY_SLUG = "email_approval_capability"

# revoke_scope values EmailApprovalCapabilityPolicy may set. Unrecognised or
# absent always falls back to the middle, not the permissive, option -- see
# _revoke_scope.
_REVOKE_SCOPE_ANY = "any_approver"
_REVOKE_SCOPE_OWN_ONLY = "own_only"
_REVOKE_SCOPE_OWN_OR_HIGHER = "own_or_higher_rank"
_REVOKE_SCOPES = {_REVOKE_SCOPE_ANY, _REVOKE_SCOPE_OWN_ONLY, _REVOKE_SCOPE_OWN_OR_HIGHER}

# self_approval values EmailApprovalCapabilityPolicy may set. Absent, unreadable
# or unrecognised all mean deny -- see _self_approval_rule.
_SELF_APPROVAL_DENY = "deny"
_SELF_APPROVAL_ALLOW = "allow"
_SELF_APPROVAL_VALUES = {_SELF_APPROVAL_DENY, _SELF_APPROVAL_ALLOW}

# _required_role_rank's fail-restrictive sentinel: higher than any real
# role's rank can ever be, so `role_rank(role) > _RANK_DENY_ALL` is False for
# every role and own_or_higher_rank's override never fires. An unreadable
# policy or a missing required_role must deny the override, not admit it --
# see I8.
_RANK_DENY_ALL = 1 << 30


class ApproveRequest(BaseModel):
    """What is being approved. Deliberately carries no actor field."""

    reason: Optional[str] = None
    # The version the approver read (``content_hash`` from GET /approvals/drafts/...).
    # When given, an approval of a draft that has changed since is refused, so nobody
    # approves words they did not see. Absent, the draft as it stands is approved.
    content_hash: Optional[str] = None


class RevokeRequest(BaseModel):
    reason: Optional[str] = None


class FinalOfferDecisionRequest(BaseModel):
    # "accept" or "decline": what the person has decided about the supplier's final offer.
    decision: str
    # The version of the email the person was looking at. When given, a draft that has
    # changed since is refused, so a decision is never applied to words they did not see.
    content_hash: Optional[str] = None


def _load_draft(unique_id: str, conn: Any = None) -> Optional[Dict[str, Any]]:
    """The stored draft being approved, hydrated the same way the send path
    hydrates it, or ``None``.

    Reading only a handful of raw columns here (as this used to) is the C2
    bug: the send path hydrates payload-over-columns, so an approver could
    see and approve one subject/body while the send path would transmit a
    different one for the identical row -- the two never hashed the same
    thing and a genuinely approved draft could never pass its own
    content-binding check.
    """

    from src.services.db import get_conn

    sql = (
        "SELECT " + ", ".join(DRAFT_COLUMNS) + " "
        "  FROM proc.draft_rfq_emails "
        " WHERE unique_id = %s AND sent IS NOT TRUE "
        " ORDER BY id DESC LIMIT 1"
    )

    def _run(connection: Any) -> Optional[Dict[str, Any]]:
        cur = approval_store._dict_cursor(connection)
        cur.execute(sql, (unique_id,))
        row = cur.fetchone()
        return hydrate_draft(dict(row)) if row else None

    if conn is not None:
        return _run(conn)
    with get_conn() as own:
        return _run(own)


def _capability_policy(policy_engine: Any) -> Optional[Dict[str, Any]]:
    """The EmailApprovalCapabilityPolicy row itself, read once per caller.

    ``_revoke_scope``, ``_required_role_rank`` and the approval-recording
    code below all need facts from this same row; centralising the read
    means they can no longer disagree about whether it was readable.
    """

    try:
        policy = policy_engine.get_policy(_CAPABILITY_POLICY_SLUG) if policy_engine else None
    except Exception as exc:  # noqa: BLE001 - callers apply their own strict default
        logger.error("approvals: could not read %s: %s", _CAPABILITY_POLICY_SLUG, exc)
        return None
    return policy if isinstance(policy, dict) else None


def _capability_policy_db_id(policy_engine: Any) -> Optional[int]:
    """``proc.bp_policy.policy_id`` (bigint) for EmailApprovalCapabilityPolicy.

    ``guardrail.Decision.policy_id`` carries ``PolicyEngine``'s slug string
    (e.g. ``"email_approval_capability"``), which is a *different* value
    from this database column of the same name -- see
    ``approval_store.record_approval``'s module docstring. Passing the slug
    straight into ``bp_approval.policy_id`` (bigint) would fail the INSERT,
    so the real numeric id is read from the policy's own raw row instead.
    """

    policy = _capability_policy(policy_engine)
    raw_row = (policy or {}).get("raw_row") or {}
    value = raw_row.get("policy_id") if isinstance(raw_row, dict) else None
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _require_capability(
    principal: Any, context: Dict[str, Any], *, actioned_by: Optional[str] = None
) -> guardrail.Decision:
    """Deny unless policy grants this caller the approve_email capability.

    Every verdict -- allow or deny -- is written to ``bp_agent_actions``
    before returning (G8): this surface used to produce no audit row at
    all, for either outcome, which is invisible either way an auditor needs
    it. Returns the ``Decision`` so a caller that goes on to record an
    approval can attribute it to the policy that actually decided, rather
    than a hardcoded name that happened to belong to a different policy.
    """

    engine = rbac.policy_engine()
    decision = guardrail.authorize(_ACTION, _ACTION_CLASS, principal, context, policy_engine=engine)
    record_action_or_fail(
        phase="approve",
        action_type=_ACTION,
        agent="ApprovalsRouter",
        status="allowed" if decision.allowed else "denied",
        summary=decision.reason,
        details={
            **context,
            "principal": actioned_by,
            "policy_id": decision.policy_id,
            "policy_name": decision.policy_name,
            "policy_version": decision.policy_version,
            "decision": "allow" if decision.allowed else "deny",
            "evidence": decision.evidence,
            # Allowed or not: which groups the sign-in carried and any direct grant.
            # An allow that rests on a stop-gap grant looks the same as one that
            # rests on the person's real groups unless this says which it was.
            "role_from": rbac.role_basis(principal, policy_engine=engine),
        },
    )
    if not decision.allowed:
        raise HTTPException(status_code=403, detail=decision.reason)
    return decision


def _self_approval_rule(policy_engine: Any) -> str:
    """Whether a person may approve their own request. A policy row, not code.

    Absent, unreadable and unrecognised all resolve to ``deny`` — the rule this
    surface already applies to ``revoke_scope``, for the same reason. The August
    migration ``i6_fix_remove_self_approval_allowed`` removed a key nothing read;
    if a missing key meant permission, every deployment that had not yet run the
    migration adding it back would silently be running without the bar, and the
    guard would look identical to one that works.
    """

    policy = _capability_policy(policy_engine)
    rules = ((policy or {}).get("details") or {}).get("rules") or {}
    value = str(rules.get("self_approval") or "").strip().lower()
    return value if value in _SELF_APPROVAL_VALUES else _SELF_APPROVAL_DENY


def _refuse_self_approval(
    policy_engine: Any,
    *,
    approver: str,
    requester: Optional[str],
    context: Dict[str, Any],
) -> None:
    """Refuse when the person signing is the person who asked (M18.72).

    ``requester`` must come from the STORED record — the draft row, the workflow
    row. Never from the request body: a caller who can name the requester can
    name somebody else and approve freely, which is the same forgery the
    approver field was locked down to prevent, one field along.

    A requester that is absent is not a match. Agents draft most of these and an
    agent-created draft has no human requester to collide with; treating unknown
    as a collision would make every such draft unapprovable, which is not a
    stricter version of this rule, it is a broken surface.
    """

    if _self_approval_rule(policy_engine) == _SELF_APPROVAL_ALLOW:
        return

    requested_by = str(requester or "").strip()
    if not requested_by or requested_by != str(approver or "").strip():
        return

    policy = _capability_policy(policy_engine) or {}
    raw_row = policy.get("raw_row") or {}
    decision = guardrail.Decision(
        allowed=False,
        reason=(
            "an approval cannot be signed by the person who requested it; "
            "someone else must approve this"
        ),
        policy_id=_CAPABILITY_POLICY_SLUG,
        policy_name=policy.get("policyName") or "EmailApprovalCapabilityPolicy",
        policy_version=raw_row.get("version") if isinstance(raw_row, dict) else None,
        evidence={"rule": "self_approval", "requested_by": requested_by,
                  "actioned_by": approver},
    )
    record_action_or_fail(
        phase="approve",
        action_type=_ACTION,
        agent="ApprovalsRouter",
        status="denied",
        summary=decision.reason,
        details={
            **context,
            "principal": approver,
            "policy_id": decision.policy_id,
            "policy_name": decision.policy_name,
            "policy_version": decision.policy_version,
            "decision": "deny",
            "evidence": decision.evidence,
        },
    )
    logger.info("self-approval refused: %s tried to approve their own request (%s)",
                approver, context)
    raise HTTPException(status_code=403, detail=decision.reason)


def _subject(principal: Any) -> str:
    subject = str(getattr(principal, "subject", "") or "").strip()
    if not subject:
        # Belt and braces: require_user already refuses an unidentified caller
        # when auth is enforced, and an approval without a person is exactly
        # what this surface exists to prevent.
        raise HTTPException(
            status_code=401, detail="an approval must name an authenticated person"
        )
    return subject


@router.get("/pending")
def list_pending(principal=Depends(require_user)) -> Dict[str, Any]:
    """Drafts awaiting a decision, each with the hash a caller would approve."""

    _require_capability(
        principal, {"action": "list_pending"},
        actioned_by=str(getattr(principal, "subject", "") or "") or None,
    )
    rows = approval_store.list_pending_dispatch_approvals(limit=200)
    return {"pending": rows, "count": len(rows)}


def _describe_draft(
    draft: Dict[str, Any],
    approval: Optional[Dict[str, Any]],
    *,
    viewer: str,
    created_on: Any = None,
    with_text: bool = False,
) -> Dict[str, Any]:
    """One unsent draft as an approver needs to see it.

    Everything is read off what a send with no overrides would transmit right now --
    the same resolution the approval and the send guard both hash -- so the subject,
    recipients and words shown are the ones the approval will be bound to.
    """

    effective = resolve_effective_content(draft)
    digest = content_hash(effective)
    requested_by = str(draft.get("requested_by") or "").strip() or None

    state: Optional[Dict[str, Any]] = None
    if approval:
        grounding = approval.get("grounding")
        approved_hash = grounding.get("content_hash") if isinstance(grounding, dict) else None
        state = {
            "approval_id": approval.get("approval_id"),
            "actioned_by": approval.get("actioned_by"),
            "actioned_at": approval.get("actioned_at"),
            # The send guard refuses an approval whose recorded content is not the
            # content about to go. Said here so a screen can ask for a fresh approval
            # instead of offering a Send that will be refused.
            "covers_this_version": bool(approved_hash) and approved_hash == digest,
        }

    out: Dict[str, Any] = {
        "unique_id": draft.get("unique_id"),
        "workflow_id": draft.get("workflow_id"),
        "supplier_id": draft.get("supplier_id"),
        "supplier_name": draft.get("supplier_name"),
        "recipients": effective["recipients"],
        "subject": effective["subject"],
        "created_on": created_on,
        "requested_by": requested_by,
        # Self-approval is refused on the stored requester; a screen says so up front.
        "requested_by_you": bool(requested_by) and requested_by == viewer,
        "content_hash": digest,
        "approval": state,
        # A reply to a supplier's final offer: whether the price is within the
        # limit on record, and why, so the approver is not shown a holding email
        # with no reason for it.
        "final_offer": draft.get("final_offer") if isinstance(draft.get("final_offer"), dict) else None,
    }
    if with_text:
        attachments = effective.get("attachments")
        out["body_text"] = _email_text(effective["body"])
        out["attachment_count"] = len(attachments) if isinstance(attachments, list) else 0
    return out


def _email_text(body: Optional[str]) -> str:
    """The email's words, one paragraph per line: no markup, no hidden marker."""

    from src.services.draft_assurance.capture import plain

    lines = (line.strip() for line in plain(body).splitlines())
    return "\n".join(line for line in lines if line)


@router.get("/drafts")
def list_drafts(principal=Depends(require_user)) -> Dict[str, Any]:
    """Every unsent draft: waiting for approval, or approved and waiting to be sent."""

    viewer = str(getattr(principal, "subject", "") or "")
    _require_capability(principal, {"action": "list_drafts"}, actioned_by=viewer or None)
    rows = [
        _describe_draft(
            entry["draft"], entry.get("approval"),
            viewer=viewer, created_on=entry.get("created_on"),
        )
        for entry in approval_store.list_unsent_drafts(limit=200)
    ]
    return {"drafts": rows, "count": len(rows)}


@router.get("/drafts/{unique_id}")
def read_draft(unique_id: str, principal=Depends(require_user)) -> Dict[str, Any]:
    """One unsent draft in full, as the words the supplier would receive."""

    viewer = str(getattr(principal, "subject", "") or "")
    _require_capability(
        principal, {"action": "read_draft", "unique_id": unique_id}, actioned_by=viewer or None
    )
    draft = _load_draft(unique_id)
    if draft is None:
        raise HTTPException(
            status_code=404, detail=f"no unsent draft with unique_id {unique_id}"
        )
    approval = approval_store.find_dispatch_approval(
        rfq_id=draft.get("rfq_id"),
        workflow_id=draft.get("workflow_id"),
        unique_id=draft.get("unique_id"),
    )
    return _describe_draft(draft, approval, viewer=viewer, with_text=True)


def _drafting_agent(request: Request) -> Any:
    """The email drafting agent, which owns how a draft is assured and stored."""

    agent_nick = getattr(request.app.state, "agent_nick", None)
    if not agent_nick:
        raise HTTPException(status_code=503, detail="the drafting service is not available")
    from agents.email_drafting_agent import EmailDraftingAgent

    return EmailDraftingAgent(agent_nick)


@router.post("/drafts/{unique_id}/final-offer")
def decide_final_offer(
    unique_id: str,
    body: FinalOfferDecisionRequest,
    request: Request,
    principal=Depends(require_user),
) -> Dict[str, Any]:
    """A person accepts or declines a supplier's final offer; the waiting reply becomes their answer.

    When the agent escalates a final offer it leaves a holding reply that commits to
    nothing. This turns that same draft into the acceptance or the decline, in fixed
    wording, and sends NOTHING: the email still has to be approved and sent like any
    other. The person deciding is recorded as the one who asked for it, taken from
    their sign-in, so the existing rule on approving your own request applies to it.
    """

    from src.services.negotiation import final_offer

    decided_by = _subject(principal)
    decision = str(body.decision or "").strip().lower()
    if decision not in final_offer.DECISIONS:
        raise HTTPException(status_code=422, detail="the decision must be accept or decline")

    gate("email.draft", principal, agent="approvals_api",
         context={"unique_id": unique_id, "final_offer_decision": decision})

    draft = _load_draft(unique_id)
    if draft is None:
        raise HTTPException(status_code=404, detail=f"no unsent draft with unique_id {unique_id}")
    advice = draft.get("final_offer")
    if not isinstance(advice, dict):
        raise HTTPException(
            status_code=409,
            detail="this email is not a reply to a supplier's final offer, so there is nothing to decide here",
        )
    if body.content_hash and body.content_hash != content_hash(resolve_effective_content(draft)):
        raise HTTPException(
            status_code=409,
            detail="this email changed after you opened it; read it again before deciding",
        )

    agent = _drafting_agent(request)
    supplier_id = draft.get("supplier_id")
    contact = agent._master_contact(supplier_id)
    try:
        html = final_offer.decision_html(
            decision, offer=advice.get("offer"), currency=advice.get("currency"),
            contact_name=getattr(contact, "name", None),
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    recipients = list(draft.get("recipients") or [])
    blocked, assurance = agent.assure_final_offer_reply(
        {"current_offer": advice.get("offer"), "currency": advice.get("currency"),
         "target_price": advice.get("limit") if advice.get("limit_kind") == "target" else None,
         "walkaway_price": advice.get("limit") if advice.get("limit_kind") == "walk-away" else None,
         "contact_name": getattr(contact, "name", None)},
        text=html, recipients=recipients, supplier_id=supplier_id, workflow_id=draft.get("workflow_id"),
        requested_by=decided_by,
    )
    if blocked:
        raise HTTPException(status_code=409, detail=blocked)

    from datetime import datetime, timezone

    decided = dict(draft)
    decided["body"] = html
    decided["requested_by"] = decided_by
    decided["assurance"] = assurance
    decided["final_offer"] = {
        **advice,
        "decision": {"decision": decision, "decided_by": decided_by,
                     "decided_at": datetime.now(timezone.utc).isoformat()},
    }
    decided.pop("draft_record_id", None)
    try:
        agent._store_draft(decided)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if decided.get("draft_record_id") is None:
        # _store_draft logs and swallows database errors; the id is stamped only on a commit.
        raise HTTPException(status_code=500, detail="the decision could not be saved; the email is unchanged")

    logger.info("final offer decided: unique_id=%s decision=%s by=%s", unique_id, decision, decided_by)
    stored = _load_draft(unique_id) or decided
    approval = approval_store.find_dispatch_approval(
        rfq_id=stored.get("rfq_id"), workflow_id=stored.get("workflow_id"), unique_id=stored.get("unique_id"),
    )
    return _describe_draft(stored, approval, viewer=decided_by, with_text=True)


@router.post("/dispatch/{unique_id}")
def approve_dispatch(
    unique_id: str,
    body: ApproveRequest,
    principal=Depends(require_user),
) -> Dict[str, Any]:
    """Approve a drafted email for sending."""

    actioned_by = _subject(principal)
    decision = _require_capability(
        principal, {"unique_id": unique_id}, actioned_by=actioned_by
    )

    draft = _load_draft(unique_id)
    if draft is None:
        raise HTTPException(
            status_code=404, detail=f"no unsent draft with unique_id {unique_id}"
        )

    # Who asked for this draft, off the stored row. `POST /workflows/email/prepare`
    # lets a person persist an edited email; without this, the same person could
    # then approve it.
    _refuse_self_approval(
        rbac.policy_engine(),
        approver=actioned_by,
        requester=draft.get("requested_by"),
        context={"unique_id": unique_id},
    )

    # Hash the resolved material a send with no overrides would transmit
    # right now -- the same computation the send path uses (see C1), so an
    # unedited draft's approval and its eventual send agree (see C2).
    digest = content_hash(resolve_effective_content(draft))
    if body.content_hash and body.content_hash != digest:
        raise HTTPException(
            status_code=409,
            detail="this email changed after you opened it; read it again before approving",
        )
    approval_id = approval_store.record_approval(
        rfq_id=draft.get("rfq_id"),
        workflow_id=draft.get("workflow_id"),
        unique_id=unique_id,
        supplier_id=draft.get("supplier_id"),
        actioned_by=actioned_by,
        deal_id=draft.get("deal_id"),
        policy_id=_capability_policy_db_id(rbac.policy_engine()),
        policy_name=decision.policy_name or "EmailApprovalCapabilityPolicy",
        grounding_extra={
            "content_hash": digest,
            "reason": body.reason,
            "policy_version": decision.policy_version,
        },
    )
    logger.info(
        "approval recorded: unique_id=%s approval_id=%s by=%s",
        unique_id, approval_id, actioned_by,
    )
    return {
        "approval_id": approval_id,
        "unique_id": unique_id,
        "actioned_by": actioned_by,
        "content_hash": digest,
    }


@router.post("/round/{workflow_id}/{round_num}")
def approve_round(
    workflow_id: str,
    round_num: int,
    body: ApproveRequest,
    principal=Depends(require_user),
) -> Dict[str, Any]:
    """Approve a negotiation round.

    A round has no editable artefact, so there is no content hash to bind.
    """

    actioned_by = _subject(principal)
    decision = _require_capability(
        principal, {"workflow_id": workflow_id, "round": round_num},
        actioned_by=actioned_by,
    )

    if not approval_store.negotiation_workflow_exists(workflow_id=workflow_id):
        raise HTTPException(
            status_code=404, detail=f"no negotiation workflow {workflow_id}"
        )

    # A round carries no artefact, so the workflow's recorded initiator is the
    # requester. See approval_store.workflow_initiator for why this cannot fire
    # yet, and why it is still resolved from the record rather than assumed.
    _refuse_self_approval(
        rbac.policy_engine(),
        approver=actioned_by,
        requester=approval_store.workflow_initiator(workflow_id),
        context={"workflow_id": workflow_id, "round": int(round_num)},
    )

    approval_id = approval_store.record_approval(
        rfq_id=None,
        workflow_id=workflow_id,
        unique_id=None,
        supplier_id=None,
        actioned_by=actioned_by,
        policy_id=_capability_policy_db_id(rbac.policy_engine()),
        policy_name=decision.policy_name or "EmailApprovalCapabilityPolicy",
        grounding_extra={
            "round": int(round_num),
            "reason": body.reason,
            "policy_version": decision.policy_version,
        },
    )
    return {
        "approval_id": approval_id,
        "workflow_id": workflow_id,
        "round": int(round_num),
        "actioned_by": actioned_by,
    }


def _revoke_scope(policy_engine: Any) -> str:
    """The customer-set ``revoke_scope`` for EmailApprovalCapabilityPolicy.

    This is a policy row, not a Python constant -- the customer owns who may
    cancel whose approval, same as everything else on this surface. An
    absent or unrecognised value falls back to ``own_or_higher_rank``, never
    to ``any_approver``: an unenforceable/mistyped setting must narrow
    access, not silently widen it.
    """

    policy = _capability_policy(policy_engine)
    rules = ((policy or {}).get("details") or {}).get("rules") or {}
    value = str(rules.get("revoke_scope") or "").strip()
    return value if value in _REVOKE_SCOPES else _REVOKE_SCOPE_OWN_OR_HIGHER


def _required_role_rank(policy_engine: Any) -> int:
    """Rank of the role EmailApprovalCapabilityPolicy requires to approve at
    all (today, Buyer) -- the floor ``own_or_higher_rank`` measures a senior
    override against, since ``bp_approval`` does not record the original
    approver's role.

    Fails restrictive: an unreadable policy, or one with no ``required_role``
    at all, returns a rank higher than any real role so the override in
    ``_may_revoke`` never fires -- only the original approver may then
    revoke. This used to return ``rbac.role_rank(None, ...)`` == 0 in both
    cases, and since every real role ranks >= 1, ``role_rank(role) > 0`` was
    true for *any* role: a capability row with no ``required_role`` silently
    turned ``own_or_higher_rank`` into ``any_approver`` (I8). Its sibling
    ``_revoke_scope`` already fails to the restrictive option on the same
    failure; this brought the two back into agreement.
    """

    policy = _capability_policy(policy_engine)
    if not isinstance(policy, dict):
        return _RANK_DENY_ALL

    required_role = (policy.get("details") or {}).get("required_role")
    if not required_role:
        return _RANK_DENY_ALL

    rank = rbac.role_rank(required_role, policy_engine=policy_engine)
    if rank <= 0:
        # required_role is set but names a role policy does not recognise
        # (typo'd/renamed) -- an unenforceable restriction denies, exactly
        # like guardrail.authorize's own required_role handling.
        return _RANK_DENY_ALL
    return rank


def _may_revoke(principal: Any, actioned_by: str, approval: Dict[str, Any]) -> bool:
    """Whether ``principal`` may revoke ``approval``, per revoke_scope.

    Resolves the policy engine once and threads it through every rbac call,
    the same pattern guardrail.authorize follows, so a broken/unreachable
    engine fails closed rather than each helper resolving its own.
    """

    engine = rbac.policy_engine()
    scope = _revoke_scope(engine)

    if scope == _REVOKE_SCOPE_ANY:
        return True

    owner = str(approval.get("actioned_by") or "").strip()
    if actioned_by == owner:
        return True

    if scope == _REVOKE_SCOPE_OWN_ONLY:
        return False

    # own_or_higher_rank: not the owner, so only someone who strictly
    # outranks the floor required to hold approve_email at all may override.
    role = rbac.effective_role(principal, policy_engine=engine)
    return rbac.role_rank(role, policy_engine=engine) > _required_role_rank(engine)


@router.post("/{approval_id}/revoke")
def revoke(
    approval_id: int,
    body: RevokeRequest,
    principal=Depends(require_user),
) -> Dict[str, Any]:
    """Withdraw an approval. Writes a later row; history is not rewritten."""

    actioned_by = _subject(principal)
    _require_capability(
        principal, {"approval_id": approval_id}, actioned_by=actioned_by
    )

    approval = approval_store.get_approval(approval_id=approval_id)
    if approval is None:
        raise HTTPException(
            status_code=404, detail=f"no approval with id {approval_id}"
        )

    if not _may_revoke(principal, actioned_by, approval):
        raise HTTPException(
            status_code=403,
            detail=(
                "only the original approver, or someone of higher rank, "
                "may revoke this approval"
            ),
        )

    try:
        new_id = approval_store.revoke_approval(
            approval_id=approval_id, actioned_by=actioned_by, reason=body.reason
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return {
        "revoked_approval_id": approval_id,
        "revocation_id": new_id,
        "actioned_by": actioned_by,
    }
