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
from src.services import approval_store, guardrail, rbac
from src.services.approval_content import content_hash

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


class ApproveRequest(BaseModel):
    """What is being approved. Deliberately carries no actor field."""

    reason: Optional[str] = None


class RevokeRequest(BaseModel):
    reason: Optional[str] = None


def get_agent_nick(request: Request):
    nick = getattr(request.app.state, "agent_nick", None)
    if not nick:
        raise HTTPException(status_code=503, detail="AgentNick not available")
    return nick


def _load_draft(unique_id: str, conn: Any = None) -> Optional[Dict[str, Any]]:
    """The stored draft being approved, or None."""

    from src.services.db import get_conn

    sql = (
        "SELECT unique_id, rfq_id, workflow_id, supplier_id, subject, body, "
        "       recipient_email, sender, attachments, payload "
        "  FROM proc.draft_rfq_emails "
        " WHERE unique_id = %s AND sent IS NOT TRUE "
        " ORDER BY id DESC LIMIT 1"
    )

    def _run(connection: Any) -> Optional[Dict[str, Any]]:
        cur = approval_store._dict_cursor(connection)
        cur.execute(sql, (unique_id,))
        row = cur.fetchone()
        return dict(row) if row else None

    if conn is not None:
        return _run(conn)
    with get_conn() as own:
        return _run(own)


def _require_capability(principal: Any, context: Dict[str, Any]) -> None:
    """Deny unless policy grants this caller the approve_email capability."""

    decision = guardrail.authorize(_ACTION, _ACTION_CLASS, principal, context)
    if not decision.allowed:
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

    _require_capability(principal, {"action": "list_pending"})
    rows = approval_store.list_pending_dispatch_approvals(limit=200)
    return {"pending": rows, "count": len(rows)}


@router.post("/dispatch/{unique_id}")
def approve_dispatch(
    unique_id: str,
    body: ApproveRequest,
    principal=Depends(require_user),
) -> Dict[str, Any]:
    """Approve a drafted email for sending."""

    actioned_by = _subject(principal)
    _require_capability(principal, {"unique_id": unique_id})

    draft = _load_draft(unique_id)
    if draft is None:
        raise HTTPException(
            status_code=404, detail=f"no unsent draft with unique_id {unique_id}"
        )

    digest = content_hash(draft)
    approval_id = approval_store.record_approval(
        rfq_id=draft.get("rfq_id"),
        workflow_id=draft.get("workflow_id"),
        unique_id=unique_id,
        supplier_id=draft.get("supplier_id"),
        actioned_by=actioned_by,
        deal_id=draft.get("deal_id"),
        policy_name="EmailDispatchApprovalPolicy",
        grounding_extra={"content_hash": digest, "reason": body.reason},
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
    _require_capability(principal, {"workflow_id": workflow_id, "round": round_num})

    approval_id = approval_store.record_approval(
        rfq_id=None,
        workflow_id=workflow_id,
        unique_id=None,
        supplier_id=None,
        actioned_by=actioned_by,
        policy_name="EmailDispatchApprovalPolicy",
        grounding_extra={"round": int(round_num), "reason": body.reason},
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

    try:
        policy = policy_engine.get_policy(_CAPABILITY_POLICY_SLUG) if policy_engine else None
    except Exception as exc:  # noqa: BLE001 - an unreadable policy is the strict default
        logger.error("approvals: could not read %s: %s", _CAPABILITY_POLICY_SLUG, exc)
        policy = None

    rules = ((policy or {}).get("details") or {}).get("rules") or {}
    value = str(rules.get("revoke_scope") or "").strip()
    return value if value in _REVOKE_SCOPES else _REVOKE_SCOPE_OWN_OR_HIGHER


def _required_role_rank(policy_engine: Any) -> int:
    """Rank of the role EmailApprovalCapabilityPolicy requires to approve at
    all (today, Buyer) -- the floor ``own_or_higher_rank`` measures a senior
    override against, since ``bp_approval`` does not record the original
    approver's role."""

    try:
        policy = policy_engine.get_policy(_CAPABILITY_POLICY_SLUG) if policy_engine else None
    except Exception:  # noqa: BLE001 - unreadable policy denies via rank 0 below
        policy = None

    required_role = ((policy or {}).get("details") or {}).get("required_role")
    return rbac.role_rank(required_role, policy_engine=policy_engine)


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
    _require_capability(principal, {"approval_id": approval_id})

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
