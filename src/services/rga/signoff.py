"""Report sign-off: which reports need it, and who may review one before it has it.

Ruled 2026-09-24: every report needs a person's sign-off before it leaves the company, and
that is a policy point, not code. Both answers here come from proc.bp_policy
(deploy/sql/2026-09-24_report_signoff_policy.sql):

  * ``report_signoff`` -- which report types need sign-off (``"*"`` = all) and whether the
    person who asked for a report may sign it off (``self_approval``);
  * ``report_signoff_authority`` -- the stated permit for ``report.signoff``: its
    ``required_role`` is who may sign off, and so who may open a deck before sign-off to
    review it.

Nothing here names a role or a report type. Every unreadable answer is the cautious one:
a policy that cannot be read holds every report; an unreadable self-approval rule denies;
a missing authority row lets nobody review early.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Any, Dict

from src.services import actions, rbac

logger = logging.getLogger(__name__)

POLICY = "report_signoff"
AUTHORITY = "report_signoff_authority"
ACTION = "report.signoff"


def _details(slug: str, engine: Any) -> Dict[str, Any]:
    engine = engine if engine is not None else rbac.policy_engine()
    try:
        row = engine.get_policy(slug) if engine is not None else None
    except Exception:  # noqa: BLE001 - unreadable falls to the caller's cautious default
        logger.exception("rga: could not read policy %s", slug)
        row = None
    details = row.get("details") if isinstance(row, dict) else None
    return details if isinstance(details, dict) else {}


def policy(engine: Any = None) -> Dict[str, Any]:
    """``{requires, self_approval, readable}``. ``requires`` is None when unreadable,
    which every caller treats as "all reports"."""
    rules = _details(POLICY, engine).get("rules")
    rules = rules if isinstance(rules, dict) else {}
    requires = rules.get("requires_signoff")
    readable = isinstance(requires, list) and all(isinstance(x, str) for x in requires)
    if not readable:
        logger.warning("rga: policy %s is unreadable; holding every report for sign-off", POLICY)
    return {"requires": requires if readable else None,
            "self_approval": "allow" if rules.get("self_approval") == "allow" else "deny",
            "readable": readable}


def required(report_type: str, engine: Any = None) -> bool:
    requires = policy(engine)["requires"]
    return requires is None or "*" in requires or report_type in requires


def self_approval_denied(engine: Any = None) -> bool:
    return policy(engine)["self_approval"] != "allow"


def may_sign_off(principal: Any, engine: Any = None) -> bool:
    """Whether ``principal`` could sign a report off -- asked to let them REVIEW one first.

    Read-only, and so not ``guardrail.authorize``: that records an observation on every call,
    and this decides nothing. It applies the same two tests the gate would: the role holds
    the action's class, and ranks at or above the authority row's ``required_role``.
    """
    engine = engine if engine is not None else rbac.policy_engine()
    needed = _details(AUTHORITY, engine).get("required_role")
    if not needed:
        return False
    role = rbac.effective_role(principal, policy_engine=engine)
    need_rank = rbac.role_rank(needed, policy_engine=engine)
    return (need_rank > 0
            and rbac.may(role, actions.action_class(ACTION), policy_engine=engine)
            and rbac.role_rank(role, policy_engine=engine) >= need_rank)


_UNSET = object()
_BY_STATUS = {"approved": "signed_off", "refused": "refused"}


def deck_hash(content: bytes) -> str:
    """What a sign-off is bound to: the exact file that was reviewed."""
    return hashlib.sha256(content or b"").hexdigest()


def state(job: Dict[str, Any], engine: Any = None, decision: Any = _UNSET) -> Dict[str, Any]:
    """A job's sign-off state: ``not_released`` | ``not_required`` | ``awaiting`` |
    ``signed_off`` | ``refused``, with who, when, why, the approval id and the deck hash.

    Derived, never stored: policy decides whether a report needs sign-off, and the newest
    ``bp_approval`` row for the job decides whether it has one. A decision in any other status
    (a future revoke) is not a sign-off, so the report is awaiting again.
    """
    need = required(job.get("report_type") or "", engine=engine)
    out: Dict[str, Any] = {"required": need, "state": "not_required", "by": None, "at": None,
                           "reason": None, "approval_id": None, "deck_sha256": None}
    if job.get("status") != "released":
        out["state"] = "not_released"
        return out
    if not need:
        return out
    if decision is _UNSET:
        from src.services import approval_store

        decision = approval_store.find_report_decision(job["job_id"])
    if not decision:
        out["state"] = "awaiting"
        return out
    grounding = decision.get("grounding") or {}
    at = decision.get("actioned_at")
    out.update(by=decision.get("actioned_by"),
               at=at.isoformat() if hasattr(at, "isoformat") else at,
               reason=grounding.get("reason"), approval_id=decision.get("approval_id"),
               deck_sha256=grounding.get("deck_sha256"))
    out["state"] = _BY_STATUS.get(decision.get("status"), "awaiting")
    return out
