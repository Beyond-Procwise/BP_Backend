"""Who may answer the question a paused workflow is asking (decision D-60 / D-62, backlog 1.23).

A workflow that cannot continue stops and asks a person (``POST /agent-workflows/runs/{id}/input``).
Before this, any signed-in user who knew the run's id could answer, and so steer the run.

The rule is an administrator's setting, not code: the ``workflow_question_answer`` choice setting
(``services.permission_settings``), stored as a ``proc.bp_policy`` row. Its values:

  approver_only        (DEFAULT) only people with the approver permission. For an APPROVAL question the
                       person who started the run is also barred, whatever their role. A question that is
                       not an approval still needs an approver: the setting is not loosened for it.
  starter_or_approver  the person who started the run, or an approver.
  any_signed_in        anyone signed in.

FAIL CLOSED. If the row cannot be read, or holds a value that is not one of the three, the answer is
Admin only (and the starter is still barred from answering an approval). An outage must never read as
"anyone may answer".

Approval question: the request's ``field_type`` is ``approval``, or its ``required_field`` names one
(contains "approv"). No question the product asks today is typed that way; this is the seam the approval
nodes use.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from services import permission_settings, rbac

logger = logging.getLogger(__name__)

KEY = "workflow_question_answer"
APPROVER_ONLY = "approver_only"
STARTER_OR_APPROVER = "starter_or_approver"
ANY_SIGNED_IN = "any_signed_in"
ADMIN_ONLY = "admin_only"       # not a stored value: what an unreadable or unknown setting means
_VALID = (APPROVER_ONLY, STARTER_OR_APPROVER, ANY_SIGNED_IN)


@dataclass(frozen=True)
class Decision:
    allowed: bool
    reason: str
    mode: str
    role: str


def is_approval_question(request: Optional[dict]) -> bool:
    """True when the question asks for an approval. A missing request is treated as one (the strict reading)."""
    if not request:
        return True
    if str(request.get("field_type") or "").strip().lower() == "approval":
        return True
    return "approv" in str(request.get("required_field") or "").lower()


def current_mode() -> str:
    """The mode in force; ``admin_only`` if the setting cannot be read or holds an unknown value."""
    try:
        value = permission_settings.choice_value(KEY)
    except Exception as exc:  # noqa: BLE001 - any failure to read must close, not open
        logger.warning("workflow answer policy unreadable (%s): Admin only", type(exc).__name__)
        return ADMIN_ONLY
    if value not in _VALID:
        logger.warning("workflow answer policy holds %r, which is not a known mode: Admin only", value)
        return ADMIN_ONLY
    return value


def _is_approver(role: str, engine: Any) -> bool:
    need = rbac.role_rank("Approver", policy_engine=engine)
    return need > 0 and rbac.role_rank(role, policy_engine=engine) >= need


def decide(principal: Any, started_by: Optional[str], approval: bool,
           mode: Optional[str] = None, engine: Any = None) -> Decision:
    """May ``principal`` answer a question on a run started by ``started_by``?"""
    mode = mode if mode is not None else current_mode()
    engine = engine if engine is not None else rbac.policy_engine()
    subject = getattr(principal, "subject", None)
    role = rbac.effective_role(principal, policy_engine=engine) if principal is not None else ""
    is_starter = bool(subject) and bool(started_by) and subject == started_by

    def no(why: str) -> Decision:
        return Decision(False, why, mode, role)

    if principal is None or not subject:
        return no("You need to be signed in to answer this question.")
    if approval and is_starter:
        return no("You started this run, so you cannot approve it yourself. Ask an approver to answer.")
    if mode == ANY_SIGNED_IN:
        return Decision(True, "any signed-in person may answer", mode, role)
    if mode == STARTER_OR_APPROVER:
        if is_starter or _is_approver(role, engine):
            return Decision(True, "the person who started the run, or an approver, may answer", mode, role)
        return no("Only the person who started this run, or an approver, can answer this question.")
    if mode == APPROVER_ONLY:
        if _is_approver(role, engine):
            return Decision(True, "an approver may answer", mode, role)
        return no("Only an approver can answer this question. Ask an approver.")
    # ADMIN_ONLY (unreadable or unknown setting)
    if role == "Admin":
        return Decision(True, "the setting could not be read; an Admin may answer", mode, role)
    return no("The rule for who may answer this question could not be read, so only an Admin can answer it "
              "for now.")
