"""One way for an endpoint to ask the gate, so there is one way it is answered.

Before this, ``guardrail.authorize`` had four call sites and all four were on the
email path. Creating an agent, deleting one, reloading the rules that govern
everything, retraining a model -- none of them consulted anything. Granting
someone Admin granted powers nothing checked.

WHY A HELPER RATHER THAN AN INLINE BLOCK PER ENDPOINT

Because the inline version is four lines of gate and eight lines of audit, and
the audit is where this project's attribution bugs have actually happened: three
near-identical blocks that agreed until one drifted. One function, one shape,
one place to fix.

WHAT IT GUARANTEES

  * The action name is checked against the closed vocabulary FIRST. A typo would
    otherwise match no policy, defer forever, and look like a policy problem
    rather than a spelling one. It raises here instead, at the call site, where
    the mistake is.
  * Every attempt is audited before it proceeds -- allowed and refused alike --
    through ``record_action_or_fail``, so an action whose audit cannot be written
    does not happen.
  * A refusal is an HTTP 403 that says which rule refused, without disclosing
    the policy's contents.
  * An unresolved decision refuses too, and says a person has been asked. The
    caller is not told "no" as though a rule had decided; they are told nobody
    has decided yet.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import HTTPException

from src.services import actions, agent_actions, guardrail, rbac

logger = logging.getLogger(__name__)


class NotPermitted(HTTPException):
    """Raised when the gate refuses. Carries the reason, never the policy body."""

    def __init__(self, detail: str) -> None:
        super().__init__(status_code=403, detail=detail)


def require(
    action: str,
    principal: Optional[Any],
    *,
    context: Optional[Dict[str, Any]] = None,
    agent: str = "api",
    engine: Optional[Any] = None,
) -> guardrail.Decision:
    """Refuse unless ``principal`` may perform ``action``. Returns the Decision.

    ``action`` must be in :data:`services.actions.ACTIONS`; its class is taken
    from there rather than passed in, so a call site cannot quietly declare a
    ``share`` to be a ``read``.
    """

    action_class = actions.action_class(action)  # raises on an unknown name
    resolved = engine if engine is not None else rbac.policy_engine()
    ctx = dict(context or {})

    decision = guardrail.authorize(
        action, action_class, principal, ctx, policy_engine=resolved
    )

    subject = getattr(principal, "subject", None)
    agent_actions.record_action_or_fail(
        phase="authorize",
        action_type=action,
        agent=agent,
        status="allowed" if decision.allowed else "denied",
        summary=decision.reason,
        details={
            **ctx,
            "action_class": action_class,
            "principal": subject,
            "resolution": decision.resolution,
            "policy_id": decision.policy_id,
            "policy_name": decision.policy_name,
            "policy_version": decision.policy_version,
            "evidence": decision.evidence,
        },
    )

    if decision.allowed:
        return decision

    if decision.unresolved:
        logger.warning(
            "%s refused for %s: no rule settled it, raised for review",
            action,
            subject or "an anonymous caller",
        )
        raise NotPermitted(
            f"{action} is not covered by any rule yet, so it has been raised "
            "for review rather than decided here."
        )

    raise NotPermitted(decision.reason)
