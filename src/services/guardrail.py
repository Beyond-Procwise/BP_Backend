"""The single seam an irreversible action must pass through.

Evaluation order, and the reasoning behind it:

1. Resolve the caller's role. No principal means Viewer, so an unauthenticated
   environment cannot send mail just because authentication happens to be off.
2. Role cap. An agent executes as the person who invoked it and can never
   exceed them.
3. Every applicable policy is evaluated and any denial wins. Deny beats allow
   so that adding a restriction never depends on removing a permission.
4. Default-deny for irreversible classes. Silence is not permission.

Any exception is converted to a denial with the error recorded as evidence.
The gate never raises into its caller and never fails open.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.services import rbac

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Decision:
    allowed: bool
    reason: str
    policy_id: Optional[str] = None
    policy_name: Optional[str] = None
    policy_version: Optional[int] = None
    evidence: Dict[str, Any] = field(default_factory=dict)


def _deny(reason: str, **evidence: Any) -> Decision:
    return Decision(allowed=False, reason=reason, evidence=dict(evidence))


def _version_of(policy: Dict[str, Any]) -> Optional[int]:
    raw_row = policy.get("raw_row")
    if isinstance(raw_row, dict):
        try:
            return int(raw_row.get("version"))
        except (TypeError, ValueError):
            return None
    return None


def authorize(
    action: str,
    action_class: str,
    principal: Optional[Any],
    context: Optional[Dict[str, Any]] = None,
    policy_engine: Optional[Any] = None,
) -> Decision:
    """Decide whether ``principal`` may perform ``action``."""

    context = dict(context or {})
    try:
        # Resolve the engine once and thread that same instance through every
        # downstream call. Otherwise the default (no engine passed) path
        # would let each rbac helper resolve (and potentially rebuild) its
        # own engine, bypassing rbac's TTL cache and re-reading bp_policy
        # several times per authorization decision.
        engine = policy_engine if policy_engine is not None else rbac.policy_engine()

        role = rbac.effective_role(principal, policy_engine=engine)
        irreversible = rbac.is_irreversible(action_class, policy_engine=engine)

        if principal is None and irreversible:
            return _deny(
                "no authenticated principal: irreversible actions are refused",
                action=action,
                action_class=action_class,
            )

        # DO NOT move this below the `may` check: that reorders which failure
        # an exploding engine is caught by, and silently turns a broken-engine
        # denial back into a role denial with no error evidence (see
        # test_an_exploding_engine_denies_rather_than_raises).
        #
        # Fetch the applicable policies before the role-cap check below.
        # rbac's own helpers (effective_role, is_irreversible, may) each
        # catch their own engine failures internally and fall back to the
        # least-privileged answer, so a broken engine never raises through
        # them -- it just quietly becomes "Viewer can't do that". Calling
        # the engine directly here, before that early return, is what lets
        # a genuinely unreachable/broken engine surface as a denial with
        # the real error recorded as evidence, rather than being
        # indistinguishable from an ordinary role-based denial. If engine
        # construction failed upstream (engine is None), this raises
        # AttributeError, which the outer except converts into a denial
        # the same way -- fail-closed either way.
        policies: List[Dict[str, Any]] = engine.policies_for_action(action) or []

        if not rbac.may(role, action_class, policy_engine=engine):
            return _deny(
                f"role {role} may not perform {action_class}",
                action=action,
                role=role,
            )

        allowing: Optional[Dict[str, Any]] = None
        for policy in policies:
            details = policy.get("details") or {}
            rules = details.get("rules") or {}

            required_role = details.get("required_role")
            if required_role and rbac.role_rank(
                role, policy_engine=engine
            ) < rbac.role_rank(required_role, policy_engine=engine):
                return _deny(
                    f"{policy.get('policyName')} requires role {required_role}; "
                    f"caller is {role}",
                    action=action,
                    role=role,
                )

            if str(rules.get("effect") or "").lower() == "deny":
                return Decision(
                    allowed=False,
                    reason=str(rules.get("reason") or "denied by policy"),
                    policy_id=policy.get("policyId"),
                    policy_name=policy.get("policyName"),
                    policy_version=_version_of(policy),
                    evidence={"action": action, "role": role},
                )

            if allowing is None:
                allowing = policy

        if allowing is None:
            if irreversible:
                return _deny(
                    f"no policy permits {action}; irreversible actions are "
                    "default-deny",
                    action=action,
                    action_class=action_class,
                    role=role,
                )
            return Decision(
                allowed=True,
                reason=f"{action_class} is not irreversible and no policy denies it",
                evidence={"action": action, "role": role},
            )

        return Decision(
            allowed=True,
            reason=f"permitted by {allowing.get('policyName')}",
            policy_id=allowing.get("policyId"),
            policy_name=allowing.get("policyName"),
            policy_version=_version_of(allowing),
            evidence={"action": action, "role": role},
        )

    except Exception as exc:  # noqa: BLE001 - a broken gate is a closed gate
        logger.error("guardrail.authorize(%s) failed: %s", action, exc)
        return _deny(
            "policy evaluation failed; denying",
            action=action,
            error=str(exc),
        )
