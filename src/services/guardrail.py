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

This function decides role and policy applicability only. An ``allowed=True``
Decision means the caller's role is permitted the action class and no
applicable policy denies it -- it does NOT mean any policy-specific
precondition (an approval on file, a recipient allow-list, content
sensitivity, and so on) has been checked. Those preconditions are the
caller's to enforce before acting; see e.g. Task 7's email_dispatch_guard for
one such caller-side check that authorize() deliberately does not duplicate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.services import policy_observation, rbac

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


def policy_attribution(policy: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The ``policy_id``/``policy_name``/``policy_version`` triple a
    ``Decision`` should carry when it was decided by ``policy`` (a
    ``PolicyEngine.get_policy``-shaped dict, or ``None``).

    Centralised so every caller attributes a decision the same way -- this
    is exactly where the I1/C2-class "missing attribution" bugs on this
    plan have lived: three near-identical inline ``Decision(...)`` blocks
    that happened to agree until one of them drifted.
    """

    if not isinstance(policy, dict):
        return {"policy_id": None, "policy_name": None, "policy_version": None}
    return {
        "policy_id": policy.get("policyId"),
        "policy_name": policy.get("policyName"),
        "policy_version": _version_of(policy),
    }


def deny_from_policy(
    reason: str,
    policy: Optional[Dict[str, Any]] = None,
    *,
    policy_name: Optional[str] = None,
    **evidence: Any,
) -> Decision:
    """A denial attributed to ``policy``.

    ``policy_name`` is a fallback used only when ``policy`` itself could not
    be resolved (e.g. the engine is unreachable) -- ``policy_id`` and
    ``policy_version`` stay ``None`` in that case rather than being guessed.
    """

    attribution = policy_attribution(policy)
    if attribution["policy_name"] is None and policy_name:
        attribution["policy_name"] = policy_name
    return Decision(allowed=False, reason=reason, evidence=dict(evidence), **attribution)


def _evaluate(
    action: str,
    action_class: str,
    principal: Optional[Any],
    context: Optional[Dict[str, Any]] = None,
    policy_engine: Optional[Any] = None,
) -> Decision:
    """The real decision, with no knowledge of shadow mode.

    Split out of ``authorize`` unchanged. Shadow mode is decided once, at the
    boundary, rather than threaded through the evaluation logic -- a mode check
    scattered through here is how a deliberate hole becomes an accidental one.

    Decide whether ``principal`` may perform ``action``.

    ``context`` is accepted for interface symmetry with callers that carry
    request-scoped data, but is not read by this function today. It is
    intentionally left uncoerced: a coercion that is never used can only
    ever turn a caller's type error into a silently swallowed denial,
    hiding a bug in the caller rather than surfacing it.
    """

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
            if required_role:
                required_rank = rbac.role_rank(required_role, policy_engine=engine)
                if required_rank <= 0:
                    # An unresolvable required_role (typo'd, renamed, blank)
                    # means the policy cannot be enforced as written.
                    # role_rank returns 0 for any role it does not
                    # recognise, and every real role ranks >= 1, so treating
                    # this as "no cap" would silently delete the
                    # restriction instead of denying it. An unenforceable
                    # restriction denies rather than evaporating.
                    return deny_from_policy(
                        f"{policy.get('policyName')} requires role "
                        f"{required_role!r}, which no policy defines",
                        policy,
                        action=action,
                        role=role,
                    )
                if rbac.role_rank(role, policy_engine=engine) < required_rank:
                    return deny_from_policy(
                        f"{policy.get('policyName')} requires role "
                        f"{required_role}; caller is {role}",
                        policy,
                        action=action,
                        role=role,
                    )

            if str(rules.get("effect") or "").lower() == "deny":
                return deny_from_policy(
                    str(rules.get("reason") or "denied by policy"),
                    policy,
                    action=action,
                    role=role,
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
            # State only what this function actually evaluated: role and
            # policy applicability. Do not claim or imply that a
            # policy-specific precondition -- an approval on file, a
            # recipient allow-list, content sensitivity -- was checked here;
            # that is the caller's job, and wording this as "approved" would
            # let a future reader of an audit row mistake applicability for
            # full clearance.
            reason=(
                f"{allowing.get('policyName')} permits {action} for role "
                f"{role}; policy-specific preconditions are enforced by the "
                "caller"
            ),
            policy_id=allowing.get("policyId"),
            policy_name=allowing.get("policyName"),
            policy_version=_version_of(allowing),
            evidence={"action": action, "role": role},
        )

    except Exception as exc:  # noqa: BLE001 - a broken gate is a closed gate
        logger.error("guardrail evaluation of %s failed: %s", action, exc)
        return _deny(
            "policy evaluation failed; denying",
            action=action,
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# Shadow mode
#
# A deliberate hole, so the rules around it are stated in code rather than left
# to configuration:
#
#   * These two actions can never be shadowed. They are enforced today at four
#     call sites and are the only things in this product that reliably refuse.
#     A list in a policy row could enrol them by accident or by edit; this
#     cannot.
#   * An enrolment without an expiry is not an enrolment. Shadow mode must not
#     become the permanent state because nobody got round to the next step.
#   * No record, no shadow. Allowing without recording gives neither the safety
#     of the refusal nor the data it was traded for.
# ---------------------------------------------------------------------------

NEVER_SHADOW = frozenset({"email.send", "approval.email"})

_SHADOW_SLUG = "shadow_mode"


def _shadow_expiry(action: str, engine: Optional[Any]) -> Optional[datetime]:
    """When this action's shadow enrolment ends, or ``None`` if it is not enrolled."""

    if action in NEVER_SHADOW or engine is None:
        return None
    try:
        policy = engine.get_policy(_SHADOW_SLUG)
        rules = ((policy or {}).get("details") or {}).get("rules") or {}
        for entry in rules.get("shadow_actions") or []:
            if not isinstance(entry, dict) or entry.get("action") != action:
                continue
            raw = entry.get("until")
            if not raw:
                # A missing expiry is not an unlimited one.
                return None
            parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed
    except Exception as exc:  # noqa: BLE001 - unreadable config means enforce
        logger.error("shadow config unreadable for %s: %s", action, exc)
    return None


def shadow_status(engine: Optional[Any] = None) -> Dict[str, Any]:
    """What is enrolled and until when. Surfaced on /health.

    A control that is off must be visible, not something discovered by reading
    code -- the same reason ``ask_auth`` is reported there.
    """

    resolved = engine if engine is not None else rbac.policy_engine()
    out: Dict[str, Any] = {"enrolled": [], "never_shadowed": sorted(NEVER_SHADOW)}
    try:
        policy = resolved.get_policy(_SHADOW_SLUG) if resolved else None
        rules = ((policy or {}).get("details") or {}).get("rules") or {}
        now = datetime.now(timezone.utc)
        for entry in rules.get("shadow_actions") or []:
            if not isinstance(entry, dict):
                continue
            action = entry.get("action")
            expiry = _shadow_expiry(str(action or ""), resolved)
            out["enrolled"].append(
                {
                    "action": action,
                    "until": entry.get("until"),
                    "active": bool(expiry and expiry > now),
                }
            )
    except Exception as exc:  # noqa: BLE001
        out["error"] = str(exc)
    return out


def authorize(
    action: str,
    action_class: str,
    principal: Optional[Any],
    context: Optional[Dict[str, Any]] = None,
    policy_engine: Optional[Any] = None,
) -> Decision:
    """Decide whether ``principal`` may perform ``action``, and write it down.

    The decision itself is ``_evaluate``'s and is not influenced by anything
    here. This function records what was decided and, for an action explicitly
    enrolled in shadow mode with an unexpired entry, declines to apply a denial
    so the effect of a rule can be measured before it is felt.

    Every decision is recorded, allows included: "we observed no denials" and
    "we were not observing" are otherwise the same observation.
    """

    decision = _evaluate(action, action_class, principal, context, policy_engine)

    engine = policy_engine if policy_engine is not None else rbac.policy_engine()
    expiry = _shadow_expiry(action, engine) if not decision.allowed else None
    shadowed = bool(expiry and expiry > datetime.now(timezone.utc))

    try:
        recorded = policy_observation.record(
            action=action,
            action_class=action_class,
            principal_subject=getattr(principal, "subject", None),
            role=(decision.evidence or {}).get("role"),
            would_have_denied=not decision.allowed,
            shadowed=shadowed,
            policy_id=decision.policy_id,
            policy_name=decision.policy_name,
            policy_version=decision.policy_version,
            reason=decision.reason,
            evidence=decision.evidence,
        )
    except Exception as exc:  # noqa: BLE001 - bookkeeping must not break the gate
        logger.error("policy observation failed for %s: %s", action, exc)
        recorded = False

    if not shadowed:
        return decision

    if not recorded:
        # No record, no shadow. Allowing here would spend the refusal and buy
        # nothing with it.
        logger.warning(
            "shadow mode for %s could not be recorded; the denial stands", action
        )
        return decision

    logger.info(
        "SHADOW: %s would have been denied (%s) and was allowed through",
        action,
        decision.reason,
    )
    return Decision(
        allowed=True,
        reason=(
            f"shadow mode: this would have been denied -- {decision.reason} -- "
            f"and was allowed through so the effect could be measured"
        ),
        policy_id=decision.policy_id,
        policy_name=decision.policy_name,
        policy_version=decision.policy_version,
        evidence={**(decision.evidence or {}), "shadowed": True},
    )
