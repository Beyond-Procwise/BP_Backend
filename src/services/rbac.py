"""Resolve an authenticated caller to a role, and answer what that role may do.

Both answers come from proc.bp_policy, never from constants in this file. A
role's powers change by editing a policy row, which is how the rest of the
platform's governance already works.

Every helper is safe to call with ``principal=None`` and every failure path
lands on the least-privileged answer. A policy that will not load must not
become an accidental grant.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

ROLE_UNKNOWN = "Viewer"

_ROLE_DEFINITION_SLUG = "role_definition"
_ROLE_ASSIGNMENT_SLUG = "role_assignment"

_ENGINE_CACHE: Optional[Any] = None
_ENGINE_CACHED_AT: float = 0.0
_ENGINE_TTL_SECONDS = 60.0


def reset_policy_cache() -> None:
    """Drop the cached engine. For tests and for an explicit reload."""

    global _ENGINE_CACHE, _ENGINE_CACHED_AT
    _ENGINE_CACHE, _ENGINE_CACHED_AT = None, 0.0


def _build_engine() -> Optional[Any]:
    """Construct a fresh PolicyEngine. Extracted for testability."""

    try:
        from src.engines.policy_engine import PolicyEngine
        from src.services.db import get_conn

        return PolicyEngine(connection_factory=get_conn)
    except Exception as exc:  # noqa: BLE001 - resolved to deny by the callers
        logger.error("rbac: could not construct a PolicyEngine: %s", exc)
        return None


def _engine(policy_engine: Optional[Any]) -> Optional[Any]:
    if policy_engine is not None:
        return policy_engine

    global _ENGINE_CACHE, _ENGINE_CACHED_AT
    now = time.time()
    if _ENGINE_CACHE is not None and (now - _ENGINE_CACHED_AT) < _ENGINE_TTL_SECONDS:
        return _ENGINE_CACHE

    engine = _build_engine()
    if engine is not None:
        _ENGINE_CACHE = engine
        _ENGINE_CACHED_AT = now
    return engine


def _rules(slug: str, policy_engine: Optional[Any]) -> Dict[str, Any]:
    engine = _engine(policy_engine)
    if engine is None:
        return {}
    try:
        policy = engine.get_policy(slug)
    except Exception as exc:  # noqa: BLE001
        logger.error("rbac: get_policy(%s) failed: %s", slug, exc)
        return {}
    if not isinstance(policy, dict):
        return {}
    details = policy.get("details")
    if not isinstance(details, dict):
        return {}
    rules = details.get("rules")
    return rules if isinstance(rules, dict) else {}


def _roles_table(policy_engine: Optional[Any]) -> Dict[str, Any]:
    roles = _rules(_ROLE_DEFINITION_SLUG, policy_engine).get("roles")
    return roles if isinstance(roles, dict) else {}


def role_rank(role: Optional[str], policy_engine: Optional[Any] = None) -> int:
    """Rank of ``role``; 0 for anything the policy does not define."""

    entry = _roles_table(policy_engine).get(str(role or ""))
    if not isinstance(entry, dict):
        return 0
    try:
        return int(entry.get("rank") or 0)
    except (TypeError, ValueError):
        return 0


def resolve_roles(
    principal: Optional[Any], policy_engine: Optional[Any] = None
) -> List[str]:
    """Roles carried by ``principal``, mapped from its identity-provider groups."""

    rules = _rules(_ROLE_ASSIGNMENT_SLUG, policy_engine)
    if not rules:
        return []
    if principal is None:
        return []

    claim = str(rules.get("claim") or "cognito:groups")
    claims = getattr(principal, "claims", None)
    raw = claims.get(claim) if isinstance(claims, dict) else None
    if isinstance(raw, str):
        groups = [raw]
    elif isinstance(raw, (list, tuple, set)):
        groups = [str(g) for g in raw]
    else:
        groups = []

    mapping = rules.get("group_to_role")
    mapping = mapping if isinstance(mapping, dict) else {}
    unmapped = str(rules.get("unmapped_group_role") or ROLE_UNKNOWN)

    resolved: List[str] = []
    for group in groups:
        resolved.append(str(mapping.get(group) or unmapped))
    return resolved


def effective_role(
    principal: Optional[Any], policy_engine: Optional[Any] = None
) -> str:
    """The single role ``principal`` acts with.

    No principal, no groups, or an unloadable policy all resolve to the
    policy's ``no_principal_role`` (Viewer), never to something permissive.
    """

    rules = _rules(_ROLE_ASSIGNMENT_SLUG, policy_engine)
    fallback = str(rules.get("no_principal_role") or ROLE_UNKNOWN)

    roles = resolve_roles(principal, policy_engine=policy_engine)
    if not roles:
        return fallback

    if str(rules.get("multiple_groups") or "highest_rank") == "highest_rank":
        best = max(roles, key=lambda r: role_rank(r, policy_engine=policy_engine))
        return best if role_rank(best, policy_engine=policy_engine) else fallback
    candidate = roles[0]
    return candidate if role_rank(candidate, policy_engine=policy_engine) else fallback


def may(
    role: Optional[str], action_class: str, policy_engine: Optional[Any] = None
) -> bool:
    """True when ``role`` is permitted ``action_class`` by policy."""

    entry = _roles_table(policy_engine).get(str(role or ""))
    if not isinstance(entry, dict):
        return False
    allowed = entry.get("allow")
    if not isinstance(allowed, (list, tuple, set)):
        return False
    return str(action_class) in {str(a) for a in allowed}


def is_irreversible(action_class: str, policy_engine: Optional[Any] = None) -> bool:
    """True when the action class needs the stricter, default-deny handling.

    Policy names both sets explicitly. A class in neither is treated as
    irreversible: an action nobody has classified is not thereby safe, and a
    class that drifts out of the irreversible list during a policy edit must
    fail closed rather than silently become permissible.
    """

    rules = _rules(_ROLE_DEFINITION_SLUG, policy_engine)
    if not rules:
        return True

    listed = rules.get("irreversible_classes")
    if not isinstance(listed, (list, tuple, set)):
        return True
    if str(action_class) in {str(c) for c in listed}:
        return True

    reversible = rules.get("reversible_classes")
    if not isinstance(reversible, (list, tuple, set)):
        return True
    return str(action_class) not in {str(c) for c in reversible}
