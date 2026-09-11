"""The one place a governance limit is read.

Thirty-three values decided what reaches the financial record, what counts as a
match on money, who a supplier is, what an agent may put to a supplier and how
far it may reach — and every one of them was an `os.getenv` with a hardcoded
default. Changeable with no code change AND no policy edit, versioned by
nothing, audited by nothing, and on no governance screen. They live in
`proc.bp_policy` now, in seven rows grouped by subject, and this module is how
they are read.

Three rules, and the first is the one with teeth:

  1. **A missing value refuses.** Never a silent fall back to the number the
     code used to carry. Fifteen of the thirty-three had their default inside
     the `getenv` call, so "the policy is missing" and "the policy says 50" were
     the same value — and a guard that cannot tell those apart is not a guard.
     That is exactly the hole P6 found one layer down, where an outage and
     "nothing governs this" were indistinguishable.

  2. **The environment still overrides, for one release, and never silently.**
     A deployment that tuned a threshold must not have it revert underneath them
     mid-rollout. But an override nobody can see is how the environment came to
     be the real source of truth in the first place, so a value that disagrees
     with policy is logged as a warning naming both.

  3. **Present-and-null is an answer.** `neg_thread_transcript_limit: null` says
     "no limit" and is a decision somebody made. An absent key is an unanswered
     question. They must not collapse into each other.

The env override is scheduled for removal. When it goes, `env=` arguments go
with it and nothing else here changes.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

_CACHE: Dict[str, Dict[str, Any]] = {}
_LOCK = threading.Lock()

#: Overrides already warned about, so a limit read in a loop does not fill the
#: log with the same line. Cleared by reset_cache().
_WARNED: set = set()


class LimitUnavailable(RuntimeError):
    """A governed limit could not be read, so the caller must not proceed.

    Deliberately not carrying a default. A caller that could supply one would
    reintroduce the thing this module exists to remove.
    """


def _engine() -> Optional[Any]:
    """The shared PolicyEngine. A seam: tests replace this, not the whole module."""

    from src.services import rbac

    return rbac.policy_engine()


def reset_cache() -> None:
    """Forget everything read so far. For tests, and for a governance reload."""

    with _LOCK:
        _CACHE.clear()
        _WARNED.clear()


def _rules(policy: str) -> Dict[str, Any]:
    with _LOCK:
        cached = _CACHE.get(policy)
    if cached is not None:
        return cached

    try:
        engine = _engine()
        row = engine.get_policy(policy) if engine else None
    except Exception as exc:  # noqa: BLE001 - unreadable is not permission
        raise LimitUnavailable(
            f"could not read the {policy} policy: {exc}") from exc

    if not isinstance(row, dict):
        raise LimitUnavailable(
            f"no active {policy} policy — its limits are unset, and an unset "
            f"limit is not an unlimited one")

    rules = (row.get("details") or {}).get("rules")
    if not isinstance(rules, dict):
        raise LimitUnavailable(f"the {policy} policy states no rules")

    with _LOCK:
        _CACHE[policy] = rules
    return rules


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _cast(value: Any, cast: Callable[[Any], Any]) -> Any:
    return _as_bool(value) if cast is bool else cast(value)


def limit(policy: str, rule: str, *, env: Optional[str] = None,
          cast: Callable[[Any], Any] = float) -> Any:
    """The governed value of one limit, or raise.

    ``policy`` is a ``policy_identifier`` (e.g. ``promotion_thresholds``),
    ``rule`` a key under its ``rules``. ``env`` names the environment variable
    that still overrides it during the deprecation window.

    Returns ``None`` only when the policy states ``null`` for the rule, which
    means "no limit" and is not the same as the rule being absent.
    """

    rules = _rules(policy)
    if rule not in rules:
        raise LimitUnavailable(
            f"{policy} does not state {rule!r}; refusing rather than assuming a "
            f"value for it")

    stated = rules[rule]
    governed = None if stated is None else _cast(stated, cast)

    if not env:
        return governed

    raw = os.getenv(env)
    if raw is None or not str(raw).strip():
        return governed

    try:
        overridden = _cast(str(raw).strip(), cast)
    except (TypeError, ValueError):
        logger.warning(
            "%s=%r cannot be read as a limit and is being ignored; %s.%s = %r "
            "from policy", env, raw, policy, rule, governed)
        return governed

    if overridden == governed:
        return governed

    key = f"{env}={raw}"
    with _LOCK:
        first_time = key not in _WARNED
        _WARNED.add(key)
    if first_time:
        logger.warning(
            "%s=%r overrides %s.%s, which policy states as %r. The environment "
            "override is deprecated and will be removed; set the policy instead.",
            env, overridden, policy, rule, governed)
    return overridden
