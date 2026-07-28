"""What is this agent allowed to send on its own -- resolved, in parallel, fail-closed.

This is deliberately NOT `envelope.resolve_governance`, and the difference is the
whole point of a separate module:

  * `envelope` is FAIL-OPEN. A governance error there must never break a workflow,
    because what it carries is prompts and descriptive policy summaries.
  * this is FAIL-CLOSED. What it carries is an authority limit. A missing limit is
    not "carry on unlimited", it is "stop and ask a human" -- the same rule
    DecisionEngine already applies to a missing approval threshold, and the exact
    mistake ApprovalsAgent once made by defaulting a spend limit to 1000.

Do not "tidy" these two into one module. `governed: False` here means escalate --
there is no other reading of it. Every failure path (missing policy, missing
deferred spend limit, unparseable rules, a raising policy engine, a dead worker
thread) must produce `governed: False`, never a permissive default.

It also resolves through `PolicyEngine.get_policy()` -- the engine's own selection
logic -- rather than a static workflow->agent map, so the row reported is the row
that would genuinely apply at run time. `node_governance` says plainly that it
cannot make that claim; this can, because it asks the engine.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence

log = logging.getLogger(__name__)

DEFAULT_AUTONOMY_SLUG = "email_reply_autonomy"

# `reason` is USER-FACING. It is interpolated verbatim into the decision rationale
# (DecisionEngine._decide_email_reply, authority gate) and rendered on the Action Centre
# card, so nothing in it may be a config identifier: a buyer reading
# "no usable governed policy 'email_reply_autonomy'" is being handed a slug to decode,
# and it tells anyone else more about our internals than a review screen should. The slug
# that failed goes to the LOG, where the person who can act on it will look. These two
# phrases name the same two policies the module resolves, in words.
_AUTONOMY_IN_WORDS = "the policy that sets what this agent may answer unattended"
_APPROVAL_IN_WORDS = "the spend approval policy it takes its value limit from"

# Resolution is IO-bound (one policy lookup per agent) and the agent count per run
# is small; a handful of threads is plenty and keeps the DB pool calm.
_MAX_WORKERS = 8


def ungoverned_block(agent: str, reason: str) -> Dict[str, Any]:
    """The fail-closed block. Every field a caller reads is present and empty.

    Public on purpose: this is the ONE definition of "ungoverned" in the system.
    Any caller that needs a fail-closed placeholder (e.g. the orchestrator, when
    the resolver itself blows up before this module's own try/except gets a
    chance to run) must call this rather than hand-building the same 12 keys --
    two copies of a fail-closed shape will drift, and a consumer reading the
    stale copy after a key is added to one of them gets a `KeyError`.
    """
    return {
        "agent": agent,
        "governed": False,
        "slug": None,
        "policy_id": None,
        "policy_name": None,
        "auto_intents": [],
        "escalate_intents": [],
        "limit_gbp": None,
        "limit_currency": None,
        "max_auto_replies_per_thread": None,
        "min_intent_confidence": None,
        "reason": reason,
    }


def _rules(policy: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Rule body out of PolicyEngine's normalised shape (same accessor order as
    DecisionEngine._rules -- one convention, not two)."""
    if not policy:
        return {}
    details = policy.get("details") or policy.get("policy_details") or {}
    if not isinstance(details, dict):
        return {}
    rules = details.get("rules") or policy.get("rules") or {}
    return rules if isinstance(rules, dict) else {}


def _ids(policy: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    raw = (policy or {}).get("raw_row") or {}
    return {
        "policy_id": raw.get("policy_id"),
        "policy_name": raw.get("policy_name") or (policy or {}).get("policyName"),
    }


def _str_list(value: Any) -> List[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [str(v) for v in value if v is not None]


def _resolve_one(
    policy_engine: Any, agent: str, autonomy_slug: str
) -> Dict[str, Any]:
    try:
        autonomy = policy_engine.get_policy(autonomy_slug)
    except Exception:  # noqa: BLE001 - fail CLOSED, and say why
        log.exception(
            "authority: autonomy policy lookup failed for %s (slug %r)",
            agent, autonomy_slug,
        )
        return ungoverned_block(agent, f"{_AUTONOMY_IN_WORDS} could not be read")

    rules = _rules(autonomy)
    if not autonomy or not rules:
        log.warning(
            "authority: no usable autonomy policy for %s (slug %r)", agent, autonomy_slug
        )
        return ungoverned_block(
            agent,
            f"there is no usable version of {_AUTONOMY_IN_WORDS} -- it carries no "
            "rules to apply",
        )

    limit_gbp: Optional[str] = None
    limit_currency: Optional[str] = None
    # The autonomy policy names the policy its money limit comes from. There is no
    # module-level default to fall back on, deliberately: a policy that does NOT defer
    # states no value limit, and resolving one from a default approval slug anyway would
    # invent an authority the governed row never delegated -- the fail-OPEN direction
    # this module exists to prevent. limit_gbp stays None, which the decision engine
    # escalates on. (This replaces a dead `approval_slug` parameter: it sat inside this
    # `if deferred:` block behind `str(deferred) or approval_slug`, where `str()` of a
    # truthy value is never empty, so the `or` could not fire and the public keyword
    # argument looked configurable while being inert.)
    deferred = rules.get("defer_value_limit_to")
    if deferred:
        try:
            approval = policy_engine.get_policy(str(deferred))
        except Exception:  # noqa: BLE001
            log.exception(
                "authority: approval policy lookup failed for %s (slug %r)",
                agent, deferred,
            )
            return ungoverned_block(
                agent, f"{_APPROVAL_IN_WORDS} could not be read"
            )
        approval_rules = _rules(approval)
        threshold = approval_rules.get("default_threshold_gbp")
        if threshold is None:
            log.warning(
                "authority: approval policy %r sets no default_threshold_gbp for %s",
                deferred, agent,
            )
            return ungoverned_block(
                agent,
                f"{_APPROVAL_IN_WORDS} sets no threshold amount -- so there is no "
                "limit to enforce",
            )
        limit_gbp = str(threshold)
        # NOT `or "GBP"`. Defaulting the denomination fabricates one: a limit whose
        # policy states no currency would arrive at the decision engine looking like a
        # sterling limit, and its currency gate would then match a GBP reply against a
        # denomination nobody chose. None means "the policy does not say", which the
        # decision engine escalates on. governed stays True -- the escalation belongs at
        # decide time, where it can be explained to a human, not in the resolver.
        currency = approval_rules.get("currency")
        limit_currency = str(currency) if currency else None

    ids = _ids(autonomy)
    confidence = rules.get("min_intent_confidence")
    cap = rules.get("max_auto_replies_per_thread")
    return {
        "agent": agent,
        "governed": True,
        # The slug we asked for -- not the policy's own display-name-derived
        # `slug` attribute, which is a different string (PolicyEngine slugifies
        # `policy_name`, e.g. "EmailReplyAutonomyPolicy" ->
        # "email_reply_autonomy_policy"). `autonomy` only reached this point
        # because that lookup already resolved by alias, so this is accurate.
        "slug": autonomy_slug,
        "policy_id": ids["policy_id"],
        "policy_name": ids["policy_name"],
        "auto_intents": _str_list(rules.get("auto_reply_intents")),
        "escalate_intents": _str_list(rules.get("escalate_intents")),
        "limit_gbp": limit_gbp,
        "limit_currency": limit_currency,
        "max_auto_replies_per_thread": int(cap) if cap is not None else None,
        "min_intent_confidence": float(confidence) if confidence is not None else None,
        "reason": "resolved from governed policy",
    }


def resolve_authority(
    policy_engine: Any,
    agents: Sequence[str],
    *,
    autonomy_slug: str = DEFAULT_AUTONOMY_SLUG,
) -> Dict[str, Dict[str, Any]]:
    """Resolve each agent's send authority concurrently. Never raises."""
    names = [str(a) for a in agents if a]
    if not names:
        return {}
    workers = min(_MAX_WORKERS, len(names))
    out: Dict[str, Dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="authority") as pool:
        futures = {
            pool.submit(_resolve_one, policy_engine, name, autonomy_slug): name
            for name in names
        }
        for future, name in futures.items():
            try:
                out[name] = future.result()
            except Exception:  # noqa: BLE001 - a thread that died is not a licence to send
                log.exception("authority resolution thread failed for %s", name)
                out[name] = ungoverned_block(name, "authority resolution failed")
    return out
