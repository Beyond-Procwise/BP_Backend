"""Decide what live agent policies say about one action. Pure: no database, no model, no clock.

One evaluator: conditions.to_engine + policy_condition.evaluate, the same pair the examples
and the contract use, so what the reviewer saw is what is enforced.

Precedence (user rulings, stage 3): any matching block blocks; every matching approve policy
needs its own approval; matching notify policies (and blocks with a notify list) are always
listed. A condition that cannot be read fails CLOSED, as a block.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from services import policy_condition as pc
from services.agent_policy import conditions

MASK = "•••"
_ABSENT = object()


@dataclass
class Verdict:
    result: str = "allowed"                      # allowed | paused_for_approval | blocked
    blocks: List[Dict[str, Any]] = field(default_factory=list)
    approvals: List[Dict[str, Any]] = field(default_factory=list)
    notifies: List[Dict[str, Any]] = field(default_factory=list)
    to_agent: Optional[Dict[str, Any]] = None
    evaluated: List[Dict[str, Any]] = field(default_factory=list)   # every considered policy


def _sensitive(policy: Dict[str, Any]) -> Set[str]:
    return {i.get("field") for i in policy.get("inputs") or [] if isinstance(i, dict) and i.get("sensitive")}


def _mask_fields(values: Dict[str, Any], sensitive: Set[str]) -> Dict[str, Any]:
    return {k: (MASK if k in sensitive else v) for k, v in (values or {}).items()}


def mask(values: Dict[str, Any], policy: Dict[str, Any]) -> Dict[str, Any]:
    return _mask_fields(values, _sensitive(policy))


def _lookup(context: Dict[str, Any], path: str) -> Any:
    cur: Any = context
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return _ABSENT
        cur = cur[part]
    return _ABSENT if cur is None else cur


_NUM = r"(\d+(?:[.,]\d+)?)"
_DURATION = re.compile(rf"^P(?!$)(?:{_NUM}W)?(?:{_NUM}D)?(?:T(?=\d)(?:{_NUM}H)?(?:{_NUM}M)?(?:{_NUM}S)?)?$")


def _seconds(iso: Optional[str]) -> float:
    """Length of an ISO 8601 duration. Unreadable counts as the LONGEST, so the agent is
    never told a shorter wait than a real one."""
    m = _DURATION.match(iso or "")
    if not m:
        return float("inf")
    w, d, h, mi, s = (float((x or "0").replace(",", ".")) for x in m.groups())
    return (((w * 7 + d) * 24 + h) * 60 + mi) * 60 + s


def _respond_within(policy: Dict[str, Any]) -> Optional[str]:
    return (((policy.get("enforcement") or {}).get("intervention") or {}).get("sla") or {}).get("respondWithin")


def _evaluate(policy: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    trigger = policy.get("trigger") or {}
    cond = trigger.get("condition")
    outcome = (policy.get("enforcement") or {}).get("outcome")
    fields = sorted(conditions.condition_fields(cond))
    present = {f: val for f in fields if (val := _lookup(context, f)) is not _ABSENT}
    missing: List[str] = []
    unreadable = False
    try:
        matched = pc.evaluate(conditions.to_engine(cond), context)
    except pc.MissingField as exc:
        missing = sorted({exc.field, *(f for f in fields if f not in present)})
        # anything but an explicit fail_open fails closed
        matched = trigger.get("onMissingData") != "fail_open"
    except pc.ConditionError:
        matched, unreadable, outcome = True, True, "block"
    to_agent = (policy.get("outputs") or {}).get("toAgent") or {}
    pid = policy.get("id")
    return {
        "id": pid,
        "version": policy.get("version"),
        "outcome": outcome,
        "matched": bool(matched),
        "unreadable": unreadable,
        "reasonCode": f"{pid}.condition_unreadable" if unreadable else to_agent.get("reasonCode"),
        "reason": ("This action's policy check could not be read, so it was not run."
                   if unreadable else to_agent.get("reason")),
        "messageForPerson": to_agent.get("messageForPerson"),
        "notify": list(((policy.get("outputs") or {}).get("toNotify") or {}).get("to") or []),
        "matched_values": present,   # masked in check(), with every policy's sensitive fields
        "missing": missing,
        # stage 1 stores scope.limit as free text; nothing can enforce it, so say so.
        "limitIgnored": bool((policy.get("scope") or {}).get("limit")),
        "policy": policy,
    }


def check(ctx: Dict[str, Any], policies: List[Dict[str, Any]], *,
          default_response_time: str = "PT4H") -> Verdict:
    """The verdict for one action. May raise on a malformed policy list or context; the
    caller (the gate) must treat ANY exception as policy_check_unavailable and refuse."""
    checkpoint = (ctx or {}).get("checkpoint")
    context = conditions.nest({k: v for k, v in (ctx or {}).items() if k != "checkpoint"})
    v = Verdict()
    sensitive: Set[str] = set()
    for policy in policies or []:
        if not isinstance(policy, dict) or (policy.get("context") or {}).get("checkpoint") != checkpoint:
            continue
        sensitive |= _sensitive(policy)
        hit = _evaluate(policy, context)
        v.evaluated.append(hit)
        if not hit["matched"]:
            continue
        if hit["outcome"] == "block":
            v.blocks.append(hit)
            if hit["notify"]:
                v.notifies.append(hit)
        elif hit["outcome"] == "approve":
            v.approvals.append(hit)
        elif hit["outcome"] == "notify":
            v.notifies.append(hit)

    # a field one policy marks sensitive is masked everywhere, not just in that policy's hit
    for hit in v.evaluated:
        hit["matched_values"] = _mask_fields(hit["matched_values"], sensitive)

    if v.blocks:
        first = v.blocks[0]
        v.result = "blocked"
        v.to_agent = {"result": "blocked", "reasonCode": first["reasonCode"], "reason": first["reason"],
                      "messageForPerson": first["messageForPerson"],
                      "policies": [h["id"] for h in v.blocks]}
    elif v.approvals:
        first = v.approvals[0]
        # every approval must arrive, so the agent is told the longest wait
        within = max((_respond_within(h["policy"]) or default_response_time for h in v.approvals), key=_seconds)
        v.result = "paused_for_approval"
        v.to_agent = {"result": "paused_for_approval", "requestIds": [], "respondWithin": within,
                      "whilePaused": ((first["policy"].get("outputs") or {}).get("toAgent") or {}).get("whilePaused")
                      or "no_retry",
                      "reasonCode": first["reasonCode"], "reason": first["reason"],
                      "messageForPerson": first["messageForPerson"]}
    return v
