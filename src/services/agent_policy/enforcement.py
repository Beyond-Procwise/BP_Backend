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
from typing import Any, Dict, List, Optional

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


def mask(values: Dict[str, Any], policy: Dict[str, Any]) -> Dict[str, Any]:
    sensitive = {i.get("field") for i in policy.get("inputs") or [] if i.get("sensitive")}
    return {k: (MASK if k in sensitive else v) for k, v in (values or {}).items()}


def _lookup(context: Dict[str, Any], path: str) -> Any:
    cur: Any = context
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return _ABSENT
        cur = cur[part]
    return _ABSENT if cur is None else cur


def _seconds(iso: Optional[str]) -> int:
    m = re.match(r"^P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?)?$", iso or "")
    if not m:
        return -1
    d, h, mi, s = (int(x or 0) for x in m.groups())
    return ((d * 24 + h) * 60 + mi) * 60 + s


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
        matched = trigger.get("onMissingData") == "fail_closed"
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
        "matched_values": mask(present, policy),
        "missing": missing,
        # stage 1 stores scope.limit as free text; nothing can enforce it, so say so.
        "limitIgnored": bool((policy.get("scope") or {}).get("limit")),
        "policy": policy,
    }


def check(ctx: Dict[str, Any], policies: List[Dict[str, Any]]) -> Verdict:
    checkpoint = (ctx or {}).get("checkpoint")
    context = conditions.nest({k: v for k, v in (ctx or {}).items() if k != "checkpoint"})
    v = Verdict()
    for policy in policies or []:
        if not isinstance(policy, dict) or (policy.get("context") or {}).get("checkpoint") != checkpoint:
            continue
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

    if v.blocks:
        first = v.blocks[0]
        v.result = "blocked"
        v.to_agent = {"result": "blocked", "reasonCode": first["reasonCode"], "reason": first["reason"],
                      "messageForPerson": first["messageForPerson"],
                      "policies": [h["id"] for h in v.blocks]}
    elif v.approvals:
        first = v.approvals[0]
        # every approval must arrive, so the agent is told the longest wait
        within = max((_respond_within(h["policy"]) for h in v.approvals), key=_seconds, default=None)
        v.result = "paused_for_approval"
        v.to_agent = {"result": "paused_for_approval", "requestIds": [], "respondWithin": within,
                      "whilePaused": ((first["policy"].get("outputs") or {}).get("toAgent") or {}).get("whilePaused")
                      or "no_retry",
                      "reasonCode": first["reasonCode"], "reason": first["reason"],
                      "messageForPerson": first["messageForPerson"]}
    return v
