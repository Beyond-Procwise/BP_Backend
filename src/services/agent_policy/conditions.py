"""Agent-policy conditions, evaluated by the ONE existing evaluator (services.policy_condition).

Stored conditions use the brief's operator names; to_engine() is the only translation.
Example results are computed here, by code, so an example can never disagree with what the
orchestrator will enforce: the model proposes inputs, never results.
"""
from __future__ import annotations

from typing import Any, Dict, List, Set

from services import policy_condition as pc
from services.policy_condition import ConditionError, MissingField  # re-exported

OPS = {"gt": ">", "gte": ">=", "lt": "<", "lte": "<=", "eq": "==", "ne": "!=",
       "in": "in", "not_in": "not_in", "exists": "exists"}
RESULT_LABEL = {"approve": "A person decides", "block": "Blocked",
                "notify": "Someone is told", "none": "Nothing happens"}


def to_engine(cond: Any) -> Dict[str, Any]:
    if not isinstance(cond, dict) or not cond:
        raise ConditionError("a condition must be a non-empty object")
    if "all" in cond or "any" in cond:
        key = "all" if "all" in cond else "any"
        if not isinstance(cond[key], list) or not cond[key]:
            raise ConditionError(f"{key!r} needs a non-empty list of conditions")
        return {key: [to_engine(c) for c in cond[key]]}
    if "not" in cond:
        return {"not": to_engine(cond["not"])}
    op = cond.get("op")
    if op not in OPS:
        raise ConditionError(f"unknown operator {op!r}")
    out = {"field": cond.get("field"), "op": OPS[op]}
    if op != "exists":
        out["value"] = cond.get("value")
    pc.validate(out)
    return out


def _leaves(cond: Any) -> List[Dict[str, Any]]:
    if not isinstance(cond, dict):
        return []
    for key in ("all", "any"):
        if key in cond:
            if not isinstance(cond[key], list):
                return []
            return [leaf for c in cond[key] for leaf in _leaves(c)]
    if "not" in cond:
        return _leaves(cond["not"])
    return [cond]


def condition_fields(cond: Any) -> Set[str]:
    return {leaf["field"] for leaf in _leaves(cond) if leaf.get("field")}


def tool_names(cond: Any) -> Set[str]:
    out: Set[str] = set()
    for leaf in _leaves(cond):
        if leaf.get("field") != "tool.name":
            continue
        value = leaf.get("value")
        out.update(value if isinstance(value, list) else [value] if value is not None else [])
    return out


def nest(flat: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for path, value in (flat or {}).items():
        node = out
        parts = str(path).split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return out


def example_result(condition: Any, outcome: Any, example_input: Dict[str, Any],
                   on_missing: str = "fail_closed") -> str:
    if outcome not in ("approve", "block", "notify"):
        return "none"
    try:
        hit = pc.evaluate(to_engine(condition), nest(example_input))
    except MissingField:
        hit = on_missing == "fail_closed"
    return outcome if hit else "none"


def reviewer_view(form: Dict[str, Any], settings: Dict[str, Any]) -> List[Dict[str, Any]]:
    from services.agent_policy.compiler import on_missing_for

    hidden = form.get("hidden") or {}
    outcome = form.get("outcome")
    rows = []
    for ex in form.get("examples") or []:
        try:
            computed = example_result(hidden.get("condition"), outcome, ex.get("input") or {},
                                      on_missing_for(form, settings))
        except ConditionError:
            computed = "invalid"
        flipped = bool(ex.get("flipped"))
        expects = computed
        if flipped and computed in ("none",) and outcome:
            expects = outcome
        elif flipped:
            expects = "none"
        rows.append({"input": ex.get("input") or {}, "computed": computed,
                     "label": RESULT_LABEL.get(computed, "The condition could not be read"),
                     "flipped": flipped, "reviewer_expects": expects,
                     "agent_expected": ex.get("agentExpected")})
    return rows
