"""Find one example input that two policies both match (a "witness"). Pure: no database, no model, no clock.

A witness proves two policies conflict. It is found by code, using the ONE evaluator the
orchestrator uses, from boundary values of both conditions plus the policies' own examples.
"""
from __future__ import annotations

import itertools
import json
from typing import Any, Dict, Iterable, Optional, Tuple

from services import policy_condition as pc
from services.agent_policy import conditions
from services.policy_condition import ConditionError, MissingField  # noqa: F401  (ConditionError propagates)

MAX_CANDIDATES = 5000
OTHER = "__none_of_these__"
PRESENT = "present"


def pair_key(*keys: str) -> str:
    return "|".join(sorted(set(keys)))


def source_of(doc: Dict[str, Any]) -> Optional[str]:
    name = (doc.get("source") or {}).get("document")
    if not isinstance(name, str) or not name.strip():
        return None
    return name.strip().casefold()


def same_source(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    sa = source_of(a)
    return sa is not None and sa == source_of(b)


def _outcome(doc: Dict[str, Any]) -> Optional[str]:
    return (doc.get("enforcement") or {}).get("outcome")


def deciders_of(doc: Dict[str, Any]) -> Tuple[str, ...]:
    if _outcome(doc) != "approve":
        return ()
    esc = ((doc.get("enforcement") or {}).get("intervention") or {}).get("escalateTo") or []
    return tuple(e.get("name") for e in esc if isinstance(e, dict) and e.get("name"))


def deciding(doc: Dict[str, Any]) -> bool:
    return _outcome(doc) in ("block", "approve")


def outcomes_differ(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    return _outcome(a) != _outcome(b)


def deciders_differ(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    return _outcome(a) == "approve" and _outcome(b) == "approve" and deciders_of(a) != deciders_of(b)


def _checkpoint(doc: Dict[str, Any]) -> Optional[str]:
    cp = (doc.get("context") or {}).get("checkpoint")
    if cp:
        return cp
    events = (doc.get("trigger") or {}).get("events") or []
    return events[0] if events else None


def design_time_pair(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    return (a.get("id") != b.get("id") and deciding(a) and deciding(b)
            and not same_source(a, b)
            and _checkpoint(a) is not None and _checkpoint(a) == _checkpoint(b)
            and outcomes_differ(a, b))


def _tools(doc):
    return list(((doc.get("context") or {}).get("actions") or {}).get("tools") or [])


def _cond(doc):
    return conditions.to_engine((doc.get("trigger") or {}).get("condition"))


def _matches(doc, engine_cond, flat) -> bool:
    listed = _tools(doc)
    if listed and flat.get("tool.name") not in listed:
        return False          # stage 3 rule I1: a policy that lists tools applies to those only
    try:
        return bool(pc.evaluate(engine_cond, conditions.nest(flat)))
    except pc.MissingField:
        return False          # a design-time witness never relies on missing data
    except pc.ConditionError:
        return False          # a candidate the evaluator cannot compare (e.g. the sentinel vs a number) is no match;
                              # the conditions themselves were validated up front by to_engine


def _candidates(field, leaves, examples, tools):
    vals = []
    for leaf in leaves:
        if leaf.get("field") != field:
            continue
        op, v = leaf.get("op"), leaf.get("value")
        if op in ("gt", "gte", "lt", "lte", "eq", "ne") and isinstance(v, (int, float)) and not isinstance(v, bool):
            vals += [v, v + 1, v - 1, v + 0.01, v - 0.01]
        elif op in ("in", "not_in") and isinstance(v, list):
            vals += list(v) + [OTHER]
        elif op in ("eq", "ne"):
            vals += [v, OTHER]
        elif op == "exists":
            vals.append(PRESENT)
    vals += [ex[field] for ex in examples if field in ex]
    if field == "tool.name":
        vals += tools
    out, seen = [], set()
    for v in vals:
        k = json.dumps(v, sort_keys=True, default=str)
        if k not in seen:
            seen.add(k)
            out.append(v)
    return out


def witness(a: Dict[str, Any], b: Dict[str, Any], examples: Iterable[Dict[str, Any]] = ()) -> Optional[Dict[str, Any]]:
    ca, cb = _cond(a), _cond(b)            # ConditionError propagates
    examples = [dict(e) for e in examples if isinstance(e, dict)]
    tools = sorted(set(_tools(a)) | set(_tools(b)))
    fields = sorted(conditions.condition_fields((a.get("trigger") or {}).get("condition"))
                    | conditions.condition_fields((b.get("trigger") or {}).get("condition"))
                    | ({"tool.name"} if tools else set()))
    leaves = conditions.leaves((a.get("trigger") or {}).get("condition")) + \
        conditions.leaves((b.get("trigger") or {}).get("condition"))
    tried = 0
    for ex in examples:                    # 1. the examples as written, when complete
        if all(f in ex for f in fields):
            tried += 1
            flat = {f: ex[f] for f in fields}
            if _matches(a, ca, flat) and _matches(b, cb, flat):
                return flat
    pools = [_candidates(f, leaves, examples, tools) for f in fields]
    if any(not p for p in pools):
        return None
    for combo in itertools.product(*pools):  # 2. boundary values of both conditions
        tried += 1
        if tried > MAX_CANDIDATES:
            return None
        flat = dict(zip(fields, combo))
        if _matches(a, ca, flat) and _matches(b, cb, flat):
            return flat
    return None
