"""What the evidence supports loosening in a policy's condition (brief §6.2, §6.4).

Pure. A threshold is loosened by moving its number toward what people approved; a list by
removing a value people always approve. Only parts of the condition that every match must
satisfy are considered: under `any` or `not`, changing one part does not mean one thing.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from typing import Any, Dict, List, Optional, Sequence

from services import policy_condition as pc
from services.agent_policy.learning_stats import Decided, failed_guardrails, stats

UP = ("gt", "gte")       # the policy matches above the number: loosening raises it
DOWN = ("lt", "lte")     # the policy matches below the number: loosening lowers it


@dataclass(frozen=True)
class Leaf:
    kind: str
    field: str
    op: str
    value: Any


def _number(value: Any) -> Optional[Decimal]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return pc._number(value, "learning")
    except pc.ConditionError:
        return None


def _plain(d: Decimal):
    return int(d) if d == d.to_integral_value() else float(d)


def _must(cond: Any) -> List[Dict[str, Any]]:
    """The leaves every match must satisfy: the condition itself, or inside nested `all`."""
    if not isinstance(cond, dict):
        return []
    if "all" in cond:
        return [leaf for c in cond["all"] if isinstance(cond["all"], list) for leaf in _must(c)]
    if "any" in cond or "not" in cond:
        return []
    return [cond]


def eligible_leaves(condition: Any) -> List[Leaf]:
    out: List[Leaf] = []
    for leaf in _must(condition):
        fld, op, value = leaf.get("field"), leaf.get("op"), leaf.get("value")
        if not fld or fld == "tool.name":
            continue
        if op in UP + DOWN and isinstance(value, (int, float)) and not isinstance(value, bool):
            out.append(Leaf("threshold", fld, op, value))
        elif op == "in" and isinstance(value, list) and len(value) >= 2:
            out.append(Leaf("list", fld, op, value))
    return out


def _nice(c: Decimal, up: bool) -> Decimal:
    """c at two significant figures, rounded toward the current limit (never past the evidence)."""
    if c == 0:
        return c
    step = Decimal(1).scaleb(c.adjusted() - 1)
    return c.quantize(step, rounding=ROUND_FLOOR if up else ROUND_CEILING)


def _band(points, frm: Decimal, to: Decimal, up: bool) -> List[Decided]:
    return [d for v, d in points if (frm <= v <= to if up else to <= v <= frm)]


def _threshold(leaf: Leaf, decided: Sequence[Decided], cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    frm, up = _number(leaf.value), leaf.op in UP
    points = [(v, d) for d in decided if (v := _number(d.values.get(leaf.field))) is not None
              and (v >= frm if up else v <= frm)]
    best = None
    for cut in sorted({v for v, _ in points}, reverse=not up):
        if not failed_guardrails(stats(_band(points, frm, cut, up)), cfg):
            best = cut                       # the furthest band that holds
    if best is None:
        return None
    nice = _nice(best, up)
    if nice != best and (nice > frm if up else nice < frm) \
            and not failed_guardrails(stats(_band(points, frm, nice, up)), cfg):
        best = nice
    if best == frm:
        return None
    return {"kind": "threshold", "field": leaf.field, "op": leaf.op, "from": _plain(frm), "to": _plain(best),
            "changeKey": f"threshold:{leaf.field}:{leaf.op}:{_plain(frm)}",
            "stats": stats(_band(points, frm, best, up))}


def _same(a: Any, b: Any) -> bool:
    return not isinstance(a, (list, dict, bool)) and a is not None and str(a) == str(b)


def _list(leaf: Leaf, decided: Sequence[Decided], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for value in leaf.value:
        group = [d for d in decided if _same(d.values.get(leaf.field), value)]
        s = stats(group)
        if not failed_guardrails(s, cfg):
            out.append({"kind": "list", "field": leaf.field, "op": "in", "remove": value,
                        "changeKey": f"list:{leaf.field}:{value}", "stats": s})
    return out


def propose(condition: Any, decided: Sequence[Decided], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for leaf in eligible_leaves(condition):
        if leaf.kind == "threshold":
            p = _threshold(leaf, decided, cfg)
            out.extend([p] if p else [])
        else:
            out.extend(_list(leaf, decided, cfg))
    return out


def group_for(proposal: Dict[str, Any], decided: Sequence[Decided]) -> List[Decided]:
    fld = proposal["field"]
    if proposal["kind"] == "list":
        return [d for d in decided if _same(d.values.get(fld), proposal["remove"])]
    frm, to, up = _number(proposal["from"]), _number(proposal["to"]), proposal["op"] in UP
    points = [(v, d) for d in decided if (v := _number(d.values.get(fld))) is not None]
    return _band(points, frm, to, up)


def _leaf_of(proposal: Dict[str, Any], condition: Any) -> Optional[Leaf]:
    for leaf in eligible_leaves(condition):
        if leaf.field == proposal["field"] and leaf.op == proposal["op"] and leaf.kind == proposal["kind"]:
            return leaf
    return None


def targets(proposal: Dict[str, Any], condition: Any) -> bool:
    leaf = _leaf_of(proposal, condition)
    if leaf is None:
        return False
    if proposal["kind"] == "list":
        return any(_same(v, proposal["remove"]) for v in leaf.value)
    return _number(leaf.value) == _number(proposal["from"])


def applied(proposal: Dict[str, Any], condition: Any) -> bool:
    """The condition is looser than the proposal's starting state, in the proposal's direction."""
    if proposal["kind"] == "list":
        leaf = next((l for l in _must(condition) if l.get("field") == proposal["field"] and l.get("op") == "in"
                     and isinstance(l.get("value"), list)), None)
        return leaf is not None and not any(_same(v, proposal["remove"]) for v in leaf["value"])
    leaf = _leaf_of(proposal, condition)
    if leaf is None:
        return False
    now, frm = _number(leaf.value), _number(proposal["from"])
    return now > frm if proposal["op"] in UP else now < frm
