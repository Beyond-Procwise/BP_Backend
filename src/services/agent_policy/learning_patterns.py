"""What the evidence supports loosening in a policy's condition (brief §6.2, §6.4).

Pure. A threshold is loosened by moving its number toward what people approved; a list by
removing a value people always approve. Only parts of the condition that every match must
satisfy are considered: under `any` or `not`, changing one part does not mean one thing.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from math import ceil
from typing import Any, Dict, List, Optional, Sequence

from services import policy_condition as pc
from services.agent_policy.learning_stats import Decided, _num, failed_guardrails, stats

UP = ("gt", "gte")       # the policy matches above the number: loosening raises it
DOWN = ("lt", "lte")     # the policy matches below the number: loosening lowers it


@dataclass(frozen=True)
class Leaf:
    kind: str
    field: str
    op: str
    value: Any


def _number(value: Any) -> Optional[Decimal]:
    """The value as the evaluator reads a number, or None. NaN and Infinity are not amounts."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = pc._number(value, "learning")
    except pc.ConditionError:
        return None
    return number if number.is_finite() else None


def _plain(d: Decimal):
    return int(d) if d == d.to_integral_value() else float(d)


def _same(a: Any, b: Any) -> bool:
    """Whether a decided value and a listed value are the same, as the evaluator's `actual in expected`
    sees it (1.0 is 1; "1" is not). Stricter in one place on purpose: a boolean is never a number."""
    if a is None or b is None or isinstance(a, (list, dict)) or isinstance(b, (list, dict)):
        return False
    if isinstance(a, bool) != isinstance(b, bool):
        return False
    return a == b


def _must(cond: Any) -> List[Dict[str, Any]]:
    """The leaves every match must satisfy: the condition itself, or inside nested `all`."""
    if not isinstance(cond, dict):
        return []
    if "all" in cond:
        if not isinstance(cond["all"], list):
            return []
        return [leaf for c in cond["all"] for leaf in _must(c)]
    if "any" in cond or "not" in cond:
        return []
    return [cond]


def _distinct(values: List[Any]) -> int:
    seen: List[Any] = []
    for v in values:
        if not isinstance(v, (list, dict)) and v is not None and not any(_same(v, s) for s in seen):
            seen.append(v)
    return len(seen)


def eligible_leaves(condition: Any) -> List[Leaf]:
    must = _must(condition)
    out: List[Leaf] = []
    for leaf in must:
        fld, op, value = leaf.get("field"), leaf.get("op"), leaf.get("value")
        if not fld or fld == "tool.name":
            continue
        # The same field and operator twice ("over 500" and "over 1000"): which one a suggestion is
        # about would depend on their order in the condition. Neither is touched.
        if sum(1 for other in must if other.get("field") == fld and other.get("op") == op) > 1:
            continue
        if op in UP + DOWN and isinstance(value, (int, float)) and _number(value) is not None:
            out.append(Leaf("threshold", fld, op, value))
        elif op == "in" and isinstance(value, list) and _distinct(value) >= 2:
            out.append(Leaf("list", fld, op, value))
    return out


def _nice(c: Decimal, up: bool) -> Decimal:
    """c at two significant figures, rounded toward the current limit (never past the evidence)."""
    if c == 0:
        return c
    step = Decimal(1).scaleb(c.adjusted() - 1)
    return c.quantize(step, rounding=ROUND_FLOOR if up else ROUND_CEILING)


def _evidence(v: Decimal, frm: Decimal, op: str) -> bool:
    """A decision the policy gated, on the loosening side of the current limit."""
    return {"gt": v > frm, "gte": v >= frm, "lt": v < frm, "lte": v <= frm}[op]


def _released(points, frm: Decimal, to: Decimal, op: str) -> List[Decided]:
    """Exactly the decisions that moving the limit from `frm` to `to` stops asking about.

    `gt T` -> `gt T'` releases T < v <= T'; `gte` releases T <= v < T' (a request for exactly T' is still
    gated). Mirrored for `lt` (T' <= v < T) and `lte` (T' < v <= T). The one place the edges are written.
    """
    inside = {"gt": lambda v: frm < v <= to, "gte": lambda v: frm <= v < to,
              "lt": lambda v: to <= v < frm, "lte": lambda v: to < v <= frm}[op]
    return [d for v, d in points if inside(v)]


def _bands(points, size: int) -> List[list]:
    """`points` (nearest the limit first) cut every `size` decisions. Equal values are never split: a
    band takes every decision with the same value as its last. A shorter tail is not a band."""
    bands, i = [], 0
    while len(points) - i >= size:
        j = i + size
        while j < len(points) and points[j][0] == points[j - 1][0]:
            j += 1
        bands.append(points[i:j])
        i = j
    return bands


def _threshold(leaf: Leaf, decided: Sequence[Decided], cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The furthest limit the evidence supports band by band (brief §6.2: "use bands of the amount").

    A total can look fine while its far end is one decision, or a cluster of rejections, carried by
    the approvals near the limit. So each band of `min_decisions` must pass every guardrail on its own,
    and the first band that fails ends the run however good the bands beyond it look.
    """
    frm, op, up = _number(leaf.value), leaf.op, leaf.op in UP
    points = sorted(((v, d) for d in decided
                     if (v := _number(d.values.get(leaf.field))) is not None and _evidence(v, frm, op)),
                    key=lambda p: p[0], reverse=not up)
    bands = _bands(points, max(1, ceil(_num(cfg, "min_decisions"))))
    supported = 0
    for band in bands:
        if failed_guardrails(stats([d for _, d in band]), cfg):
            break
        supported += 1
    # The released set for a band's far edge lies inside the supported bands by construction: the
    # bands are an unbroken run from the limit and a value is never split across two of them. It
    # can be smaller than the bands (`gte`/`lte` do not release the edge value itself), so it is
    # judged as a whole again; if that fails, step back a band.
    for used in range(supported, 0, -1):
        to = bands[used - 1][-1][0]
        if failed_guardrails(stats(_released(points, frm, to, op)), cfg):
            continue
        nice = _nice(to, up)
        if nice != to and (nice > frm if up else nice < frm) \
                and not failed_guardrails(stats(_released(points, frm, nice, op)), cfg):
            to = nice
        return {"kind": "threshold", "field": leaf.field, "op": op, "from": _plain(frm), "to": _plain(to),
                "changeKey": f"threshold:{leaf.field}:{op}:{_plain(frm)}",
                "stats": stats(_released(points, frm, to, op)), "bands": used}
    return None


def _list(leaf: Leaf, decided: Sequence[Decided], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    out, seen = [], []
    for value in leaf.value:
        if any(_same(value, s) for s in seen):       # [1, 1.0] is one value, not two suggestions
            continue
        seen.append(value)
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
    frm, to = _number(proposal["from"]), _number(proposal["to"])
    points = [(v, d) for d in decided if (v := _number(d.values.get(fld))) is not None]
    return _released(points, frm, to, proposal["op"])


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
        # not eligible_leaves: once the value is gone the list may hold a single value
        found = [l for l in _must(condition) if l.get("field") == proposal["field"] and l.get("op") == "in"]
        return len(found) == 1 and isinstance(found[0].get("value"), list) \
            and not any(_same(v, proposal["remove"]) for v in found[0]["value"])
    leaf = _leaf_of(proposal, condition)
    if leaf is None:
        return False
    now, frm = _number(leaf.value), _number(proposal["from"])
    return now > frm if proposal["op"] in UP else now < frm
