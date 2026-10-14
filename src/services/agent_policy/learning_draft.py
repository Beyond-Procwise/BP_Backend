"""'Draft this change' (brief §6.4): the next version as an unsaved form, built by code.

Pure. The condition is changed, the wording follows where that is unambiguous, the examples
are moved to the new boundary and the confirmation is cleared. Nothing is saved here: only
the owner's save, and then the normal activate step, changes a policy.
"""
from __future__ import annotations

import copy
import re
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional

from services.agent_policy import learning_patterns as patterns
from services.agent_policy.conditions import ConditionError, example_result
from services.agent_policy.learning_patterns import UP

WORDING = ("situation", "messageForAgent", "messageForPerson")
# A number with thousands commas (1,250.50) or without (1250.50). "500," is 500 then punctuation.
_NUMBER = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")
_NOT_BEFORE = "-/."      # ... or a letter or digit: A500, 2026-10, 4.500
_NOT_AFTER = "-/%"       # ... or a letter or digit: 500-B, 10/2026, 500%
_PLACES = Decimal("0.000001")


def _dec(value: Any) -> Optional[Decimal]:
    """A number for showing and comparing in wording: float noise (803.9000000000001) rounded away."""
    if isinstance(value, bool):
        return None
    try:
        d = Decimal(str(value).replace(",", "").strip())
    except (InvalidOperation, ValueError):
        return None
    return d.quantize(_PLACES).normalize() if d.is_finite() else None


def _show(value: Any, commas: bool = False) -> str:
    d = _dec(value)
    return str(value) if d is None else format(d, ",f" if commas else "f")


def _input(form: Dict[str, Any], fld: str) -> Dict[str, Any]:
    for i in (form.get("hidden") or {}).get("inputs") or []:
        if i.get("field") == fld:
            return i
    return {}


def _amount(value: Any, unit: Optional[str]) -> str:
    return f"{_show(value)} {unit}" if unit else _show(value)


def plain(proposal: Dict[str, Any], form: Dict[str, Any]) -> str:
    spec = _input(form, proposal["field"])
    name = spec.get("name") or proposal["field"]
    if proposal["kind"] == "list":
        return f"Stop asking for approval when {name} is {proposal['remove']}?"
    verb = "Raise" if proposal["op"] in UP else "Lower"
    unit = spec.get("unit")
    return f"{verb} the limit on {name} from {_amount(proposal['from'], unit)} to {_amount(proposal['to'], unit)}?"


def evidence(proposal: Dict[str, Any], form: Dict[str, Any]) -> str:
    s, spec = proposal["stats"], _input(form, proposal["field"])
    if proposal["kind"] == "list":
        what = f"requests where {spec.get('name') or proposal['field']} is {proposal['remove']}"
    else:
        lo, hi = sorted([proposal["from"], proposal["to"]], key=_dec)
        what = f"requests between {_amount(lo, spec.get('unit'))} and {_amount(hi, spec.get('unit'))}"
    return (f"{s['approved']} of {s['decisions']} {what} were approved, "
            f"by {s['approvers']} different people over {s['days']} {'day' if s['days'] == 1 else 'days'}.")


def _candidate(text: str, m: "re.Match[str]") -> bool:
    """Whether a number in the wording stands alone, so it could be the limit. Part of a reference
    (A500), a date (2026-10-01), a code (500-B) or a percentage (500%) is not."""
    before, after = text[m.start() - 1:m.start()], text[m.end():m.end() + 1]
    return not (before.isalnum() or (before and before in _NOT_BEFORE)
                or after.isalnum() or (after and after in _NOT_AFTER))


def _swap_number(text: Any, old: Any, new: Any) -> Optional[str]:
    """`text` with its one mention of `old` replaced by `new`; None unless there is exactly one."""
    if not isinstance(text, str):
        return None
    hits = [m for m in _NUMBER.finditer(text) if _candidate(text, m) and _dec(m.group()) == _dec(old)]
    if len(hits) != 1:
        return None
    m = hits[0]
    return text[:m.start()] + _show(new, commas="," in m.group()) + text[m.end():]


def _replace_leaf(cond: Any, proposal: Dict[str, Any]) -> None:
    if not isinstance(cond, dict):
        return
    if "all" in cond:
        for c in cond["all"] if isinstance(cond["all"], list) else []:
            _replace_leaf(c, proposal)
        return
    if "any" in cond or "not" in cond:
        return
    if cond.get("field") != proposal["field"] or cond.get("op") != proposal["op"]:
        return
    if proposal["kind"] == "list":
        cond["value"] = [v for v in cond["value"] if not patterns._same(v, proposal["remove"])]
    else:
        cond["value"] = proposal["to"]


def _leaf_value(cond: Any, proposal: Dict[str, Any]) -> Any:
    for leaf in patterns._must(cond):
        if leaf.get("field") == proposal["field"] and leaf.get("op") == proposal["op"]:
            return leaf.get("value")
    return None


def _examples(form: Dict[str, Any], proposal: Dict[str, Any], on_missing: str) -> List[Dict[str, Any]]:
    fld, old = proposal["field"], form.get("examples") or []
    kept = old
    if proposal["kind"] == "threshold":
        whole = isinstance(proposal["from"], int) and isinstance(proposal["to"], int)
        step = 1 if whole else 0.01
        near = lambda v, c: isinstance(v, (int, float)) and abs(v - c) <= step * 1.5   # noqa: E731
        kept = [e for e in old if not near((e.get("input") or {}).get(fld), proposal["from"])]
        base = dict(((kept or old or [{}])[0].get("input")) or {})
        # the base must be an input the policy applies to, or the boundary would show nothing
        for e in old:
            if e.get("agentExpected") not in (None, "none"):
                base = dict(e.get("input") or {})
                break
        trio = [round(proposal["to"] + d * step, 2) if not whole else proposal["to"] + d for d in (-1, 0, 1)]
        kept = kept[:3] + [{"input": {**base, fld: v}} for v in trio]
    cond, outcome = form["hidden"]["condition"], form.get("outcome")
    return [{"input": e.get("input") or {}, "flipped": False,
             "agentExpected": example_result(cond, outcome, e.get("input") or {}, on_missing)} for e in kept]


def build(form: Dict[str, Any], proposal: Dict[str, Any], on_missing: str = "fail_closed") -> Optional[Dict[str, Any]]:
    """The unsaved next version, or None when the form no longer holds what the proposal is about,
    the change would leave a list empty, or the condition or an example cannot be read."""
    condition = (form.get("hidden") or {}).get("condition")
    if not patterns.targets(proposal, condition):
        return None
    out = copy.deepcopy(form)
    _replace_leaf(out["hidden"]["condition"], proposal)
    if proposal["kind"] == "list" and not _leaf_value(out["hidden"]["condition"], proposal):
        return None                      # an `in` with nothing in it matches nothing: not a loosening
    changed: List[str] = []
    check: List[str] = []
    for key in WORDING:
        swapped = _swap_number(out.get(key), proposal["from"], proposal["to"]) \
            if proposal["kind"] == "threshold" else None
        if swapped is None:
            check.append(key)
        else:
            out[key] = swapped
            changed.append(key)
    try:
        out["examples"] = _examples(out, proposal, on_missing)
    except ConditionError:
        return None
    out["checked"] = None
    note = f"Suggested by learning: {plain(proposal, form)} Evidence: {evidence(proposal, form)}"
    out["changeNote"] = note
    return {"form": out, "changeNote": note, "checkWording": check, "changedWording": changed}
