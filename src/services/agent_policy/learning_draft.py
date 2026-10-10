"""'Draft this change' (brief §6.4): the next version as an unsaved form, built by code.

Pure. The condition is changed, the wording follows where that is unambiguous, the examples
are moved to the new boundary and the confirmation is cleared. Nothing is saved here: only
the owner's save, and then the normal activate step, changes a policy.
"""
from __future__ import annotations

import copy
import re
from typing import Any, Dict, List, Optional

from services.agent_policy import learning_patterns as patterns
from services.agent_policy.conditions import example_result
from services.agent_policy.learning_patterns import UP

WORDING = ("situation", "messageForAgent", "messageForPerson")
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _input(form: Dict[str, Any], fld: str) -> Dict[str, Any]:
    for i in (form.get("hidden") or {}).get("inputs") or []:
        if i.get("field") == fld:
            return i
    return {}


def _amount(value: Any, unit: Optional[str]) -> str:
    return f"{value} {unit}" if unit else str(value)


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
        lo, hi = sorted([proposal["from"], proposal["to"]])
        what = f"requests between {_amount(lo, spec.get('unit'))} and {_amount(hi, spec.get('unit'))}"
    return (f"{s['approved']} of {s['decisions']} {what} were approved, "
            f"by {s['approvers']} different people over {s['days']} days.")


def _swap_number(text: Any, old: Any, new: Any) -> Optional[str]:
    """`text` with its one mention of `old` replaced by `new`; None unless there is exactly one."""
    if not isinstance(text, str):
        return None
    hits = [m for m in _NUMBER.finditer(text) if float(m.group().replace(",", "")) == float(old)]
    if len(hits) != 1:
        return None
    m = hits[0]
    shown = f"{new:,}" if "," in m.group() else str(new)
    return text[:m.start()] + shown + text[m.end():]


def _replace_leaf(cond: Any, proposal: Dict[str, Any]) -> None:
    if not isinstance(cond, dict):
        return
    if "all" in cond:
        for c in cond["all"]:
            _replace_leaf(c, proposal)
        return
    if cond.get("field") != proposal["field"] or cond.get("op") != proposal["op"]:
        return
    if proposal["kind"] == "list":
        cond["value"] = [v for v in cond["value"] if str(v) != str(proposal["remove"])]
    else:
        cond["value"] = proposal["to"]


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
    condition = (form.get("hidden") or {}).get("condition")
    if not patterns.targets(proposal, condition):
        return None
    out = copy.deepcopy(form)
    _replace_leaf(out["hidden"]["condition"], proposal)
    check: List[str] = []
    for key in WORDING:
        if proposal["kind"] == "threshold":
            swapped = _swap_number(out.get(key), proposal["from"], proposal["to"])
            if swapped is None:
                check.append(key)
            else:
                out[key] = swapped
        else:
            check.append(key)
    out["examples"] = _examples(out, proposal, on_missing)
    out["checked"] = None
    note = f"Suggested by learning: {plain(proposal, form)} Evidence: {evidence(proposal, form)}"
    out["changeNote"] = note
    return {"form": out, "changeNote": note, "checkWording": check}
