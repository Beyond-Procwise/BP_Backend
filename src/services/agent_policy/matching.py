"""Which extracted policy is which existing policy, and did it change? (pure)

A policy extracted from a document keeps its id across the document's revisions. The
existing policies are the ones whose ``source_document_id`` is the document; each proposed
form (in document order) is matched to at most one of them:

1. exact: same reference and same split key (the stored one, or the latest form's);
2. same reference, and the only unmatched existing policy there with the same outcome;
3. excerpt similarity >= 0.85, same outcome, unmatched (the clause was renumbered).

Each pass runs over every proposed form before the next pass starts, so a loose match can
never take an existing policy that a later form matches exactly.

Decisions (the revised-document ruling): a matched form that says the same thing is
``unchanged`` (nothing is written); one that differs is ``changed`` (a new draft, the live
version stays live); an unmatched form is ``new``; an existing policy nothing matched is
``proposed_retire`` (a suggestion for a person; never retired automatically).
"""
from __future__ import annotations

import difflib
import json
import re
from typing import Any, Dict, List, Optional

SIMILARITY = 0.85


def _number(value: Any) -> Optional[str]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return str(int(value)) if float(value).is_integer() else repr(float(value))


def _boundaries(node: Any, out: List[str]) -> None:
    if isinstance(node, list):
        for n in node:
            _boundaries(n, out)
    elif isinstance(node, dict):
        if "op" in node and "field" in node:
            num = _number(node.get("value"))
            if num is not None:
                out.append(f"{node['op']}:{num}")
            return
        for v in node.values():
            _boundaries(v, out)


def split_key(form: Dict[str, Any]) -> str:
    """``outcome|op:number,...`` -- what tells the tiers of one clause apart."""
    nums: List[str] = []
    _boundaries(((form or {}).get("hidden") or {}).get("condition"), nums)
    return f"{(form or {}).get('outcome') or ''}|{','.join(sorted(nums))}"


def _text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def _excerpt(form: Dict[str, Any]) -> str:
    return _text(((form or {}).get("source") or {}).get("excerpt"))


def _reference(form: Dict[str, Any]) -> Optional[str]:
    return ((form or {}).get("source") or {}).get("reference")


def _canon(node: Any) -> Any:
    """A condition with its all/any members in a fixed order: reordering is not a change."""
    if isinstance(node, list):
        return sorted((_canon(n) for n in node), key=lambda n: json.dumps(n, sort_keys=True))
    if isinstance(node, dict):
        return {k: (_canon(v) if k in ("all", "any", "not") else v) for k, v in node.items()}
    return node


def _floats(node: Any) -> Any:
    """500 and 500.0 are the same number (a form read back from JSON may hold either)."""
    if isinstance(node, bool):
        return node
    if isinstance(node, int):
        return float(node)
    if isinstance(node, list):
        return [_floats(n) for n in node]
    if isinstance(node, dict):
        return {k: _floats(v) for k, v in node.items()}
    return node


def _substance(form: Dict[str, Any]) -> Dict[str, Any]:
    form = form or {}
    hidden = form.get("hidden") or {}
    examples = sorted((_floats(e.get("input") or {}) for e in form.get("examples") or []),
                      key=lambda i: json.dumps(i, sort_keys=True))
    return _floats({
        "situation": _text(form.get("situation")),
        "outcome": form.get("outcome"),
        "condition": _canon(_floats(hidden.get("condition"))),
        "deciders": list(form.get("deciders") or []),
        "notify": list(form.get("notify") or []),
        "excerpt": _excerpt(form),
        "inputs": sorted(i.get("field") or "" for i in hidden.get("inputs") or []),
        "checkpoint": hidden.get("checkpoint"),
        "examples": examples,
    })


def substantively_equal(old_form: Dict[str, Any], new_form: Dict[str, Any]) -> bool:
    """Same situation, outcome, condition, deciders, notify, excerpt, input fields, checkpoint and
    example inputs. What the agent expected, the confirmation, the change note, the owner and the
    dates are not the policy's substance."""
    return _substance(old_form) == _substance(new_form)


def match(existing: List[Dict[str, Any]], proposed: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """``existing``: ``[{policyKey, reference, split, form}]``; ``proposed``: forms in document order.

    Returns one ``{policyKey, decision}`` per proposed form, in order, then one
    ``{policyKey, decision: "proposed_retire"}`` per existing policy left unmatched.
    """
    taken: Dict[int, int] = {}          # proposed index -> existing index
    used: set = set()

    def claim(pi: int, ei: int) -> None:
        taken[pi] = ei
        used.add(ei)

    # 1. exact (reference, split key)
    for pi, form in enumerate(proposed):
        ref, key = _reference(form), split_key(form)
        for ei, ex in enumerate(existing):
            if ei in used or ex.get("reference") != ref:
                continue
            if key in (ex.get("split"), split_key(ex.get("form") or {})):
                claim(pi, ei)
                break

    # 2. same reference, the only unmatched existing policy there with the same outcome
    for pi, form in enumerate(proposed):
        if pi in taken:
            continue
        ref, outcome = _reference(form), form.get("outcome")
        same = [ei for ei, ex in enumerate(existing)
                if ei not in used and ex.get("reference") == ref
                and (ex.get("form") or {}).get("outcome") == outcome]
        if len(same) == 1:
            claim(pi, same[0])

    # 3. the excerpt says nearly the same thing (renumbered clause)
    for pi, form in enumerate(proposed):
        if pi in taken:
            continue
        text, outcome = _excerpt(form), form.get("outcome")
        if not text:
            continue
        best, best_ratio = None, SIMILARITY
        for ei, ex in enumerate(existing):
            exf = ex.get("form") or {}
            if ei in used or exf.get("outcome") != outcome or not _excerpt(exf):
                continue
            ratio = difflib.SequenceMatcher(None, _excerpt(exf), text).ratio()
            if ratio >= best_ratio:
                best, best_ratio = ei, ratio
        if best is not None:
            claim(pi, best)

    out: List[Dict[str, Any]] = []
    for pi, form in enumerate(proposed):
        if pi not in taken:
            out.append({"policyKey": None, "decision": "new"})
            continue
        ex = existing[taken[pi]]
        same = substantively_equal(ex.get("form") or {}, form)
        out.append({"policyKey": ex["policyKey"], "decision": "unchanged" if same else "changed"})
    for ei, ex in enumerate(existing):
        if ei not in used:
            out.append({"policyKey": ex["policyKey"], "decision": "proposed_retire"})
    return out
