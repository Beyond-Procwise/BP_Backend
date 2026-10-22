"""Step 3b: the figures a model passes INTO a tool.

The verify step checks what AgentNick says. This checks what it asks to have run. A model that is
told never to supply a rate will still, now and then, pass ``cost_of_capital: 0.08`` or a seven year
horizon that nobody gave it, and the tool will compute on it as if it were a fact.

The check is deliberately narrow. It covers *assumptions*: the rates, horizons, tolerances and
thresholds a person or a governed policy has to supply (``ASSUMPTION_ARGS``). Each one must appear in
the request, in something an earlier tool call returned, or in the skill's own settings (allowing for
``10 percent`` stated as ``0.1``, ``120k`` as ``120000`` and numbers written as words). Amounts,
scores, weights and totals are not checked here: a model adding up the orders into a total is doing
arithmetic, not inventing a fact, and the answer it finally gives is traced by ``verifier``.

A call that fails is not run. The model is told which figure, and to ask the user instead.
This module is a pure check: it calls no model and no tool, and changes nothing it is given.
"""
from __future__ import annotations

import json
import re
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

# Inputs that are assumptions: someone has to supply them, and the model may not. Every name here must
# be a real input of some calculation or tool (a test checks), so a rename cannot silently empty it.
ASSUMPTION_ARGS = frozenset({
    "cost_of_capital", "default_insurance_rate", "duty_rate", "fx_rates", "fx_rate_to_policy_currency",
    "horizon_years", "margin_pct", "overhead_pct", "sga_pct", "price_reference", "target_payment_days",
    "early_tolerance_days", "late_tolerance_days", "in_full_tolerance", "sensitivity_pct",
    # deal.pct_change: the two figures compared must be figures the run holds, never a pair the model made up
    # (live, 2026-10-22: invoiced_total 100,000 against po_total 95,000 from nowhere).
    "new", "base",
    # deal.share_pct and deal.scaled_total: the same rule; a share or a scaled total is only as honest as
    # the two figures that go in.
    "part", "whole", "unit_value", "multiplier",
})

#: Identifiers a model passes to a tool. An id is never worked out: it comes from the request or from
#: something an earlier tool returned. Live, 2026-10-22: asked about "our steel supplier", the model called
#: get_opportunity_bucket with item_id "STEEL-PLATE-3MM", an id nothing had returned.
IDENTIFIER_ARGS = frozenset({
    "supplier_id", "item_id", "document_id", "waiver_id", "po_id",
})
MIN_IDENTIFIER_CHARS = 3

#: Names a model passes to a lookup. Live, 2026-10-22: given only "the supplier we are leaving", the model
#: called search_contracts with "Acme Logistics", supplier_lookup with "large_commodity_supplier" and
#: calc_supplier_scorecard with "ABC Corp": names from nowhere, each costing a round. A name is sourced
#: when every word of it (three letters or more) appears in the request, an earlier result or the settings;
#: "Kestrel" in the request sources "Kestrel", and a lookup that returned "Kestrel Supplies 8" sources that.
NAME_ARGS = frozenset({"supplier", "supplier_name"})
_NAME_WORD = re.compile(r"[a-z]{3,}")

# Tools whose results describe what exists (inputs, ranges, playbooks), not what a value is. Their numbers
# are never a source: "range [0, 1]" in a catalogue must not make a duty rate of 0 look supplied.
NON_SOURCE_TOOLS = frozenset({"list_calculations", "list_governance", "list_skills", "use_skill", "get_prompt",
                              "list_opportunity_levers"})

_NUMBER = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s?(k|m|bn|thousand|million|billion)?\b", re.IGNORECASE)
_SCALE = {"k": Decimal(1000), "thousand": Decimal(1000), "m": Decimal(1_000_000), "million": Decimal(1_000_000),
          "bn": Decimal(1_000_000_000), "billion": Decimal(1_000_000_000)}
_WORDS = {w: Decimal(n) for n, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen twenty".split())}
_WORDS.update({"thirty": Decimal(30), "forty": Decimal(40), "fifty": Decimal(50), "sixty": Decimal(60),
               "seventy": Decimal(70), "eighty": Decimal(80), "ninety": Decimal(90), "hundred": Decimal(100)})


def _dec(value: Any) -> Optional[Decimal]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return Decimal(str(value).replace(",", "").strip())
    except (InvalidOperation, ValueError):
        return None


def _known(texts: Iterable[str]) -> Set[Decimal]:
    out: Set[Decimal] = set()
    for text in texts:
        for m in _NUMBER.finditer(text or ""):
            n = _dec(m.group(1))
            if n is None:
                continue
            out.add(n)
            if m.group(2):
                out.add(n * _SCALE[m.group(2).lower()])
        for word in re.findall(r"[a-z]+", (text or "").lower()):
            if word in _WORDS:
                out.add(_WORDS[word])
    return out


def _supported(value: Decimal, known: Set[Decimal]) -> bool:
    q = Decimal("0.000001")
    norm = {k.quantize(q) for k in known}
    for candidate in (value, value * 100, value / 100):
        if candidate.quantize(q) in norm:
            return True
    return False


def _leaves(node: Any, path: str) -> Iterable[Tuple[str, Any]]:
    if isinstance(node, Mapping):
        for k, v in node.items():
            yield from _leaves(v, f"{path}.{k}")
    elif isinstance(node, (list, tuple)):
        for i, v in enumerate(node):
            yield from _leaves(v, f"{path}[{i}]")
    else:
        yield path, node


def _assumption_values(node: Any, path: str = "") -> Iterable[Tuple[str, Any]]:
    """Every numeric value sitting under an assumption name, anywhere in the arguments."""
    if isinstance(node, Mapping):
        for k, v in node.items():
            p = f"{path}.{k}" if path else str(k)
            if k in ASSUMPTION_ARGS:
                for leaf_path, leaf in _leaves(v, p):
                    if _dec(leaf) is not None:
                        yield leaf_path, leaf
            else:
                yield from _assumption_values(v, p)
    elif isinstance(node, (list, tuple)):
        for i, v in enumerate(node):
            yield from _assumption_values(v, f"{path}[{i}]")


def _identifier_values(node: Any, path: str = "") -> Iterable[Tuple[str, str]]:
    """Every identifier string sitting under an identifier name, anywhere in the arguments."""
    if isinstance(node, Mapping):
        for k, v in node.items():
            p = f"{path}.{k}" if path else str(k)
            if k in IDENTIFIER_ARGS:
                for leaf_path, leaf in _leaves(v, p):
                    if isinstance(leaf, str) and len(leaf.strip()) >= MIN_IDENTIFIER_CHARS:
                        yield leaf_path, leaf.strip()
            else:
                yield from _identifier_values(v, p)
    elif isinstance(node, (list, tuple)):
        for i, v in enumerate(node):
            yield from _identifier_values(v, f"{path}[{i}]")


def _source_texts(task: str, calls: Sequence[Any], skill_config: Optional[Mapping[str, Any]]) -> List[str]:
    texts = [task or ""]
    for c in calls:
        if (getattr(c, "ok", False) and getattr(c, "result", None) is not None
                and getattr(c, "name", None) not in NON_SOURCE_TOOLS):
            texts.append(json.dumps(c.result, default=str))
    if skill_config:
        texts.append(json.dumps(skill_config, default=str))
    return texts


def _name_values(node: Any, path: str = "") -> Iterable[Tuple[str, str]]:
    """Every name string sitting under a name argument, anywhere in the arguments."""
    if isinstance(node, Mapping):
        for k, v in node.items():
            p = f"{path}.{k}" if path else str(k)
            if k in NAME_ARGS:
                for leaf_path, leaf in _leaves(v, p):
                    if isinstance(leaf, str) and _NAME_WORD.search(leaf.lower()):
                        yield leaf_path, leaf.strip()
            else:
                yield from _name_values(v, p)
    elif isinstance(node, (list, tuple)):
        for i, v in enumerate(node):
            yield from _name_values(v, f"{path}[{i}]")


def untraced_names(args: Mapping[str, Any], *, task: str, calls: Sequence[Any],
                   skill_config: Optional[Mapping[str, Any]] = None) -> List[Tuple[str, str]]:
    """The names in ``args`` with a word that appears in none of the request, a successful earlier tool
    result or the skill's settings. Words are compared case-insensitively; short words are ignored."""
    haystack = set(_NAME_WORD.findall("\n".join(_source_texts(task, calls, skill_config)).lower()))
    return [(path, value) for path, value in _name_values(args)
            if not set(_NAME_WORD.findall(value.lower())) <= haystack]


def name_refusal(tool: str, found: Sequence[Tuple[str, str]]) -> Dict[str, Any]:
    listed = "; ".join(f"{p} = {v}" for p, v in found)
    return {
        "result": "blocked",
        "reasonCode": "name_not_sourced",
        "reason": (f"{tool} was not run: {listed} is a name that appears in neither the request nor any earlier "
                   "tool result. Never invent a supplier. Use the name the request gives, find it with "
                   "supplier_lookup or supplier_discovery, or ask the user which supplier they mean."),
    }


def untraced_identifiers(args: Mapping[str, Any], *, task: str, calls: Sequence[Any],
                         skill_config: Optional[Mapping[str, Any]] = None) -> List[Tuple[str, str]]:
    """The identifiers in ``args`` that appear in neither the request nor any successful earlier tool
    result (nor the skill's settings). Compared case-insensitively, as whole strings."""
    haystack = "\n".join(_source_texts(task, calls, skill_config)).lower()
    return [(path, value) for path, value in _identifier_values(args) if value.lower() not in haystack]


def identifier_refusal(tool: str, found: Sequence[Tuple[str, str]]) -> Dict[str, Any]:
    listed = "; ".join(f"{p} = {v}" for p, v in found)
    return {
        "result": "blocked",
        "reasonCode": "identifier_not_sourced",
        "reason": (f"{tool} was not run: {listed} appears in neither the request nor any earlier tool result. "
                   "Never invent an id. Find it with a lookup or search tool first, or ask the user for it."),
    }


def untraced_assumptions(args: Mapping[str, Any], *, task: str, calls: Sequence[Any],
                         skill_config: Optional[Mapping[str, Any]] = None,
                         today: Optional[date] = None) -> List[Tuple[str, Any]]:
    """The assumption figures in ``args`` that came from none of the request, a successful earlier
    tool result, the skill's settings, or today's date."""
    now = today or date.today()
    texts = [task or "", now.isoformat(), str(now.year)]
    for c in calls:
        if (getattr(c, "ok", False) and getattr(c, "result", None) is not None
                and getattr(c, "name", None) not in NON_SOURCE_TOOLS):
            texts.append(json.dumps(c.result, default=str))
    if skill_config:
        texts.append(json.dumps(skill_config, default=str))
    known = _known(texts)
    out: List[Tuple[str, Any]] = []
    for path, value in _assumption_values(args):
        n = _dec(value)
        if n is not None and not _supported(n, known):
            out.append((path, value))
    return out


def refusal(tool: str, found: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    listed = "; ".join(f"{p} = {v}" for p, v in found)
    return {
        "result": "blocked",
        "reasonCode": "figure_not_sourced",
        "reason": (f"{tool} was not run: {listed} is not in the request or in any earlier tool result. "
                   "Do not assume or estimate a rate, horizon or threshold. Ask the user for it, or leave the "
                   "argument out if it is optional, and say that you did."),
    }


def make_checker(task: str, skill_config: Optional[Mapping[str, Any]], *,
                 today: Optional[date] = None) -> Callable[[str, Mapping[str, Any], Any], Optional[Dict[str, Any]]]:
    """The check for one run: ``(tool_name, args, result_so_far) -> refusal or None``."""

    def check(name: str, args: Mapping[str, Any], result: Any) -> Optional[Dict[str, Any]]:
        calls = getattr(result, "calls", []) or []
        found = untraced_assumptions(args, task=task, calls=calls, skill_config=skill_config, today=today)
        if found:
            return refusal(name, found)
        ids = untraced_identifiers(args, task=task, calls=calls, skill_config=skill_config)
        if ids:
            return identifier_refusal(name, ids)
        names = untraced_names(args, task=task, calls=calls, skill_config=skill_config)
        return name_refusal(name, names) if names else None

    return check
