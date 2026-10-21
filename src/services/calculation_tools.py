"""The formula register, as something an agent consults and is held to.

Every deterministic number this system computes is a registered formula
(``src/services/formulas``): named, versioned, contract-checked and pinned by
golden vectors. Until now a formula could be reached one way only — a line of
Python in the agent that needed it. That leaves two gaps:

  * **An agent a person created has no Python of its own.** It could not be
    made to run a calculation at all, short of someone writing a class for it.
  * **A question nobody scheduled** ("what if the terms were 60 days?") had no
    route to the maths, so the model did the arithmetic itself.

Three things live here, and they are deliberately different in kind:

  * ``run_required`` — the calculations an agent was *declared* to need. They
    are run before the agent reasons, from the inputs the run arrived with, and
    handed to it as facts. The agent is not asked and cannot skip them.
  * ``list_calculations`` / ``run_calculation`` — the same register as a pair of
    tools, for the calculation that only a conversation can call for. Here the
    choice is the point. Inputs an agent typed in are marked as asserted, so the
    result never reads as better evidence than what went into it.
  * ``untraced_figures`` — the check that a figure in an answer appears in
    something the run was actually given. It reports; it does not block.

Nothing here computes anything. Every value comes from ``formulas.evaluate``.
"""
from __future__ import annotations

import json
import re
from decimal import ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple


def _formulas():
    # ``src.services.formulas``, as every caller of the register spells it: the
    # same package imported as ``services.formulas`` would be a second register.
    from src.services import formulas

    formulas.ensure_registered()
    return formulas


def _family(name: str) -> str:
    return name.split(".", 1)[0]


def _plain(value: Any) -> Any:
    """``value`` as JSON-safe data: what a model can be shown and a record can hold."""
    try:
        return json.loads(json.dumps(value, default=str))
    except Exception:  # noqa: BLE001 - a value that cannot be serialised is still shown
        return str(value)


def _input_row(term: Any) -> Dict[str, Any]:
    row = {"name": term.name, "unit": term.unit.symbol, "range": term.range_text(),
           "required": term.required}
    if term.description:
        row["description"] = term.description
    if term.shared:
        row["shared"] = True
    return row


def _describe(spec: Any) -> Dict[str, Any]:
    out = {
        "name": spec.name,
        "purpose": spec.purpose,
        "version": spec.version,
        "kind": spec.kind.value,
        "inputs": [_input_row(t) for t in spec.contract.inputs],
        "returns": spec.contract.output.to_dict(),
    }
    if spec.kind.value == "set":
        out["note"] = ("Scores a whole set at once: pass one object per member in "
                       "'rows', and the inputs marked shared in 'inputs'.")
    return out


# ---------------------------------------------------------------------------
# The two tools
# ---------------------------------------------------------------------------
def list_calculations(family: Optional[str] = None, name: Optional[str] = None) -> Dict[str, Any]:
    """What can be calculated. With no argument, every calculation by name and
    purpose; with a ``family`` or a ``name``, the inputs each one needs."""
    registry = _formulas().REGISTRY
    names = sorted(registry)
    families = sorted({_family(n) for n in names})

    if name:
        wanted = str(name).strip()
        if wanted not in registry:
            return {"found": False, "note": f"no calculation is named {wanted!r}",
                    "families": families}
        return {"found": True, "calculations": [_describe(registry[wanted])]}

    if family:
        wanted = str(family).strip().lower()
        chosen = [n for n in names if _family(n) == wanted]
        if not chosen:
            return {"found": False, "note": f"no family is named {wanted!r}",
                    "families": families}
        return {"found": True, "family": wanted,
                "calculations": [_describe(registry[n]) for n in chosen]}

    return {
        "count": len(names),
        "families": families,
        # The input names are here because a model that has to ask twice often
        # does not: it guesses them, and a guessed name is refused.
        "calculations": [{"name": n, "purpose": registry[n].purpose,
                          "inputs": [t.name for t in registry[n].contract.inputs]}
                         for n in names],
        "note": ("Use the input names exactly as listed. Call again with a family or "
                 "a name for each input's unit, range and meaning."),
    }


def _as_mapping(value: Any, what: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """A model often sends an object as a JSON string. Accept both; guess neither."""
    if value is None or value == "":
        return {}, None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return None, f"{what} was not valid JSON"
    if not isinstance(value, Mapping):
        return None, f"{what} must be an object of input names and values"
    return dict(value), None


def _findings(result: Any) -> List[str]:
    return [f.detail for f in result.findings]


def _not_given(spec: Any, given: Mapping[str, Any]) -> List[str]:
    return [t.name for t in spec.contract.inputs
            if not t.shared and given.get(t.name) is None]


def _outcome(spec: Any, result: Any, given: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"calculation": spec.name, "version": spec.version}
    absent = _not_given(spec, given)
    if result.unassessed:
        # Not a zero and not a null: there is no value, and this is why.
        out["assessed"] = False
        out["why"] = result.why() or "the calculation's inputs were not satisfied"
        if any(f.code == "undeclared_input" for f in result.findings):
            # A wrong name is a mistake that can be put right in the next call.
            out["expects"] = [_input_row(t) for t in spec.contract.inputs]
            out["retry"] = "Call run_calculation again using exactly these input names."
        return out
    if result.value is None:
        # Many formulas take every input as optional and answer None when what
        # they need is not there. To an agent that is not a value to report.
        out["assessed"] = False
        out["why"] = "the calculation has no value on the inputs supplied" + (
            f" (not given: {', '.join(absent)})" if absent else "")
        return out
    out["assessed"] = True
    out["value"] = _plain(result.value)
    out["unit"] = spec.contract.output.unit.symbol
    if result.confidence is not None:
        out["confidence"] = result.confidence.value
    if absent:
        # The value stands, on the formula's own rule for a missing input.
        out["not_given"] = absent
    notes = _findings(result)
    if notes:
        out["notes"] = notes
    return out


def run_calculation(
    name: str,
    inputs: Any = None,
    rows: Any = None,
    *,
    trace_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Run one registered calculation on inputs an agent supplied.

    The inputs are marked as asserted: a model put them there, and a result
    cannot be better evidence than what it was computed from.
    """
    formulas = _formulas()
    wanted = str(name or "").strip()
    spec = formulas.REGISTRY.get(wanted)
    if spec is None:
        return {"assessed": False, "why": f"no calculation is named {wanted!r}",
                "hint": "list_calculations names the ones that exist"}

    given, problem = _as_mapping(inputs, "inputs")
    if problem:
        return {"calculation": spec.name, "assessed": False, "why": problem}

    def _asserted(keys: Iterable[str]) -> Dict[str, Any]:
        return {k: formulas.Provenance(id=f"agent:{k}", confidence=formulas.Confidence.ASSERTED,
                                       source="supplied by an agent in a tool call")
                for k in keys}

    if spec.kind is formulas.FormulaKind.SET:
        if isinstance(rows, str):
            try:
                rows = json.loads(rows)
            except ValueError:
                return {"calculation": spec.name, "assessed": False,
                        "why": "rows was not valid JSON"}
        if not isinstance(rows, list) or not rows or not all(isinstance(r, Mapping) for r in rows):
            return {"calculation": spec.name, "assessed": False,
                    "why": ("this calculation scores a whole set at once: pass one object "
                            "per member in 'rows'")}
        batch = formulas.evaluate_many(
            spec.name, rows, shared=given,
            provenance=[_asserted(r) for r in rows], trace_id=trace_id,
        )
        return {
            "calculation": spec.name, "version": spec.version,
            "assessed": not any(r.unassessed or r.value is None for r in batch),
            "results": [_outcome(spec, r, row) for r, row in zip(batch, rows)],
        }

    if rows:
        return {"calculation": spec.name, "assessed": False,
                "why": "this calculation takes one set of inputs, not rows"}
    return _outcome(spec, formulas.evaluate(spec.name, given, provenance=_asserted(given),
                                            trace_id=trace_id), given)


# ---------------------------------------------------------------------------
# The calculations an agent was declared to need
# ---------------------------------------------------------------------------
def checked_required(names: Sequence[Any]) -> List[str]:
    """The list as it will be stored, or a ``ValueError`` naming what cannot be in it.

    Refused at the door rather than skipped at run time: a list naming a
    calculation the agent will never be given reads, to whoever reviews the
    agent, as a statement of what it always does.
    """
    formulas = _formulas()
    wanted = list(dict.fromkeys(str(n).strip() for n in names if str(n).strip()))
    unknown = [n for n in wanted if n not in formulas.REGISTRY]
    if unknown:
        raise ValueError(f"not a registered calculation: {', '.join(unknown)}")
    sets = [n for n in wanted if formulas.REGISTRY[n].kind is formulas.FormulaKind.SET]
    if sets:
        raise ValueError(
            "scores a whole set of rows and cannot be a required calculation yet: "
            + ", ".join(sets)
        )
    return wanted


def required_options() -> List[Dict[str, Any]]:
    """The calculations an agent may be declared to need, with what each reads."""
    formulas = _formulas()
    return [
        {"key": spec.name, "family": _family(spec.name), "purpose": spec.purpose,
         "reads": [t.name for t in spec.contract.inputs]}
        for spec in (formulas.REGISTRY[n] for n in sorted(formulas.REGISTRY))
        if spec.kind is not formulas.FormulaKind.SET
    ]


def run_required(
    names: Sequence[str],
    input_data: Any,
    *,
    trace_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Run each declared calculation on the inputs the run arrived with.

    A calculation reads the inputs it declares, by name, from ``input_data``. One
    whose inputs are not there is reported as not assessed, with what was
    missing — never skipped without a word, and never run on a stand-in value.
    """
    formulas = _formulas()
    data = input_data if isinstance(input_data, Mapping) else {}
    results: List[Dict[str, Any]] = []
    for name in names:
        spec = formulas.REGISTRY.get(name)
        if spec is None:
            # Declared, then removed from the register. Said, not swallowed.
            results.append({"calculation": name, "assessed": False,
                            "why": "this calculation is no longer registered"})
            continue
        given = {t.name: data[t.name] for t in spec.contract.inputs
                 if t.name in data and data[t.name] is not None}
        outcome = _outcome(spec, formulas.evaluate(spec.name, given, trace_id=trace_id), given)
        outcome["inputs"] = _plain(given)
        results.append(outcome)
    return results


def render_required(results: Sequence[Mapping[str, Any]]) -> str:
    """The declared calculations, as the part of the task that states them."""
    if not results:
        return ""
    lines = [
        "Calculations already run for this task. They are exact: use each value as "
        "given, do not recompute it, and do not replace one that was not assessed "
        "with a figure of your own — say it could not be assessed, and why."
    ]
    for r in results:
        if r.get("assessed"):
            value = json.dumps(r.get("value"), ensure_ascii=False, default=str)
            unit = f" ({r['unit']})" if r.get("unit") else ""
            absent = (f" — computed without: {', '.join(r['not_given'])}"
                      if r.get("not_given") else "")
            lines.append(f"- {r['calculation']}{unit}: {value}{absent}")
        else:
            lines.append(f"- {r['calculation']}: NOT ASSESSED — {r.get('why')}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Does a figure in the answer trace to something the run was given?
# ---------------------------------------------------------------------------
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
_MONEY_BEFORE = re.compile(r"(?:[£$€¥₹]|\b[A-Z]{3})\s?$")
_MONEY_AFTER = re.compile(r"^\s?(?:%|percent\b|[A-Z]{3}\b|[kKmM]\b|million\b|thousand\b)")


def _decimal(token: str) -> Optional[Decimal]:
    try:
        return Decimal(token.replace(",", ""))
    except (InvalidOperation, ValueError):
        return None


def _numbers(text: str) -> Set[Decimal]:
    found = (_decimal(m.group(0)) for m in _NUMBER.finditer(text or ""))
    return {n for n in found if n is not None}


# "ISO 9001", "ISO/IEC 27001", "BS EN 1090": the number names a standard, it does not state a quantity.
_STANDARD_BEFORE = re.compile(r"\b(?:ISO|IEC|EN|BS|PAS|GS1|OHSAS)[\s/:-]*(?:IEC[\s/:-]*)?$")


def _is_figure(text: str, start: int, end: int, token: str) -> bool:
    """A number that states an amount, a rate or a measure — not "round 2",
    "3 suppliers" or "ISO 9001". Marked as money or a percentage, carrying a
    decimal part, or four digits and more."""
    if _STANDARD_BEFORE.search(text[max(0, start - 14):start]):
        return False
    if "." in token or len(token.replace(",", "")) >= 4:
        return True
    return bool(_MONEY_BEFORE.search(text[max(0, start - 4):start])
                or _MONEY_AFTER.match(text[end:end + 9]))


def _supported(token: str, value: Decimal, percent: bool, known: Set[Decimal]) -> bool:
    if value in known:
        return True
    places = len(token.split(".", 1)[1]) if "." in token else 0
    quantum = Decimal(1).scaleb(-places)
    for candidate in known:
        # 0.125 stated as 12.5% is the same figure; so is 10812.4999 stated as 10,812.50.
        for source in ((candidate, candidate * 100) if percent else (candidate,)):
            for rounding in (ROUND_HALF_UP, ROUND_HALF_EVEN):
                try:
                    if source.quantize(quantum, rounding=rounding) == value:
                        return True
                except InvalidOperation:
                    continue
    return False


def untraced_figures(answer: str, sources: Iterable[Any]) -> List[str]:
    """The figures in ``answer`` that appear in none of ``sources``.

    ``sources`` is everything the run was given: its task, its governing text,
    the calculations run for it and what each tool returned. A figure found in
    none of them was not read from anywhere — the model produced it. Rounding to
    the precision the answer states, and a ratio restated as a percentage, are
    the same figure and are not reported.
    """
    text = answer or ""
    known: Set[Decimal] = set()
    for source in sources:
        if source is None:
            continue
        known |= _numbers(source if isinstance(source, str)
                          else json.dumps(source, ensure_ascii=False, default=str))

    untraced: List[str] = []
    for match in _NUMBER.finditer(text):
        token = match.group(0).rstrip(",")
        value = _decimal(token)
        if value is None or not _is_figure(text, match.start(), match.start() + len(token), token):
            continue
        percent = bool(re.match(r"\s?(?:%|percent\b)", text[match.start() + len(token):][:9]))
        if not _supported(token, value, percent, known) and token not in untraced:
            untraced.append(token)
    return untraced
