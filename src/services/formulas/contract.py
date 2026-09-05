"""The typed input contract a formula declares, and the check it must pass.

A contract is three things: what each input *means* (its concept), what scale
it is on (its unit), and what values are admissible (its range). The third is
what makes the first two enforceable. This codebase already carries a live
example of why that matters: ``risk_score`` is consumed on a 0-1 scale in
``risk_intelligence_service``, on a 0-100 scale in ``negotiation_advice``, and
on a coerce-whichever scale in ``opportunity_miner``. Three conventions for one
field, and nothing that could have complained. A declared range of ``[0, 1]``
turns a 0-100 value into a refusal instead of a silent 100x error.

Concept codes bind to ``src.services.facts.concept_codes``, which derives the
vocabulary from ``extraction_schemas/*.yaml`` at import time. There is no GPSS
dictionary in this project (see docs/remediation/00_seam_map.md B3), so a term
naming a concept is checked against the vocabulary that actually exists.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


class Dimension(str, Enum):
    """What kind of quantity a unit measures."""

    MONEY = "money"
    COUNT = "count"
    TIME = "time"
    DIMENSIONLESS = "dimensionless"
    CATEGORICAL = "categorical"
    TEMPORAL_POINT = "temporal_point"
    STRUCTURE = "structure"


@dataclass(frozen=True)
class Unit:
    """A scale. Two values are only comparable if their units are identical."""

    symbol: str
    dimension: Dimension
    note: str = ""

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.symbol


# The units actually in use in this codebase. Deliberately small: a unit that
# no formula declares is a unit nobody has thought about.
GBP = Unit("GBP", Dimension.MONEY, "pounds sterling")
USD = Unit("USD", Dimension.MONEY, "US dollars")
MONEY = Unit("money", Dimension.MONEY, "currency-bearing amount, currency carried alongside")
COUNT = Unit("count", Dimension.COUNT, "a whole number of things")
DAYS = Unit("days", Dimension.TIME)
HOURS = Unit("hours", Dimension.TIME)
SECONDS = Unit("seconds", Dimension.TIME)
RATIO = Unit("ratio", Dimension.DIMENSIONLESS, "0-1 fraction")
FACTOR = Unit("factor", Dimension.DIMENSIONLESS, "multiplier, 1.0 = no change")
PERCENT = Unit("percent", Dimension.DIMENSIONLESS, "0-100 percentage points")
SCORE_100 = Unit("score_0_100", Dimension.DIMENSIONLESS, "0-100 score")
PROBABILITY = Unit("probability", Dimension.DIMENSIONLESS, "0-1 probability")
CURRENCY_CODE = Unit("currency_code", Dimension.CATEGORICAL, "ISO 4217 code")
LABEL = Unit("label", Dimension.CATEGORICAL, "one of a fixed vocabulary")
TEXT = Unit("text", Dimension.CATEGORICAL, "free text, compared not measured")
DATE = Unit("date", Dimension.TEMPORAL_POINT)
TIMESTAMP = Unit("timestamp", Dimension.TEMPORAL_POINT)
BOOLEAN = Unit("boolean", Dimension.CATEGORICAL)
ROWS = Unit("rows", Dimension.STRUCTURE, "a sequence of records")
RECORD = Unit("record", Dimension.STRUCTURE, "one mapping of fields")

_ALL_UNITS = {
    u.symbol: u
    for u in (
        GBP, USD, MONEY, COUNT, DAYS, HOURS, SECONDS, RATIO, FACTOR, PERCENT,
        SCORE_100, PROBABILITY, CURRENCY_CODE, LABEL, TEXT, DATE, TIMESTAMP,
        BOOLEAN, ROWS, RECORD,
    )
}


def unit_by_symbol(symbol: str) -> Optional[Unit]:
    return _ALL_UNITS.get(symbol)


@dataclass(frozen=True)
class Quantity:
    """A number that carries its own unit.

    Optional. A caller passing a bare float is *asserting* it is already in the
    declared unit, which cannot be checked; passing a ``Quantity`` lets the
    contract check it. Range validation applies either way, and for the scale
    collisions that actually occur here the range is the effective unit check.
    """

    value: Any
    unit: Unit

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.value} {self.unit.symbol}"


class Severity(str, Enum):
    CONTRACT_VIOLATION = "contract_violation"
    WARNING = "warning"


@dataclass(frozen=True)
class Finding:
    """Why a formula refused, in a form that can be logged and counted."""

    formula: str
    term: Optional[str]
    code: str
    detail: str
    severity: Severity = Severity.CONTRACT_VIOLATION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "formula": self.formula,
            "term": self.term,
            "code": self.code,
            "detail": self.detail,
            "severity": self.severity.value,
        }


@dataclass(frozen=True)
class Term:
    """One declared input.

    ``required=False`` is not a licence to invent a value. It means the formula
    body has a defined, deliberate behaviour when the term is absent --- which,
    for every formula migrated here, is the behaviour it already had. Making
    such a term required would change live numbers, and that is a separate,
    explicit decision (see docs/formula-registry-gap-report.md, C-1).
    """

    name: str
    unit: Unit
    description: str = ""
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    required: bool = True
    allowed: Optional[Tuple[Any, ...]] = None
    concept_code: Optional[str] = None
    #: Terms whose value is a container are range-checked element-wise when
    #: this names the element unit; otherwise the container is passed through.
    element_of: Optional[str] = None
    #: SET formulas only. A shared term is supplied once for the whole
    #: population rather than once per context --- weights, thresholds and
    #: settings are properties of the evaluation, not of a row, and repeating
    #: them across five thousand contexts would be noise in the audit record
    #: and a chance for them to disagree.
    shared: bool = False

    def range_text(self) -> str:
        if self.allowed is not None:
            return "one of " + ", ".join(repr(a) for a in self.allowed)
        if self.minimum is None and self.maximum is None:
            return "unbounded"
        lo = "-inf" if self.minimum is None else f"{self.minimum:g}"
        hi = "+inf" if self.maximum is None else f"{self.maximum:g}"
        return f"[{lo}, {hi}]"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "unit": self.unit.symbol,
            "dimension": self.unit.dimension.value,
            "description": self.description,
            "range": self.range_text(),
            "required": self.required,
            "concept_code": self.concept_code,
            "shared": self.shared,
        }


@dataclass(frozen=True)
class Output:
    """What the formula hands back."""

    type: str
    unit: Unit
    description: str = ""
    minimum: Optional[float] = None
    maximum: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "unit": self.unit.symbol,
            "dimension": self.unit.dimension.value,
            "description": self.description,
        }


@dataclass(frozen=True)
class Contract:
    """The full input/output contract of one formula."""

    inputs: Tuple[Term, ...]
    output: Output

    def term(self, name: str) -> Optional[Term]:
        for t in self.inputs:
            if t.name == name:
                return t
        return None

    @property
    def per_context(self) -> Tuple[Term, ...]:
        return tuple(t for t in self.inputs if not t.shared)

    @property
    def shared_terms(self) -> Tuple[Term, ...]:
        return tuple(t for t in self.inputs if t.shared)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "inputs": [t.to_dict() for t in self.inputs],
            "output": self.output.to_dict(),
        }

    def signature(self) -> str:
        """A stable text form, used in the version hash."""
        parts = [
            f"{t.name}:{t.unit.symbol}:{t.range_text()}"
            f":{'req' if t.required else 'opt'}{':shared' if t.shared else ''}"
            for t in self.inputs
        ]
        return "|".join(parts) + "=>" + f"{self.output.type}:{self.output.unit.symbol}"


_MISSING = object()


def _numeric(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    from decimal import Decimal

    if isinstance(value, Decimal):
        return float(value)
    return None


def validate(
    formula_name: str,
    contract: Contract,
    inputs: Mapping[str, Any],
    terms: Optional[Sequence[Term]] = None,
) -> Tuple[Dict[str, Any], List[Finding]]:
    """Check ``inputs`` against ``contract`` before anything is evaluated.

    Returns the resolved inputs and the findings. A non-empty finding list means
    the formula must not run: the caller gets UNASSESSED. Unknown keys are a
    violation too --- a caller passing ``risk`` where the contract says
    ``risk_score`` is a caller whose value is being silently dropped.
    """
    findings: List[Finding] = []
    resolved: Dict[str, Any] = {}
    checked = tuple(terms) if terms is not None else contract.inputs
    declared = {t.name for t in checked}

    for extra in sorted(set(inputs) - declared):
        findings.append(
            Finding(
                formula=formula_name,
                term=extra,
                code="undeclared_input",
                detail=(
                    f"{extra!r} is not declared by this formula's contract; "
                    f"declared terms are {sorted(declared)}"
                ),
            )
        )

    for term in checked:
        raw = inputs.get(term.name, _MISSING)

        if raw is _MISSING or raw is None:
            if term.required:
                findings.append(
                    Finding(
                        formula=formula_name,
                        term=term.name,
                        code="missing_required_input",
                        detail=(
                            f"{term.name} is required "
                            f"({term.unit.symbol}, {term.range_text()}) and was not supplied"
                        ),
                    )
                )
            resolved[term.name] = None
            continue

        value = raw
        if isinstance(raw, Quantity):
            if raw.unit != term.unit:
                findings.append(
                    Finding(
                        formula=formula_name,
                        term=term.name,
                        code="unit_mismatch",
                        detail=(
                            f"{term.name} declared in {term.unit.symbol} "
                            f"but supplied in {raw.unit.symbol}"
                        ),
                    )
                )
                continue
            value = raw.value

        findings.extend(_check_value(formula_name, term, value))
        resolved[term.name] = value

    return resolved, findings


def _check_value(formula_name: str, term: Term, value: Any) -> List[Finding]:
    out: List[Finding] = []

    if term.allowed is not None and value not in term.allowed:
        out.append(
            Finding(
                formula=formula_name,
                term=term.name,
                code="value_not_allowed",
                detail=f"{term.name}={value!r} is not {term.range_text()}",
            )
        )
        return out

    if term.minimum is None and term.maximum is None:
        return out

    candidates: Sequence[Any]
    if term.element_of and isinstance(value, (list, tuple)):
        candidates = value
    else:
        candidates = (value,)

    for item in candidates:
        num = _numeric(item)
        if num is None:
            # Not a number and not declared as one to range-check. Structures
            # (rows, records) legitimately land here; a bad type inside them is
            # the formula body's business, not the range check's.
            continue
        if math.isnan(num):
            out.append(
                Finding(
                    formula=formula_name,
                    term=term.name,
                    code="not_a_number",
                    detail=f"{term.name} is NaN; a range cannot be checked against it",
                )
            )
            continue
        if term.minimum is not None and num < term.minimum:
            out.append(
                Finding(
                    formula=formula_name,
                    term=term.name,
                    code="out_of_range",
                    detail=(
                        f"{term.name}={num:g} {term.unit.symbol} is below the declared "
                        f"minimum {term.minimum:g} (range {term.range_text()})"
                    ),
                )
            )
        if term.maximum is not None and num > term.maximum:
            out.append(
                Finding(
                    formula=formula_name,
                    term=term.name,
                    code="out_of_range",
                    detail=(
                        f"{term.name}={num:g} {term.unit.symbol} is above the declared "
                        f"maximum {term.maximum:g} (range {term.range_text()})"
                    ),
                )
            )
    return out


__all__ = [
    "Dimension", "Unit", "Quantity", "Term", "Output", "Contract",
    "Finding", "Severity", "validate", "unit_by_symbol",
    "GBP", "USD", "MONEY", "COUNT", "DAYS", "HOURS", "SECONDS", "RATIO",
    "FACTOR", "PERCENT", "SCORE_100", "PROBABILITY", "CURRENCY_CODE", "LABEL",
    "TEXT", "DATE", "TIMESTAMP", "BOOLEAN", "ROWS", "RECORD",
]
