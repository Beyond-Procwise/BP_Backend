"""V3 runner over invariants. Wraps the v2 ValidatorChain.

The YAML schema references invariants by name (e.g. 'subtotal_closure',
'tax_closure', 'scale_mismatch'). This runner builds a ValidatorChain
from the names listed in `schema.document_invariants` (top-level) plus
per-field `invariants` lists, and runs it against the bound record.

Note: ``line_sum_closure`` (used in invoice.yaml line_items.invariants)
is an alias for ``SubtotalClosure``; the logic is identical — verify that
Σ line_amounts ≈ header subtotal. It is registered here rather than in v2
to keep v2 unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from src.services.extraction_v2.invariants import (
    Validator,
    ValidatorChain,
    Severity,
    ValidatorResult,
    LineArithmetic,
    SubtotalClosure,
    TaxClosure,
    GrandTotalClosure,
    CurrencyConsistency,
    DateSanity,
    VendorIdentity,
    QuantitySign,
    RoundOffBucket,
    TaxTotalConfusion,
)
from src.services.extraction_v3.binding.scale_mismatch import ScaleMismatch
from src.services.extraction_v3.yaml_schema.loader import DocSchema


class _LineSumClosure(SubtotalClosure):
    """Alias of SubtotalClosure registered under the name 'line_sum_closure'.

    invoice.yaml references this name in line_items.invariants; the underlying
    check is identical to subtotal_closure (Σ line_amounts ≈ header subtotal).
    """
    name = "line_sum_closure"


# Name → Validator class
_VALIDATORS: dict[str, type[Validator]] = {
    "line_arithmetic": LineArithmetic,
    "subtotal_closure": SubtotalClosure,
    "tax_closure": TaxClosure,
    "grand_total_closure": GrandTotalClosure,
    "currency_consistency": CurrencyConsistency,
    "date_sanity": DateSanity,
    "vendor_identity": VendorIdentity,
    "line_sum_closure": _LineSumClosure,
    "quantity_sign": QuantitySign,
    "round_off_bucket": RoundOffBucket,
    "scale_mismatch": ScaleMismatch,
    "tax_total_confusion": TaxTotalConfusion,
}


@dataclass
class InvariantResult:
    name: str
    severity: str          # "PASS" | "INFO" | "WARNING" | "CRITICAL" | "NA"
    message: str | None    # human-readable detail


def _collect_invariant_names(schema: DocSchema) -> list[str]:
    names: list[str] = list(schema.document_invariants or [])
    for field in schema.fields:
        for inv in (field.invariants or []):
            if inv not in names:
                names.append(inv)
    if schema.line_items:
        for inv in (schema.line_items.invariants or []):
            if inv not in names:
                names.append(inv)
    return names


def run_invariants(
    header: dict,
    line_items: list[dict],
    schema: DocSchema,
) -> list[InvariantResult]:
    """Run all invariants referenced in the schema; return per-invariant results.

    Unknown invariant names cause a runtime error (loader's startup consistency
    check should have caught these). For Plan 1 we accept the late-fail; in
    Plan 2 the loader will validate names against ``_VALIDATORS.keys()``.
    """
    chain_validators: list[Validator] = []
    for name in _collect_invariant_names(schema):
        cls = _VALIDATORS.get(name)
        if cls is None:
            raise ValueError(
                f"unknown invariant name: {name!r} (not in registry)"
            )
        chain_validators.append(cls())

    chain = ValidatorChain(chain_validators)
    report = chain.run(header, line_items, schema.doc_type)

    out: list[InvariantResult] = []
    for r in report.results:
        # Severity is a plain class with string constants (not an Enum),
        # so r.severity is already a str. Guard against future Enum migration.
        sev_str = (
            r.severity
            if isinstance(r.severity, str)
            else getattr(r.severity, "name", str(r.severity))
        )
        out.append(
            InvariantResult(
                name=r.name,
                severity=_severity_for(r, sev_str),
                message=getattr(r, "message", None) or None,
            )
        )
    return out


def _severity_for(result, raw: str) -> str:
    """Map a ValidatorResult onto the vocabulary InvariantResult declares.

    That vocabulary — "PASS" | "INFO" | "WARNING" | "CRITICAL" | "NA" — has been
    in the dataclass comment since this file was written, and was never
    implemented: the raw v2 severity was passed straight through. Two things
    went wrong as a result, and consumers could not tell them apart.

    ``ValidatorResult.ok()`` does not set a severity, so it inherits the
    dataclass default of WARNING. A satisfied invariant therefore arrived
    claiming "warning" — identical to a failed one. dispatch.py files a
    discrepancy per WARNING, so honouring the raw value would have filed one for
    every invariant that HELD.

    And the raw values are lowercase while every consumer compares against the
    uppercase names in the declared vocabulary, so nothing matched at all. Live
    on 2026-08-11: 0 rows of issue_type='invariant_failed' against 821
    discrepancies of every other type.

    So: a pass is PASS, an abstention is NA, and only a genuine failure carries a
    severity that asks anyone to act.
    """
    if getattr(result, "passed", False):
        # ValidatorResult.na() is a pass carrying message="not_applicable".
        # Abstaining is not the same as passing — the check could not run, and a
        # reader who cannot tell those apart cannot tell coverage from success.
        if (getattr(result, "message", "") or "") == "not_applicable":
            return "NA"
        return "PASS"
    return (raw or "").upper() or "WARNING"
