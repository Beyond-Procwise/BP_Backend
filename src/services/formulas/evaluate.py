"""The only two ways to run a formula.

Callers do not import formula functions. They name one. The name is what gets
recorded, what ``model_inventory()`` reports on, and what a stored evaluation
can be traced back to; a direct call is none of those things.

``evaluate_many`` is not a loop with a nicer name. For a ``SET`` formula it is
the *only* correct entry point, because the calculation is a property of the
population --- a min-max normalised score and a ratio-to-cheapest price score
cannot be reproduced one row at a time. For a ``SCALAR`` formula it is a loop,
but a loop that writes one audit record instead of N.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from . import audit as _audit
from .audit import EvaluationRecord, Provenance
from .contract import Finding, Severity, validate
from .registry import FormulaKind, FormulaSpec, get
from .unassessed import UNASSESSED, Confidence, UnassessedError, weakest

logger = logging.getLogger(__name__)


class EvaluationRefused(RuntimeError):
    """Raised by ``Result.unwrap()`` when there is no value to unwrap."""


@dataclass(frozen=True)
class Result:
    """One evaluation's outcome.

    ``value`` is ``UNASSESSED`` when the contract was not satisfied. It is not
    ``None`` and not ``0.0``: both of those are values, and this is the absence
    of one.
    """

    formula: str
    value: Any
    confidence: Optional[Confidence]
    findings: Tuple[Finding, ...]
    record: EvaluationRecord

    @property
    def unassessed(self) -> bool:
        return self.value is UNASSESSED

    @property
    def ok(self) -> bool:
        return not self.unassessed

    def unwrap(self) -> Any:
        """The value, or an exception. Use when there is no sensible fallback."""
        if self.unassessed:
            detail = self.findings[0].detail if self.findings else "contract not satisfied"
            raise EvaluationRefused(f"{self.formula} could not be evaluated: {detail}")
        return self.value

    def or_else(self, default: Any) -> Any:
        """The value, or ``default`` --- an explicit, visible choice to substitute.

        Exists so that a caller which genuinely wants a fallback has to write
        down that it wants one, at the call site, where a reader can see it.
        """
        return default if self.unassessed else self.value

    def why(self) -> str:
        if not self.unassessed:
            return ""
        return "; ".join(f.detail for f in self.findings)

    def __repr__(self) -> str:  # pragma: no cover - diagnostics
        state = "UNASSESSED" if self.unassessed else repr(self.value)
        return f"<Result {self.formula} {state}>"


@dataclass(frozen=True)
class BatchResult:
    """A set evaluation, or a batch of scalar evaluations, plus its one record."""

    formula: str
    results: Tuple[Result, ...]
    record: EvaluationRecord

    @property
    def values(self) -> List[Any]:
        return [r.value for r in self.results]

    @property
    def any_unassessed(self) -> bool:
        return any(r.unassessed for r in self.results)

    @property
    def assessed(self) -> List[Result]:
        return [r for r in self.results if r.ok]

    def __len__(self) -> int:
        return len(self.results)

    def __iter__(self):
        return iter(self.results)

    def __getitem__(self, i: int) -> Result:
        return self.results[i]


def _provenance_confidences(
    provenance: Optional[Mapping[str, Provenance]]
) -> Tuple[List[str], Confidence]:
    if not provenance:
        return [], Confidence.UNVERIFIED
    ids = [p.id for p in provenance.values() if getattr(p, "id", None)]
    return ids, weakest(p.confidence for p in provenance.values())


def _refuse(
    spec: FormulaSpec,
    resolved: Mapping[str, Any],
    findings: Sequence[Finding],
    prov_ids: Sequence[str],
    started: float,
    context: Mapping[str, Any],
) -> Result:
    record = EvaluationRecord(
        formula=spec.name,
        version=spec.version,
        version_hash=spec.version_hash,
        evaluated_at=_audit.now(),
        inputs=dict(resolved),
        provenance_ids=list(prov_ids),
        output=UNASSESSED,
        confidence=None,
        status="unassessed",
        findings=list(findings),
        duration_ms=(time.perf_counter() - started) * 1000.0,
        **context,
    )
    return Result(spec.name, UNASSESSED, None, tuple(findings), record)


def _call(spec: FormulaSpec, resolved: Mapping[str, Any]) -> Any:
    if spec.kind is FormulaKind.SET:
        raise TypeError(f"{spec.name} is a SET formula; call evaluate_many()")
    return spec.fn(**resolved)


def evaluate(
    name: str,
    ctx: Optional[Mapping[str, Any]] = None,
    *,
    provenance: Optional[Mapping[str, Provenance]] = None,
    trace_id: Optional[str] = None,
    deal_id: Optional[str] = None,
    document_id: Optional[str] = None,
    audit: bool = True,
) -> Result:
    """Validate ``ctx`` against the named formula's contract, then evaluate it.

    Nothing is computed until the contract passes. A missing required term, a
    value outside its declared range, a unit mismatch or an undeclared key all
    produce ``UNASSESSED`` and a finding --- never a zero, a ``None`` or a
    default.
    """
    spec = get(name)
    started = time.perf_counter()
    ctx = dict(ctx or {})
    context = {"trace_id": trace_id, "deal_id": deal_id, "document_id": document_id}
    prov_ids, input_confidence = _provenance_confidences(provenance)

    resolved, findings = validate(spec.name, spec.contract, ctx)
    blocking = [f for f in findings if f.severity is Severity.CONTRACT_VIOLATION]
    if blocking:
        result = _refuse(spec, resolved, findings, prov_ids, started, context)
        if audit:
            _audit.emit(result.record)
        return result

    try:
        value = _call(spec, resolved)
    except UnassessedError:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("formula %s raised during evaluation", spec.name)
        findings = list(findings) + [
            Finding(
                formula=spec.name,
                term=None,
                code="evaluation_error",
                detail=f"{type(exc).__name__}: {exc}",
            )
        ]
        result = _refuse(spec, resolved, findings, prov_ids, started, context)
        if audit:
            _audit.emit(result.record)
        return result

    if value is UNASSESSED:
        # A body may decide, on evidence, that it cannot answer --- the benchmark
        # engine's evidence gate is exactly this. That is a refusal, not a value.
        findings = list(findings) + [
            Finding(
                formula=spec.name,
                term=None,
                code="refused_by_body",
                detail="the formula declined to produce a value on the evidence supplied",
            )
        ]
        result = _refuse(spec, resolved, findings, prov_ids, started, context)
        if audit:
            _audit.emit(result.record)
        return result

    record = EvaluationRecord(
        formula=spec.name,
        version=spec.version,
        version_hash=spec.version_hash,
        evaluated_at=_audit.now(),
        inputs=dict(resolved),
        provenance_ids=list(prov_ids),
        output=value,
        confidence=input_confidence,
        status="ok",
        findings=list(findings),
        duration_ms=(time.perf_counter() - started) * 1000.0,
        **context,
    )
    if audit:
        _audit.emit(record)
    return Result(spec.name, value, input_confidence, tuple(findings), record)


def evaluate_many(
    name: str,
    contexts: Sequence[Mapping[str, Any]],
    *,
    shared: Optional[Mapping[str, Any]] = None,
    provenance: Optional[Sequence[Optional[Mapping[str, Provenance]]]] = None,
    trace_id: Optional[str] = None,
    deal_id: Optional[str] = None,
    document_id: Optional[str] = None,
    audit: bool = True,
) -> BatchResult:
    """Evaluate a whole set, writing one aggregated audit record.

    For a ``SET`` formula this is a single evaluation over the population: every
    context is validated first, and a contract violation anywhere refuses the
    whole set. That is fail-closed by design --- a population statistic computed
    from a partly-invalid population is not a smaller truth, it is a wrong one.

    For a ``SCALAR`` formula each context is independent, so one bad context
    yields ``UNASSESSED`` for that context alone and the rest still evaluate.
    """
    spec = get(name)
    started = time.perf_counter()
    contexts = [dict(c) for c in contexts]
    shared = dict(shared or {})
    context_meta = {"trace_id": trace_id, "deal_id": deal_id, "document_id": document_id}

    if shared and spec.kind is not FormulaKind.SET:
        raise TypeError(
            f"{spec.name} is a SCALAR formula; shared parameters belong in each context"
        )

    prov_list: List[Optional[Mapping[str, Provenance]]] = list(provenance or [])
    while len(prov_list) < len(contexts):
        prov_list.append(None)

    all_ids: List[str] = []
    confidences: List[Confidence] = []
    for p in prov_list:
        ids, conf = _provenance_confidences(p)
        all_ids.extend(ids)
        confidences.append(conf)
    batch_confidence = weakest(confidences) if contexts else Confidence.UNVERIFIED

    if spec.kind is FormulaKind.SET:
        resolved_rows: List[Dict[str, Any]] = []
        findings: List[Finding] = []

        resolved_shared, shared_findings = validate(
            spec.name, spec.contract, shared, terms=spec.contract.shared_terms
        )
        for f in shared_findings:
            findings.append(
                Finding(spec.name, f.term, f.code, f"shared: {f.detail}", f.severity)
            )

        for i, ctx in enumerate(contexts):
            resolved, fs = validate(
                spec.name, spec.contract, ctx, terms=spec.contract.per_context
            )
            resolved_rows.append(resolved)
            for f in fs:
                findings.append(
                    Finding(spec.name, f.term, f.code, f"context[{i}]: {f.detail}", f.severity)
                )

        blocking = [f for f in findings if f.severity is Severity.CONTRACT_VIOLATION]
        if blocking:
            record = _batch_record(
                spec, resolved_rows, all_ids, UNASSESSED, None, "unassessed",
                findings, len(contexts), started, context_meta, shared=resolved_shared,
            )
            if audit:
                _audit.emit(record)
            results = tuple(
                Result(spec.name, UNASSESSED, None, tuple(findings), record) for _ in contexts
            )
            return BatchResult(spec.name, results, record)

        try:
            values = spec.fn(resolved_rows, **resolved_shared)
        except Exception as exc:  # noqa: BLE001
            logger.exception("set formula %s raised during evaluation", spec.name)
            findings.append(
                Finding(spec.name, None, "evaluation_error", f"{type(exc).__name__}: {exc}")
            )
            record = _batch_record(
                spec, resolved_rows, all_ids, UNASSESSED, None, "unassessed",
                findings, len(contexts), started, context_meta, shared=resolved_shared,
            )
            if audit:
                _audit.emit(record)
            results = tuple(
                Result(spec.name, UNASSESSED, None, tuple(findings), record) for _ in contexts
            )
            return BatchResult(spec.name, results, record)

        values = list(values)
        if len(values) != len(contexts):
            raise RuntimeError(
                f"{spec.name}: set formula returned {len(values)} values for "
                f"{len(contexts)} contexts; a set formula must be length-preserving"
            )

        record = _batch_record(
            spec, resolved_rows, all_ids, values, batch_confidence, "ok",
            findings, len(contexts), started, context_meta, shared=resolved_shared,
        )
        if audit:
            _audit.emit(record)
        results = tuple(
            Result(
                spec.name,
                v,
                None if v is UNASSESSED else batch_confidence,
                tuple(findings),
                record,
            )
            for v in values
        )
        return BatchResult(spec.name, results, record)

    # SCALAR: independent contexts, one aggregated record.
    results: List[Result] = []
    all_findings: List[Finding] = []
    outputs: List[Any] = []
    resolved_rows = []
    for i, ctx in enumerate(contexts):
        r = evaluate(
            name, ctx, provenance=prov_list[i], trace_id=trace_id,
            deal_id=deal_id, document_id=document_id, audit=False,
        )
        results.append(r)
        resolved_rows.append(r.record.inputs)
        outputs.append(r.value)
        for f in r.findings:
            all_findings.append(
                Finding(spec.name, f.term, f.code, f"context[{i}]: {f.detail}", f.severity)
            )

    status = "ok" if any(r.ok for r in results) or not results else "unassessed"
    record = _batch_record(
        spec, resolved_rows, all_ids, outputs, batch_confidence, status,
        all_findings, len(contexts), started, context_meta,
    )
    if audit:
        _audit.emit(record)
    rebound = tuple(
        Result(r.formula, r.value, r.confidence, r.findings, record) for r in results
    )
    return BatchResult(spec.name, rebound, record)


def _batch_record(
    spec: FormulaSpec,
    resolved_rows: Sequence[Mapping[str, Any]],
    prov_ids: Sequence[str],
    output: Any,
    confidence: Optional[Confidence],
    status: str,
    findings: Sequence[Finding],
    batch_size: int,
    started: float,
    context_meta: Mapping[str, Any],
    shared: Optional[Mapping[str, Any]] = None,
) -> EvaluationRecord:
    payload: Dict[str, Any] = {"contexts": list(resolved_rows)}
    if shared:
        payload["shared"] = dict(shared)
    return EvaluationRecord(
        formula=spec.name,
        version=spec.version,
        version_hash=spec.version_hash,
        evaluated_at=_audit.now(),
        inputs=payload,
        provenance_ids=list(dict.fromkeys(prov_ids)),
        output=output,
        confidence=confidence,
        status=status,
        findings=list(findings),
        batch_size=batch_size,
        duration_ms=(time.perf_counter() - started) * 1000.0,
        **context_meta,
    )


__all__ = ["evaluate", "evaluate_many", "Result", "BatchResult", "EvaluationRefused"]
