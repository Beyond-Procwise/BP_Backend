"""The evaluation record, and where it goes.

Every evaluation produces a record. What happens to that record is a sink
decision, and it is deliberately pluggable: some of these formulas run over
pairs (a batch of 12,408 invoices is 77 million candidate pairs before
blocking), and a synchronous INSERT per evaluation would be a denial-of-service
against the audit table rather than an audit trail. The default sink keeps a
bounded in-process ring buffer and logs at debug; a caller that wants durability
installs ``DbAuditSink`` and gets rows in ``proc.bp_agent_actions``.

Batch evaluation writes ONE record for the whole set, which is both what the
specification asks for and what keeps the volume honest.
"""
from __future__ import annotations

import json
import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Mapping, Optional, Sequence

from .contract import Finding
from .unassessed import UNASSESSED, Confidence

logger = logging.getLogger(__name__)

#: How much of a resolved input set is kept verbatim in a record. A frame of
#: 5,000 supplier rows is evidence of what was evaluated, not something to
#: serialise into every audit row.
MAX_INLINE_INPUT_CHARS = 4000


@dataclass(frozen=True)
class Provenance:
    """Where one input value came from, and how far it can be trusted."""

    id: str
    confidence: Confidence = Confidence.UNVERIFIED
    source: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "confidence": self.confidence.value, "source": self.source}


def _summarise(value: Any) -> Any:
    """A JSON-safe, size-bounded rendering of a resolved input."""
    try:
        if value is UNASSESSED:
            return "UNASSESSED"
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, Mapping):
            return {str(k): _summarise(v) for k, v in list(value.items())[:50]}
        if isinstance(value, (list, tuple)):
            if len(value) > 20:
                return {"__len__": len(value), "head": [_summarise(v) for v in value[:5]]}
            return [_summarise(v) for v in value]
        text = repr(value)
        return text if len(text) <= 200 else text[:200] + "…"
    except Exception:  # pragma: no cover - a summary must never break a run
        return "<unrenderable>"


def _bounded(payload: Any) -> Any:
    try:
        text = json.dumps(payload, default=str)
    except Exception:  # pragma: no cover
        return "<unserialisable>"
    if len(text) <= MAX_INLINE_INPUT_CHARS:
        return payload
    return {"__truncated__": True, "chars": len(text), "preview": text[:MAX_INLINE_INPUT_CHARS]}


@dataclass
class EvaluationRecord:
    """One audit-spine entry for one evaluation (or one batch)."""

    formula: str
    version: str
    version_hash: str
    evaluated_at: datetime
    inputs: Dict[str, Any]
    provenance_ids: List[str]
    output: Any
    confidence: Optional[Confidence]
    status: str  # "ok" | "unassessed"
    findings: List[Finding] = field(default_factory=list)
    batch_size: int = 1
    trace_id: Optional[str] = None
    deal_id: Optional[str] = None
    document_id: Optional[str] = None
    duration_ms: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "formula": self.formula,
            "version": self.version,
            "version_hash": self.version_hash,
            "qualified_version": f"{self.version}+{self.version_hash}",
            "evaluated_at": self.evaluated_at.isoformat(),
            "inputs": _bounded({k: _summarise(v) for k, v in self.inputs.items()}),
            "provenance_ids": self.provenance_ids,
            "output": _bounded(_summarise(self.output)),
            "confidence": self.confidence.value if self.confidence else None,
            "status": self.status,
            "findings": [f.to_dict() for f in self.findings],
            "batch_size": self.batch_size,
            "trace_id": self.trace_id,
            "deal_id": self.deal_id,
            "document_id": self.document_id,
            "duration_ms": self.duration_ms,
        }

    def summary(self) -> str:
        if self.status == "unassessed":
            first = self.findings[0].detail if self.findings else "contract not satisfied"
            return f"{self.formula} UNASSESSED: {first}"
        if self.batch_size != 1:
            return (
                f"{self.formula} evaluated over {self.batch_size} contexts "
                f"({self.confidence.value if self.confidence else 'unknown'})"
            )
        return (
            f"{self.formula} = {_summarise(self.output)!r} "
            f"({self.confidence.value if self.confidence else 'unknown'})"
        )


class AuditSink:
    """Where evaluation records go. Never raises into the caller."""

    def write(self, record: EvaluationRecord) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class MemoryAuditSink(AuditSink):
    """Bounded ring buffer plus a debug log line. The default.

    Bounded because an unbounded list behind a pairwise scorer is a memory leak
    with an audit trail's name on it.
    """

    def __init__(self, capacity: int = 5000) -> None:
        self._records: Deque[EvaluationRecord] = deque(maxlen=capacity)
        self._lock = threading.Lock()

    def write(self, record: EvaluationRecord) -> None:
        with self._lock:
            self._records.append(record)
        logger.debug("formula-eval %s", record.summary())

    @property
    def records(self) -> List[EvaluationRecord]:
        with self._lock:
            return list(self._records)

    def clear(self) -> None:
        with self._lock:
            self._records.clear()


class NullAuditSink(AuditSink):
    """Discards records. For hot loops that have opted out deliberately."""

    def write(self, record: EvaluationRecord) -> None:
        return None


class DbAuditSink(AuditSink):
    """Writes to ``proc.bp_agent_actions`` via the existing action recorder.

    The evaluation-specific fields ride in ``details`` because the table has no
    columns for them. ``docs/adr/0002-formula-registry.md`` carries the proposed
    migration that would give them real columns; it is deliberately not run.
    """

    ACTION_TYPE = "formula_eval"
    PHASE = "validation"

    def __init__(self, conn: Any = None, agent: str = "formula_registry") -> None:
        self.conn = conn
        self.agent = agent

    def write(self, record: EvaluationRecord) -> None:
        try:
            from src.services.agent_actions import record_action

            record_action(
                phase=self.PHASE,
                action_type=self.ACTION_TYPE,
                conn=self.conn,
                deal_id=record.deal_id,
                document_id=record.document_id,
                trace_id=record.trace_id,
                agent=self.agent,
                field_name=record.formula,
                status="ok" if record.status == "ok" else "unassessed",
                summary=record.summary()[:1000],
                details=record.to_dict(),
                confidence=None,
                pipeline_version=f"{record.formula}@{record.version}+{record.version_hash}",
            )
        except Exception:  # noqa: BLE001
            # An audit write must never take down the calculation it audits.
            logger.warning(
                "formula audit write failed for %s; record kept in logs only",
                record.formula, exc_info=True,
            )


_sink: AuditSink = MemoryAuditSink()
_sink_lock = threading.Lock()


def set_audit_sink(sink: AuditSink) -> AuditSink:
    """Install a sink; returns the previous one so it can be restored."""
    global _sink
    with _sink_lock:
        previous, _sink = _sink, sink
    return previous


def get_audit_sink() -> AuditSink:
    return _sink


def emit(record: EvaluationRecord) -> None:
    try:
        _sink.write(record)
    except Exception:  # noqa: BLE001 - pragma: no cover
        logger.warning("audit sink raised for %s", record.formula, exc_info=True)


def now() -> datetime:
    return datetime.now(timezone.utc)


__all__ = [
    "Provenance", "EvaluationRecord", "AuditSink", "MemoryAuditSink",
    "NullAuditSink", "DbAuditSink", "set_audit_sink", "get_audit_sink", "emit",
]
