"""Registry-as-code formula layer.

Every number this system computes from domain inputs is registered here: named,
versioned, contract-checked, audited and pinned by golden vectors that run at
import time. Callers name a formula and call ``evaluate``; they do not import
the function.

    from src.services.formulas import evaluate, UNASSESSED

    result = evaluate("supplier.payment_terms_score", {"payment_terms_days": 30})
    if result.unassessed:
        ...          # handle it; do NOT treat it as zero
    score = result.value

See ``docs/adr/0002-formula-registry.md`` for what is deliberately deferred.
"""
from __future__ import annotations

from .audit import (
    AuditSink, DbAuditSink, EvaluationRecord, MemoryAuditSink, NullAuditSink,
    Provenance, get_audit_sink, install_db_audit_sink, set_audit_sink,
)
from .contract import Contract, Finding, Output, Quantity, Term, Unit
from .evaluate import BatchResult, EvaluationRefused, Result, evaluate, evaluate_many
from .inventory import model_inventory, render_markdown
from .registry import (
    REGISTRY, FormulaError, FormulaKind, FormulaSpec, GoldenVector,
    GoldenVectorFailure, ensure_registered, formula, get, names, verify_all,
)
from .unassessed import UNASSESSED, Confidence, UnassessedError, weakest

__all__ = [
    "UNASSESSED", "UnassessedError", "Confidence", "weakest",
    "formula", "FormulaKind", "FormulaSpec", "GoldenVector", "GoldenVectorFailure",
    "FormulaError", "REGISTRY", "get", "names", "verify_all", "ensure_registered",
    "Term", "Output", "Contract", "Unit", "Quantity", "Finding",
    "evaluate", "evaluate_many", "Result", "BatchResult", "EvaluationRefused",
    "Provenance", "EvaluationRecord", "AuditSink", "MemoryAuditSink",
    "NullAuditSink", "DbAuditSink", "set_audit_sink", "get_audit_sink",
    "install_db_audit_sink",
    "model_inventory", "render_markdown",
]
