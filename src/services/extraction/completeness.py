# src/services/extraction/completeness.py
"""Completeness assessment for extracted documents.

Pure functions — no I/O. Given the built header columns + line items for a
document, decide whether the extraction is COMPLETE enough to promote, or has a
gap: missing required header field, missing line items, or line items that do
not reconcile to the header subtotal. The dispatch loop uses this to trigger a
bounded recovery pass before promotion, so gaps are recovered or surfaced —
never silently promoted.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any, Optional

# Header column holding the line-item subtotal, per doc type.
_SUBTOTAL_COL = {
    "invoice": "invoice_amount",
    "purchase_order": "total_amount",
    "quote": "total_amount",
}
# Line-item column holding the per-line amount, per doc type.
_LINE_AMOUNT_COL = {
    "invoice": "line_amount",
    "purchase_order": "line_total",
    "quote": "line_total",
}

_RECONCILE_TOLERANCE = 0.05  # 5%


@dataclass
class CompletenessReport:
    header_complete: bool
    lines_expected: bool
    lines_present: bool
    lines_reconcile: bool
    is_complete: bool
    status: str  # complete | missing_required | no_line_items | line_sum_mismatch
    gaps: list[str] = field(default_factory=list)


def _to_float(v: Any) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(Decimal(str(v)))
    except (InvalidOperation, ValueError, TypeError):
        return None


def line_sum(doc_type: str, line_items: list[dict[str, Any]]) -> Optional[float]:
    """Sum the per-line amount column for this doc type. None if no usable amounts."""
    amt_col = _LINE_AMOUNT_COL.get(doc_type)
    if not amt_col or not line_items:
        return None
    vals = [_to_float(li.get(amt_col)) for li in line_items]
    vals = [v for v in vals if v is not None]
    return sum(vals) if vals else None


def header_subtotal(doc_type: str, columns: dict[str, Any]) -> Optional[float]:
    """Public: the document's header subtotal as a float, or None if absent/unparseable."""
    col = _SUBTOTAL_COL.get(doc_type)
    return _to_float(columns.get(col)) if col else None


def _reconciles(lsum: Optional[float], header_total: Optional[float]) -> bool:
    # Can't check without a header total -> don't flag a gap.
    if header_total in (None, 0) or header_total == 0.0:
        return True
    if lsum is None:
        return False
    return abs(lsum - header_total) <= max(0.01, _RECONCILE_TOLERANCE * abs(header_total))


def assess(
    doc_type: str,
    columns: dict[str, Any],
    line_items: list[dict[str, Any]],
    *,
    has_line_schema: bool,
    missing_required: list[str] | None = None,
) -> CompletenessReport:
    """Assess extraction completeness. Pure — no I/O."""
    missing_required = missing_required or []
    header_complete = not missing_required

    lines_expected = bool(has_line_schema)
    lines_present = bool(line_items)

    sub_col = _SUBTOTAL_COL.get(doc_type)
    header_total = _to_float(columns.get(sub_col)) if sub_col else None
    lsum = line_sum(doc_type, line_items)
    lines_reconcile = (
        _reconciles(lsum, header_total)
        if (lines_expected and lines_present)
        else True
    )

    gaps: list[str] = []
    if not header_complete:
        gaps.append("missing_required:" + ",".join(missing_required))
    if lines_expected and not lines_present:
        gaps.append("no_line_items")
    if lines_expected and lines_present and not lines_reconcile:
        gaps.append(f"line_sum_mismatch(lines={lsum},header={header_total})")

    if not header_complete:
        status = "missing_required"
    elif lines_expected and not lines_present:
        status = "no_line_items"
    elif lines_expected and lines_present and not lines_reconcile:
        status = "line_sum_mismatch"
    else:
        status = "complete"

    return CompletenessReport(
        header_complete=header_complete,
        lines_expected=lines_expected,
        lines_present=lines_present,
        lines_reconcile=lines_reconcile,
        is_complete=(status == "complete"),
        status=status,
        gaps=gaps,
    )
