"""Calibrated extraction-confidence score.

Replaces the static ``0.85`` with a multiplicative score derived from
observable signals that correlate with extraction quality:

  - critical-fields-present rate
  - lines-vs-total invariant (zero lines on a non-zero total tanks the score)
  - sanitizer-rejection penalty (each nulled field reduces it)
  - filename-rescue penalty (the LLM didn't get this on its own)
  - template-applied bonus / non-event (we trust template overrides)

The output is a float in [0.0, 1.0]. Anything below ``0.65`` (default)
is "needs review" — the orchestrator routes those to the review queue
even when no individual critical field was missing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional


__all__ = ["calibrated_confidence", "ConfidenceBreakdown"]


_REQUIRED_FIELDS = {
    "Invoice": ["invoice_id", "supplier_id", "invoice_total_incl_tax",
                "invoice_date"],
    "Purchase_Order": ["po_id", "supplier_id", "total_amount", "order_date"],
    "Quote": ["quote_id", "supplier_id", "total_amount"],
    "Contract": ["contract_id", "supplier_id"],
}


@dataclass
class ConfidenceBreakdown:
    score: float
    required_fill: float           # 0.0 .. 1.0
    lines_invariant_pass: bool
    sanitizer_penalty: float       # >= 0
    rescue_penalty: float          # >= 0
    template_bonus: float          # >= 0
    notes: list[str]


def _coerce_amount(v) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def calibrated_confidence(
    *,
    header: dict,
    line_items: list,
    doc_type: str,
    sanitizer_rejections: Iterable = (),
    rescued_fields: Iterable[str] = (),
    template_overrides: Iterable[str] = (),
) -> ConfidenceBreakdown:
    """Return the calibrated confidence + breakdown for a completed extraction.

    The math:

        score = required_fill
              * (1.0 if lines_invariant_pass else 0.65)
              - sanitizer_penalty
              - rescue_penalty
              + template_bonus
              clamped to [0.0, 1.0]

    where:
      - ``required_fill`` is the fraction of doc-type-required fields populated
        (each missing field drops 1/N).
      - ``lines_invariant_pass`` is False when the doc has total>0 but lines==0
        — a strong signal of silent line-data loss.
      - Each sanitizer rejection contributes 0.05 of penalty.
      - Each filename-rescued field contributes 0.05 of penalty.
      - Each template override contributes 0.05 of bonus, capped at 0.10.
    """
    notes: list[str] = []
    required = _REQUIRED_FIELDS.get(doc_type, [])
    if required:
        present = sum(
            1 for f in required
            if header.get(f) not in (None, "", 0)
        )
        required_fill = present / len(required)
        if present < len(required):
            missing = [f for f in required if header.get(f) in (None, "", 0)]
            notes.append(f"required_missing={missing}")
    else:
        required_fill = 1.0

    # Lines-vs-total invariant — only relevant for line-bearing docs.
    lines_invariant_pass = True
    if doc_type in {"Invoice", "Quote", "Purchase_Order"}:
        total = (
            _coerce_amount(header.get("total_amount"))
            or _coerce_amount(header.get("total_amount_incl_tax"))
            or _coerce_amount(header.get("invoice_total_incl_tax"))
            or _coerce_amount(header.get("invoice_amount"))
            or 0.0
        )
        if total > 0 and not line_items:
            lines_invariant_pass = False
            notes.append("zero_lines_on_nonzero_total")

    rejection_count = sum(1 for _ in sanitizer_rejections)
    sanitizer_penalty = 0.05 * rejection_count
    if rejection_count:
        notes.append(f"sanitizer_rejections={rejection_count}")

    rescued_count = sum(1 for _ in rescued_fields)
    rescue_penalty = 0.05 * rescued_count
    if rescued_count:
        notes.append(f"rescued_fields={rescued_count}")

    template_count = sum(1 for _ in template_overrides)
    template_bonus = min(0.05 * template_count, 0.10)
    if template_count:
        notes.append(f"template_overrides={template_count}")

    base = required_fill * (1.0 if lines_invariant_pass else 0.65)
    score = base - sanitizer_penalty - rescue_penalty + template_bonus
    # Clamp to [0.0, 1.0]
    score = max(0.0, min(1.0, score))

    return ConfidenceBreakdown(
        score=round(score, 3),
        required_fill=round(required_fill, 3),
        lines_invariant_pass=lines_invariant_pass,
        sanitizer_penalty=round(sanitizer_penalty, 3),
        rescue_penalty=round(rescue_penalty, 3),
        template_bonus=round(template_bonus, 3),
        notes=notes,
    )
