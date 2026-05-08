"""Line-item recovery for documents the LLM/structural extractor missed.

Production observation: docs with line items printed AFTER the totals
(unusual layout, e.g. CITY OF NEWPORT PO502004 and NEXASPARK invoices)
get persisted with ``line_items=[]`` even though ``total_amount`` is
non-zero. The LLM's training set didn't include this layout, so it
returns no line items.

This module sweeps the parsed text for ``description+amount`` pairs:

  Tier 3 Marketing Services
  (Months 1-10) 3-5 Posts Per Week
  £83,330
  Tier 3 Marketing Services
  (Months 10-12) 3-5 Posts Per Week
  £16,670

It then filters obvious totals/tax/subtotal lines, validates by checking
the recovered line amounts roughly sum to the header subtotal/total, and
emits ``line_items`` only when the sum-check passes — so we never insert
fabricated lines that don't reconcile with the header.

Hard constraints (per the no-fabrication rule):
- A line item is only emitted when its amount is literally in the text.
- Description must be the lines preceding the amount, not invented text.
- If the recovered set doesn't sum to a plausible subtotal/total, we
  decline to emit anything (returning ``[]``) rather than insert
  questionable data.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field as dc_field
from typing import List, Optional

logger = logging.getLogger(__name__)

__all__ = ["recover_line_items", "LineRecoveryReport"]


# Currency-prefixed amount: £1,234, £1,234.56, $99.99, €1.50.
_AMOUNT_LINE_RE = re.compile(
    r"^[\s]*([£€\$])\s*"
    r"([0-9]{1,3}(?:[,\s][0-9]{3})*(?:\.[0-9]{1,2})?|[0-9]+\.[0-9]{1,2})\s*$",
    re.MULTILINE,
)
# Standalone amount without currency, used when the doc has a single-currency layout.
_BARE_AMOUNT_RE = re.compile(
    r"^[\s]*([0-9]{1,3}(?:[,][0-9]{3})*(?:\.[0-9]{1,2})?|[0-9]+\.[0-9]{1,2})\s*$",
    re.MULTILINE,
)
# Lines that are NEVER line-item amounts — totals, taxes, fees, etc.
# Keep this conservative: anchor on word boundaries, prefer phrases over
# bare verbs. ("payment" alone matched "(FINAL PAYMENT)" in a real
# invoice description, causing the line item to be rejected — that's
# why we use 'payment terms' rather than 'payment'.)
_TOTAL_LABEL_RE = re.compile(
    r"(?:"
    # Sub-total / SUBTOTAL / sub total
    r"\bsub[-\s]?total\b|"
    # Standalone "Total" / "TAX" / "VAT" / "GST" lines (with optional ':',
    # '-', or parenthesised rate). The line-anchored forms are handled
    # below — these match ALWAYS-bounded variants.
    r"^total(?:\s*[:\-]\s*)?$|"
    r"^tax(?:\s*\([^)]*\))?(?:\s*[:\-])?\s*$|"
    r"^vat(?:\s*\([^)]*\))?(?:\s*[:\-])?\s*$|"
    r"^gst(?:\s*\([^)]*\))?(?:\s*[:\-])?\s*$|"
    # Multi-word totals
    r"\bgrand\s+total\b|\bnet\s+total\b|\bnet\s+amount\b|"
    # Other never-line-item labels
    r"\bdiscount\b|\bshipping\b|\bfreight\b|"
    r"\bamount\s+(?:due|payable|paid)\b|\bbalance\s+due\b|"
    r"\bpayment\s+terms\b"
    r")",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass
class LineRecoveryReport:
    items_recovered: int = 0
    sum_recovered: float = 0.0
    target_total: Optional[float] = None
    skipped_reason: Optional[str] = None
    items: list = dc_field(default_factory=list)


def _to_float(s: str) -> Optional[float]:
    if not s:
        return None
    cleaned = s.replace(",", "").replace(" ", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def _is_label_line(line: str) -> bool:
    """True if the line looks like a totals/tax/discount label, not a description."""
    return bool(_TOTAL_LABEL_RE.search(line))


def _description_above(
    lines: list, amount_idx: int, max_lines: int = 3,
    max_labels_to_skip: int = 2,
) -> Optional[str]:
    """Return up to ``max_lines`` non-blank description lines above the amount.

    Walks upward from ``amount_idx - 1`` collecting non-blank lines.
    PDF parsers sometimes interleave labels (e.g. "Total" or "Subtotal")
    BETWEEN the description and the amount — so we tolerate up to
    ``max_labels_to_skip`` such labels by skipping past them rather than
    bailing out. After that many labels (or hitting another amount line),
    we stop.

    Returns the joined description or None if nothing usable.
    """
    desc_parts: list = []
    labels_skipped = 0
    i = amount_idx - 1
    while i >= 0 and len(desc_parts) < max_lines:
        line = lines[i].strip()
        if not line:
            i -= 1
            if desc_parts:
                # blank between amount and description — stop
                break
            continue
        if _is_label_line(line):
            # Tolerate a few interleaved label lines, then bail out
            labels_skipped += 1
            if labels_skipped > max_labels_to_skip:
                break
            i -= 1
            continue
        # Skip pure-amount lines (don't include other line-item amounts as desc)
        if _AMOUNT_LINE_RE.match(line) or _BARE_AMOUNT_RE.match(line):
            break
        desc_parts.insert(0, line)
        i -= 1
    if not desc_parts:
        return None
    desc = " ".join(desc_parts).strip()
    # Trim trivial whitespace / control chars
    desc = re.sub(r"\s+", " ", desc)
    return desc or None


def _expected_subtotal(header: dict) -> Optional[float]:
    """Pull the most-trusted subtotal/total from the header for sum-checking."""
    for key in (
        "subtotal", "invoice_amount", "total_amount", "total_amount_excl_tax",
        "invoice_total_incl_tax", "total_amount_incl_tax",
    ):
        v = header.get(key)
        if v is not None:
            try:
                f = float(v)
                if f > 0:
                    return f
            except (ValueError, TypeError):
                continue
    return None


def recover_line_items(
    header: dict,
    *,
    parsed_text: Optional[str],
    doc_type: str,
    file_path: Optional[str] = None,
) -> LineRecoveryReport:
    """Best-effort line-item recovery from parsed text.

    Returns a report; the caller decides whether to merge ``items`` into
    the persisted line_items. Emits items ONLY when the recovered sum is
    within 5% of a plausible header subtotal/total.
    """
    report = LineRecoveryReport()
    if not parsed_text:
        report.skipped_reason = "no_parsed_text"
        return report
    target = _expected_subtotal(header)
    if target is None:
        report.skipped_reason = "no_target_total"
        return report
    report.target_total = target

    lines = parsed_text.splitlines()
    candidates: list = []
    seen_amount_indices: set = set()
    for idx, line in enumerate(lines):
        m = _AMOUNT_LINE_RE.match(line)
        if not m:
            continue
        amount = _to_float(m.group(2))
        if amount is None or amount <= 0:
            continue
        # Tax-amount and grand-total are NEVER line items.
        for k in ("tax_amount", "invoice_total_incl_tax",
                  "total_amount_incl_tax"):
            v = header.get(k)
            if v is not None:
                try:
                    if abs(amount - float(v)) < 0.005:
                        amount = None
                        break
                except (ValueError, TypeError):
                    pass
        if amount is None:
            continue
        # Subtotal can match a line item amount in single-item docs
        # (NEXASPARK), so we don't blanket-reject on subtotal match —
        # but if the IMMEDIATE line above is a clear "Sub-total" /
        # "TAX" label, that's the subtotal/tax row, not a line item.
        if idx > 0 and _is_label_line(lines[idx - 1]):
            line_above = lines[idx - 1].strip().lower()
            # "Total" alone is often a column header in tables, not a
            # row label. Distinguish: skip only if it's specifically
            # subtotal/tax/discount/etc. — keep "Total" because it can
            # be a column header above a line-item amount.
            if any(kw in line_above for kw in (
                "sub-total", "subtotal", "sub total",
                "tax", "vat", "gst",
                "discount", "shipping", "freight",
                "amount due", "amount payable", "amount paid",
                "balance due", "grand total", "net total",
            )):
                # Subtotal/tax match against header → AND amount matches
                # the header subtotal value → skip; else keep.
                if header.get("subtotal") is not None:
                    try:
                        if abs(amount - float(header["subtotal"])) < 0.005:
                            continue
                    except (ValueError, TypeError):
                        pass
                # Common case: just skip when label is unambiguously a totals row.
                continue
        desc = _description_above(lines, idx)
        if not desc:
            continue
        candidates.append({
            "line_idx": idx,
            "amount": amount,
            "description": desc,
        })
        seen_amount_indices.add(idx)

    if not candidates:
        report.skipped_reason = "no_candidates_found"
        return report

    # Sum-check: recovered amounts should approximate the target.
    total_amount = sum(c["amount"] for c in candidates)
    diff = abs(total_amount - target)
    tolerance = max(target * 0.05, 0.50)  # 5% or 50p, whichever larger
    if diff > tolerance:
        report.skipped_reason = (
            f"sum_mismatch: recovered={total_amount:.2f} vs target={target:.2f} "
            f"(diff={diff:.2f}, tol={tolerance:.2f})"
        )
        report.sum_recovered = total_amount
        return report

    # Build minimal items: description + line_amount + line_no.
    # Derive tax_percent for per-line use, ONLY if it's explicit in the header.
    # Per-line tax is: line_amount × tax_percent. This is derivation from
    # real document evidence (the header tax rate IS in the doc), not
    # fabrication.
    tax_pct = None
    raw_tp = header.get("tax_percent")
    if raw_tp is not None:
        try:
            tax_pct = float(raw_tp)
            if tax_pct > 1:
                tax_pct = tax_pct / 100.0
        except (ValueError, TypeError):
            tax_pct = None

    items = []
    for i, c in enumerate(candidates, start=1):
        item: dict = {
            "line_no": i,
            "line_number": i,  # PO/Quote schemas use line_number
            "item_description": c["description"][:500],
            "line_amount": c["amount"],
            "line_total": c["amount"],
        }
        if tax_pct is not None and tax_pct > 0:
            tax = round(c["amount"] * tax_pct, 2)
            item["tax_percent"] = float(raw_tp)
            item["tax_amount"] = tax
            # PO/Quote line schemas use ``total_amount`` for the
            # post-tax line total (= line_total + tax_amount).
            item["total_amount"] = round(c["amount"] + tax, 2)
            # Invoice line schema uses ``total_amount_incl_tax`` (same
            # value, different column name).
            item["total_amount_incl_tax"] = round(c["amount"] + tax, 2)
        else:
            # No tax_percent in header: total_amount = line_total.
            item["total_amount"] = c["amount"]
        items.append(item)
    report.items = items
    report.items_recovered = len(items)
    report.sum_recovered = total_amount
    logger.info(
        "[LineRecovery] %s %s: recovered %d line item(s) summing to %.2f "
        "(target=%.2f, diff=%.2f)",
        doc_type, file_path or "", len(items), total_amount, target, diff,
    )
    return report
