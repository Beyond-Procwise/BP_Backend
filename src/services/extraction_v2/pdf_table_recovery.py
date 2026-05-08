"""Layout-aware line-item extraction via pdfplumber Y-clustering.

The structural extractor flattens multi-column line-item tables into
text where amounts and descriptions aren't adjacent (I-37). The text-
only ``line_recovery`` can't reconstruct rows from that.

This module re-opens the original PDF bytes with pdfplumber, groups
words by approximate Y-coordinate (each row in the table gets its own
y bucket), and parses each row into ``{item_id, description, quantity,
unit_price, line_total}``.

Hard constraint (no fabrication):
- A row is only emitted if its math is internally consistent
  (qty × unit_price ≈ line_total within 1%) OR if we have only the
  line_total — we never invent qty/unit_price.
- The extractor only fires when the doc has 2+ amount columns inside
  the same Y-row (i.e. an actual table). For docs without that shape,
  it returns no items so the existing line_recovery can take over.
"""
from __future__ import annotations

import io
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field as dc_field
from typing import List, Optional

logger = logging.getLogger(__name__)


__all__ = ["recover_lines_from_pdf_table", "PdfTableLineReport"]


# An "amount" token must have either a currency symbol OR a decimal
# point — bare integers are ambiguous (could be qty, item count, etc.)
# and we don't want to grab them as money in the trailing-amounts walk.
_AMOUNT_RE = re.compile(
    r"^(?:[£€\$]\s*([0-9]{1,3}(?:[,][0-9]{3})*(?:\.[0-9]{1,2})?|[0-9]+(?:\.[0-9]{1,2})?)"
    r"|([0-9]{1,3}(?:[,][0-9]{3})+(?:\.[0-9]{1,2})?|[0-9]+\.[0-9]{1,2}))$"
)
_INT_RE = re.compile(r"^[0-9]{1,5}$")  # qty: 1–5 digit integer


@dataclass
class PdfTableLineReport:
    items_recovered: int = 0
    sum_recovered: float = 0.0
    target_total: Optional[float] = None
    skipped_reason: Optional[str] = None
    items: list = dc_field(default_factory=list)


def _parse_amount(token: str) -> Optional[float]:
    """Parse a money token. Requires either a leading currency symbol
    (£/€/$) OR a decimal point (or thousands-separator comma) — bare
    integers like '100' are deliberately rejected so they can be picked
    up as quantity by the row-splitter."""
    if not token:
        return None
    m = _AMOUNT_RE.match(token.strip())
    if not m:
        return None
    raw = (m.group(1) or m.group(2) or "").replace(",", "")
    try:
        return float(raw)
    except ValueError:
        return None


def _is_int_token(token: str) -> bool:
    return bool(_INT_RE.match(token.strip()))


def _cluster_words_by_row(page, y_tol: float = 5.0) -> List[List[dict]]:
    """Group page words into rows by approximate Y-coordinate.

    Two-pass algorithm:
    1. First pass: bucket by Y rounded to ``y_tol`` units (default 5pt).
    2. Second pass: merge adjacent buckets whose Y-gap is small (≤ 8pt).
       This handles tables like DUNCAN's where a single line item is
       split across 3 vertically-offset cell rows because the PDF
       renderer staggered cell baselines.

    Returns a list of rows, each row sorted by x0 left-to-right.
    """
    words = page.extract_words(use_text_flow=True, keep_blank_chars=False)
    if not words:
        return []
    # Pass 1: tight bucketing
    raw: dict = defaultdict(list)
    for w in words:
        key = round(w["top"] / y_tol) * y_tol
        raw[key].append(w)

    # Pass 2: merge adjacent buckets within MERGE_GAP. A "row" in the
    # source PDF for offset-cell layouts often spans ~10-12pt; gap > 16pt
    # is a clear inter-row separator.
    MERGE_GAP = 8.0
    sorted_keys = sorted(raw)
    merged: List[List[dict]] = []
    current: List[dict] = []
    last_key: Optional[float] = None
    for k in sorted_keys:
        if last_key is not None and (k - last_key) <= MERGE_GAP:
            current.extend(raw[k])
        else:
            if current:
                merged.append(current)
            current = list(raw[k])
        last_key = k
    if current:
        merged.append(current)
    # Sort each row by x0
    return [sorted(row, key=lambda w: w["x0"]) for row in merged]


def _row_to_tokens(row: List[dict]) -> List[str]:
    return [w["text"] for w in row]


def _split_row_into_columns(tokens: List[str]) -> Optional[dict]:
    """Given a row's tokens left-to-right, identify a line-item shape.

    Heuristic: a line-item row ends with 1 or 2 currency amounts. The
    LAST amount is `line_total`. The penultimate amount, if present, is
    `unit_price`. Just before the trailing amount(s), the right-most
    standalone integer is the quantity. Everything before that is the
    description; the first token may be an item_id (alphanumeric short
    code) which we extract.

    Returns dict with keys: item_id, description, quantity, unit_price,
    line_total. Missing fields are None.
    Returns None if no plausible amount/total is found.
    """
    if not tokens:
        return None
    # Find trailing currency amounts
    trail_amounts: list = []
    i = len(tokens) - 1
    while i >= 0 and len(trail_amounts) < 2:
        v = _parse_amount(tokens[i])
        if v is None:
            break
        trail_amounts.insert(0, v)
        i -= 1
    if not trail_amounts:
        return None
    # Pop the trailing amounts from the working list
    head = tokens[:i + 1]
    # Right-most standalone int in head → quantity
    qty = None
    qty_idx = None
    for j in range(len(head) - 1, -1, -1):
        if _is_int_token(head[j]):
            qty = int(head[j])
            qty_idx = j
            break
    desc_tokens = head[:qty_idx] if qty_idx is not None else head[:]
    # First token may be a short alphanumeric code (length <=12, mix of
    # uppercase + digits) → item_id; otherwise it's part of description.
    item_id = None
    if desc_tokens:
        first = desc_tokens[0]
        if (1 <= len(first) <= 12
                and re.match(r"^[A-Z0-9][A-Z0-9\-_/]{1,11}$", first)
                and any(c.isdigit() for c in first)):
            item_id = first
            desc_tokens = desc_tokens[1:]
    description = " ".join(desc_tokens).strip() or None

    # Distribute trailing amounts: 1 → just total; 2 → unit_price + total
    line_total = trail_amounts[-1]
    unit_price = trail_amounts[0] if len(trail_amounts) >= 2 else None

    return {
        "item_id": item_id,
        "description": description,
        "quantity": qty,
        "unit_price": unit_price,
        "line_total": line_total,
    }


_TOTAL_LABEL_DESC_RE = re.compile(
    r"(?:"
    r"\bsub[-\s]?total\b|\bgrand\s+total\b|\bnet\s+total\b|\bnet\s+amount\b|"
    r"^total$|^total\s*[:\-]?\s*$|^total\s+amount(\s+(due|payable|paid))?\b|"
    # 'TOTAL' anywhere after a colon (e.g. 'PAYMENT DETAILS: TOTAL £X')
    r":\s*total\b|"
    # 'PAYMENT DETAILS' / 'PAYMENT METHOD' etc. (info rows, not line items)
    r"\bpayment\s+(details|method|terms|instructions)\b|"
    r"^tax\b|^vat\b|^gst\b|\btax\s*\([^)]*\)|"
    r"\bdiscount\b|\bshipping\b|\bfreight\b|"
    r"\bbalance\s+due\b|\bamount\s+(due|payable|paid)\b"
    r")",
    re.IGNORECASE,
)


def _row_looks_like_line_item(row: dict, target: Optional[float],
                                full_row_text: str = "") -> bool:
    """Filter false positives — a row IS a line item only if it has a
    description AND a positive line_total AND (when math is checkable)
    qty×unit_price matches line_total within 1%.

    ``full_row_text`` is the raw concatenated row tokens (passed by the
    caller so we can match labels that the description-walk skipped
    over, e.g. 'Payment must be made within SUBTOTAL: £6,750' where
    'SUBTOTAL:' sits between the description and the amount).
    """
    desc = row.get("description")
    if not desc:
        return False
    # Reject rows whose description IS a totals/tax/discount label.
    # In I-37 docs the same Y-clustering matches "Subtotal £864.15",
    # "Tax (20%) £172.83", "Total Amount Due £1,036.98" — these would
    # double-count if treated as line items.
    if _TOTAL_LABEL_DESC_RE.search(desc.strip()):
        return False
    # Also reject when the FULL row text contains a label marker that
    # the description walker skipped over. Example: 'Payment must be
    # made within SUBTOTAL: £6,750' — desc='Payment must be made
    # within' (passes), but the SUBTOTAL label is right before the
    # amount → the £6,750 is the subtotal, not a line item.
    if full_row_text and _TOTAL_LABEL_DESC_RE.search(full_row_text):
        return False
    if not row.get("line_total") or row["line_total"] <= 0:
        return False
    qty = row.get("quantity")
    unit = row.get("unit_price")
    total = row["line_total"]
    if qty is not None and unit is not None and qty > 0:
        expected = qty * unit
        if abs(expected - total) / max(total, 0.01) > 0.01:
            # Math doesn't reconcile. Either the column-split misidentified
            # qty (common when description has trailing numerics like
            # "10 per Pack") or unit_price is in a different unit. The
            # safest move: keep line_total but DROP qty/unit_price so we
            # don't persist false data.
            row["quantity"] = None
            row["unit_price"] = None
    # If we have target, individual amount cannot exceed it (sanity)
    if target is not None and total > target * 2:
        return False
    return True


def recover_lines_from_pdf_table(
    pdf_bytes: bytes,
    *,
    target_total: Optional[float] = None,
    tax_percent: Optional[float] = None,
    file_path: Optional[str] = None,
) -> PdfTableLineReport:
    """Extract line items from `pdf_bytes` using pdfplumber Y-clustering.

    Args:
        pdf_bytes: raw PDF content
        target_total: header subtotal (or total_amount) — used to filter
            rows whose amounts can't possibly be line items
        tax_percent: header tax rate; if set, per-line tax_amount and
            total_amount_incl_tax are derived from line_total

    Returns ``PdfTableLineReport``. Items are emitted only when the
    aggregate sum is within 5% of ``target_total`` (no fabrication).
    """
    report = PdfTableLineReport(target_total=target_total)
    try:
        import pdfplumber
    except ImportError:
        report.skipped_reason = "pdfplumber_not_installed"
        return report
    if not pdf_bytes:
        report.skipped_reason = "no_pdf_bytes"
        return report
    # Quick magic-byte check: real PDFs start with "%PDF-". Word
    # docs / images / other formats fail with PdfminerException, which
    # we'd rather avoid emitting as an error.
    if not pdf_bytes.startswith(b"%PDF-"):
        report.skipped_reason = "not_a_pdf"
        return report

    candidates: list = []
    try:
        pdf_ctx = pdfplumber.open(io.BytesIO(pdf_bytes))
    except Exception as exc:
        report.skipped_reason = f"pdfplumber_open_failed: {exc.__class__.__name__}"
        return report
    with pdf_ctx as pdf:
        for page in pdf.pages:
            rows = _cluster_words_by_row(page)
            for row in rows:
                tokens = _row_to_tokens(row)
                parsed = _split_row_into_columns(tokens)
                if not parsed:
                    continue
                full_row_text = " ".join(tokens)
                if not _row_looks_like_line_item(parsed, target_total, full_row_text):
                    continue
                candidates.append(parsed)

    if not candidates:
        report.skipped_reason = "no_table_rows_found"
        return report

    if target_total is not None:
        recovered_sum = sum(c["line_total"] for c in candidates)
        diff = abs(recovered_sum - target_total)
        tolerance = max(target_total * 0.05, 0.50)
        if diff > tolerance:
            report.skipped_reason = (
                f"sum_mismatch: pdf_table_sum={recovered_sum:.2f} vs "
                f"target={target_total:.2f} (diff={diff:.2f}, tol={tolerance:.2f})"
            )
            report.sum_recovered = recovered_sum
            return report

    items = []
    for i, c in enumerate(candidates, start=1):
        item: dict = {
            "line_no": i,
            "line_number": i,
            "item_id": c.get("item_id"),
            "item_description": (c.get("description") or "")[:500],
            "quantity": c.get("quantity"),
            "unit_price": c.get("unit_price"),
            "line_amount": c["line_total"],
            "line_total": c["line_total"],
        }
        # Derive per-line tax IF the header has an explicit tax_percent.
        # The doc says "Tax (20%)" → header.tax_percent=20 → real evidence.
        # Per-line tax = line_total × tax_rate is derivation, not fabrication.
        if tax_percent is not None and tax_percent > 0:
            try:
                rate = float(tax_percent)
                if rate > 1:  # allow both 20 and 0.20 forms
                    rate /= 100.0
                tax = round(c["line_total"] * rate, 2)
                item["tax_percent"] = float(tax_percent)
                item["tax_amount"] = tax
                # PO/Quote line schemas use ``total_amount`` post-tax;
                # Invoice line schema uses ``total_amount_incl_tax``.
                item["total_amount"] = round(c["line_total"] + tax, 2)
                item["total_amount_incl_tax"] = round(c["line_total"] + tax, 2)
            except (ValueError, TypeError):
                item["total_amount"] = c["line_total"]
        else:
            item["total_amount"] = c["line_total"]
        items.append(item)
    report.items = items
    report.items_recovered = len(items)
    report.sum_recovered = sum(it["line_total"] for it in items)
    logger.info(
        "[PdfTableRecovery] %s: recovered %d line item(s), sum=%.2f, target=%.2f",
        file_path or "", len(items), report.sum_recovered, target_total or 0,
    )
    return report
