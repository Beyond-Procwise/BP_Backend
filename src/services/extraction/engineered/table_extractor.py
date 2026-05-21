"""L2 — line-item extraction from ParsedDocument.tables.

Reads the table(s) on each page, matches the header row's cell text to a
line-item field via canonical_labels (case-insensitive substring match),
and emits one Candidate per (line_index, field). Skips tables that lack
an identified header row.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from src.services.extraction.types import Candidate, Span
from src.services.extraction_v3.yaml_schema.loader import DocSchema, FieldSpec

log = logging.getLogger(__name__)


def _header_to_field(header_text: str, line_fields: list[FieldSpec]) -> str | None:
    """Match header cell text to a line-item field by canonical_labels."""
    h = (header_text or "").strip().lower()
    if not h:
        return None
    # exact match preferred
    for f in line_fields:
        for lbl in (f.canonical_labels or []):
            if h == lbl.lower():
                return f.name
    # substring fallback (handles "Unit Price ($)" → "Unit Price")
    for f in line_fields:
        for lbl in (f.canonical_labels or []):
            if lbl.lower() in h or h in lbl.lower():
                return f.name
    return None


_AMOUNT_CLEAN_RE = re.compile(r"[^\d.\-]")
_DIGIT_RE = re.compile(r"\d")

# Summary-label patterns that procurement docs put in the DESCRIPTION
# column on the totals rows (Sub-Total/Tax/Discount/Grand Total). These
# are NOT line items — they're the financial summary block. When a row's
# item_description matches one of these, skip the whole row.
_SUMMARY_DESC_RE = re.compile(
    r"^\s*"
    r"(?:"
    r"sub\s*-?\s*total"
    r"|grand\s*-?\s*total"
    r"|tax(?:\s*\(?\s*\d{1,3}(?:\.\d+)?%?\s*\)?)?"
    r"|vat(?:\s*\(?\s*\d{1,3}(?:\.\d+)?%?\s*\)?)?"
    r"|gst(?:\s*\(?\s*\d{1,3}(?:\.\d+)?%?\s*\)?)?"
    r"|sales\s+tax"
    r"|discount(?:\s*\(?\s*\d{1,3}(?:\.\d+)?%?\s*\)?)?"
    r"|net\s+(?:total|amount)"
    r"|amount\s+due"
    r"|balance\s+due"
    r"|payable"
    r"|total\s+payable"
    r"|total\s+amount(?:\s+due)?"
    r"|total"
    r")"
    r"\s*[:$£€¥]*\s*$",
    re.IGNORECASE,
)


def _is_useful_value(field_name: str, value: str) -> bool:
    """Reject obviously-empty / heading-only cells."""
    v = (value or "").strip()
    if not v:
        return False
    # Numeric line-item fields require at least one digit
    if field_name in ("quantity", "unit_price", "line_amount", "tax_amount",
                      "tax_percent", "total_amount_incl_tax", "total_amount", "line_total"):
        return bool(_DIGIT_RE.search(v))
    return True


def extract_line_items(parsed: Any, schema: DocSchema) -> list[Candidate]:
    """Walk parsed.tables, emit per-line Candidates keyed `line_items[i].<field>`."""
    if not schema.line_items or not schema.line_items.fields:
        return []
    line_fields = schema.line_items.fields

    out: list[Candidate] = []
    line_index = 0
    for page in parsed.pages:
        for tbl in page.tables:
            if tbl.header_row_index is None or tbl.header_row_index >= len(tbl.rows):
                continue
            header_row = tbl.rows[tbl.header_row_index]
            # Map column index → field name
            col_to_field: dict[int, str] = {}
            for cell in header_row:
                fld = _header_to_field(cell.text, line_fields)
                if fld:
                    col_to_field[cell.col_index] = fld
            if not col_to_field:
                continue  # no recognised columns

            for ri, row in enumerate(tbl.rows):
                if ri == tbl.header_row_index:
                    continue
                # First pass: gather candidate values for this row so we can
                # decide whether it qualifies as a line item BEFORE emitting.
                row_values: dict[str, tuple[str, Any]] = {}
                for cell in row:
                    fld = col_to_field.get(cell.col_index)
                    if not fld:
                        continue
                    value = (cell.text or "").strip()
                    if not _is_useful_value(fld, value):
                        continue
                    row_values[fld] = (value, cell)
                # A "line item" is a row that itemises something — so it must
                # have an item_description. Summary rows like
                # "| | Sub Total | £5,000" carry values in the amount column
                # but have no description.
                if "item_description" not in row_values:
                    continue
                # And the item_description must not itself be a summary
                # label ("Sub-Total" / "Tax (20%)" / "Grand Total" / etc.) —
                # some templates put those labels in the description column.
                desc_value = row_values["item_description"][0]
                if _SUMMARY_DESC_RE.match(desc_value):
                    continue
                local_idx = line_index
                for fld, (value, cell) in row_values.items():
                    out.append(Candidate(
                        field=f"line_items[{local_idx}].{fld}",
                        value=value,
                        span=Span(page=tbl.page, bbox=tuple(cell.bbox), text=value),
                        source="table",
                        pattern_name=None,
                        confidence=0.88,  # tables are structurally reliable
                    ))
                if row_values:
                    line_index += 1
    return out
