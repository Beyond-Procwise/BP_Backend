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


#: A column header that names money. Matched against the HEADER TEXT, not a
#: value, so "Unit Price" and "Line Total" are caught whatever is under them.
#: `value` and `rate` are here because the SQL and YAML comments both list
#: them and the first version of this regex did not: a "Unit Rate" or "Unit
#: Value" column still matched the bare "Unit" label and landed money in
#: unit_of_measure -- the same defect the live run of 2026-10-05 found on
#: "Unit Price", wearing a different word.
_MONEY_HEADER_RE = re.compile(
    r"(price|amount|total|cost|currency|tax|vat|value|rate)", re.I)


def _declares_money(line_fields: list[FieldSpec]) -> bool:
    """Does this doc type have any priced line field at all?

    A goods receipt does not, by design (extraction_schemas/goods_receipt.yaml):
    it proves DELIVERY and carries quantities only.
    """
    return any(
        f.type == "money" or _MONEY_HEADER_RE.search(f.name or "")
        for f in line_fields
    )


def _header_match(header_text: str, line_fields: list[FieldSpec]) -> tuple[str | None, int]:
    """Match header cell text to a line-item field by canonical_labels.

    Substring matches pick the LONGEST (most specific) matching label, so a
    generic label like "Unit" (unit_of_measure) cannot steal a specific column
    like "Unit Price" (unit_price) just because its field is listed first.

    That rule depends on a MORE specific label existing to be preferred. On a
    doc type with no priced field there is none, so "Unit Price" matched "Unit"
    and 'GBP 13.59' was about to be stored as the unit a line was delivered in
    -- found on the live run of 2026-10-05. A doc type that declares no money
    field refuses a money-named header outright: the same rule as the absent
    column and the absent schema field, one layer further out.
    """
    h = (header_text or "").strip().lower()
    if not h:
        return None, 0
    if _MONEY_HEADER_RE.search(h) and not _declares_money(line_fields):
        return None, 0
    # exact match preferred -- it outranks any substring match (see _find_header)
    for f in line_fields:
        for lbl in (f.canonical_labels or []):
            if h == lbl.lower():
                return f.name, 1000 + len(lbl)
    # substring: choose the most specific (longest) matching label
    best_field: str | None = None
    best_len = -1
    for f in line_fields:
        for lbl in (f.canonical_labels or []):
            ll = lbl.lower()
            if ll in h or h in ll:
                if len(ll) > best_len:
                    best_len = len(ll)
                    best_field = f.name
    return best_field, max(best_len, 0)


def _header_to_field(header_text: str, line_fields: list[FieldSpec]) -> str | None:
    return _header_match(header_text, line_fields)[0]


def _find_header(tbl: Any, line_fields: list[FieldSpec]) -> tuple[int | None, dict[int, str]]:
    """Locate the column-header row and its column→field map.

    Tries the table's declared header row first; if that yields no recognised
    columns (spreadsheets put the real "Description | Qty | Unit Price" header
    below title/metadata rows), scans for the row with the most field matches.
    """
    n = len(tbl.rows)
    order: list[int] = []
    if tbl.header_row_index is not None and 0 <= tbl.header_row_index < n:
        order.append(tbl.header_row_index)
    order += [i for i in range(n) if i not in order]

    best_i: int | None = None
    best_map: dict[int, str] = {}
    for i in order:
        # One column per field: the best-matching header wins (an exact label over a partial
        # one, then the leftmost). Two columns used to feed one field and the LAST won, so
        # "Provision | Detail" took the detail text ("Standard 1.5x; weekend 2x.") as the
        # description of the Overtime line.
        best: dict[str, tuple[int, int]] = {}
        for cell in tbl.rows[i]:
            fld, score = _header_match(cell.text, line_fields)
            if fld and (fld not in best or score > best[fld][0]):
                best[fld] = (score, cell.col_index)
        cmap: dict[int, str] = {col: fld for fld, (_score, col) in best.items()}
        if len(cmap) > len(best_map):
            best_i, best_map = i, cmap
        # A declared header with 2+ recognised columns is trusted as-is.
        if i == tbl.header_row_index and len(cmap) >= 2:
            break
    return best_i, best_map


_AMOUNT_CLEAN_RE = re.compile(r"[^\d.\-]")
_DIGIT_RE = re.compile(r"\d")

# Summary-label patterns that procurement docs put in the DESCRIPTION
# column on the totals rows (Sub-Total/Tax/Discount/Grand Total). These
# are NOT line items — they're the financial summary block. When a row's
# item_description matches one of these, skip the whole row.
#: A section total named after its section: "Staffing subtotal", "Fixed-fee subtotal",
#: "Provisions sub-total". _SUMMARY_DESC_RE only knew the bare word, so these were kept as
#: lines and the line sum double-counted every section.
_SECTION_SUBTOTAL_RE = re.compile(r"\bsub\s*-?\s*totals?\s*[:$£€¥]*\s*$", re.IGNORECASE)
#: "Total (ex-VAT)", "Grand total (incl. VAT)": the word total qualified only by a bracket.
_QUALIFIED_TOTAL_RE = re.compile(r"^\s*(?:grand\s+)?total\s*\([^)]*\)\s*[:$£€¥]*\s*$", re.IGNORECASE)


def _is_summary_row(desc_value: str, row: list) -> bool:
    """A totals/subtotal row, not an item.

    Besides the label patterns, a label that the document MERGED across the row's other
    columns ("Staffing subtotal | Staffing subtotal | ... | 1,995,700" -- a docx merged cell
    repeats its text in every column it spans) is a heading or a total: an item puts
    different things in its columns."""
    if (_SUMMARY_DESC_RE.match(desc_value) or _SECTION_SUBTOTAL_RE.search(desc_value)
            or _QUALIFIED_TOTAL_RE.match(desc_value)):
        return True
    d = desc_value.strip().lower()
    repeats = sum(1 for cell in row if (cell.text or "").strip().lower() == d)
    return bool(d) and repeats >= 3


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
            if not tbl.rows:
                continue
            header_idx, col_to_field = _find_header(tbl, line_fields)
            if header_idx is None or not col_to_field:
                continue  # no recognised columns

            for ri, row in enumerate(tbl.rows):
                # Skip the header and any title/metadata rows above it — line
                # items only appear BELOW the column header.
                if ri <= header_idx:
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
                if _is_summary_row(desc_value, row):
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
