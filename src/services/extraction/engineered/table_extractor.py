"""L2 — line-item extraction from ParsedDocument.tables.

Reads the table(s) on each page, matches the header row's cell text to a
line-item field via canonical_labels (case-insensitive substring match),
and emits one Candidate per (line_index, field). When a table has no
recognisable header at index 0, we rescan all rows and pick the row with
the most label matches as the header — this rescues docs whose first
row is a section title or a summary block.

When the table-based pass yields zero line items but the document text
contains a markdown 'Description' / 'Item' / 'Service' heading followed
by a paragraph and a currency amount, we fall back to extracting one or
more line items from that text. This rescues paragraph-layout invoices /
POs / quotes that docling failed to recognise as tables.
"""
from __future__ import annotations

import html
import logging
import re
from typing import Any

from src.services.extraction.types import Candidate, Span
from src.services.extraction_v3.yaml_schema.loader import DocSchema, FieldSpec


def _clean_text_value(v: str) -> str:
    """HTML-unescape a value pulled from docling's markdown export.

    Docling escapes `&`, `|`, `<`, `>` etc. in `full_text` markdown; the
    structural .tables cells keep the literal char. Whichever fallback
    we pull from, the persisted value should be the source-document form
    (the human-readable one) — so always decode entities before emitting.
    """
    if not v:
        return v
    return html.unescape(v).strip()

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


def _best_header_row(tbl: Any, line_fields: list[FieldSpec]) -> tuple[int | None, dict[int, str]]:
    """Return (row_index, col_to_field) for the row in this table whose
    cells match the most line-item labels.

    Falls back from the parser-reported header_row_index when that row
    yields zero matches. Returns (None, {}) when no row matches anything.
    """
    best_ri: int | None = None
    best_map: dict[int, str] = {}
    for ri, row in enumerate(tbl.rows or []):
        col_to_field: dict[int, str] = {}
        for cell in row:
            fld = _header_to_field(cell.text, line_fields)
            if fld:
                col_to_field[cell.col_index] = fld
        if len(col_to_field) > len(best_map):
            best_map = col_to_field
            best_ri = ri
    return best_ri, best_map


def extract_line_items(parsed: Any, schema: DocSchema) -> list[Candidate]:
    """Walk parsed.tables, emit per-line Candidates keyed `line_items[i].<field>`.

    Falls back to a markdown-text scan when no structural table yields
    any candidates (paragraph-layout docs).
    """
    if not schema.line_items or not schema.line_items.fields:
        return []
    line_fields = schema.line_items.fields

    out: list[Candidate] = []
    line_index = 0
    for page in parsed.pages:
        for tbl in page.tables:
            if not tbl.rows:
                continue
            header_ri, col_to_field = _best_header_row(tbl, line_fields)
            if header_ri is None or not col_to_field:
                continue

            for ri, row in enumerate(tbl.rows):
                if ri == header_ri:
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
                # Drop rows where ANY other cell carries a summary label —
                # docling sometimes wraps a Sub-Total row's label into the
                # qty column when the previous row's description spilled
                # over. Treat the whole row as the summary block.
                other_cell_is_summary = False
                for cell in row:
                    if cell.col_index == 0:
                        continue
                    txt = (cell.text or "").strip()
                    if not txt:
                        continue
                    if _SUMMARY_DESC_RE.match(txt):
                        other_cell_is_summary = True
                        break
                if other_cell_is_summary:
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

    # Text fallback. Only run when nothing came back from tables — keeps
    # this conservative so we never compete with a clean structural pass.
    if not out:
        out.extend(_extract_line_items_from_markdown_table(parsed, line_fields))
    if not out:
        out.extend(_extract_line_items_from_text(parsed, line_fields))
    # Drop a trailing mis-captured subtotal/total block (table parsers often lay
    # the Subtotal/Tax/Total values into the amount column with garbled or
    # payment-terms text in the description column — see TechWorld Q-005-41).
    out = _trim_trailing_summary_candidates(out, line_fields)
    return out


def _line_index_of(field: str) -> int | None:
    m = re.match(r"line_items\[(\d+)\]\.", field)
    return int(m.group(1)) if m else None


def _amount_to_float(value: Any) -> float | None:
    """Parse a currency string ("£6,750" / "1,250.00") to float; None if not numeric."""
    s = _AMOUNT_CLEAN_RE.sub("", str(value if value is not None else ""))
    if s in ("", "-", ".", "-."):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _trim_trailing_summary_candidates(
    out: list[Candidate], line_fields: list[FieldSpec],
) -> list[Candidate]:
    """Subtotal-closure trim at the candidate level: if one of the LAST TWO line
    groups has an amount equal to the running sum of the preceding (>=2) line
    amounts, that row is the mis-captured Subtotal — drop it and every row after
    (Tax/Total/payment-terms bleed). Mirrors completeness.derive_subtotal_from_lines
    so summary rows never reach persistence. Conservative: only the last two
    positions are eligible, so a real line whose amount coincidentally equals the
    running sum mid-list is never trimmed.
    """
    amt_name = next(
        (f.name for f in line_fields
         if f.name in ("line_amount", "line_total", "total_amount")),
        None,
    )
    if not amt_name or not out:
        return out
    idx_amt: dict[int, float | None] = {}
    max_idx = -1
    for c in out:
        i = _line_index_of(c.field)
        if i is None:
            continue
        max_idx = max(max_idx, i)
        if c.field.endswith(f".{amt_name}"):
            idx_amt[i] = _amount_to_float(c.value)
    if max_idx < 1:
        return out
    n = max_idx + 1
    amts = [idx_amt.get(i) for i in range(n)]
    cut: int | None = None
    for i in range(n):
        if amts[i] is None or i < n - 2:  # only the last two rows are closure-eligible
            continue
        prior = [a for a in amts[:i] if a is not None]
        if len(prior) >= 2:
            s = sum(prior)
            if s > 0 and abs(amts[i] - s) <= max(0.01, 0.01 * s):
                cut = i
                break
    if cut is None:
        return out
    return [c for c in out
            if (_line_index_of(c.field) is None) or (_line_index_of(c.field) < cut)]


def _extract_line_items_from_markdown_table(
    parsed: Any, line_fields: list[FieldSpec],
) -> list[Candidate]:
    """Parse markdown-style pipe tables that docling left in full_text but
    did NOT promote to parsed.tables[]. These look like:

        | SERVICES                | TOTAL  |
        |-------------------------|--------|
        | Social Media Management | £2,000 |
        | SUBTOTAL                | £2,000 |
        | VAT 20%                 | £400   |

    Reads the header row to map columns; emits one candidate per cell on
    each data row. Summary rows are filtered by _SUMMARY_DESC_RE.
    """
    full_text = getattr(parsed, "full_text", "") or ""
    if "|" not in full_text:
        return []
    lines = full_text.splitlines()
    out: list[Candidate] = []
    i = 0
    line_index = 0
    while i < len(lines):
        ln = lines[i].strip()
        # A header row: starts with '|', has more than one column, and the
        # next line is a separator like '|----|----|'
        if (
            ln.startswith("|") and ln.count("|") >= 3
            and i + 1 < len(lines)
            and re.match(r"^\s*\|[\s\-\|:]+\|\s*$", lines[i + 1])
        ):
            header_cells = [c.strip() for c in ln.strip("|").split("|")]
            col_to_field: dict[int, str] = {}
            for ci, ctext in enumerate(header_cells):
                fld = _header_to_field(ctext, line_fields)
                if fld:
                    col_to_field[ci] = fld
            if not col_to_field:
                i += 2
                continue
            j = i + 2
            while j < len(lines):
                row_ln = lines[j].strip()
                if not row_ln.startswith("|") or row_ln.count("|") < 3:
                    break
                cells = [c.strip() for c in row_ln.strip("|").split("|")]
                row_values: dict[str, str] = {}
                for ci, val in enumerate(cells):
                    fld = col_to_field.get(ci)
                    if not fld or not _is_useful_value(fld, val):
                        continue
                    row_values[fld] = val
                if "item_description" in row_values:
                    desc = row_values["item_description"]
                    if not _SUMMARY_DESC_RE.match(desc):
                        for fld, val in row_values.items():
                            cleaned = _clean_text_value(val)
                            out.append(Candidate(
                                field=f"line_items[{line_index}].{fld}",
                                value=cleaned,
                                span=Span(page=0, bbox=(0.0, 0.0, 0.0, 0.0), text=cleaned),
                                source="table",
                                pattern_name="md_pipe_table",
                                confidence=0.85,
                            ))
                        line_index += 1
                j += 1
            i = j
        else:
            i += 1
    return out


# ---------------------------------------------------------------------------
# Markdown-text fallback for paragraph-layout documents
# ---------------------------------------------------------------------------
#
# Some procurement docs (single-service invoices, simple consulting POs)
# render their line items as a section heading + paragraph + a currency
# amount rather than as a table. Docling outputs nothing in `.tables` for
# these. The fallback locates a description heading, takes the next
# non-empty content paragraph as `item_description`, and pairs it with
# the nearest currency amount within a small window as `line_amount`.
#
# The fallback writes ONE candidate per line item it identifies. It does
# NOT try to fabricate `quantity` / `unit_price` when those are not in
# the text — leaving them NULL is the explicit policy
# (no_fabrication_null_when_absent).

_AMOUNT_INLINE_RE = re.compile(
    r"(?:£|\$|€|¥|GBP|USD|EUR|JPY|AUD|CAD|INR|SGD|HKD|CHF|NZD)\s*"
    r"-?\d{1,3}(?:[,\s]\d{3})*(?:\.\d+)?",
)
_SECTION_BREAK_RE = re.compile(r"^\s*##+\s+", re.IGNORECASE)
# SOW-scope sections that look like description headings but aren't line items.
# These typically lead into a bulleted list of deliverables, not the priced
# line item — skip them.
_SOW_SCOPE_HEADING_RE = re.compile(
    r"^\s*##+\s*(?:description\s+of\s+services|"
    r"scope\s+of\s+work|"
    r"deliverables|"
    r"statement\s+of\s+work)\s*[:\.]?\s*$",
    re.IGNORECASE,
)


def _looks_like_summary_line(text: str) -> bool:
    """Reject paragraphs that are themselves summary labels (Sub-Total / Tax / …)."""
    return bool(_SUMMARY_DESC_RE.match((text or "").strip()))


def _build_desc_heading_re(line_fields: list[FieldSpec]) -> re.Pattern[str]:
    """Compile a heading regex from the schema's item_description labels."""
    desc_meta = next((f for f in line_fields if f.name == "item_description"), None)
    labels = list(desc_meta.canonical_labels or []) if desc_meta else []
    # Always include a few minimal generics so DOCX exports that lack a
    # markdown heading prefix (e.g. just "Item") still match.
    labels = list(dict.fromkeys(labels + ["Item Description", "Description", "Item"]))
    escaped = "|".join(re.escape(lbl) for lbl in sorted(labels, key=len, reverse=True))
    # Trailing "s?" so a singular label ("Service", "Item") still matches the
    # plural column header docs actually print ("SERVICES", "ITEMS").
    return re.compile(
        rf"^\s*(?:##+\s*)?(?:{escaped})s?\s*[:\.]?\s*$",
        re.IGNORECASE,
    )


def _is_column_header_line(line: str, line_fields: list[FieldSpec]) -> bool:
    """True when a line is purely linearised table column-headers — e.g. docling
    flattens a "SERVICES | TOTAL | RATE | QTY" header into separate lines
    "TOTAL RATE" / "QTY". Every whitespace token must map to a line-item field
    label. Used to walk PAST these from a "SERVICES"/"Item" heading to the real
    priced row, without mistaking a real description that merely contains a
    header word ("Rate Card Design") for a header.
    """
    stripped = re.sub(r"^[\s#>*\-]+", "", line or "").strip()
    if not stripped:
        return False
    tokens = [t for t in re.split(r"\s+", stripped) if t]
    if not tokens:
        return False
    return all(_header_to_field(t, line_fields) for t in tokens)


def _extract_line_items_from_text(parsed: Any, line_fields: list[FieldSpec]) -> list[Candidate]:
    """Markdown-text fallback. Returns a list of Candidates; empty when
    nothing usable found.

    Conservative by design: a candidate is only emitted when the heading
    matches the schema's `item_description.canonical_labels`, the content
    paragraph is not a summary label, AND a currency amount is found
    within a small window. No amount → no candidate (we never fabricate).
    """
    full_text = getattr(parsed, "full_text", "") or ""
    if not full_text:
        return []
    desc_meta = next((f for f in line_fields if f.name == "item_description"), None)
    amount_meta = next(
        (f for f in line_fields if f.name in ("line_amount", "line_total", "total_amount")),
        None,
    )
    if desc_meta is None or amount_meta is None:
        return []

    desc_heading_re = _build_desc_heading_re(line_fields)
    lines = full_text.splitlines()
    found: list[tuple[str, str]] = []  # (description, amount)
    consumed: set[int] = set()

    i = 0
    while i < len(lines):
        if i in consumed:
            i += 1
            continue
        if _SOW_SCOPE_HEADING_RE.match(lines[i]):
            i += 1
            continue
        if not desc_heading_re.match(lines[i]):
            i += 1
            continue
        # Walk forward past blanks, image markers, and stacked summary-label
        # sub-headings (## SUBTOTAL / ## TOTAL ...) — docling sometimes
        # interleaves column-header markdown above the actual content.
        j = i + 1
        while j < len(lines):
            stripped = lines[j].strip()
            if not stripped or stripped.startswith("<!--"):
                j += 1
                continue
            if _SECTION_BREAK_RE.match(lines[j]):
                head = lines[j].lstrip("#").strip()
                if _looks_like_summary_line(head):
                    j += 1
                    continue
            # Walk past linearised column-header rows ("TOTAL RATE", "QTY")
            # that sit between the "SERVICES" heading and the priced item.
            if _is_column_header_line(lines[j], line_fields):
                j += 1
                continue
            break
        if j >= len(lines):
            break
        next_line = lines[j].strip()
        # Case 1: next non-empty line is itself a sub-heading naming the item
        # ("## Enterprise IT Consulting Package"). Use the heading text as desc.
        if _SECTION_BREAK_RE.match(lines[j]):
            heading_text = lines[j].lstrip("#").strip()
            if heading_text and not _looks_like_summary_line(heading_text):
                amount = _nearest_amount(lines, j, window=12)
                if not amount:
                    # Single-item fallback: Subtotal IS the line total.
                    amount = _find_subtotal_amount(lines)
                if amount:
                    found.append((heading_text, amount))
                    consumed.update(range(i, j + 1))
            i = j + 1
            continue
        # Case 2: plain paragraph text. Collect consecutive non-empty
        # non-summary content lines into a single description until we
        # hit a summary line (Sub-Total/Tax/Total/etc.) or a currency
        # amount. Description = the joined non-amount content.
        desc_parts: list[str] = []
        amount_for_line: str | None = None
        k = j
        while k < len(lines):
            ln_stripped = lines[k].strip()
            if not ln_stripped or ln_stripped.startswith("<!--"):
                k += 1
                continue
            # Stop on another markdown section heading.
            if _SECTION_BREAK_RE.match(lines[k]):
                break
            # Stop on summary-label line (Sub-Total: / Tax: ...).
            if _looks_like_summary_line(ln_stripped):
                break
            # Stop and capture amount when the line IS purely a currency value.
            m_amt = _AMOUNT_INLINE_RE.search(ln_stripped)
            if m_amt and not re.sub(_AMOUNT_INLINE_RE, "", ln_stripped).strip():
                amount_for_line = m_amt.group(0).strip()
                k += 1
                break
            desc_parts.append(ln_stripped)
            k += 1
            # Safety: don't run away
            if len(desc_parts) > 5:
                break
        if desc_parts:
            desc = " ".join(desc_parts).strip()
            if desc and not _looks_like_summary_line(desc):
                amount = amount_for_line or _nearest_amount(lines, k - 1, window=12)
                if not amount:
                    amount = _find_subtotal_amount(lines)
                if amount:
                    found.append((desc, amount))
                    consumed.update(range(i, k))
        i = max(k, j + 1)

    if not found:
        return []

    out: list[Candidate] = []
    for idx, (desc, amount) in enumerate(found):
        # HTML-unescape the user-facing string values: docling's markdown
        # export carries `&amp;` / `&#124;` etc. The structural .tables
        # path keeps the literal char; for consistency we always store
        # the decoded form. The candidate's span.text mirrors the value
        # so the L3 grounding gate (which html-unescapes full_text too)
        # accepts it.
        clean_desc = _clean_text_value(desc)
        clean_amount = _clean_text_value(amount)
        # No raw-substring check here — dispatch.py's L3 grounding gate
        # re-checks each candidate.span.text against full_text with the
        # same whitespace-normalisation rules as the judge layer.
        out.append(Candidate(
            field=f"line_items[{idx}].item_description",
            value=clean_desc,
            span=Span(page=0, bbox=(0.0, 0.0, 0.0, 0.0), text=clean_desc),
            source="table",
            pattern_name="text_fallback",
            confidence=0.72,
        ))
        out.append(Candidate(
            field=f"line_items[{idx}].{amount_meta.name}",
            value=clean_amount,
            span=Span(page=0, bbox=(0.0, 0.0, 0.0, 0.0), text=clean_amount),
            source="table",
            pattern_name="text_fallback",
            confidence=0.72,
        ))
    return out


def _nearest_amount(lines: list[str], anchor_idx: int, window: int = 8) -> str | None:
    """Find the closest currency amount within ±window lines of anchor.
    Skips amounts that are explicitly part of a Tax / Sub-Total / Grand Total /
    Total Amount line — those are summary, not line totals.
    """
    candidates: list[tuple[int, str]] = []  # (distance, amount_str)
    for offset in range(1, window + 1):
        for direction in (1, -1):
            j = anchor_idx + direction * offset
            if not (0 <= j < len(lines)):
                continue
            line = lines[j].strip()
            if not line:
                continue
            # Skip lines that are themselves summary labels w/ amounts
            if re.search(r"(?i)\b(tax|vat|gst|sub\s*-?\s*total|grand\s*-?\s*total|"
                         r"total\s+amount|discount|payable|amount\s+due)\b", line):
                continue
            m = _AMOUNT_INLINE_RE.search(line)
            if m:
                candidates.append((offset, m.group(0).strip()))
                break  # first hit at this offset wins for this direction
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


_SUBTOTAL_LABEL_RE = re.compile(
    r"(?i)(?:^|\s)(?:sub\s*-?\s*total|subtotal)\b",
)


def _find_subtotal_amount(lines: list[str]) -> str | None:
    """Locate the document's Subtotal value. Used as line_amount when
    text-fallback identified a single-item description but no nearby
    amount — for single-item docs, Subtotal IS the line total.

    Strategy:
      - Find lines containing 'Subtotal' / 'Sub-Total'.
      - First look on the same line for an inline currency value.
      - Otherwise scan forward up to 4 non-empty lines for the next
        currency value, stopping if a different summary label
        (tax/discount/total) appears first.
    """
    for i, raw in enumerate(lines):
        line = raw.strip()
        if not line:
            continue
        if not _SUBTOTAL_LABEL_RE.search(line):
            continue
        # Skip "## SUBTOTAL"-style heading where the value is on the next line.
        m = _AMOUNT_INLINE_RE.search(line)
        if m:
            return m.group(0).strip()
        # Look ahead a few lines for the amount
        non_empty_seen = 0
        for j in range(i + 1, min(len(lines), i + 12)):
            nxt = lines[j].strip()
            if not nxt or nxt.startswith("<!--"):
                continue
            non_empty_seen += 1
            if non_empty_seen > 4:
                break
            # Stop if a different summary section starts before we found a value
            if re.search(
                r"(?i)\b(tax|vat|gst|grand\s*-?\s*total|total\s+amount|"
                r"discount|payable|amount\s+due|total)\b",
                nxt,
            ):
                break
            m2 = _AMOUNT_INLINE_RE.search(nxt)
            if m2:
                return m2.group(0).strip()
    return None
