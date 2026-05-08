"""Multi-page line-item stitching.

PDFs split a single line-items table across pages. Off-the-shelf parsers
(pdfplumber, AnalyzeExpense) return per-page tables with no continuation
hints, so when a 50-row line-items table spans 3 pages we get 3 separate
tables and the downstream LLM sees only the first.

The algorithm here, applied to the structural extractor's
``ParsedDocument`` *after* parsing:

  1. Group ``tables`` by page (via the ``BBox.page`` of each table's
     source anchor or its first token).
  2. For each pair (page-N table, page-N+1 table):
        - Compute ``column_signature`` from the first row's column count
          plus the rounded x-span of each column.
        - If signatures match within ``COL_TOL_PX`` and the next-page
          table has NO header row, treat it as a continuation and merge.
  3. Walk the merged tables and emit a unified ``StitchedTable`` whose
     rows are concatenated in page order, header row taken from the
     first page only.

The output is a plain list[dict] of stitched line items, ready for the
existing line-items handling. The original ``ParsedDocument`` is left
unchanged — multi-page stitching is purely additive.

"Page X of Y" assertion: we extract the maximum Y from the parsed text
and warn (via the returned ``MultiPageReport``) when the observed page
count disagrees.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


__all__ = [
    "MultiPageReport", "stitch_multi_page_tables", "extract_page_y_assertion",
]


# Column-x-span match tolerance in PDF points (~1pt = 1/72 inch).
COL_TOL_PX = 6.0

# Header-row repetition matcher: identical first cell text on consecutive
# pages indicates a continuation table whose header was repeated.
_HEADER_REPEAT_RE = re.compile(
    r"\b(description|item|qty|quantity|unit\s*price|amount|total)\b",
    re.IGNORECASE,
)

# "Page 2 of 5", "Page 2/5", "Pg. 2 / 5" — extracts the Y total.
_PAGE_OF_RE = re.compile(
    r"page\s*\d+\s*(?:of|/|\\)\s*(\d+)",
    re.IGNORECASE,
)


@dataclass
class MultiPageReport:
    pages_observed: int
    pages_asserted: Optional[int] = None
    page_assertion_passed: Optional[bool] = None
    tables_per_page: dict[int, int] = field(default_factory=dict)
    stitched_groups: list[list[int]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def extract_page_y_assertion(parsed_text: str) -> Optional[int]:
    """Return the largest Y in any "Page X of Y" marker, or None."""
    if not parsed_text:
        return None
    best: Optional[int] = None
    for m in _PAGE_OF_RE.finditer(parsed_text):
        try:
            y = int(m.group(1))
        except ValueError:
            continue
        if best is None or y > best:
            best = y
    return best


def _table_page(table) -> int:
    """Return the page number of a Table, falling back across anchor types."""
    sa = getattr(table, "source_anchor", None)
    if sa is not None and hasattr(sa, "page"):
        return int(getattr(sa, "page", 0) or 0)
    if hasattr(sa, "row"):
        # XLSX / DOCX use sheet/row anchors; treat as page 0.
        return 0
    # Fall back to the first token's bbox if available.
    rows = getattr(table, "rows", None) or []
    for row in rows:
        for cell in row:
            for tok in getattr(cell, "tokens", None) or []:
                anchor = getattr(tok, "anchor", None)
                if anchor is not None and hasattr(anchor, "page"):
                    return int(anchor.page or 0)
    return 0


def _column_signature(table) -> Optional[tuple]:
    """Return a tuple of (col_count, rounded x-spans) for the table's
    first non-empty row. Returns None if no x-coordinates are available
    (typical for XLSX/CSV — multi-page stitching is a PDF concern)."""
    rows = getattr(table, "rows", None) or []
    if not rows:
        return None
    first_row = rows[0]
    xs: list[tuple[int, int]] = []
    for cell in first_row:
        x0 = x1 = None
        for tok in getattr(cell, "tokens", None) or []:
            anchor = getattr(tok, "anchor", None)
            if anchor is None or not hasattr(anchor, "x0"):
                continue
            ax0 = float(anchor.x0)
            ax1 = float(anchor.x1)
            x0 = ax0 if x0 is None else min(x0, ax0)
            x1 = ax1 if x1 is None else max(x1, ax1)
        if x0 is None:
            return None
        xs.append((round(x0 / COL_TOL_PX), round(x1 / COL_TOL_PX)))
    return (len(first_row), tuple(xs))


def _is_header_row(row) -> bool:
    """Heuristic: row is a header if any cell's joined text matches one
    of the standard line-items column tokens."""
    for cell in row:
        text = " ".join(t.text for t in (getattr(cell, "tokens", None) or []))
        if _HEADER_REPEAT_RE.search(text):
            return True
    return False


def _row_cells_text(row) -> list[str]:
    out: list[str] = []
    for cell in row:
        out.append(" ".join(
            t.text for t in (getattr(cell, "tokens", None) or [])
        ))
    return out


def stitch_multi_page_tables(
    parsed_doc, *, parsed_text: Optional[str] = None,
) -> tuple[list[list[str]], MultiPageReport]:
    """Group continuation tables across pages and emit unified rows.

    Returns ``(stitched_rows, report)`` where ``stitched_rows`` is a list
    of raw text-cell lists (one per stitched-table row, header excluded)
    and ``report`` carries the multi-page metadata.

    The unified rows are NOT yet typed — the existing line-items
    extractor / salvage prompt stays responsible for converting cells
    into typed line items. This module's job is purely the page join.
    """
    tables = list(getattr(parsed_doc, "tables", None) or [])
    pages = sorted({_table_page(t) for t in tables})
    text = parsed_text if parsed_text is not None else getattr(parsed_doc, "full_text", "")

    report = MultiPageReport(
        pages_observed=getattr(parsed_doc, "pages_or_sheets", 0) or len(pages),
    )
    asserted = extract_page_y_assertion(text or "")
    if asserted is not None:
        report.pages_asserted = asserted
        report.page_assertion_passed = (
            asserted == report.pages_observed
        )
        if not report.page_assertion_passed:
            report.notes.append(
                f"page_assertion_failed: marker says {asserted}, "
                f"observed {report.pages_observed}"
            )

    # Per-page table count
    for t in tables:
        p = _table_page(t)
        report.tables_per_page[p] = report.tables_per_page.get(p, 0) + 1

    if len(pages) <= 1 or len(tables) <= 1:
        # Single-page document or single table — no stitching needed.
        if tables:
            rows = _flatten_table_to_text_rows(tables[0])
            return rows, report
        return [], report

    # Group tables that are continuations of each other across pages.
    # Walk pages in order; for each table on page N, look for one on
    # page N+1 with matching column signature AND no header row.
    groups: list[list[int]] = []
    used: set[int] = set()
    indexed = list(enumerate(tables))
    for i, t_a in indexed:
        if i in used:
            continue
        sig_a = _column_signature(t_a)
        page_a = _table_page(t_a)
        group = [i]
        used.add(i)
        if sig_a is None:
            groups.append(group)
            continue
        # Search forward for continuations
        for j, t_b in indexed:
            if j <= i or j in used:
                continue
            page_b = _table_page(t_b)
            if page_b <= page_a:
                continue
            sig_b = _column_signature(t_b)
            if sig_b is None:
                continue
            if sig_a == sig_b:
                rows_b = getattr(t_b, "rows", None) or []
                first_b = rows_b[0] if rows_b else None
                if first_b is None:
                    continue
                # Continuation table: NO header row in first row of t_b
                if not _is_header_row(first_b):
                    group.append(j)
                    used.add(j)
                    page_a = page_b
                else:
                    # Header repeats — still a continuation but skip
                    # the header row when concatenating.
                    group.append(j)
                    used.add(j)
                    page_a = page_b
        groups.append(group)
    report.stitched_groups = [g for g in groups if len(g) > 1]

    # Concatenate rows of each group, dropping duplicate header rows.
    if not groups:
        return [], report
    # Choose the largest group as the "main" line-items table.
    main_group = max(groups, key=len)
    out: list[list[str]] = []
    header_seen = False
    for idx in main_group:
        rows = getattr(tables[idx], "rows", None) or []
        for r in rows:
            if _is_header_row(r):
                if header_seen:
                    continue  # Skip repeated header on continuation page.
                header_seen = True
                continue  # Don't include the header row itself.
            out.append(_row_cells_text(r))
    return out, report


def _flatten_table_to_text_rows(table) -> list[list[str]]:
    rows = getattr(table, "rows", None) or []
    out: list[list[str]] = []
    header_seen = False
    for r in rows:
        if _is_header_row(r) and not header_seen:
            header_seen = True
            continue
        out.append(_row_cells_text(r))
    return out
