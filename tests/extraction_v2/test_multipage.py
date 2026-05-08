"""Tests for multi-page table stitching."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.services.extraction_v2.multipage import (  # noqa: E402
    extract_page_y_assertion, stitch_multi_page_tables,
)


@dataclass
class _BBox:
    page: int
    x0: float
    y0: float
    x1: float
    y1: float


@dataclass
class _Token:
    text: str
    anchor: object


@dataclass
class _Region:
    tokens: List[_Token]
    kind: str = "cell"


@dataclass
class _Table:
    rows: List[List[_Region]]
    source_anchor: Optional[object] = None
    header_row_index: Optional[int] = None


@dataclass
class _Doc:
    tables: List[_Table]
    full_text: str = ""
    pages_or_sheets: int = 1


def _cell(text: str, page: int, x0: float, x1: float) -> _Region:
    bb = _BBox(page=page, x0=x0, y0=0.0, x1=x1, y1=10.0)
    return _Region(tokens=[_Token(text=text, anchor=bb)])


def _row(cells: List[tuple], page: int) -> List[_Region]:
    """cells: list of (text, x0, x1)."""
    return [_cell(t, page, x0, x1) for (t, x0, x1) in cells]


def _table(rows_data, page: int) -> _Table:
    """rows_data: list of list of (text, x0, x1)."""
    rows = [_row(r, page) for r in rows_data]
    return _Table(rows=rows, source_anchor=_BBox(page, 0.0, 0.0, 600.0, 800.0))


# -- extract_page_y_assertion ----------------------------------------------

def test_page_y_assertion_finds_max_y():
    text = "Page 1 of 5\n... line 1 ...\nPage 2 of 5\n..."
    assert extract_page_y_assertion(text) == 5


def test_page_y_assertion_handles_slash_form():
    assert extract_page_y_assertion("Page 1 / 3") == 3


def test_page_y_assertion_returns_none_when_absent():
    assert extract_page_y_assertion("just some text") is None


# -- stitch_multi_page_tables ----------------------------------------------

def test_single_page_table_passes_through_unchanged():
    t = _table([
        [("Description", 50, 200), ("Qty", 220, 270), ("Amount", 280, 350)],
        [("Widget", 50, 200), ("2", 220, 270), ("100.00", 280, 350)],
    ], page=1)
    doc = _Doc(tables=[t], full_text="", pages_or_sheets=1)
    rows, report = stitch_multi_page_tables(doc)
    assert len(rows) == 1  # header dropped
    assert rows[0][0] == "Widget"
    assert report.stitched_groups == []


def test_two_page_continuation_table_is_merged():
    # Page 1: header + 2 rows
    p1 = _table([
        [("Description", 50, 200), ("Qty", 220, 270), ("Amount", 280, 350)],
        [("Widget A", 50, 200), ("1", 220, 270), ("10.00", 280, 350)],
        [("Widget B", 50, 200), ("2", 220, 270), ("20.00", 280, 350)],
    ], page=1)
    # Page 2: continuation — no header, 2 more rows, same column x-spans
    p2 = _table([
        [("Widget C", 50, 200), ("3", 220, 270), ("30.00", 280, 350)],
        [("Widget D", 50, 200), ("4", 220, 270), ("40.00", 280, 350)],
    ], page=2)
    doc = _Doc(tables=[p1, p2], full_text="Page 1 of 2\n", pages_or_sheets=2)
    rows, report = stitch_multi_page_tables(doc)
    # Header dropped, 2+2 data rows joined
    assert len(rows) == 4
    assert [r[0] for r in rows] == ["Widget A", "Widget B", "Widget C", "Widget D"]
    assert len(report.stitched_groups) == 1
    assert len(report.stitched_groups[0]) == 2  # two tables in the group


def test_repeated_header_on_continuation_is_dropped():
    p1 = _table([
        [("Description", 50, 200), ("Qty", 220, 270), ("Amount", 280, 350)],
        [("Widget A", 50, 200), ("1", 220, 270), ("10.00", 280, 350)],
    ], page=1)
    p2 = _table([
        [("Description", 50, 200), ("Qty", 220, 270), ("Amount", 280, 350)],
        [("Widget B", 50, 200), ("2", 220, 270), ("20.00", 280, 350)],
    ], page=2)
    doc = _Doc(tables=[p1, p2], full_text="", pages_or_sheets=2)
    rows, report = stitch_multi_page_tables(doc)
    # Both headers dropped, just the 2 data rows
    assert len(rows) == 2
    assert {r[0] for r in rows} == {"Widget A", "Widget B"}


def test_different_column_signatures_are_NOT_merged():
    """If page 2's table has different column count it's a different
    table, not a continuation — must not be stitched."""
    p1 = _table([
        [("Description", 50, 200), ("Qty", 220, 270), ("Amount", 280, 350)],
        [("Widget A", 50, 200), ("1", 220, 270), ("10.00", 280, 350)],
    ], page=1)
    p2 = _table([
        [("Other field 1", 50, 300), ("Other field 2", 320, 500)],
        [("not a line item", 50, 300), ("just text", 320, 500)],
    ], page=2)
    doc = _Doc(tables=[p1, p2], full_text="", pages_or_sheets=2)
    rows, report = stitch_multi_page_tables(doc)
    # Different signatures → no group stitch; main group = the larger one
    assert report.stitched_groups == []


def test_page_y_assertion_failure_recorded_in_report():
    p1 = _table([
        [("Description", 50, 200), ("Amount", 280, 350)],
        [("Widget", 50, 200), ("10.00", 280, 350)],
    ], page=1)
    # Marker says 5 pages but we only see 1
    doc = _Doc(tables=[p1], full_text="Page 1 of 5", pages_or_sheets=1)
    _, report = stitch_multi_page_tables(doc)
    assert report.pages_asserted == 5
    assert report.page_assertion_passed is False
    assert any("page_assertion_failed" in n for n in report.notes)


def test_xlsx_csv_documents_skip_stitching_gracefully():
    """Tables without bbox anchors (XLSX/CSV) shouldn't crash; they
    just fall through to the single-table path."""
    # Make a table with no x-coordinate anchors
    no_xy_token = _Token(text="A", anchor=object())  # bare object — no .x0
    no_xy_row = [_Region(tokens=[no_xy_token])]
    t = _Table(rows=[no_xy_row], source_anchor=None)
    doc = _Doc(tables=[t], full_text="", pages_or_sheets=1)
    rows, report = stitch_multi_page_tables(doc)
    # Doesn't crash; nothing stitched
    assert isinstance(rows, list)


def test_no_tables_returns_empty():
    doc = _Doc(tables=[], full_text="", pages_or_sheets=1)
    rows, report = stitch_multi_page_tables(doc)
    assert rows == []
    assert report.stitched_groups == []
