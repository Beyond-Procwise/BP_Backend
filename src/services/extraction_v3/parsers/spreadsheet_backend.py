"""Spreadsheet parser backend — .xlsx / .xls / .csv → ParsedDocument.

Spreadsheets carry no layout/OCR uncertainty, so parsing is deterministic: each
sheet becomes a Page with one Table (real cell grid, for the L2 table_extractor)
and the whole workbook is rendered as GitHub-flavoured markdown pipe tables in
``full_text`` (what the context_layer LLM and the md-pipe line-item fallback read).
"""
from __future__ import annotations

import csv as _csv
import logging
from pathlib import Path

from src.services.extraction_v3.schemas.parsed_document import (
    Cell,
    Page,
    ParsedDocument,
    Table,
)

log = logging.getLogger(__name__)

_ZERO_BBOX = (0.0, 0.0, 0.0, 0.0)


def _cell_text(value) -> str:
    if value is None:
        return ""
    # Render whole-number floats as ints (12000.0 -> "12000") so values match
    # how they read in the source and stay clean for grounding.
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _rows_to_markdown(sheet_name: str, rows: list[list[str]]) -> str:
    """Render a sheet's rows as a markdown pipe table under a heading."""
    non_empty = [r for r in rows if any(c.strip() for c in r)]
    if not non_empty:
        return f"## Sheet: {sheet_name}\n\n(empty)\n"
    width = max(len(r) for r in non_empty)
    norm = [r + [""] * (width - len(r)) for r in non_empty]
    header, *body = norm
    lines = [f"## Sheet: {sheet_name}", ""]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * width) + " |")
    for r in body:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines) + "\n"


def _build_page(index: int, rows: list[list[str]]) -> Page:
    non_empty = [r for r in rows if any(c.strip() for c in r)]
    table_cells: list[list[Cell]] = []
    for ri, row in enumerate(non_empty):
        table_cells.append([
            Cell(page=index, bbox=_ZERO_BBOX, text=c, row_index=ri, col_index=ci)
            for ci, c in enumerate(row)
        ])
    tables = []
    if table_cells:
        tables.append(Table(
            page=index, bbox=_ZERO_BBOX, rows=table_cells,
            header_row_index=0,
        ))
    return Page(index=index, width=1000.0, height=1000.0, rotation=0,
                regions=[], tables=tables, tokens=[])


def _sheets_to_parsed(source_path: str, backend: str,
                      sheets: list[tuple[str, list[list[str]]]]) -> ParsedDocument:
    pages = [_build_page(i, rows) for i, (_, rows) in enumerate(sheets)]
    full_text = "\n\n".join(_rows_to_markdown(name, rows) for name, rows in sheets).strip()
    return ParsedDocument(
        source_path=source_path,
        file_format="spreadsheet",
        pages=pages,
        full_text=full_text,
        parser_backend=backend,
        parser_confidence=1.0,
    )


def parse_xlsx(path: Path | str) -> ParsedDocument:
    """Parse an .xlsx/.xls workbook (all sheets) into a ParsedDocument."""
    import openpyxl
    p = Path(path)
    wb = openpyxl.load_workbook(p, read_only=True, data_only=True)
    sheets: list[tuple[str, list[list[str]]]] = []
    try:
        for ws in wb.worksheets:
            rows = [[_cell_text(v) for v in row] for row in ws.iter_rows(values_only=True)]
            sheets.append((ws.title, rows))
    finally:
        wb.close()
    if not sheets:
        sheets = [("Sheet1", [])]
    return _sheets_to_parsed(str(p), "openpyxl", sheets)


def parse_csv(path: Path | str) -> ParsedDocument:
    """Parse a .csv file into a single-sheet ParsedDocument."""
    p = Path(path)
    with open(p, newline="", encoding="utf-8-sig", errors="replace") as fh:
        rows = [[_cell_text(c) for c in row] for row in _csv.reader(fh)]
    return _sheets_to_parsed(str(p), "csv-reader", [(p.stem, rows)])
