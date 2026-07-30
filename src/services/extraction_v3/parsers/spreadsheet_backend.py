"""Spreadsheet parser backend — .xlsx / .xls / .csv → ParsedDocument.

Spreadsheets carry no layout/OCR uncertainty, so parsing is deterministic: each
sheet becomes a Page with one Table (real cell grid, for the L2 table_extractor)
and the whole workbook is rendered as GitHub-flavoured markdown pipe tables in
``full_text`` (what the context_layer LLM and the md-pipe line-item fallback read).
"""
from __future__ import annotations

import csv as _csv
import logging
import re
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


class _FormulaEvaluator:
    """Fill values for formula cells that carry no cached result.

    Workbooks generated programmatically (never opened in Excel) store
    formulas without cached values, so a data_only load reads their totals
    as blank. This evaluates the small, safe subset that covers financial
    documents — SUM over ranges/args, + - * / on cell refs and numeric
    constants, parentheses, chained references — and yields None for
    anything else (never the formula text, never a guessed number).
    """

    _MAX_EXPR_LEN = 300
    _MAX_RANGE_CELLS = 20000
    _TOKEN_RE = re.compile(
        r"\s*(?:(\d+\.\d+|\d+)"                     # number
        r"|(SUM)\s*\("                              # SUM(
        r"|(\$?[A-Za-z]{1,3}\$?\d+\s*:\s*\$?[A-Za-z]{1,3}\$?\d+)"  # range
        r"|(\$?[A-Za-z]{1,3}\$?\d+)"                # cell ref
        r"|([()+\-*/,]))",                          # operators
        re.IGNORECASE,
    )

    def __init__(self, vals: list[list], forms: list[list]):
        self._vals = vals
        self._forms = forms
        self._memo: dict = {}
        self._busy: set = set()

    def value(self, row: int, col: int):
        key = (row, col)
        if key in self._memo:
            return self._memo[key]
        if key in self._busy:
            return None  # circular reference
        v = None
        if 0 <= row < len(self._vals) and 0 <= col < len(self._vals[row]):
            v = self._vals[row][col]
        if v is None:
            f = None
            if 0 <= row < len(self._forms) and 0 <= col < len(self._forms[row]):
                f = self._forms[row][col]
            if isinstance(f, str) and f.startswith("="):
                self._busy.add(key)
                try:
                    v = self._eval_expr(f[1:])
                except Exception:
                    v = None
                finally:
                    self._busy.discard(key)
                if isinstance(v, float):
                    v = round(v, 10)
        self._memo[key] = v
        return v

    # -- expression evaluation (recursive descent, whitelisted grammar) --

    def _eval_expr(self, expr: str):
        if len(expr) > self._MAX_EXPR_LEN or "!" in expr:
            raise ValueError("unsupported formula")
        self._tokens = self._tokenize(expr)
        self._pos = 0
        result = self._parse_sum_of_terms()
        if self._pos != len(self._tokens):
            raise ValueError("trailing tokens")
        if result is None:
            raise ValueError("unresolved operand")
        return result

    def _tokenize(self, expr: str) -> list:
        tokens, pos = [], 0
        while pos < len(expr):
            m = self._TOKEN_RE.match(expr, pos)
            if not m or (not m.group(0).strip() and m.end() == pos):
                if expr[pos:].strip() == "":
                    break
                raise ValueError(f"bad token at {pos}")
            num, sum_kw, rng, ref, op = m.groups()
            if num:
                tokens.append(("num", float(num) if "." in num else int(num)))
            elif sum_kw:
                tokens.append(("sum", None))
            elif rng:
                tokens.append(("range", rng.replace("$", "").replace(" ", "")))
            elif ref:
                tokens.append(("ref", ref.replace("$", "")))
            elif op:
                tokens.append(("op", op))
            pos = m.end()
        return tokens

    def _peek(self):
        return self._tokens[self._pos] if self._pos < len(self._tokens) else (None, None)

    def _next(self):
        tok = self._peek()
        self._pos += 1
        return tok

    def _parse_sum_of_terms(self):
        left = self._parse_term()
        while self._peek() == ("op", "+") or self._peek() == ("op", "-"):
            _, op = self._next()
            right = self._parse_term()
            if left is None or right is None:
                raise ValueError("unresolved operand")
            left = left + right if op == "+" else left - right
        return left

    def _parse_term(self):
        left = self._parse_factor()
        while self._peek() == ("op", "*") or self._peek() == ("op", "/"):
            _, op = self._next()
            right = self._parse_factor()
            if left is None or right is None:
                raise ValueError("unresolved operand")
            left = left * right if op == "*" else left / right
        return left

    def _parse_factor(self):
        kind, val = self._next()
        if kind == "num":
            return val
        if kind == "op" and val == "-":
            inner = self._parse_factor()
            return None if inner is None else -inner
        if kind == "op" and val == "(":
            inner = self._parse_sum_of_terms()
            if self._next() != ("op", ")"):
                raise ValueError("unbalanced parens")
            return inner
        if kind == "ref":
            return self._ref_value(val)
        if kind == "sum":
            return self._parse_sum_call()
        raise ValueError(f"unexpected token {kind}")

    def _parse_sum_call(self):
        # SUM( was consumed; args are ranges/exprs separated by commas.
        total: float | int = 0
        while True:
            kind, val = self._peek()
            if kind == "range":
                self._next()
                total += self._range_sum(val)
            else:
                part = self._parse_sum_of_terms()
                total += part if part is not None else 0
            kind, val = self._next()
            if (kind, val) == ("op", ")"):
                return total
            if (kind, val) != ("op", ","):
                raise ValueError("bad SUM args")

    def _ref_value(self, ref: str):
        row, col = self._ref_to_rc(ref)
        v = self.value(row, col)
        return v if isinstance(v, (int, float)) else None

    def _range_sum(self, rng: str):
        start, end = rng.split(":")
        r1, c1 = self._ref_to_rc(start)
        r2, c2 = self._ref_to_rc(end)
        r1, r2 = sorted((r1, r2))
        c1, c2 = sorted((c1, c2))
        if (r2 - r1 + 1) * (c2 - c1 + 1) > self._MAX_RANGE_CELLS:
            raise ValueError("range too large")
        total: float | int = 0
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                v = self.value(r, c)
                if isinstance(v, (int, float)):
                    total += v
        return total

    @staticmethod
    def _ref_to_rc(ref: str) -> tuple[int, int]:
        from openpyxl.utils import column_index_from_string
        m = re.fullmatch(r"([A-Za-z]{1,3})(\d+)", ref)
        if not m:
            raise ValueError(f"bad ref {ref}")
        return int(m.group(2)) - 1, column_index_from_string(m.group(1).upper()) - 1


def _fill_formula_gaps(vals: list[list], forms: list[list]) -> list[list]:
    """Where a cell's cached value is None but a formula exists, evaluate it.

    Cached values always win; evaluation only fills gaps, so workbooks with
    cached results parse byte-identically to before.
    """
    has_gap = any(
        v is None and isinstance(f, str) and f.startswith("=")
        for vrow, frow in zip(vals, forms)
        for v, f in zip(vrow, frow)
    )
    if not has_gap:
        return vals
    ev = _FormulaEvaluator(vals, forms)
    return [
        [ev.value(r, c) for c in range(len(vrow))]
        for r, vrow in enumerate(vals)
    ]


def parse_xlsx(path: Path | str) -> ParsedDocument:
    """Parse an .xlsx/.xls workbook (all sheets) into a ParsedDocument."""
    import openpyxl
    p = Path(path)
    wb = openpyxl.load_workbook(p, read_only=True, data_only=True)
    raw_sheets: list[tuple[str, list[list]]] = []
    try:
        for ws in wb.worksheets:
            raw_sheets.append(
                (ws.title, [list(row) for row in ws.iter_rows(values_only=True)]))
    finally:
        wb.close()

    # Second pass for formula source, only consulted where values are absent.
    formula_sheets: dict[str, list[list]] = {}
    try:
        wb_f = openpyxl.load_workbook(p, read_only=True, data_only=False)
        try:
            for ws in wb_f.worksheets:
                formula_sheets[ws.title] = [
                    list(row) for row in ws.iter_rows(values_only=True)]
        finally:
            wb_f.close()
    except Exception:  # pragma: no cover - formula pass is best-effort
        log.warning("formula pass failed for %s; using cached values only", p)

    sheets: list[tuple[str, list[list[str]]]] = []
    for title, vals in raw_sheets:
        forms = formula_sheets.get(title)
        if forms:
            vals = _fill_formula_gaps(vals, forms)
        sheets.append((title, [[_cell_text(v) for v in row] for row in vals]))
    if not sheets:
        sheets = [("Sheet1", [])]
    return _sheets_to_parsed(str(p), "openpyxl", sheets)


def parse_csv(path: Path | str) -> ParsedDocument:
    """Parse a .csv file into a single-sheet ParsedDocument."""
    p = Path(path)
    with open(p, newline="", encoding="utf-8-sig", errors="replace") as fh:
        rows = [[_cell_text(c) for c in row] for row in _csv.reader(fh)]
    return _sheets_to_parsed(str(p), "csv-reader", [(p.stem, rows)])
