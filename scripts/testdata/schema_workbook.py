"""Schema coverage workbook: every live table and column, and what fills it.

Reads the live schema at build time, so the workbook cannot drift from the
database it describes. Read-only against live.

    .venv/bin/python -m scripts.testdata.schema_workbook

Writes docs/testdata/BP_Schema_Coverage.xlsx.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from scripts.testdata.coverage import BACKUP, OUTPUT as BUCKET_OUTPUT, SEED, classify
from scripts.testdata.db import connect
from scripts.testdata.reference import REFERENCE_TABLES
from scripts.testdata.suppliers import SUPPLIER_COLUMNS

BUCKET_MEANING = {
    SEED: "Fill — business data the seeder synthesises",
    "REFERENCE": "Fill — configuration copied verbatim from live",
    BUCKET_OUTPUT: "Leave empty — the product derives this",
    BACKUP: "Leave empty — historical or dated copy",
    "UNCLEAR": "Undecided — must be classified",
}

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = ROOT / "docs" / "testdata" / "BP_Schema_Coverage.xlsx"

PAIRS = (("bp_sqldb", "bp_testdb"), ("uicanvas", "uicanvas_test"))

HEADER_FILL = PatternFill("solid", fgColor="1F3864")
HEADER_FONT = Font(color="FFFFFF", bold=True)
GROUP_FONT = Font(bold=True)

# How each table is filled. Anything absent is "Not covered".
STATUS_GENERATED = "Generated — full row set"
STATUS_VERBATIM = "Copied verbatim from live"
STATUS_PLANNED = "Planned — persistence spec 2026-07-27"
STATUS_NONE = "Not covered"

GENERATED: dict[str, dict[str, str]] = {
    "bp_sqldb": {
        "bp_supplier": "5,000 suppliers, all 51 columns",
        "bp_supplier_id_crosswalk": "5,000 rows joining both ID conventions",
    },
    "uicanvas": {
        "supplier": "5,000 suppliers, all 51 columns (SI###### convention)",
    },
}

PLANNED: dict[str, dict[str, str]] = {
    "bp_sqldb": {
        "bp_quote_trgt": "~21,020 quotes",
        "bp_purchase_order_trgt": "~5,037 purchase orders",
        "bp_invoice_trgt": "~12,398 invoices",
        "bp_quote_line_items_trgt": "~115,000 quote lines",
        "bp_po_line_items_trgt": "~28,000 PO lines",
        "bp_invoice_line_items_trgt": "~50,000 invoice lines",
        "bp_requirement": "6,000 requirements",
    },
    "uicanvas": {
        "business_unit": "400 business units (6/40/120/240/400 tree)",
        "cost_centre": "500 cost centres across 6 entities",
        "item": "5,000 catalogue items on the real L5 leaves",
    },
}


def _status(source_db: str, table: str) -> tuple[str, str]:
    if table in GENERATED.get(source_db, {}):
        return STATUS_GENERATED, GENERATED[source_db][table]
    if table in REFERENCE_TABLES.get(source_db, ()):
        return STATUS_VERBATIM, "Every live row copied unchanged"
    if table in PLANNED.get(source_db, {}):
        return STATUS_PLANNED, PLANNED[source_db][table]
    return STATUS_NONE, ""


def _fetch(conn, sql: str, args: tuple = ()) -> list[tuple]:
    with conn.cursor() as cur:
        cur.execute(sql, args)
        return cur.fetchall()


def _tables(conn) -> list[str]:
    return [
        row[0]
        for row in _fetch(
            conn,
            "select table_name from information_schema.tables "
            "where table_schema = 'proc' and table_type = 'BASE TABLE' "
            "order by table_name",
        )
    ]


def _columns(conn) -> dict[str, list[tuple]]:
    rows = _fetch(
        conn,
        "select table_name, ordinal_position, column_name, data_type, "
        "       character_maximum_length, is_nullable, column_default "
        "from information_schema.columns where table_schema = 'proc' "
        "order by table_name, ordinal_position",
    )
    by_table: dict[str, list[tuple]] = {}
    for row in rows:
        by_table.setdefault(row[0], []).append(row[1:])
    return by_table


def _row_counts(conn, tables: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for table in tables:
        try:
            counts[table] = _fetch(conn, f'select count(*) from proc."{table}"')[0][0]
        except Exception:
            conn.rollback()
            counts[table] = -1
    return counts


def _write_header(sheet, headings: list[str]) -> None:
    sheet.append(headings)
    for cell in sheet[1]:
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(vertical="center", wrap_text=True)
    sheet.freeze_panes = "A2"


def _autosize(sheet, widths: dict[int, int]) -> None:
    for index, width in widths.items():
        sheet.column_dimensions[get_column_letter(index)].width = width


def _type_label(data_type: str, length: Any) -> str:
    if length:
        return f"{data_type}({length})"
    return data_type


def build() -> Path:
    book = Workbook()
    book.remove(book.active)

    summary = book.create_sheet("Summary")
    coverage = book.create_sheet("Coverage by Table")
    schema = book.create_sheet("All Columns")

    _write_header(
        coverage,
        ["Database", "Table", "Columns", "Live rows", "Test-DB rows", "Plan", "Why", "Status", "What fills it"],
    )
    _write_header(
        schema,
        ["Database", "Table", "#", "Column", "Data type", "Nullable", "Default", "Plan", "Status"],
    )

    totals: list[tuple] = []
    focus: dict[str, list[tuple]] = {}

    for source_db, target_db in PAIRS:
        live = connect(source_db)
        try:
            tables = _tables(live)
            columns = _columns(live)
            live_counts = _row_counts(live, tables)
        finally:
            live.close()

        try:
            target = connect(target_db)
            try:
                target_counts = _row_counts(target, tables)
            finally:
                target.close()
        except Exception:
            target_counts = {}

        covered = 0
        for table in tables:
            status, note = _status(source_db, table)
            if status != STATUS_NONE:
                covered += 1
            cols = columns.get(table, [])
            verdict = classify(table)
            coverage.append(
                [
                    source_db,
                    table,
                    len(cols),
                    live_counts.get(table, -1),
                    target_counts.get(table, -1),
                    BUCKET_MEANING.get(verdict.bucket, verdict.bucket),
                    verdict.reason,
                    status,
                    note,
                ]
            )
            for ordinal, name, data_type, length, nullable, default in cols:
                schema.append(
                    [
                        source_db,
                        table,
                        ordinal,
                        name,
                        _type_label(data_type, length),
                        "yes" if nullable == "YES" else "NO",
                        (default or "")[:60],
                        BUCKET_MEANING.get(classify(table).bucket, ""),
                        status,
                    ]
                )

        totals.append(
            (
                source_db,
                target_db,
                len(tables),
                sum(len(columns.get(t, [])) for t in tables),
                covered,
                len(tables) - covered,
                sum(v for v in live_counts.values() if v > 0),
            )
        )
        focus[source_db] = [
            (t, columns.get(t, []), live_counts.get(t, -1), target_counts.get(t, -1))
            for t in tables
        ]

    _autosize(coverage, {1: 14, 2: 38, 3: 9, 4: 12, 5: 13, 6: 44, 7: 40, 8: 32, 9: 44})
    _autosize(schema, {1: 14, 2: 38, 3: 5, 4: 34, 5: 24, 6: 9, 7: 30, 8: 44, 9: 32})

    # --- Summary -----------------------------------------------------------
    _write_header(
        summary,
        ["Live database", "Test database", "Tables", "Columns", "Tables filled",
         "Tables empty", "Live rows"],
    )
    for row in totals:
        summary.append(list(row))
    _autosize(summary, {1: 16, 2: 16, 3: 9, 4: 10, 5: 14, 6: 13, 7: 12})

    summary.append([])
    summary.append(["Status key"])
    summary.cell(summary.max_row, 1).font = GROUP_FONT
    for status, meaning in (
        (STATUS_GENERATED, "Synthesised by the seeder; every column populated"),
        (STATUS_VERBATIM, "Every live row copied unchanged, so behaviour matches production config"),
        (STATUS_PLANNED, "Designed and specified; not yet loaded"),
        (STATUS_NONE, "Empty in the test databases — see note below"),
    ):
        summary.append([status, meaning])

    summary.append([])
    summary.append(["Why most tables are intentionally empty"])
    summary.cell(summary.max_row, 1).font = GROUP_FONT
    for line in (
        "The seeder writes inputs, never outputs.",
        "Rankings, evaluations, decisions, summaries, opportunities, findings and deal_id are",
        "produced by the product's own agents. Anything the seeder writes there, no test can",
        "afterwards prove — the test would only check that our labels match themselves.",
        "Backup tables (_bkp, _june12, _old) are historical copies and are deliberately left empty.",
    ):
        summary.append([line])

    # --- Focus tabs the question was about ---------------------------------
    _focus_tab(
        book, "Supplier Master",
        [("bp_sqldb", "bp_supplier"), ("uicanvas", "supplier"),
         ("bp_sqldb", "bp_supplier_alias"), ("bp_sqldb", "bp_supplier_ranking"),
         ("bp_sqldb", "bp_supplier_enrichment"), ("bp_sqldb", "bp_supplier_review"),
         ("bp_sqldb", "bp_tprm_supplier")],
        focus,
    )
    _focus_tab(
        book, "Category Data",
        [("uicanvas", "bp_category"), ("uicanvas", "category"),
         ("uicanvas", "category_mapping"), ("uicanvas", "category_denorm"),
         ("uicanvas", "bp_category_product_mapping"), ("bp_sqldb", "bp_category")],
        focus,
    )
    _focus_tab(
        book, "Cost Centre and Org",
        [("uicanvas", "cost_centre"), ("uicanvas", "business_unit"),
         ("uicanvas", "item"), ("uicanvas", "bp_products")],
        focus,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    book.save(OUTPUT_PATH)
    return OUTPUT_PATH


def _focus_tab(book: Workbook, title: str, wanted: list[tuple[str, str]], focus: dict) -> None:
    """One tab per area the reviewer asked about: every column, spelled out."""
    sheet = book.create_sheet(title)
    _write_header(
        sheet,
        ["Database", "Table", "#", "Column", "Data type", "Nullable", "Status", "What fills it"],
    )
    for source_db, table in wanted:
        entries = {name: (cols, live, test) for name, cols, live, test in focus.get(source_db, [])}
        if table not in entries:
            continue
        cols, live_rows, test_rows = entries[table]
        status, note = _status(source_db, table)

        sheet.append([f"{source_db}.proc.{table}",
                      f"{len(cols)} columns", f"live rows: {live_rows}",
                      f"test rows: {test_rows}", "", "", status, note])
        for cell in sheet[sheet.max_row]:
            cell.font = GROUP_FONT

        for ordinal, name, data_type, length, nullable, _default in cols:
            populated = ""
            if status == STATUS_GENERATED and table in ("bp_supplier", "supplier"):
                populated = "populated" if name in SUPPLIER_COLUMNS else "not populated"
            elif status == STATUS_VERBATIM:
                populated = "copied verbatim"
            elif status == STATUS_PLANNED:
                populated = "planned"
            sheet.append([
                source_db, table, ordinal, name,
                _type_label(data_type, length),
                "yes" if nullable == "YES" else "NO",
                status, populated,
            ])
        sheet.append([])
    _autosize(sheet, {1: 22, 2: 30, 3: 5, 4: 34, 5: 24, 6: 9, 7: 32, 8: 24})


if __name__ == "__main__":
    path = build()
    print(f"wrote {path}")
