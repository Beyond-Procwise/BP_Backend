"""CatalogImportService. Every test drives a fake psycopg2 connection, so the suite
never needs a database; one test drives a real CSV through the real spreadsheet
parser, because a mapping that only ever sees a hand-built table proves nothing
about the cells the parser actually emits.

Spec: docs/superpowers/specs/2026-09-09-reseller-catalog-and-sell-side-design.md
"""
from __future__ import annotations

import hashlib
from decimal import Decimal

import pytest

from src.services import catalog_import
from src.services.extraction_v3.schemas.parsed_document import (
    Cell, Page, ParsedDocument, Table,
)

_ZERO = (0.0, 0.0, 0.0, 0.0)


# --- fakes ------------------------------------------------------------------

def _table(rows: list[list[str]]) -> Table:
    cells = [
        [Cell(page=0, bbox=_ZERO, text=v, row_index=r, col_index=c)
         for c, v in enumerate(row)]
        for r, row in enumerate(rows)
    ]
    return Table(page=0, bbox=_ZERO, rows=cells, header_row_index=0)


def _parsed(sheets: list[list[list[str]]]) -> ParsedDocument:
    return ParsedDocument(
        source_path="feed.csv", file_format="spreadsheet",
        pages=[Page(index=i, width=1000.0, height=1000.0, rotation=0,
                    regions=[], tables=[_table(rows)], tokens=[])
               for i, rows in enumerate(sheets)],
        full_text="", parser_backend="spreadsheet", parser_confidence=1.0,
    )


class FakeCursor:
    """Dispatches on the statement rather than replaying a fixed queue: the
    importer issues several different reads and their order is an implementation
    detail no test should be pinned to."""

    def __init__(self, conn):
        self.conn = conn

    def execute(self, sql, params=None):
        flat = " ".join(sql.split())
        self.conn.calls.append((flat, params))
        self._result = self.conn.answer(flat, params)

    def fetchone(self):
        r = self._result
        return r[0] if isinstance(r, list) and r else (r if not isinstance(r, list) else None)

    def fetchall(self):
        return self._result if isinstance(self._result, list) else []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeConn:
    def __init__(self, *, mapping, existing_source=None, unspsc=(), current=None,
                 fail_on=None):
        self.mapping = mapping
        self.existing_source = existing_source
        self.unspsc = list(unspsc)
        self.current = dict(current or {})     # distributor_sku -> current row dict
        self.fail_on = fail_on
        self.calls = []
        self.committed = False
        self.rolled_back = False
        self._next_source_id = 7001

    def answer(self, flat, params):
        if self.fail_on and self.fail_on in flat:
            raise RuntimeError("boom: simulated database failure")
        if "FROM proc.bp_catalog_source" in flat and "SELECT" in flat:
            return [self.existing_source] if self.existing_source else []
        if "FROM proc.bp_catalog_mapping" in flat:
            return list(self.mapping)
        if "FROM proc.bp_category_master" in flat:
            return [{"unspsc_code": c} for c in self.unspsc]
        if "INSERT INTO proc.bp_catalog_source" in flat:
            sid = self._next_source_id
            self._next_source_id += 1
            return {"source_id": sid}
        if "FROM proc.bp_catalog_item" in flat and "SELECT" in flat:
            sku = params[1] if params and len(params) > 1 else None
            row = self.current.get(sku)
            return [row] if row else []
        return []

    def cursor(self, *a, **k):
        return FakeCursor(self)

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _map(**overrides):
    """A minimal viable mapping profile: the three columns the schema makes NOT NULL."""
    base = [
        {"target_column": "distributor_sku", "source_header": "SKU",
         "transform": None, "is_required": True},
        {"target_column": "item_description", "source_header": "Description",
         "transform": None, "is_required": True},
        {"target_column": "currency", "source_header": "Ccy",
         "transform": None, "is_required": True},
    ]
    extra = overrides.pop("extra", [])
    return base + list(extra)


def _run(conn, sheets, **kw):
    kw.setdefault("distributor_id", "SUP-001")
    kw.setdefault("feed_name", "March list")
    kw.setdefault("mapping_profile", "ingram_v1")
    kw.setdefault("price_effective", "2026-03-01")
    kw.setdefault("imported_by", "am@example.com")
    return catalog_import.import_catalog(
        file_bytes=b"irrelevant-when-parse-is-patched",
        file_name="feed.csv",
        parsed=_parsed(sheets),
        conn=conn,
        **kw,
    )


def _inserted_items(conn):
    return [p for sql, p in conn.calls if "INSERT INTO proc.bp_catalog_item" in sql]


def _receipt_updates(conn):
    return [p for sql, p in conn.calls if "UPDATE proc.bp_catalog_source" in sql]


# --- idempotency ------------------------------------------------------------

def test_the_same_file_twice_loads_nothing_the_second_time():
    conn = FakeConn(mapping=_map(),
                    existing_source={"source_id": 42, "status": "imported"})
    res = _run(conn, [[["SKU", "Description", "Ccy"], ["A1", "Widget", "GBP"]]])

    assert res.status == "duplicate"
    assert res.source_id == 42
    assert _inserted_items(conn) == []


def test_content_hash_is_of_the_file_not_the_parse():
    payload = b"SKU,Description,Ccy\nA1,Widget,GBP\n"
    conn = FakeConn(mapping=_map())
    catalog_import.import_catalog(
        file_bytes=payload, file_name="f.csv",
        parsed=_parsed([[["SKU", "Description", "Ccy"], ["A1", "W", "GBP"]]]),
        distributor_id="SUP-001", feed_name="f", mapping_profile="ingram_v1",
        price_effective="2026-03-01", imported_by="x", conn=conn,
    )
    expected = hashlib.sha256(payload).hexdigest()
    dup_check = [p for sql, p in conn.calls
                 if "FROM proc.bp_catalog_source" in sql and "SELECT" in sql][0]
    assert expected in dup_check


# --- mapping guards ---------------------------------------------------------

def test_no_mapping_profile_fails_the_import_loudly():
    conn = FakeConn(mapping=[])
    res = _run(conn, [[["SKU", "Description", "Ccy"], ["A1", "Widget", "GBP"]]])

    assert res.status == "failed"
    assert "mapping" in (res.error or "").lower()
    assert _inserted_items(conn) == []


def test_a_sheet_missing_a_required_heading_is_skipped_not_guessed():
    conn = FakeConn(mapping=_map())
    # First sheet is a cover page; the second carries the data.
    res = _run(conn, [
        [["Distributor price file", "March 2026"]],
        [["SKU", "Description", "Ccy"], ["A1", "Widget", "GBP"]],
    ])

    assert res.rows_loaded == 1
    assert res.sheets_skipped == 1


def test_no_sheet_matching_the_mapping_is_a_failure_not_an_empty_success():
    conn = FakeConn(mapping=_map())
    res = _run(conn, [[["Part", "Name"], ["A1", "Widget"]]])

    assert res.status == "failed"
    assert res.rows_loaded == 0
    assert _receipt_updates(conn), "a failed import must still leave a receipt"


# --- absence stays absent ---------------------------------------------------

def test_an_absent_cost_column_yields_null_not_zero():
    conn = FakeConn(mapping=_map())
    _run(conn, [[["SKU", "Description", "Ccy"], ["A1", "Widget", "GBP"]]])

    (params,) = _inserted_items(conn)
    cols = catalog_import.ITEM_COLUMNS
    row = dict(zip(cols, params[: len(cols)]))
    assert row["cost_price"] is None
    assert row["list_price"] is None
    assert row["stock_qty"] is None
    assert row["lifecycle_status"] is None


def test_a_blank_cell_in_a_mapped_optional_column_is_null():
    conn = FakeConn(mapping=_map(extra=[
        {"target_column": "cost_price", "source_header": "Cost",
         "transform": None, "is_required": False}]))
    _run(conn, [[["SKU", "Description", "Ccy", "Cost"], ["A1", "Widget", "GBP", "  "]]])

    (params,) = _inserted_items(conn)
    row = dict(zip(catalog_import.ITEM_COLUMNS, params))
    assert row["cost_price"] is None


# --- row-level rejection ----------------------------------------------------

def test_a_row_with_no_sku_is_rejected_and_the_import_continues():
    conn = FakeConn(mapping=_map())
    res = _run(conn, [[["SKU", "Description", "Ccy"],
                       ["", "Orphan", "GBP"],
                       ["A2", "Widget", "GBP"]]])

    assert res.rows_loaded == 1
    assert res.rows_rejected == 1
    assert res.status == "partial"
    assert any("distributor_sku" in r.reason for r in res.rejects)


def test_a_row_with_no_currency_is_rejected_because_the_column_is_not_null():
    conn = FakeConn(mapping=_map())
    res = _run(conn, [[["SKU", "Description", "Ccy"], ["A1", "Widget", ""]]])

    assert res.rows_loaded == 0
    assert res.rows_rejected == 1
    assert any("currency" in r.reason for r in res.rejects)


def test_an_unknown_unspsc_rejects_the_row_not_the_file():
    conn = FakeConn(
        mapping=_map(extra=[{"target_column": "unspsc_code",
                             "source_header": "UNSPSC", "transform": None,
                             "is_required": False}]),
        unspsc=["43211503"],
    )
    res = _run(conn, [[["SKU", "Description", "Ccy", "UNSPSC"],
                       ["A1", "Widget", "GBP", "43211503"],
                       ["A2", "Gadget", "GBP", "99999999"]]])

    assert res.rows_loaded == 1
    assert res.rows_rejected == 1
    assert any("unspsc" in r.reason.lower() for r in res.rejects)


def test_an_unparseable_price_rejects_the_row():
    conn = FakeConn(mapping=_map(extra=[
        {"target_column": "cost_price", "source_header": "Cost",
         "transform": None, "is_required": False}]))
    res = _run(conn, [[["SKU", "Description", "Ccy", "Cost"],
                       ["A1", "Widget", "GBP", "call us"]]])

    assert res.rows_rejected == 1
    assert any("cost_price" in r.reason for r in res.rejects)


# --- transforms -------------------------------------------------------------

def test_pence_to_major_divides_by_one_hundred_exactly():
    conn = FakeConn(mapping=_map(extra=[
        {"target_column": "cost_price", "source_header": "CostPence",
         "transform": "pence_to_major", "is_required": False}]))
    _run(conn, [[["SKU", "Description", "Ccy", "CostPence"],
                 ["A1", "Widget", "GBP", "12345"]]])

    row = dict(zip(catalog_import.ITEM_COLUMNS, _inserted_items(conn)[0]))
    assert row["cost_price"] == Decimal("123.45")


def test_trim_currency_strips_symbols_and_separators():
    conn = FakeConn(mapping=_map(extra=[
        {"target_column": "list_price", "source_header": "RRP",
         "transform": "trim_currency", "is_required": False}]))
    _run(conn, [[["SKU", "Description", "Ccy", "RRP"],
                 ["A1", "Widget", "GBP", "£1,234.50"]]])

    row = dict(zip(catalog_import.ITEM_COLUMNS, _inserted_items(conn)[0]))
    assert row["list_price"] == Decimal("1234.50")


def test_pack_split_takes_the_only_number_and_refuses_ambiguity():
    """A guessed pack size reaches a margin calculation, so two numbers in the
    cell yield NULL rather than whichever one appeared first."""
    conn = FakeConn(mapping=_map(extra=[
        {"target_column": "pack_size", "source_header": "Pack",
         "transform": "pack_split", "is_required": False}]))
    _run(conn, [[["SKU", "Description", "Ccy", "Pack"],
                 ["A1", "Widget", "GBP", "Box of 10"],
                 ["A2", "Gadget", "GBP", "10 per pack"],
                 ["A3", "Doodad", "GBP", "no pack info"],
                 ["A4", "Thing", "GBP", "Box of 10 x 5"]]])

    rows = [dict(zip(catalog_import.ITEM_COLUMNS, p)) for p in _inserted_items(conn)]
    assert [r["pack_size"] for r in rows] == [
        Decimal("10"), Decimal("10"), None, None,
    ]


# --- versioning -------------------------------------------------------------

def test_a_repriced_row_closes_the_prior_version_and_opens_a_new_one():
    conn = FakeConn(
        mapping=_map(extra=[{"target_column": "cost_price", "source_header": "Cost",
                             "transform": None, "is_required": False}]),
        current={"A1": {"catalog_item_id": 900, "distributor_sku": "A1",
                        "item_description": "Widget", "currency": "GBP",
                        "cost_price": Decimal("10.00")}},
    )
    res = _run(conn, [[["SKU", "Description", "Ccy", "Cost"],
                       ["A1", "Widget", "GBP", "11.00"]]])

    closes = [p for sql, p in conn.calls
              if "UPDATE proc.bp_catalog_item" in sql and "valid_to" in sql]
    assert closes == [(900,)], "the prior version must be closed by id"
    assert len(_inserted_items(conn)) == 1
    assert res.rows_versioned == 1


def test_an_unchanged_row_creates_no_new_version():
    conn = FakeConn(
        mapping=_map(extra=[{"target_column": "cost_price", "source_header": "Cost",
                             "transform": None, "is_required": False}]),
        current={"A1": {"catalog_item_id": 900, "distributor_sku": "A1",
                        "item_description": "Widget", "currency": "GBP",
                        "cost_price": Decimal("10.00")}},
    )
    res = _run(conn, [[["SKU", "Description", "Ccy", "Cost"],
                       ["A1", "Widget", "GBP", "10.00"]]])

    assert _inserted_items(conn) == []
    assert res.rows_unchanged == 1
    assert res.rows_versioned == 0


# --- the receipt ------------------------------------------------------------

def test_a_clean_import_is_recorded_as_imported_with_its_counts():
    conn = FakeConn(mapping=_map())
    res = _run(conn, [[["SKU", "Description", "Ccy"],
                       ["A1", "Widget", "GBP"], ["A2", "Gadget", "GBP"]]])

    assert (res.status, res.rows_seen, res.rows_loaded, res.rows_rejected) == \
        ("imported", 2, 2, 0)
    (params,) = _receipt_updates(conn)
    assert "imported" in params


def test_a_database_failure_mid_import_still_leaves_a_receipt():
    """The green zero this guards against: 'we hold no catalog for this
    distributor' and 'the import died' must never look the same."""
    conn = FakeConn(mapping=_map(), fail_on="INSERT INTO proc.bp_catalog_item")
    res = _run(conn, [[["SKU", "Description", "Ccy"], ["A1", "Widget", "GBP"]]])

    assert res.status == "failed"
    assert res.error and "boom" in res.error
    assert _receipt_updates(conn), "the receipt is the whole point"
    assert conn.rolled_back


# --- through the real parser ------------------------------------------------

def test_a_real_csv_through_the_real_parser_loads(tmp_path):
    csv = tmp_path / "ingram.csv"
    csv.write_text(
        "SKU,Description,Ccy,Cost,RRP,Lifecycle\n"
        "IN-1001,Cisco Catalyst 9200 24-port,GBP,1420.00,1899.00,active\n"
        "IN-1002,Cisco Catalyst 2960X 24-port,GBP,610.00,899.00,end_of_sale\n"
    )
    from src.services.extraction_v3.parsers.router import parse

    conn = FakeConn(mapping=_map(extra=[
        {"target_column": "cost_price", "source_header": "Cost",
         "transform": None, "is_required": False},
        {"target_column": "list_price", "source_header": "RRP",
         "transform": None, "is_required": False},
        {"target_column": "lifecycle_status", "source_header": "Lifecycle",
         "transform": None, "is_required": False},
    ]))
    res = catalog_import.import_catalog(
        file_bytes=csv.read_bytes(), file_name="ingram.csv",
        parsed=parse(str(csv)),
        distributor_id="SUP-001", feed_name="March list",
        mapping_profile="ingram_v1", price_effective="2026-03-01",
        imported_by="am@example.com", conn=conn,
    )

    assert (res.status, res.rows_seen, res.rows_loaded) == ("imported", 2, 2)
    rows = [dict(zip(catalog_import.ITEM_COLUMNS, p)) for p in _inserted_items(conn)]
    assert rows[0]["cost_price"] == Decimal("1420.00")
    assert rows[1]["lifecycle_status"] == "end_of_sale"
