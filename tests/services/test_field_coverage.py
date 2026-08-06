"""Coverage must be measured from the extraction record, never from column NULL
counts — seeded data makes a never-extracted column look fully populated."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.field_coverage import field_coverage  # noqa: E402


class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows
        self.description = [("doc_type",), ("field_path",),
                            ("documents_with_field",), ("documents_total",)]
        self.executed = None

    def execute(self, sql, params=None):
        self.executed = (sql, params)

    def fetchall(self):
        return self._rows


class _FakeConn:
    def __init__(self, rows):
        self._cur = _FakeCursor(rows)

    def cursor(self):
        return self._cur


def test_coverage_pct_is_computed_not_read():
    conn = _FakeConn([("quote", "line_items[].unit_of_measure", 3, 120)])
    out = field_coverage(conn)
    assert out[0]["coverage_pct"] == 2.5


def test_zero_total_does_not_divide_by_zero():
    conn = _FakeConn([("contract", "contract_id", 0, 0)])
    out = field_coverage(conn)
    assert out[0]["coverage_pct"] is None


def test_query_reads_the_provenance_table_not_the_trgt_tables():
    conn = _FakeConn([])
    field_coverage(conn)
    sql = conn.cursor().executed[0].lower()
    assert "bp_extraction_provenance_v3" in sql
    assert "_trgt" not in sql, "coverage must not be inferred from seeded target columns"
