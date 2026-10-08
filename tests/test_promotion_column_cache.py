"""Per-connection cache for promotion's column lookups.

``promotion._stg_columns`` and ``promotion._table_columns`` queried
information_schema.columns on every call. ``promote()`` calls ``_stg_columns``
twice per document (header table, then line-item table), so a run over N
documents is 2N identical catalogue queries against a schema that cannot
change mid-run. ``linking_engine._table_columns`` already caches this for the
life of the connection (see tests/test_linking_engine_cache.py); these tests
pin the same behaviour onto promotion's two helpers.

The cache must be keyed by connection object, not table name alone: this
codebase talks to several databases (bp_sqldb, bp_testdb, uicanvas_test) and a
table-name-only cache would hand one database's column list to another.
"""
from __future__ import annotations

import pytest

from src.services.extraction import promotion


class _Conn:
    """Weak-referenceable stand-in for a psycopg2 connection."""


class _FakeCursor:
    def __init__(self, columns, connection=None):
        self._columns = list(columns)
        self.execute_count = 0
        if connection is not None:
            self.connection = connection

    def execute(self, sql, params=()):
        self.execute_count += 1

    def fetchall(self):
        return [(c,) for c in self._columns]


@pytest.mark.parametrize("helper", ["_stg_columns", "_table_columns"])
def test_second_lookup_on_same_connection_does_not_query(helper):
    conn = _Conn()
    cur = _FakeCursor(["invoice_id", "total"], connection=conn)
    lookup = getattr(promotion, helper)

    first = lookup(cur, "proc.bp_invoice_stg")
    second = lookup(cur, "proc.bp_invoice_stg")

    assert first == ["invoice_id", "total"]
    assert second == ["invoice_id", "total"]
    assert cur.execute_count == 1


@pytest.mark.parametrize("helper", ["_stg_columns", "_table_columns"])
def test_different_tables_are_cached_separately(helper):
    conn = _Conn()
    lookup = getattr(promotion, helper)
    header = _FakeCursor(["invoice_id"], connection=conn)
    lines = _FakeCursor(["line_id", "invoice_id"], connection=conn)

    assert lookup(header, "proc.bp_invoice_stg") == ["invoice_id"]
    assert lookup(lines, "proc.bp_invoice_line_items_stg") == ["line_id", "invoice_id"]
    assert header.execute_count == 1
    assert lines.execute_count == 1


def test_cache_is_per_connection_not_per_table_name():
    live = _FakeCursor(["a", "b"], connection=_Conn())
    test_db = _FakeCursor(["a", "b", "c"], connection=_Conn())

    assert promotion._stg_columns(live, "proc.t") == ["a", "b"]
    assert promotion._stg_columns(test_db, "proc.t") == ["a", "b", "c"]
    assert live.execute_count == 1
    assert test_db.execute_count == 1


def test_cursor_without_connection_queries_every_time():
    """Fake cursors elsewhere in the suite have no .connection; the helper must
    degrade to querying rather than raise or share a global cache."""
    cur = _FakeCursor(["x"])

    promotion._stg_columns(cur, "proc.t")
    promotion._stg_columns(cur, "proc.t")

    assert cur.execute_count == 2


def test_returned_list_is_a_copy_so_callers_cannot_poison_the_cache():
    cur = _FakeCursor(["a", "b"], connection=_Conn())

    first = promotion._table_columns(cur, "proc.t")
    first.append("injected")

    assert promotion._table_columns(cur, "proc.t") == ["a", "b"]
