"""Tests for the per-connection column-list cache in linking_engine._table_columns.

Context: linking_engine._table_columns queried information_schema.columns on
every call with no caching. deal_assignment_service._persist_deal calls it in
a loop over four tables per document, which on a realistic corpus (~5,000
deals) becomes ~88,000 identical queries in one long transaction -- long
enough that the connection drops before the run completes (see
docs/issues/2026-07-27-deal-assignment-does-not-scale.md). The schema cannot
change mid-run, so the fix is to cache the lookup for the life of the
connection.

The cache MUST be keyed by connection object, not by table name alone,
because this codebase talks to multiple databases (live bp_sqldb, and test
databases bp_testdb / uicanvas_test) -- a table-name-only cache would leak
one database's column list into a query against a different database.
"""
from __future__ import annotations

import src.services.linking_engine as linking_engine


class _FakeCursor:
    """Minimal cursor: records execute() calls, returns canned column rows."""

    def __init__(self, columns, connection=None):
        self._columns = columns
        self.execute_count = 0
        self.connection = connection

    def execute(self, sql, params=()):
        self.execute_count += 1
        self._last_params = params

    def fetchall(self):
        return [(c,) for c in self._columns]


class _NoConnectionCursor:
    """Mirrors the fake cursors used elsewhere in the test suite (e.g.
    test_promotion_hitl_security.py) that have no `.connection` attribute
    at all -- _table_columns must fall back to querying every time, not
    raise AttributeError."""

    def __init__(self, columns):
        self._columns = columns
        self.execute_count = 0

    def execute(self, sql, params=()):
        self.execute_count += 1

    def fetchall(self):
        return [(c,) for c in self._columns]


class _Connection:
    """A plain, weak-referenceable stand-in for a psycopg2 connection."""
    pass


def setup_function(_fn):
    # The cache is module-level and keyed by connection object; since each
    # test builds fresh _Connection() instances, stale entries from other
    # tests can't collide, but clear it anyway for isolation/determinism.
    linking_engine._TABLE_COLUMNS_CACHE.clear()


def test_second_call_same_connection_does_not_requery():
    conn = _Connection()
    cur = _FakeCursor(["a", "b", "c"], connection=conn)

    cols1 = linking_engine._table_columns(cur, "proc.bp_purchase_order_stg")
    cols2 = linking_engine._table_columns(cur, "proc.bp_purchase_order_stg")

    assert cols1 == ["a", "b", "c"]
    assert cols2 == ["a", "b", "c"]
    assert cur.execute_count == 1  # only the first call hit the database


def test_different_connections_do_not_share_cache():
    """The correctness property that matters most: a cache keyed by table
    name alone would serve one database's columns to a query against a
    different database. Two distinct connections must each query and cache
    independently, even for the identical schema_table string."""
    conn_a = _Connection()
    conn_b = _Connection()
    cur_a = _FakeCursor(["a1", "a2"], connection=conn_a)
    cur_b = _FakeCursor(["b1", "b2", "b3"], connection=conn_b)

    cols_a = linking_engine._table_columns(cur_a, "proc.bp_purchase_order_stg")
    cols_b = linking_engine._table_columns(cur_b, "proc.bp_purchase_order_stg")

    assert cols_a == ["a1", "a2"]
    assert cols_b == ["b1", "b2", "b3"]
    assert cur_a.execute_count == 1
    assert cur_b.execute_count == 1

    # Second round: each connection still serves its own cached answer.
    assert linking_engine._table_columns(cur_a, "proc.bp_purchase_order_stg") == ["a1", "a2"]
    assert linking_engine._table_columns(cur_b, "proc.bp_purchase_order_stg") == ["b1", "b2", "b3"]
    assert cur_a.execute_count == 1
    assert cur_b.execute_count == 1


def test_cursor_without_connection_attribute_falls_back_to_querying():
    cur = _NoConnectionCursor(["x", "y"])

    cols1 = linking_engine._table_columns(cur, "proc.bp_purchase_order_stg")
    cols2 = linking_engine._table_columns(cur, "proc.bp_purchase_order_stg")

    assert cols1 == ["x", "y"]
    assert cols2 == ["x", "y"]
    assert cur.execute_count == 2  # no cache available -> queried both times


def test_mutating_returned_list_does_not_corrupt_cache():
    conn = _Connection()
    cur = _FakeCursor(["a", "b"], connection=conn)

    cols1 = linking_engine._table_columns(cur, "proc.bp_purchase_order_stg")
    cols1.append("INJECTED")
    cols1.clear()

    cols2 = linking_engine._table_columns(cur, "proc.bp_purchase_order_stg")

    assert cols2 == ["a", "b"]
    assert cur.execute_count == 1  # still served from cache, not re-queried
