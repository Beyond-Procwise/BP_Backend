import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import agents.extraction_engine as ee_module


class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows
        self.executed_sql = None

    def execute(self, sql, *args, **kwargs):
        self.executed_sql = sql

    def fetchall(self):
        return self._rows

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeConn:
    def __init__(self, rows):
        self._cursor = _FakeCursor(rows)

    def cursor(self):
        return self._cursor

    def close(self):
        pass


def _reset_cache(monkeypatch):
    monkeypatch.setattr(ee_module, "_SUPPLIER_CACHE", None)
    monkeypatch.setattr(ee_module, "_SUPPLIER_CACHE_TIME", 0.0)


def test_load_suppliers_queries_bp_supplier_not_the_nonexistent_proc_supplier(monkeypatch):
    """proc.supplier has never existed; the real table is proc.bp_supplier.

    The old query (SELECT ... FROM proc.supplier) fails on every call with
    'relation "proc.supplier" does not exist', so supplier enrichment was
    silently and permanently degraded to an empty lookup.
    """
    _reset_cache(monkeypatch)

    rows = [("Acme Ltd", "Acme"), ("Beta Supplies", None)]
    fake_conn = _FakeConn(rows)
    monkeypatch.setattr(ee_module, "_db_connection_func", lambda: fake_conn)

    result = ee_module._load_suppliers_from_db()

    executed_sql = fake_conn._cursor.executed_sql
    assert executed_sql is not None
    assert "proc.bp_supplier" in executed_sql
    assert "proc.supplier " not in executed_sql
    assert "proc.supplier\n" not in executed_sql

    # And it must actually have loaded the suppliers, not silently degraded.
    assert "Acme Ltd" in result["suppliers"]
    assert "Beta Supplies" in result["suppliers"]
