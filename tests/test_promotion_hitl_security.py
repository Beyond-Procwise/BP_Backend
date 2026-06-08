"""Security tests for the HITL allowlist guard in apply_hitl_fixes_and_promote.

The guard (added to fix a SQL-injection vulnerability) must:
  - reject any field_name that is NOT a real column of the target _raw table,
  - allow field_names that ARE real columns of the target _raw table,
  - never execute an UPDATE containing a malicious multi-part field_name.

Fake cursor style mirrors test_summary_agent.py: substring-matched canned
responses + a recorder list.
"""
from __future__ import annotations

import unittest.mock as mock
import src.services.extraction.promotion as promotion


# ---------------------------------------------------------------------------
# Fake infrastructure — same pattern as test_summary_agent.py
# ---------------------------------------------------------------------------

class _FakeCursor:
    """Substring-matched canned responses + a SQL recorder."""

    def __init__(self, table_data: dict, recorder: list):
        self._table_data = table_data  # {needle: (cols, rows)}
        self._recorder = recorder
        self.description: list = []
        self._rows: list = []

    def execute(self, sql, params=()):
        self._recorder.append((sql, params))
        for needle, (cols, rows) in self._table_data.items():
            if needle in sql:
                self.description = [(c,) for c in cols]
                self._rows = list(rows)
                return
        self.description = []
        self._rows = []

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return list(self._rows)

    def close(self):
        pass


class _FakeConn:
    def __init__(self, table_data: dict, recorder: list):
        self._cur = _FakeCursor(table_data, recorder)
        self.autocommit = False
        self.committed = False
        self.rolled_back = False

    def cursor(self):
        return self._cur

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Columns we pretend exist on proc.bp_invoice_raw.
_REAL_COLUMNS = ["raw_id", "invoice_amount", "tax_amount", "currency", "supplier_id"]


def _table_data_for(discrepancy_rows: list[tuple]) -> dict:
    """Build the canned responses dict for a _FakeCursor.

    discrepancy_rows: list of (field_name, resolved_value, resolution_action)
    """
    return {
        # Allowlist query — information_schema lookup for the _raw table columns.
        "information_schema.columns": (
            ["column_name"],
            [(c,) for c in _REAL_COLUMNS],
        ),
        # Discrepancy query.
        "bp_extraction_discrepancy": (
            ["field_name", "resolved_value", "resolution_action"],
            discrepancy_rows,
        ),
        # Row-exists check for doc_type verification (SELECT 1 FROM … WHERE raw_id).
        "bp_invoice_raw": (["exists"], [(1,)]),
    }


def _wire_fake_conn(monkeypatch, discrepancy_rows: list[tuple]):
    """Patch get_conn so apply_hitl_fixes_and_promote uses a fake connection.

    Returns (recorder, fake_conn).
    """
    rec: list = []
    fake_conn = _FakeConn(_table_data_for(discrepancy_rows), rec)

    ctx_mgr = mock.MagicMock()
    ctx_mgr.__enter__ = mock.MagicMock(return_value=fake_conn)
    ctx_mgr.__exit__ = mock.MagicMock(return_value=False)
    monkeypatch.setattr(promotion, "get_conn", lambda: ctx_mgr)
    # promote() is called at the end of apply_hitl_fixes_and_promote; stub it
    # so we don't need a second DB connection in these unit tests.
    monkeypatch.setattr(promotion, "promote", lambda *a, **k: {"ok": True})
    return rec, fake_conn


# ---------------------------------------------------------------------------
# Tests for _table_columns (the new column-discovery helper)
# ---------------------------------------------------------------------------

class TestTableColumns:
    def test_returns_column_names(self):
        rec: list = []
        cur = _FakeCursor(
            {
                "information_schema.columns": (
                    ["column_name"],
                    [("invoice_amount",), ("tax_amount",), ("raw_id",)],
                )
            },
            recorder=rec,
        )
        cols = promotion._table_columns(cur, "proc.bp_invoice_raw")
        assert set(cols) == {"invoice_amount", "tax_amount", "raw_id"}

    def test_passes_schema_and_table_as_params(self):
        rec: list = []
        cur = _FakeCursor(
            {"information_schema.columns": (["column_name"], [])},
            recorder=rec,
        )
        promotion._table_columns(cur, "proc.bp_invoice_raw")
        assert any(
            params == ("proc", "bp_invoice_raw")
            for _, params in rec
            if "information_schema" in _
        )

    def test_empty_table_returns_empty_list(self):
        rec: list = []
        cur = _FakeCursor(
            {"information_schema.columns": (["column_name"], [])},
            recorder=rec,
        )
        assert promotion._table_columns(cur, "proc.bp_invoice_raw") == []


# ---------------------------------------------------------------------------
# Tests for the allowlist guard inside apply_hitl_fixes_and_promote
# ---------------------------------------------------------------------------

class TestHitlAllowlistGuard:

    def test_malicious_field_name_rejected(self, monkeypatch):
        """A field_name with SQL metacharacters must NOT appear in any UPDATE."""
        malicious = "invoice_amount = NULL, promotion_status"
        rec, _ = _wire_fake_conn(
            monkeypatch,
            [(malicious, "99.00", "apply_value")],
        )

        promotion.apply_hitl_fixes_and_promote(raw_id=42, doc_type="invoice")

        update_sqls = [sql for sql, _ in rec if "UPDATE" in sql.upper()]
        for sql in update_sqls:
            assert malicious not in sql, (
                f"Malicious field_name leaked into UPDATE: {sql!r}"
            )

    def test_benign_field_name_allowed(self, monkeypatch):
        """A field_name that IS a real column must produce a SET UPDATE."""
        rec, _ = _wire_fake_conn(
            monkeypatch,
            [("invoice_amount", "150.00", "apply_value")],
        )

        promotion.apply_hitl_fixes_and_promote(raw_id=42, doc_type="invoice")

        update_sqls = [sql for sql, _ in rec if "UPDATE" in sql.upper()]
        assert any("invoice_amount" in sql for sql in update_sqls), (
            f"Expected UPDATE for 'invoice_amount' but got: {update_sqls}"
        )

    def test_keep_null_benign_column_executes(self, monkeypatch):
        """keep_null action on a real column must produce a SET … = NULL UPDATE."""
        rec, _ = _wire_fake_conn(
            monkeypatch,
            [("tax_amount", None, "keep_null")],
        )

        promotion.apply_hitl_fixes_and_promote(raw_id=42, doc_type="invoice")

        update_sqls = [sql for sql, _ in rec if "UPDATE" in sql.upper()]
        assert any(
            "tax_amount" in sql and "NULL" in sql for sql in update_sqls
        ), f"Expected NULL UPDATE for 'tax_amount' but got: {update_sqls}"

    def test_keep_null_malicious_column_rejected(self, monkeypatch):
        """keep_null action with a malicious field_name must also be rejected."""
        malicious = "x=NULL; DROP TABLE proc.bp_invoice_raw--"
        rec, _ = _wire_fake_conn(
            monkeypatch,
            [(malicious, None, "keep_null")],
        )

        promotion.apply_hitl_fixes_and_promote(raw_id=42, doc_type="invoice")

        update_sqls = [sql for sql, _ in rec if "UPDATE" in sql.upper()]
        for sql in update_sqls:
            assert malicious not in sql, (
                f"Malicious field_name leaked into UPDATE: {sql!r}"
            )

    def test_mixed_batch_only_benign_executes(self, monkeypatch):
        """When a batch has both valid and malicious rows, only valid ones
        should generate UPDATE statements."""
        malicious = "invoice_amount = NULL, promotion_status"
        rec, _ = _wire_fake_conn(
            monkeypatch,
            [
                (malicious, "99", "apply_value"),            # rejected
                ("invoice_amount", "42.00", "apply_value"),  # allowed
            ],
        )

        promotion.apply_hitl_fixes_and_promote(raw_id=42, doc_type="invoice")

        update_sqls = [sql for sql, _ in rec if "UPDATE" in sql.upper()]
        for sql in update_sqls:
            assert malicious not in sql
        assert any("invoice_amount" in sql for sql in update_sqls), (
            f"Legitimate column update missing from: {update_sqls}"
        )

    def test_unknown_column_name_rejected(self, monkeypatch):
        """A field_name that simply doesn't exist on the table is also rejected."""
        rec, _ = _wire_fake_conn(
            monkeypatch,
            [("nonexistent_column", "value", "apply_value")],
        )

        promotion.apply_hitl_fixes_and_promote(raw_id=42, doc_type="invoice")

        update_sqls = [sql for sql, _ in rec if "UPDATE" in sql.upper()]
        for sql in update_sqls:
            assert "nonexistent_column" not in sql
