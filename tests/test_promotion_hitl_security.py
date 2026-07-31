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

import pytest

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

    discrepancy_rows: list of (field_name, resolved_value, resolution_action), or the
    widened (field_name, resolved_value, resolution_action, raw_value, resolved_by) —
    padded to 5 with None here so existing 3-tuple callers keep working unchanged.
    """
    padded_rows = [tuple(r) + (None,) * (5 - len(r)) for r in discrepancy_rows]
    return {
        # Allowlist query — information_schema lookup for the _raw table columns.
        "information_schema.columns": (
            ["column_name"],
            [(c,) for c in _REAL_COLUMNS],
        ),
        # Discrepancy query.
        "bp_extraction_discrepancy": (
            ["field_name", "resolved_value", "resolution_action", "raw_value", "resolved_by"],
            padded_rows,
        ),
        # Row-exists check for doc_type verification (SELECT 1 FROM … WHERE raw_id),
        # also matched by the verdict-capture doc_pk lookup (SELECT <pk_col>,
        # doc_pk_candidate FROM proc.bp_invoice_raw WHERE raw_id=%s) — one canned row
        # is enough for both, since fetchone() just needs *some* pk value back.
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


# ---------------------------------------------------------------------------
# Tests for the verdict-capture allowlist guard (Task 2 fix-round-1, Minor)
# ---------------------------------------------------------------------------

class TestVerdictAllowlistGuard:
    """The verdict-capture loop must skip any field_name the _raw UPDATE loop already
    rejected as unsafe/unknown — recording a verdict for it would pollute per-reader
    accuracy stats (the very table this feature exists to build) with a bogus column."""

    def test_rejected_field_name_gets_no_verdict_row(self, monkeypatch):
        malicious = "invoice_amount = NULL, promotion_status"
        rec, _ = _wire_fake_conn(monkeypatch, [(malicious, "99.00", "apply_value")])

        promotion.apply_hitl_fixes_and_promote(raw_id=42, doc_type="invoice")

        verdict_sqls = [sql for sql, _ in rec if "INSERT INTO proc.bp_extraction_verdict" in sql]
        assert verdict_sqls == [], f"expected no verdict row for a rejected field, got {verdict_sqls}"

    def test_unknown_column_gets_no_verdict_row(self, monkeypatch):
        rec, _ = _wire_fake_conn(monkeypatch, [("nonexistent_column", "value", "apply_value")])

        promotion.apply_hitl_fixes_and_promote(raw_id=42, doc_type="invoice")

        verdict_sqls = [sql for sql, _ in rec if "INSERT INTO proc.bp_extraction_verdict" in sql]
        assert verdict_sqls == [], f"expected no verdict row for an unknown column, got {verdict_sqls}"

    def test_benign_field_name_gets_a_verdict_row(self, monkeypatch):
        rec, _ = _wire_fake_conn(monkeypatch, [("invoice_amount", "150.00", "apply_value")])

        promotion.apply_hitl_fixes_and_promote(raw_id=42, doc_type="invoice")

        verdict_sqls = [sql for sql, _ in rec if "INSERT INTO proc.bp_extraction_verdict" in sql]
        assert verdict_sqls, "expected a verdict row for a benign, allowed field"


# ---------------------------------------------------------------------------
# Tests for SAVEPOINT isolation around verdict capture (Task 2 fix-round-1, Important)
# ---------------------------------------------------------------------------
#
# A bare try/except around the verdict-capture block stops the PYTHON exception from
# propagating, but does NOT undo what Postgres itself does when a statement in a
# transaction fails: it marks the *whole transaction* aborted, and every later
# statement on that connection — including a plain conn.commit() — fails until a
# ROLLBACK (full, or TO SAVEPOINT) runs. Without a savepoint, a verdict-write failure
# would silently poison the _raw fix this function exists to commit: log.exception
# fires, then the unconditional conn.commit() raises "current transaction is
# aborted", which the OUTER except catches — rolling back and discarding the human's
# entire HITL fix, not just the verdict rows. This happened in production
# (procwise.log:343738 — an in-transaction "total_amount_incl_tax does not exist"
# error lost a whole HITL fix). These tests pin the SAVEPOINT-based fix by simulating
# that exact abort-propagation behaviour at the fake cursor/connection level.

class _PoisonAwareCursor(_FakeCursor):
    """Extends _FakeCursor with real-Postgres transaction-abort semantics: once the
    statement matching ``poison_needle`` "fails", every later statement on the shared
    connection also fails with "transaction is aborted" — until a ROLLBACK (TO
    SAVEPOINT, or full) runs, which clears it. This is precisely the behaviour a bare
    try/except cannot protect against and a SAVEPOINT can.
    """

    def __init__(self, table_data, recorder, poison_needle, conn):
        super().__init__(table_data, recorder)
        self._poison_needle = poison_needle
        self._conn = conn

    def execute(self, sql, params=()):
        if self._conn.aborted:
            self._recorder.append((sql, params))
            if "ROLLBACK" in sql.upper():
                self._conn.aborted = False
                return
            raise Exception(
                "current transaction is aborted, commands ignored until end of "
                "transaction block"
            )
        if self._poison_needle in sql:
            self._conn.aborted = True
            self._recorder.append((sql, params))
            raise Exception('column "total_amount_incl_tax" does not exist')
        super().execute(sql, params)


class _PoisonAwareConn(_FakeConn):
    def __init__(self, table_data, recorder, poison_needle):
        self.aborted = False
        self._cur = _PoisonAwareCursor(table_data, recorder, poison_needle, self)
        self.autocommit = False
        self.committed = False
        self.rolled_back = False

    def commit(self):
        if self.aborted:
            raise Exception(
                "current transaction is aborted, commands ignored until end of "
                "transaction block"
            )
        self.committed = True


def _wire_poisoning_conn(monkeypatch, discrepancy_rows, poison_needle):
    rec: list = []
    fake_conn = _PoisonAwareConn(_table_data_for(discrepancy_rows), rec, poison_needle)
    ctx_mgr = mock.MagicMock()
    ctx_mgr.__enter__ = mock.MagicMock(return_value=fake_conn)
    ctx_mgr.__exit__ = mock.MagicMock(return_value=False)
    monkeypatch.setattr(promotion, "get_conn", lambda: ctx_mgr)
    monkeypatch.setattr(promotion, "promote", lambda *a, **k: {"ok": True})
    return rec, fake_conn


class TestVerdictSavepointIsolation:

    def test_verdict_write_failure_does_not_lose_the_hitl_fix_or_promotion(self, monkeypatch):
        rec, fake_conn = _wire_poisoning_conn(
            monkeypatch,
            [("invoice_amount", "150.00", "apply_value")],
            poison_needle="INSERT INTO proc.bp_extraction_verdict",
        )

        result = promotion.apply_hitl_fixes_and_promote(raw_id=42, doc_type="invoice")

        # The promotion must still succeed — a verdict-write failure is not a reason
        # to lose the human's correction.
        assert result == {"ok": True}, result
        # The HITL fix to _raw must have been applied (and, by reaching a successful
        # commit below, retained) before the poisoned verdict write ever ran.
        update_sqls = [sql for sql, _ in rec if "UPDATE" in sql.upper()]
        assert any("invoice_amount" in sql for sql in update_sqls)
        # The outer rollback (which would discard the _raw fix entirely) must never
        # fire, and the transaction must actually commit.
        assert fake_conn.rolled_back is False
        assert fake_conn.committed is True
        # And the savepoint must genuinely have been rolled back to — not merely
        # logged — or the connection would still be aborted at commit time.
        rollback_sqls = [sql for sql, _ in rec if "ROLLBACK TO SAVEPOINT" in sql.upper()]
        assert rollback_sqls, (
            "expected a ROLLBACK TO SAVEPOINT after the poisoned verdict write"
        )

    def test_without_a_savepoint_the_failure_would_have_lost_everything(self, monkeypatch):
        """Sanity check on the fake itself: prove the failure mode is real by showing
        that skipping straight to conn.commit() after the poison — i.e. what a bare
        try/except leaves behind — does raise, which is exactly what the outer except
        in apply_hitl_fixes_and_promote would catch and roll back on."""
        rec: list = []
        fake_conn = _PoisonAwareConn(
            _table_data_for([("invoice_amount", "150.00", "apply_value")]),
            rec,
            poison_needle="INSERT INTO proc.bp_extraction_verdict",
        )
        cur = fake_conn.cursor()
        with pytest.raises(Exception):
            cur.execute("INSERT INTO proc.bp_extraction_verdict (x) VALUES (%s)", (1,))
        assert fake_conn.aborted is True
        with pytest.raises(Exception):
            fake_conn.commit()
