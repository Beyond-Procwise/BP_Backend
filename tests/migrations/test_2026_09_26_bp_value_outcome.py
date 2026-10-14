"""proc.bp_value_outcome exists, has the spec's shape, and refuses every edit.

Nothing real is touched: each test inserts a probe row and rolls back.
Spec: docs/superpowers/specs/2026-09-25-value-ledger-design.md §3
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

_LIVE = os.environ.get("PROCWISE_TEST_LIVE_DB", "").strip().lower() in ("1", "true", "yes", "on")
pytestmark = [pytest.mark.skipif(not _LIVE, reason="needs PROCWISE_TEST_LIVE_DB=1"),
              pytest.mark.integration]


@pytest.fixture()
def conn():
    from src.services.db import get_conn
    with get_conn() as c:
        c.autocommit = False
        try:
            yield c
        finally:
            c.rollback()


def _insert(cur, **over):
    row = dict(source_type="finding", source_id="probe-1", outcome_type="claimed",
               amount=100, currency="GBP", amount_gbp=100, recorded_by="pytest-value-ledger",
               valid_from="2026-09-25")
    row.update(over)
    cols = ", ".join(row)
    cur.execute(f"INSERT INTO proc.bp_value_outcome ({cols}) VALUES "
                f"({', '.join(['%s'] * len(row))}) RETURNING outcome_id", tuple(row.values()))
    return cur.fetchone()[0]


def test_columns_match_the_spec(conn):
    cur = conn.cursor()
    cur.execute("SELECT column_name FROM information_schema.columns "
                "WHERE table_schema='proc' AND table_name='bp_value_outcome'")
    cols = {r[0] for r in cur.fetchall()}
    assert cols == {"outcome_id", "tenant_id", "source_type", "source_id", "outcome_type",
                    "amount", "currency", "amount_gbp", "fx_rate", "fx_as_of", "evidence_ref",
                    "note", "supersedes_id", "recorded_by", "valid_from", "recorded_at"}


def test_tenant_defaults_to_the_constant(conn):
    cur = conn.cursor()
    oid = _insert(cur)
    cur.execute("SELECT tenant_id FROM proc.bp_value_outcome WHERE outcome_id=%s", (oid,))
    assert cur.fetchone()[0] == "default"


@pytest.mark.parametrize("stmt", [
    "UPDATE proc.bp_value_outcome SET amount = 1 WHERE outcome_id = %s",
    "DELETE FROM proc.bp_value_outcome WHERE outcome_id = %s",
])
def test_rows_cannot_be_edited_or_deleted(conn, stmt):
    import psycopg2
    cur = conn.cursor()
    oid = _insert(cur)
    with pytest.raises(psycopg2.Error, match="append-only"):
        cur.execute(stmt, (oid,))


def test_truncate_is_refused(conn):
    import psycopg2
    with pytest.raises(psycopg2.Error, match="append-only"):
        conn.cursor().execute("TRUNCATE proc.bp_value_outcome")


def test_recovered_needs_evidence(conn):
    import psycopg2
    with pytest.raises(psycopg2.errors.CheckViolation):
        _insert(conn.cursor(), outcome_type="recovered", evidence_ref=None)


def test_amount_must_be_positive_except_claim_dropped(conn):
    import psycopg2
    cur = conn.cursor()
    _insert(cur, outcome_type="claim_dropped", amount=None, currency=None, amount_gbp=None)
    with pytest.raises(psycopg2.errors.CheckViolation):
        _insert(cur, amount=0)


def test_unknown_outcome_type_is_refused(conn):
    import psycopg2
    with pytest.raises(psycopg2.errors.CheckViolation):
        _insert(conn.cursor(), outcome_type="savings")
