"""Integration test for 2026-10-04_contract_buyer_signatory.sql (runs against the .env database).

A contract has two signatories and proc.bp_contracts had one pair of columns, so
the signature-block reader had to discard the buyer's name. This migration gives
it a home. The asymmetric naming is deliberate and is asserted here so nobody
"tidies" it into supplier_signatory_* without reading why.
"""
from __future__ import annotations

import sys
from pathlib import Path

import psycopg2
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import Settings  # noqa: E402

MIGRATION = (Path(__file__).resolve().parents[2]
             / "deploy" / "sql" / "2026-10-04_contract_buyer_signatory.sql")
ROLLBACK = (Path(__file__).resolve().parents[2]
            / "deploy" / "sql" / "2026-10-04_contract_buyer_signatory_rollback.sql")
NEW = {"buyer_signatory_name", "buyer_signatory_role"}


def _conn():
    s = Settings()
    c = psycopg2.connect(host=s.db_host, dbname=s.db_name,
                         user=s.db_user, password=s.db_password, port=s.db_port)
    c.autocommit = True
    return c


@pytest.fixture(scope="module")
def applied():
    conn = _conn()
    conn.cursor().execute(MIGRATION.read_text())
    conn.cursor().execute(MIGRATION.read_text())   # idempotent: a second apply is harmless
    yield conn
    conn.close()


def _cols(cur, table):
    cur.execute("""SELECT column_name FROM information_schema.columns
                   WHERE table_schema='proc' AND table_name=%s""", (table,))
    return {r[0] for r in cur.fetchall()}


def test_both_tiers_carry_the_buyer_signatory_columns(applied):
    cur = applied.cursor()
    for table in ("bp_contract_raw", "bp_contracts"):
        assert NEW <= _cols(cur, table), table


def test_the_suppliers_pair_is_untouched(applied):
    """The existing columns are NOT renamed: the gateway and the Obligations
    screen read contract_signatory_name, so a rename is a breaking change."""
    cur = applied.cursor()
    for table in ("bp_contract_raw", "bp_contracts"):
        cols = _cols(cur, table)
        assert "contract_signatory_name" in cols, table
        assert "supplier_signatory_name" not in cols, (
            f"{table}: the supplier's pair was renamed; the migration says why not")


def test_the_columns_are_text_and_nullable(applied):
    """NULL is the honest answer for a document that names no signatory, so a NOT
    NULL constraint here would force a guess."""
    cur = applied.cursor()
    cur.execute("""SELECT column_name, data_type, is_nullable
                     FROM information_schema.columns
                    WHERE table_schema='proc' AND table_name='bp_contracts'
                      AND column_name = ANY(%s)""", (sorted(NEW),))
    rows = {r[0]: (r[1], r[2]) for r in cur.fetchall()}
    assert rows == {"buyer_signatory_name": ("text", "YES"),
                    "buyer_signatory_role": ("text", "YES")}, rows


def test_the_asymmetry_is_documented_on_the_column(applied):
    """A future reader must be able to learn from the database which pair is
    which, without finding this migration."""
    cur = applied.cursor()
    cur.execute("""SELECT col_description(c.oid, a.attnum)
                     FROM pg_class c
                     JOIN pg_namespace n ON n.oid = c.relnamespace
                     JOIN pg_attribute a ON a.attrelid = c.oid
                    WHERE n.nspname='proc' AND c.relname='bp_contracts'
                      AND a.attname='contract_signatory_name'""")
    comment = (cur.fetchone() or [None])[0] or ""
    assert "SUPPLIER" in comment, comment


def test_the_rollback_removes_exactly_the_two_columns():
    """Proven by doing it, then putting them back -- a rollback nobody has run is
    a rollback nobody can rely on."""
    conn = _conn()
    cur = conn.cursor()
    cur.execute(MIGRATION.read_text())
    assert NEW <= _cols(cur, "bp_contracts")
    try:
        cur.execute(ROLLBACK.read_text())
        assert not (NEW & _cols(cur, "bp_contracts"))
        assert not (NEW & _cols(cur, "bp_contract_raw"))
        assert "contract_signatory_name" in _cols(cur, "bp_contracts")
    finally:
        cur.execute(MIGRATION.read_text())
        assert NEW <= _cols(cur, "bp_contracts")
        conn.close()
