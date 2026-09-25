"""Integration test for 2026-09-26_contract_line_terms.sql (runs against the .env database)."""
from __future__ import annotations

import sys
from pathlib import Path

import psycopg2
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import Settings  # noqa: E402

MIGRATION = (Path(__file__).resolve().parents[2]
             / "deploy" / "sql" / "2026-09-26_contract_line_terms.sql")


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


def test_both_tables_carry_the_line_term_columns(applied):
    cur = applied.cursor()
    want = {"line_number", "item_description", "term_basis", "unit_price", "unit_of_measure", "currency", "qualifier"}
    assert want | {"line_raw_id", "raw_id"} <= _cols(cur, "bp_contract_line_items_raw")
    assert want | {"contract_line_id", "contract_id"} <= _cols(cur, "bp_contract_line_items")


def test_term_basis_is_constrained(applied):
    cur = applied.cursor()
    with pytest.raises(psycopg2.errors.CheckViolation):
        cur.execute("""INSERT INTO proc.bp_contract_line_items (contract_line_id, contract_id, term_basis)
                       VALUES ('__t-L1', '__t', 'discount')""")


def test_contracts_upsert_on_contract_id(applied):
    # promotion.py: INSERT ... ON CONFLICT (contract_id). Without a unique key on
    # contract_id Postgres refuses the statement outright.
    cur = applied.cursor()
    cur.execute("BEGIN")
    try:
        for _ in range(2):
            cur.execute("""INSERT INTO proc.bp_contracts (contract_id) VALUES ('__t-upsert')
                           ON CONFLICT (contract_id) DO NOTHING""")
        cur.execute("SELECT count(*) FROM proc.bp_contracts WHERE contract_id = '__t-upsert'")
        assert cur.fetchone()[0] == 1
    finally:
        cur.execute("ROLLBACK")
