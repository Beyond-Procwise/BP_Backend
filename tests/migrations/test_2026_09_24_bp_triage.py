"""Integration test for 2026-09-24_bp_triage.sql (runs against the .env database)."""
from __future__ import annotations

import sys
from pathlib import Path

import psycopg2
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import Settings  # noqa: E402
from tests.conftest import GOVERNED_LIMIT_SEED  # noqa: E402

MIGRATION = (Path(__file__).resolve().parents[2]
             / "deploy" / "sql" / "2026-09-24_bp_triage.sql")


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


def test_tables_exist(applied):
    cur = applied.cursor()
    cur.execute("""SELECT table_name FROM information_schema.tables
                   WHERE table_schema='proc' AND table_name LIKE 'bp_triage_%'""")
    assert {r[0] for r in cur.fetchall()} >= {
        "bp_triage_run", "bp_triage_result", "bp_triage_finding"}


def test_policy_row_matches_the_test_seed(applied):
    cur = applied.cursor()
    cur.execute("""SELECT policy_details->'rules' FROM proc.bp_policy
                   WHERE policy_type='limit'
                     AND policy_details->>'policy_identifier'='triage_tolerances'""")
    rows = cur.fetchall()
    assert len(rows) == 1
    assert rows[0][0] == GOVERNED_LIMIT_SEED["triage_tolerances"]
