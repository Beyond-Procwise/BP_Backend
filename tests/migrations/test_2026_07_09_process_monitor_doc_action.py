"""Integration test for 2026-07-09_process_monitor_doc_action.sql (bp_sqldb)."""
from __future__ import annotations

import sys
from pathlib import Path

import psycopg2
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import Settings  # noqa: E402

MIGRATION = (Path(__file__).resolve().parents[2]
             / "deploy" / "sql" / "2026-07-09_process_monitor_doc_action.sql")


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
    yield conn
    conn.close()


def test_columns_exist(applied):
    cur = applied.cursor()
    cur.execute("""SELECT column_name FROM information_schema.columns
                   WHERE table_schema='proc' AND table_name='process_monitor'""")
    cols = {r[0] for r in cur.fetchall()}
    assert "content_hash" in cols
    assert "doc_action" in cols


def test_doc_action_constraint_rejects_unknown(applied):
    cur = applied.cursor()
    with pytest.raises(psycopg2.errors.CheckViolation):
        cur.execute("""INSERT INTO proc.process_monitor
                       (process_name, type, status, doc_action)
                       VALUES ('t','t','Completed','bogus')""")
    applied.rollback()


def test_doc_action_accepts_known(applied):
    cur = applied.cursor()
    cur.execute("""INSERT INTO proc.process_monitor
                   (process_name, type, status, doc_action)
                   VALUES ('t','t','Completed','duplicate') RETURNING id""")
    new_id = cur.fetchone()[0]
    cur.execute("DELETE FROM proc.process_monitor WHERE id=%s", (new_id,))


def test_resolver_function_present(applied):
    cur = applied.cursor()
    cur.execute("SELECT proc.fn_try_resolve_session('__nonexistent_session__')")
    # no exception == success (returns early on unknown session)
