"""proc.bp_agent_run can be made append-only by deploy/sql/2026-10-20_bp_agent_run_immutable.sql.

The migration is applied INSIDE each test's transaction and rolled back, so this proves the SQL
does what it says without changing the database: DDL is transactional in Postgres, and so is
TRUNCATE. No existing run is touched; each test inserts a row of its own to act on.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

_LIVE = os.environ.get("PROCWISE_TEST_LIVE_DB", "").strip().lower() in ("1", "true", "yes", "on")
pytestmark = [pytest.mark.skipif(not _LIVE, reason="needs PROCWISE_TEST_LIVE_DB=1"),
              pytest.mark.integration]

MIGRATION = ROOT / "deploy/sql/2026-10-20_bp_agent_run_immutable.sql"
ROLLBACK = ROOT / "deploy/sql/2026-10-20_bp_agent_run_immutable_rollback.sql"


def _body(path: Path) -> str:
    """The statements without the surrounding BEGIN/COMMIT, so they join the test's transaction."""
    return re.sub(r"^\s*(BEGIN|COMMIT);\s*$", "", path.read_text(), flags=re.M)


@pytest.fixture()
def cur():
    from repositories import agent_run_repo
    from src.services.db import get_conn

    agent_run_repo.ensure_schema()
    with get_conn() as conn:
        conn.autocommit = False  # get_conn() is AUTOCOMMIT; rollback would be a no-op
        try:
            c = conn.cursor()
            yield c
        finally:
            conn.rollback()


def _insert_probe(c) -> str:
    c.execute(
        """INSERT INTO proc.bp_agent_run (run_id, agent_slug, system_prompt, system_sha256, task,
                                          tools_offered, calls, started_at, finished_at)
           VALUES ('probe-append-only', 'probe', 's', 'h', 't', '[]'::jsonb, '[]'::jsonb,
                   now(), now()) RETURNING run_id""")
    return c.fetchone()[0]


def _refused(c, sql: str) -> str:
    c.execute("SAVEPOINT s")
    try:
        c.execute(sql)
    except Exception as exc:  # noqa: BLE001
        c.execute("ROLLBACK TO SAVEPOINT s")
        return str(exc)
    c.execute("ROLLBACK TO SAVEPOINT s")
    return ""


def test_the_migration_makes_update_delete_and_truncate_refuse(cur):
    cur.execute(_body(MIGRATION))
    run_id = _insert_probe(cur)          # INSERT stays open
    for sql in (f"UPDATE proc.bp_agent_run SET answer = 'edited' WHERE run_id = '{run_id}'",
                f"DELETE FROM proc.bp_agent_run WHERE run_id = '{run_id}'",
                "TRUNCATE proc.bp_agent_run"):
        assert "append-only" in _refused(cur, sql), sql
    cur.execute("SELECT answer FROM proc.bp_agent_run WHERE run_id = %s", (run_id,))
    assert cur.fetchone()[0] is None


def test_without_the_migration_an_owner_can_still_edit_a_run(cur):
    """The control: this is why the trigger is needed."""
    cur.execute(_body(ROLLBACK))
    run_id = _insert_probe(cur)
    assert _refused(cur, f"UPDATE proc.bp_agent_run SET answer = 'x' WHERE run_id = '{run_id}'") == ""


def test_the_rollback_undoes_the_migration(cur):
    cur.execute(_body(MIGRATION))
    cur.execute(_body(ROLLBACK))
    run_id = _insert_probe(cur)
    assert _refused(cur, f"DELETE FROM proc.bp_agent_run WHERE run_id = '{run_id}'") == ""


def test_the_migration_is_idempotent(cur):
    cur.execute(_body(MIGRATION))
    cur.execute(_body(MIGRATION))
    cur.execute("SELECT count(*) FROM pg_trigger WHERE tgrelid = 'proc.bp_agent_run'::regclass "
                "AND NOT tgisinternal")
    assert cur.fetchone()[0] == 2
