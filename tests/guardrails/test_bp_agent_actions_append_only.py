"""proc.bp_agent_actions is append-only, and the database is what enforces it.

The audit spine is only worth reading if it cannot be rewritten. Before
deploy/sql/2026-09-16_bp_agent_actions_immutable.sql it could be: there was no
trigger, and the application role `procwisedb123` is the table's OWNER in both
bp_testdb and bp_sqldb. REVOKE does not help against an owner -- an owner may
re-grant to itself at will -- so a privilege change here would have been theatre.
A BEFORE trigger that raises is the only thing that actually holds.

TRUNCATE is covered separately and deliberately. Postgres does not route TRUNCATE
through a row-level trigger, so a table guarded only against UPDATE and DELETE can
still be emptied in one statement -- 54,769 rows in bp_sqldb -- which is precisely
the move that matters most to prevent.

Nothing is destroyed to prove this. Each test inserts its own row, acts on that,
and rolls back; TRUNCATE is transactional in Postgres, so the rollback restores the
table. No historical audit row is touched.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

_LIVE = os.environ.get("PROCWISE_TEST_LIVE_DB", "").strip().lower() in (
    "1", "true", "yes", "on")
pytestmark = [
    pytest.mark.skipif(not _LIVE, reason="needs PROCWISE_TEST_LIVE_DB=1"),
    pytest.mark.integration,
]


@pytest.fixture()
def conn():
    """A transactional connection. get_conn() hands back an AUTOCOMMIT connection,
    on which rollback is a no-op, so autocommit is switched off explicitly -- without
    that every probe below would be a real write."""
    from src.services.db import get_conn

    with get_conn() as c:
        c.autocommit = False
        try:
            yield c
        finally:
            c.rollback()


def _probe_row(cur) -> int:
    """Insert one row of our own to act on, so no real audit row is ever the subject."""
    cur.execute(
        "INSERT INTO proc.bp_agent_actions (phase, action_type, summary) "
        "VALUES ('test', 'append_only_probe', 'probe') RETURNING action_id"
    )
    return cur.fetchone()[0]


def test_an_audit_row_cannot_be_updated(conn):
    cur = conn.cursor()
    action_id = _probe_row(cur)
    with pytest.raises(Exception) as exc:
        cur.execute(
            "UPDATE proc.bp_agent_actions SET summary = 'rewritten' WHERE action_id = %s",
            (action_id,),
        )
    assert "append-only" in str(exc.value).lower(), (
        f"an audit row was rewritten, or refused for the wrong reason: {exc.value}"
    )


def test_an_audit_row_cannot_be_deleted(conn):
    cur = conn.cursor()
    action_id = _probe_row(cur)
    with pytest.raises(Exception) as exc:
        cur.execute(
            "DELETE FROM proc.bp_agent_actions WHERE action_id = %s", (action_id,)
        )
    assert "append-only" in str(exc.value).lower(), (
        f"an audit row was deleted, or refused for the wrong reason: {exc.value}"
    )


def test_the_audit_table_cannot_be_truncated(conn):
    """The hole a row-level trigger alone would leave: one statement, the whole log."""
    cur = conn.cursor()
    with pytest.raises(Exception) as exc:
        cur.execute("TRUNCATE proc.bp_agent_actions")
    assert "append-only" in str(exc.value).lower(), (
        f"the audit log was truncated, or refused for the wrong reason: {exc.value}"
    )


def test_audit_rows_can_still_be_written(conn):
    """The guard must not be so broad that it stops the spine recording anything.

    record_action_or_fail refuses to let an irreversible action proceed when its
    audit row cannot be written, so a trigger that blocked INSERT would not merely
    lose the log -- it would stop the product working.
    """
    cur = conn.cursor()
    action_id = _probe_row(cur)
    cur.execute(
        "SELECT summary FROM proc.bp_agent_actions WHERE action_id = %s", (action_id,)
    )
    assert cur.fetchone()[0] == "probe"
