"""An irreversible action that cannot be logged must not happen.

The existing record_action is deliberately best-effort so a logging blip
cannot halt extraction. That trade is wrong for sending mail or approving
spend, so those callers use record_action_or_fail instead. This proves the
two behave differently under exactly the same failure.
"""

import os

import psycopg2
import pytest
from dotenv import load_dotenv

from src.services import agent_actions

load_dotenv()


@pytest.fixture(scope="module")
def db_conn():
    """Module-level database connection for success tests."""
    try:
        c = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT", 5432),
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            connect_timeout=8,
        )
    except Exception as exc:
        pytest.skip(f"bp_sqldb not reachable: {exc}")
    yield c
    c.close()


class ExplodingConn:
    """A connection whose cursor always fails, like a dropped session."""

    def cursor(self):
        raise RuntimeError("connection is gone")


def test_best_effort_writer_swallows_a_failure():
    """record_action must never raise, even on connection failure."""
    agent_actions.record_action(
        phase="extraction",
        action_type="parse",
        conn=ExplodingConn(),
        summary="should not raise",
    )


def test_mandatory_writer_raises_on_a_failure():
    """record_action_or_fail must raise AuditWriteError on any failure."""
    with pytest.raises(agent_actions.AuditWriteError):
        agent_actions.record_action_or_fail(
            phase="communicate",
            action_type="email.send",
            conn=ExplodingConn(),
            summary="must raise",
        )


def test_mandatory_writer_names_the_action_in_the_error():
    """The error message must include the action_type for debugging."""
    with pytest.raises(agent_actions.AuditWriteError) as excinfo:
        agent_actions.record_action_or_fail(
            phase="communicate",
            action_type="email.send",
            conn=ExplodingConn(),
        )
    assert "email.send" in str(excinfo.value)


def test_mandatory_writer_succeeds_and_persists(db_conn):
    """The strict writer must successfully insert and commit on healthy connection."""
    trace_id = "test-success-audit-trace-12345"

    # Write with the strict writer using the provided connection
    agent_actions.record_action_or_fail(
        phase="communicate",
        action_type="email.send",
        trace_id=trace_id,
        summary="test audit success",
        conn=db_conn,
    )

    # Commit the transaction to make the write persistent
    db_conn.commit()

    # Verify the row exists in the database
    with db_conn.cursor() as cur:
        cur.execute(
            "SELECT trace_id, phase, action_type FROM proc.bp_agent_actions WHERE trace_id = %s",
            (trace_id,),
        )
        row = cur.fetchone()

    assert row is not None, f"audit row with trace_id '{trace_id}' was not persisted"
    persisted_trace_id, persisted_phase, persisted_action_type = row
    assert persisted_trace_id == trace_id
    assert persisted_phase == "communicate"
    assert persisted_action_type == "email.send"

    # Clean up: roll back so test data doesn't persist
    db_conn.rollback()
