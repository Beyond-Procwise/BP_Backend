"""An irreversible action that cannot be logged must not happen.

The existing record_action is deliberately best-effort so a logging blip
cannot halt extraction. That trade is wrong for sending mail or approving
spend, so those callers use record_action_or_fail instead. This proves the
two behave differently under exactly the same failure.
"""

import os
import uuid

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
    """The strict writer must successfully insert and persist on healthy connection."""
    trace_id = uuid.uuid4().hex

    agent_actions.record_action_or_fail(
        phase="communicate",
        action_type="email.send",
        conn=db_conn,
        trace_id=trace_id,
        agent="probe",
        status="allowed",
    )

    # Visible on our own connection without committing. Verify here and roll
    # back: this is the audit table on a shared cluster, and a test that seeds
    # it undermines the record this layer exists to make trustworthy.
    cur = db_conn.cursor()
    cur.execute(
        "SELECT phase, action_type, agent FROM proc.bp_agent_actions "
        "WHERE trace_id = %s",
        (trace_id,),
    )
    row = cur.fetchone()
    assert row is not None, "the strict writer did not persist the row"
    assert row[0] == "communicate"
    assert row[1] == "email.send"

    db_conn.rollback()
