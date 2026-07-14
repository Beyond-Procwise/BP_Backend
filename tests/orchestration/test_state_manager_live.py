# tests/orchestration/test_state_manager_live.py
"""Proves proc.workflow_execution / proc.node_execution / proc.workflow_events
actually exist in the live DB and that StateManager writes real rows into
them -- not just that the class methods run without raising.

The DDL for these tables lives at
src/orchestration/migrations/001_workflow_execution.sql but was never
applied, so every workflow run's per-node execution trail was silently
discarded (StateManager.create_workflow_execution etc. would raise
"relation ... does not exist" against the live DB).
"""
import os
import sys

import psycopg2
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from orchestration.state_manager import StateManager

pytestmark = pytest.mark.integration  # touches the live proc schema


def _connect():
    return psycopg2.connect(
        host=os.environ["DB_HOST"],
        dbname=os.environ["DB_NAME"],
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
        port=os.environ.get("DB_PORT", "5432"),
    )


@pytest.fixture
def _cleanup_execution_ids():
    """Hard-delete every row this test creates, on success AND on failure."""
    created_workflow_ids = ["state-mgr-live-test"]
    created_execution_ids: list[int] = []

    yield created_workflow_ids, created_execution_ids

    conn = _connect()
    try:
        with conn.cursor() as cur:
            for wid in created_workflow_ids:
                cur.execute(
                    "DELETE FROM proc.workflow_events WHERE workflow_id = %s", (wid,)
                )
                cur.execute(
                    """DELETE FROM proc.node_execution
                       WHERE execution_id IN (
                           SELECT execution_id FROM proc.workflow_execution
                           WHERE workflow_id = %s
                       )""",
                    (wid,),
                )
                cur.execute(
                    "DELETE FROM proc.workflow_execution WHERE workflow_id = %s", (wid,)
                )
            for exec_id in created_execution_ids:
                cur.execute(
                    "DELETE FROM proc.node_execution WHERE execution_id = %s", (exec_id,)
                )
                cur.execute(
                    "DELETE FROM proc.workflow_execution WHERE execution_id = %s", (exec_id,)
                )
        conn.commit()
    finally:
        conn.close()


def test_state_manager_persists_a_real_node_execution_trail(_cleanup_execution_ids):
    """A workflow run's per-node trail must land in the live DB, not vanish."""
    created_workflow_ids, created_execution_ids = _cleanup_execution_ids

    mgr = StateManager(get_connection=_connect)

    execution_id = mgr.create_workflow_execution(
        workflow_id="state-mgr-live-test",
        workflow_name="live_verification_workflow",
        user_id="pytest",
    )
    created_execution_ids.append(execution_id)

    mgr.create_node_executions(
        execution_id,
        nodes=[{"node_name": "extract", "agent_type": "data_extraction"}],
        round_num=0,
    )
    mgr.update_node_status(
        execution_id, "extract", round_num=0, status="running"
    )
    mgr.record_node_result(
        execution_id,
        "extract",
        round_num=0,
        status="completed",
        output_data={"rows": 3},
        pass_fields={"supplier_id": "S1"},
    )
    mgr.record_event(
        workflow_id="state-mgr-live-test",
        event_type="node_completed",
        node_name="extract",
        agent_type="data_extraction",
        payload={"ok": True},
    )

    # Read back through StateManager's own accessors...
    execution = mgr.get_workflow_execution(execution_id)
    assert execution is not None
    assert execution["workflow_name"] == "live_verification_workflow"

    node_result = mgr.get_node_result(execution_id, "extract", round_num=0)
    assert node_result is not None
    assert node_result["status"] == "completed"
    assert node_result["output_data"] == {"rows": 3}

    events = mgr.get_events("state-mgr-live-test")
    assert any(e["event_type"] == "node_completed" for e in events)

    # ...and independently, straight off the tables, so this cannot pass
    # against a mock or a table that silently accepted writes into the void.
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT status FROM proc.node_execution WHERE execution_id = %s AND node_name = %s",
                (execution_id, "extract"),
            )
            row = cur.fetchone()
    finally:
        conn.close()
    assert row is not None
    assert row[0] == "completed"
