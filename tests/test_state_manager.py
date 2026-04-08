# tests/test_state_manager.py
"""State Manager tests using a mock database connection.

The State Manager owns all reads/writes to proc.workflow_execution,
proc.node_execution, and proc.workflow_events.
"""
import pytest
from unittest.mock import MagicMock, patch, call
from datetime import datetime, timezone


class FakeCursor:
    """Simulates psycopg2 cursor with fetchone/fetchall."""

    def __init__(self, rows=None):
        self._rows = rows or []
        self.description = None
        self.rowcount = len(self._rows)

    def execute(self, sql, params=None):
        pass

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return self._rows

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class FakeConnection:
    def __init__(self, cursor):
        self._cursor = cursor

    def cursor(self):
        return self._cursor

    def commit(self):
        pass

    def rollback(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def _make_state_manager(cursor_rows=None):
    from orchestration.state_manager import StateManager

    cursor = FakeCursor(cursor_rows)
    conn = FakeConnection(cursor)
    get_conn = MagicMock(return_value=conn)
    sm = StateManager(get_connection=get_conn)
    return sm, cursor, get_conn


def test_create_workflow_execution():
    sm, cursor, _ = _make_state_manager([(42,)])
    cursor.execute = MagicMock()
    cursor.fetchone = MagicMock(return_value=(42,))

    exec_id = sm.create_workflow_execution(
        workflow_id="wf-001",
        workflow_name="supplier_ranking",
        user_id="user-1",
    )
    assert exec_id == 42
    # Verify INSERT was called
    assert cursor.execute.call_count >= 1
    sql = cursor.execute.call_args_list[0][0][0]
    assert "proc.workflow_execution" in sql
    assert "INSERT" in sql.upper()


def test_create_node_executions():
    sm, cursor, _ = _make_state_manager()
    cursor.execute = MagicMock()

    nodes = [
        {"node_name": "extract", "agent_type": "data_extraction"},
        {"node_name": "rank", "agent_type": "supplier_ranking"},
    ]
    sm.create_node_executions(execution_id=42, nodes=nodes, round_num=0)
    assert cursor.execute.call_count >= 2


def test_get_ready_nodes():
    sm, cursor, _ = _make_state_manager()
    cursor.fetchall = MagicMock(return_value=[
        (1, "extract", "data_extraction", 0),
    ])
    cursor.execute = MagicMock()

    ready = sm.get_ready_nodes(execution_id=42, current_round=0)
    assert len(ready) == 1
    assert ready[0]["node_name"] == "extract"


def test_update_node_status():
    sm, cursor, _ = _make_state_manager()
    cursor.execute = MagicMock()

    sm.update_node_status(
        execution_id=42,
        node_name="extract",
        round_num=0,
        status="running",
        dispatched_at=datetime(2026, 3, 31, 12, 0, 0, tzinfo=timezone.utc),
    )
    assert cursor.execute.call_count >= 1
    sql = cursor.execute.call_args[0][0]
    assert "UPDATE" in sql.upper()
    assert "proc.node_execution" in sql


def test_record_node_result():
    sm, cursor, _ = _make_state_manager()
    cursor.execute = MagicMock()

    sm.record_node_result(
        execution_id=42,
        node_name="extract",
        round_num=0,
        status="completed",
        output_data={"details": "extracted"},
        pass_fields={"details": "extracted"},
        duration_ms=1500,
    )
    assert cursor.execute.call_count >= 1


def test_merge_shared_data():
    sm, cursor, _ = _make_state_manager()
    cursor.execute = MagicMock()

    sm.merge_shared_data(execution_id=42, new_fields={"ranking": [1, 2, 3]})
    sql = cursor.execute.call_args[0][0]
    assert "shared_data" in sql
    assert "proc.workflow_execution" in sql


def test_record_event():
    sm, cursor, _ = _make_state_manager()
    cursor.execute = MagicMock()

    sm.record_event(
        workflow_id="wf-001",
        event_type="node:completed",
        node_name="extract",
        agent_type="data_extraction",
        payload={"duration_ms": 1500},
        round_num=0,
    )
    sql = cursor.execute.call_args[0][0]
    assert "proc.workflow_events" in sql
