# tests/orchestration/test_state_manager_payload_serialisation_live.py
"""Reproduces the run-trail persistence bug seen live on run awf-140-ac728008
(a data_extraction workflow).

StateManager.record_node_result did:
    json.dumps(output_data) if output_data else None
which THROWS whenever a real agent's output carries a value the stdlib json
module cannot encode -- data_extraction output routinely carries numpy
scalars / Decimal / datetime-like values. workflow_engine._safe_persist
catches that exception so the workflow itself doesn't crash, but the
node_execution row is inserted with status='pending' at dispatch time and is
ONLY ever updated to the real outcome by record_node_result -- so when that
call raises, the row is left forever reading status='pending', duration_ms
NULL, even though the workflow_execution row (and the node itself) actually
completed. A trail that lies about the state it reached is worse than no
trail at all.

These tests hit the live proc schema (bp_sqldb, per repo-root .env) and hard
-delete every row they create, on success AND failure, per the pattern in
tests/orchestration/test_state_manager_live.py.
"""
import os
import sys
from datetime import datetime, timezone
from decimal import Decimal

import numpy as np
import psycopg2
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from orchestration.state_manager import StateManager

pytestmark = pytest.mark.integration  # touches the live proc schema

WORKFLOW_ID = "state-mgr-serialisation-live-test"


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
    created_execution_ids: list[int] = []

    yield created_execution_ids

    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM proc.workflow_events WHERE workflow_id = %s", (WORKFLOW_ID,)
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


def _dispatch_a_node(mgr: StateManager, node_name: str = "extract") -> int:
    """Mirror what workflow_engine does before a node runs: insert the
    node_execution row at status='pending', then flip it to 'running' the
    way _persist_node_created / update_node_status do on dispatch.
    """
    execution_id = mgr.create_workflow_execution(
        workflow_id=WORKFLOW_ID,
        workflow_name="live_serialisation_test_workflow",
        user_id="pytest",
    )
    mgr.create_node_executions(
        execution_id,
        nodes=[{"node_name": node_name, "agent_type": "data_extraction"}],
        round_num=0,
    )
    mgr.update_node_status(execution_id, node_name, round_num=0, status="running")
    return execution_id


def _fetch_node_row(execution_id: int, node_name: str):
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT status, duration_ms, completed_at
                   FROM proc.node_execution
                   WHERE execution_id = %s AND node_name = %s""",
                (execution_id, node_name),
            )
            return cur.fetchone()
    finally:
        conn.close()


def _non_json_serialisable_output_data() -> dict:
    """Shaped like real data_extraction output: numpy scalar, Decimal, and a
    datetime, including one buried inside a nested dict/list -- none of
    which json.dumps can natively encode.
    """
    return {
        "rows_extracted": np.int64(42),
        "confidence": np.float64(0.973),
        "total_amount": Decimal("1234.56"),
        "extracted_at": datetime(2026, 7, 13, 10, 30, 0, tzinfo=timezone.utc),
        "line_items": [
            {"qty": np.int32(3), "unit_price": Decimal("9.99")},
        ],
    }


def test_record_node_result_with_non_serialisable_output_does_not_raise_and_completes(
    _cleanup_execution_ids,
):
    """(a) A payload containing numpy/Decimal/datetime values (including
    nested) must not raise, and the row must persist as status='completed'
    with a real duration.
    """
    created_execution_ids = _cleanup_execution_ids
    mgr = StateManager(get_connection=_connect)

    execution_id = _dispatch_a_node(mgr, "extract")
    created_execution_ids.append(execution_id)

    # Must not raise.
    mgr.record_node_result(
        execution_id,
        "extract",
        round_num=0,
        status="completed",
        output_data=_non_json_serialisable_output_data(),
        pass_fields={"supplier_id": "S1"},
        duration_ms=2450,
    )

    row = _fetch_node_row(execution_id, "extract")
    assert row is not None
    status, duration_ms, completed_at = row
    assert status == "completed"
    assert duration_ms == 2450
    assert completed_at is not None


def test_record_node_result_never_leaves_node_reading_pending(
    _cleanup_execution_ids,
):
    """(b) After a record_node_result call carrying an unserialisable
    payload, the node row must NOT be left at status='pending' -- that is
    the actual damage: an audit trail that lies about a completed node.
    """
    created_execution_ids = _cleanup_execution_ids
    mgr = StateManager(get_connection=_connect)

    execution_id = _dispatch_a_node(mgr, "extract")
    created_execution_ids.append(execution_id)

    mgr.record_node_result(
        execution_id,
        "extract",
        round_num=0,
        status="completed",
        output_data=_non_json_serialisable_output_data(),
        duration_ms=1800,
    )

    row = _fetch_node_row(execution_id, "extract")
    assert row is not None
    status, duration_ms, _completed_at = row
    assert status != "pending"
    assert duration_ms is not None


def test_record_node_result_json_safe_payload_still_round_trips(
    _cleanup_execution_ids,
):
    """(c) The working path (a normal JSON-safe payload) must not regress."""
    created_execution_ids = _cleanup_execution_ids
    mgr = StateManager(get_connection=_connect)

    execution_id = _dispatch_a_node(mgr, "rank")
    created_execution_ids.append(execution_id)

    mgr.record_node_result(
        execution_id,
        "rank",
        round_num=0,
        status="completed",
        output_data={"suppliers": ["A", "B", "C"], "count": 3},
        pass_fields={"top_supplier": "A"},
        duration_ms=500,
    )

    result = mgr.get_node_result(execution_id, "rank", round_num=0)
    assert result is not None
    assert result["status"] == "completed"
    assert result["output_data"] == {"suppliers": ["A", "B", "C"], "count": 3}
    assert result["pass_fields"] == {"top_supplier": "A"}

    row = _fetch_node_row(execution_id, "rank")
    assert row[0] == "completed"
    assert row[1] == 500
