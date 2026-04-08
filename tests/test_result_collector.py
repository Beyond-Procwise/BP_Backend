# tests/test_result_collector.py
import pytest
from unittest.mock import MagicMock, patch, call
import json
from datetime import datetime, timezone


def _make_result_payload(task_id="t-001", workflow_id="wf-123", node_name="extract",
                          agent_type="data_extraction", status="SUCCESS"):
    from orchestration.message_protocol import ResultMessage
    return ResultMessage(
        task_id=task_id, workflow_id=workflow_id, node_name=node_name,
        agent_type=agent_type, status=status,
        data={"details": "extracted"}, pass_fields={"details": "extracted"},
        next_agents=[], error=None, confidence=0.95,
        completed_at=datetime(2026, 3, 31, 12, 0, 5, tzinfo=timezone.utc),
        duration_ms=1500,
    )


def test_process_success_result_updates_state():
    from orchestration.result_collector import ResultCollector

    state_manager = MagicMock()
    dag_scheduler = MagicMock()
    state_manager.get_workflow_execution.return_value = {
        "execution_id": 42, "workflow_id": "wf-123",
        "current_round": 0, "status": "running",
    }

    collector = ResultCollector(
        state_manager=state_manager,
        dag_scheduler=dag_scheduler,
        workflow_graphs={},
    )
    result = _make_result_payload()
    collector.process_result(result, execution_id=42)

    state_manager.record_node_result.assert_called_once()
    state_manager.merge_shared_data.assert_called_once_with(42, {"details": "extracted"})


def test_process_failure_result_calls_on_node_failed():
    from orchestration.result_collector import ResultCollector

    state_manager = MagicMock()
    dag_scheduler = MagicMock()
    state_manager.get_workflow_execution.return_value = {
        "execution_id": 42, "workflow_id": "wf-123",
        "current_round": 0, "status": "running",
    }

    collector = ResultCollector(
        state_manager=state_manager,
        dag_scheduler=dag_scheduler,
        workflow_graphs={"test": MagicMock()},
    )
    result = _make_result_payload(status="FAILED")
    collector.process_result(result, execution_id=42, graph_name="test")

    state_manager.record_node_result.assert_called_once()
    dag_scheduler.on_node_failed.assert_called_once()


def test_process_result_records_event():
    from orchestration.result_collector import ResultCollector

    state_manager = MagicMock()
    dag_scheduler = MagicMock()
    state_manager.get_workflow_execution.return_value = {
        "execution_id": 42, "workflow_id": "wf-123",
        "current_round": 0, "status": "running",
    }

    collector = ResultCollector(
        state_manager=state_manager,
        dag_scheduler=dag_scheduler,
        workflow_graphs={},
    )
    result = _make_result_payload()
    collector.process_result(result, execution_id=42)

    state_manager.record_event.assert_called()
    event_call = state_manager.record_event.call_args
    assert event_call.kwargs["event_type"] == "node:completed"
