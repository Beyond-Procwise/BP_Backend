import json
import pytest
from datetime import datetime, timezone


def test_task_message_round_trip():
    from orchestration.message_protocol import TaskMessage

    msg = TaskMessage(
        task_id="t-001",
        workflow_id="wf-123",
        node_name="rank_suppliers",
        agent_type="supplier_ranking",
        context={
            "workflow_id": "wf-123",
            "agent_id": "supplier_ranking",
            "user_id": "user-1",
            "input_data": {"query": "find suppliers"},
            "policy_context": [],
            "knowledge_base": {},
            "routing_history": ["data_extraction"],
            "task_profile": {},
        },
        priority="normal",
        dispatched_at=datetime(2026, 3, 31, 12, 0, 0, tzinfo=timezone.utc),
        timeout_seconds=300,
        attempt=1,
    )
    serialized = msg.to_dict()
    restored = TaskMessage.from_dict(serialized)
    assert restored.task_id == "t-001"
    assert restored.agent_type == "supplier_ranking"
    assert restored.context["input_data"]["query"] == "find suppliers"
    assert restored.timeout_seconds == 300
    json.dumps(serialized)


def test_result_message_round_trip():
    from orchestration.message_protocol import ResultMessage

    msg = ResultMessage(
        task_id="t-001",
        workflow_id="wf-123",
        node_name="rank_suppliers",
        agent_type="supplier_ranking",
        status="SUCCESS",
        data={"ranking": [{"supplier": "Acme", "score": 0.95}]},
        pass_fields={"ranking": [{"supplier": "Acme", "score": 0.95}]},
        next_agents=["quote_evaluation"],
        error=None,
        confidence=0.92,
        completed_at=datetime(2026, 3, 31, 12, 0, 5, tzinfo=timezone.utc),
        duration_ms=4230,
    )
    serialized = msg.to_dict()
    restored = ResultMessage.from_dict(serialized)
    assert restored.status == "SUCCESS"
    assert restored.duration_ms == 4230
    assert restored.data["ranking"][0]["supplier"] == "Acme"
    json.dumps(serialized)


def test_event_message_round_trip():
    from orchestration.message_protocol import EventMessage

    msg = EventMessage(
        event="node:completed",
        workflow_id="wf-123",
        node_name="rank_suppliers",
        timestamp=datetime(2026, 3, 31, 12, 0, 5, tzinfo=timezone.utc),
        metadata={"duration_ms": 4230, "agent_type": "supplier_ranking"},
    )
    serialized = msg.to_dict()
    restored = EventMessage.from_dict(serialized)
    assert restored.event == "node:completed"
    assert restored.metadata["duration_ms"] == 4230
    json.dumps(serialized)
