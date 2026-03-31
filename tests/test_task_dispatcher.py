# tests/test_task_dispatcher.py
import pytest
from unittest.mock import MagicMock, call
from datetime import datetime, timezone


def test_dispatch_task_publishes_to_correct_stream():
    from orchestration.task_dispatcher import TaskDispatcher
    from orchestration.message_protocol import TaskMessage

    redis_client = MagicMock()
    redis_client.xlen = MagicMock(return_value=0)
    dispatcher = TaskDispatcher(redis_client=redis_client, max_stream_len=10000)

    msg = TaskMessage(
        task_id="t-001",
        workflow_id="wf-123",
        node_name="rank_suppliers",
        agent_type="supplier_ranking",
        context={"workflow_id": "wf-123", "agent_id": "supplier_ranking",
                 "user_id": "u1", "input_data": {}},
        priority="normal",
        dispatched_at=datetime(2026, 3, 31, tzinfo=timezone.utc),
        timeout_seconds=300,
        attempt=1,
    )
    dispatcher.dispatch(msg)
    assert redis_client.xadd.call_count >= 1
    stream_name = redis_client.xadd.call_args_list[0][0][0]
    assert stream_name == "agent:tasks:supplier_ranking"


def test_dispatch_publishes_event():
    from orchestration.task_dispatcher import TaskDispatcher
    from orchestration.message_protocol import TaskMessage

    redis_client = MagicMock()
    redis_client.xlen = MagicMock(return_value=0)
    dispatcher = TaskDispatcher(redis_client=redis_client, max_stream_len=10000)

    msg = TaskMessage(
        task_id="t-001",
        workflow_id="wf-123",
        node_name="rank_suppliers",
        agent_type="supplier_ranking",
        context={},
        priority="normal",
        dispatched_at=datetime(2026, 3, 31, tzinfo=timezone.utc),
        timeout_seconds=300,
        attempt=1,
    )
    dispatcher.dispatch(msg)
    # Should also publish to workflow:events stream
    event_call = [c for c in redis_client.xadd.call_args_list if c[0][0] == "workflow:events"]
    assert len(event_call) == 1


def test_dispatch_respects_maxlen():
    from orchestration.task_dispatcher import TaskDispatcher
    from orchestration.message_protocol import TaskMessage

    redis_client = MagicMock()
    redis_client.xlen = MagicMock(return_value=0)
    dispatcher = TaskDispatcher(redis_client=redis_client, max_stream_len=5000)

    msg = TaskMessage(
        task_id="t-001", workflow_id="wf-123", node_name="n", agent_type="a",
        context={}, priority="normal",
        dispatched_at=datetime(2026, 3, 31, tzinfo=timezone.utc),
        timeout_seconds=300, attempt=1,
    )
    dispatcher.dispatch(msg)
    xadd_kwargs = redis_client.xadd.call_args_list[0]
    # maxlen should be passed
    assert "maxlen" in xadd_kwargs.kwargs or len(xadd_kwargs.args) > 2


def test_publish_cancellation():
    from orchestration.task_dispatcher import TaskDispatcher

    redis_client = MagicMock()
    dispatcher = TaskDispatcher(redis_client=redis_client, max_stream_len=10000)

    dispatcher.publish_cancellation(workflow_id="wf-123", agent_types=["supplier_ranking", "negotiation"])
    assert redis_client.xadd.call_count == 2
