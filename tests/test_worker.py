import pytest
from unittest.mock import MagicMock, patch
import json
import time
from datetime import datetime, timezone


def test_worker_context_provides_settings():
    from orchestration.worker_context import WorkerContext

    settings = MagicMock()
    settings.db_host = "localhost"
    ctx = WorkerContext(settings=settings, agent_types=["data_extraction"])
    assert ctx.settings.db_host == "localhost"


def test_worker_context_creates_agent_instances():
    from orchestration.worker_context import WorkerContext

    settings = MagicMock()
    mock_agent = MagicMock()
    agent_factory = MagicMock()
    agent_factory.create.return_value = mock_agent

    ctx = WorkerContext(
        settings=settings,
        agent_types=["data_extraction"],
        agent_factory=agent_factory,
    )
    agent = ctx.get_agent("data_extraction")
    assert agent == mock_agent
    agent_factory.create.assert_called_with("data_extraction")


def test_worker_executes_agent_and_publishes_result():
    from orchestration.worker import AgentWorker
    from orchestration.message_protocol import TaskMessage
    from agents.base_agent import AgentOutput, AgentStatus

    mock_agent = MagicMock()
    mock_agent.execute.return_value = AgentOutput(
        status=AgentStatus.SUCCESS,
        data={"details": "extracted"},
        pass_fields={"details": "extracted"},
    )

    worker_ctx = MagicMock()
    worker_ctx.get_agent.return_value = mock_agent
    redis_client = MagicMock()

    worker = AgentWorker(
        agent_type="data_extraction",
        worker_context=worker_ctx,
        redis_client=redis_client,
    )

    task = TaskMessage(
        task_id="t-001", workflow_id="wf-123", node_name="extract",
        agent_type="data_extraction",
        context={"workflow_id": "wf-123", "agent_id": "data_extraction",
                 "user_id": "u1", "input_data": {"file": "test.pdf"},
                 "policy_context": [], "knowledge_base": {},
                 "routing_history": [], "task_profile": {}},
        priority="normal",
        dispatched_at=datetime(2026, 3, 31, tzinfo=timezone.utc),
        timeout_seconds=300, attempt=1,
    )

    worker.execute_task(task)

    mock_agent.execute.assert_called_once()
    # Should publish result to agent:results stream
    result_call = [c for c in redis_client.xadd.call_args_list if c[0][0] == "agent:results"]
    assert len(result_call) == 1


def test_worker_handles_agent_failure():
    from orchestration.worker import AgentWorker
    from orchestration.message_protocol import TaskMessage

    mock_agent = MagicMock()
    mock_agent.execute.side_effect = RuntimeError("LLM timeout")

    worker_ctx = MagicMock()
    worker_ctx.get_agent.return_value = mock_agent
    redis_client = MagicMock()

    worker = AgentWorker(
        agent_type="data_extraction",
        worker_context=worker_ctx,
        redis_client=redis_client,
    )

    task = TaskMessage(
        task_id="t-002", workflow_id="wf-123", node_name="extract",
        agent_type="data_extraction",
        context={"workflow_id": "wf-123", "agent_id": "data_extraction",
                 "user_id": "u1", "input_data": {},
                 "policy_context": [], "knowledge_base": {},
                 "routing_history": [], "task_profile": {}},
        priority="normal",
        dispatched_at=datetime(2026, 3, 31, tzinfo=timezone.utc),
        timeout_seconds=300, attempt=1,
    )

    worker.execute_task(task)

    result_call = [c for c in redis_client.xadd.call_args_list if c[0][0] == "agent:results"]
    assert len(result_call) == 1
    payload = json.loads(result_call[0].args[1]["payload"])
    assert payload["status"] == "FAILED"
    assert "LLM timeout" in payload["error"]
