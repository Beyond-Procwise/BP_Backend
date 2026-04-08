"""Agent worker process: consumes tasks from Redis Streams, executes agents.

Each worker serves one agent type. It reads from agent:tasks:{agent_type},
executes the agent, and publishes results to agent:results.

Spec reference: Section 4 of orchestration-rearchitecture-design.md
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Optional

from agents.base_agent import AgentContext, AgentOutput, AgentStatus
from orchestration.message_protocol import ResultMessage, TaskMessage

logger = logging.getLogger(__name__)

RESULT_STREAM = "agent:results"


class AgentWorker:
    def __init__(
        self,
        agent_type: str,
        worker_context: Any,
        redis_client: Any,
        consumer_group: str = "workers",
        consumer_name: Optional[str] = None,
    ):
        self.agent_type = agent_type
        self._ctx = worker_context
        self._redis = redis_client
        self._group = consumer_group
        self._consumer = consumer_name or f"worker-{agent_type}-{id(self)}"
        self._stream = f"agent:tasks:{agent_type}"

    def execute_task(self, task: TaskMessage) -> None:
        start = time.monotonic()
        try:
            agent = self._ctx.get_agent(task.agent_type)
            context = AgentContext(
                workflow_id=task.context.get("workflow_id", ""),
                agent_id=task.context.get("agent_id", task.agent_type),
                user_id=task.context.get("user_id", ""),
                input_data=task.context.get("input_data", {}),
                policy_context=task.context.get("policy_context", []),
                knowledge_base=task.context.get("knowledge_base", {}),
                routing_history=task.context.get("routing_history", []),
                task_profile=task.context.get("task_profile", {}),
            )
            result = agent.execute(context)
            duration_ms = int((time.monotonic() - start) * 1000)
            self._publish_result(task, result, duration_ms)
        except Exception as exc:
            duration_ms = int((time.monotonic() - start) * 1000)
            logger.exception("Agent %s failed for task %s", task.agent_type, task.task_id)
            error_result = AgentOutput(
                status=AgentStatus.FAILED,
                data={},
                error=str(exc),
            )
            self._publish_result(task, error_result, duration_ms)

    def _publish_result(
        self, task: TaskMessage, result: AgentOutput, duration_ms: int
    ) -> None:
        msg = ResultMessage(
            task_id=task.task_id,
            workflow_id=task.workflow_id,
            node_name=task.node_name,
            agent_type=task.agent_type,
            status=result.status.value.upper() if hasattr(result.status, "value") else str(result.status).upper(),
            data=result.data or {},
            pass_fields=result.pass_fields or {},
            next_agents=result.next_agents or [],
            error=result.error,
            confidence=result.confidence,
            completed_at=datetime.now(timezone.utc),
            duration_ms=duration_ms,
        )
        self._redis.xadd(RESULT_STREAM, msg.to_redis())
        logger.info(
            "Published result for %s/%s: %s (%dms)",
            task.workflow_id, task.node_name, msg.status, duration_ms,
        )
