# src/orchestration/task_dispatcher.py
"""Publishes agent tasks to Redis Streams for worker consumption.

Each agent type gets its own stream: agent:tasks:{agent_type}
Also publishes lifecycle events to workflow:events stream.

Spec reference: Section 2 of orchestration-rearchitecture-design.md
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List

from orchestration.message_protocol import EventMessage, TaskMessage

logger = logging.getLogger(__name__)


class TaskDispatcher:
    TASK_STREAM_PREFIX = "agent:tasks:"
    EVENT_STREAM = "workflow:events"

    def __init__(self, redis_client: Any, max_stream_len: int = 10000):
        self._redis = redis_client
        self._max_stream_len = max_stream_len

    def dispatch(self, task: TaskMessage) -> str:
        stream = f"{self.TASK_STREAM_PREFIX}{task.agent_type}"
        msg_id = self._redis.xadd(
            stream,
            task.to_redis(),
            maxlen=self._max_stream_len,
        )
        event = EventMessage(
            event="node:dispatched",
            workflow_id=task.workflow_id,
            node_name=task.node_name,
            timestamp=datetime.now(timezone.utc),
            metadata={
                "agent_type": task.agent_type,
                "task_id": task.task_id,
                "attempt": task.attempt,
            },
        )
        self._redis.xadd(
            self.EVENT_STREAM,
            event.to_redis(),
            maxlen=self._max_stream_len * 10,
        )
        logger.info(
            "Dispatched task %s to %s (workflow=%s, attempt=%d)",
            task.task_id, stream, task.workflow_id, task.attempt,
        )
        return msg_id

    def publish_cancellation(
        self, workflow_id: str, agent_types: List[str]
    ) -> None:
        for agent_type in agent_types:
            stream = f"{self.TASK_STREAM_PREFIX}{agent_type}"
            cancel_msg = {
                "payload": json.dumps({
                    "cancel": True,
                    "workflow_id": workflow_id,
                })
            }
            self._redis.xadd(stream, cancel_msg, maxlen=self._max_stream_len)
            logger.info("Published cancellation for %s to %s", workflow_id, stream)

    def publish_event(self, event: EventMessage) -> None:
        self._redis.xadd(
            self.EVENT_STREAM,
            event.to_redis(),
            maxlen=self._max_stream_len * 10,
        )
