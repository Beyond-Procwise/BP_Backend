"""Lightweight replacement for agent_nick in worker processes.

Only initializes dependencies needed by the specific agent types
this worker serves. Implements the same interface agents expect.

Spec reference: Section 4 of orchestration-rearchitecture-design.md
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


class WorkerContext:
    def __init__(
        self,
        settings: Any,
        agent_types: List[str],
        agent_factory: Any = None,
        db_pool: Any = None,
        redis_client: Any = None,
        process_routing_service: Any = None,
    ):
        self.settings = settings
        self._agent_types = agent_types
        self._agent_factory = agent_factory
        self._db_pool = db_pool
        self._redis_client = redis_client
        self.process_routing_service = process_routing_service
        self._agents: Dict[str, Any] = {}

    def get_agent(self, agent_type: str) -> Any:
        if agent_type not in self._agents:
            if self._agent_factory:
                self._agents[agent_type] = self._agent_factory.create(agent_type)
            else:
                raise ValueError(f"No factory to create agent: {agent_type}")
        return self._agents[agent_type]

    def get_connection(self) -> Any:
        if self._db_pool:
            return self._db_pool.getconn()
        raise RuntimeError("No database pool configured")
