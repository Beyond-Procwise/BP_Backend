"""Dataclasses for Redis Streams message protocol.

Three message types:
- TaskMessage: Dispatcher -> Workers (via agent:tasks:{agent_type} streams)
- ResultMessage: Workers -> Result Collector (via agent:results stream)
- EventMessage: All components -> Observability (via workflow:events stream)
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _iso(dt: Optional[datetime]) -> Optional[str]:
    return dt.isoformat() if dt else None


def _parse_iso(val: Optional[str]) -> Optional[datetime]:
    if val is None:
        return None
    return datetime.fromisoformat(val)


@dataclass(frozen=True)
class TaskMessage:
    task_id: str
    workflow_id: str
    node_name: str
    agent_type: str
    context: Dict[str, Any]
    priority: str  # "critical" | "normal"
    dispatched_at: datetime
    timeout_seconds: int
    attempt: int

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["dispatched_at"] = _iso(self.dispatched_at)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> TaskMessage:
        d = dict(d)
        d["dispatched_at"] = _parse_iso(d["dispatched_at"])
        return cls(**d)

    def to_redis(self) -> Dict[str, str]:
        return {"payload": json.dumps(self.to_dict())}

    @classmethod
    def from_redis(cls, fields: Dict[bytes, bytes]) -> TaskMessage:
        raw = fields.get(b"payload") or fields.get("payload")
        if isinstance(raw, bytes):
            raw = raw.decode()
        return cls.from_dict(json.loads(raw))


@dataclass(frozen=True)
class ResultMessage:
    task_id: str
    workflow_id: str
    node_name: str
    agent_type: str
    status: str  # "SUCCESS" | "FAILED"
    data: Dict[str, Any]
    pass_fields: Dict[str, Any]
    next_agents: List[str]
    error: Optional[str]
    confidence: Optional[float]
    completed_at: datetime
    duration_ms: int

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["completed_at"] = _iso(self.completed_at)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ResultMessage:
        d = dict(d)
        d["completed_at"] = _parse_iso(d["completed_at"])
        return cls(**d)

    def to_redis(self) -> Dict[str, str]:
        return {"payload": json.dumps(self.to_dict())}

    @classmethod
    def from_redis(cls, fields: Dict[bytes, bytes]) -> ResultMessage:
        raw = fields.get(b"payload") or fields.get("payload")
        if isinstance(raw, bytes):
            raw = raw.decode()
        return cls.from_dict(json.loads(raw))


@dataclass(frozen=True)
class EventMessage:
    event: str
    workflow_id: str
    node_name: Optional[str]
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["timestamp"] = _iso(self.timestamp)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> EventMessage:
        d = dict(d)
        d["timestamp"] = _parse_iso(d["timestamp"])
        return cls(**d)

    def to_redis(self) -> Dict[str, str]:
        return {"payload": json.dumps(self.to_dict())}

    @classmethod
    def from_redis(cls, fields: Dict[bytes, bytes]) -> EventMessage:
        raw = fields.get(b"payload") or fields.get("payload")
        if isinstance(raw, bytes):
            raw = raw.decode()
        return cls.from_dict(json.loads(raw))
