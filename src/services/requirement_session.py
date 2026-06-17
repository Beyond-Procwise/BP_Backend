from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class RequirementSession:
    """Multi-turn elicitation state for a single procurement requirement.

    Hot state lives in Redis (best-effort); the durable source of truth is the
    ``proc.bp_requirement`` row written by ``RequirementService``.
    """

    session_id: str
    requirement_id: str
    created_by: str = ""
    status: str = "gathering"
    requirement: Dict[str, Any] = field(default_factory=dict)
    turn_history: List[Dict[str, Any]] = field(default_factory=list)
    missing_fields: List[str] = field(default_factory=list)
    completeness_score: float = 0.0

    def __post_init__(self) -> None:
        self.redis_key = f"requirement_session:{self.session_id}"

    def add_turn(self, role: str, content: str) -> None:
        self.turn_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def apply_fields(self, updates: Dict[str, Any]) -> None:
        """Merge non-empty field values. Empty/None values are ignored so the
        agent never fabricates or blanks an already-known field."""
        if not isinstance(updates, dict):
            return
        for key, value in updates.items():
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            self.requirement[key] = value

    def mark_complete(self) -> None:
        self.status = "complete"

    def mark_abandoned(self) -> None:
        self.status = "abandoned"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "requirement_id": self.requirement_id,
            "created_by": self.created_by,
            "status": self.status,
            "requirement": self.requirement,
            "turn_history": self.turn_history,
            "missing_fields": self.missing_fields,
            "completeness_score": self.completeness_score,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RequirementSession":
        return cls(
            session_id=data["session_id"],
            requirement_id=data["requirement_id"],
            created_by=data.get("created_by", ""),
            status=data.get("status", "gathering"),
            requirement=data.get("requirement", {}),
            turn_history=data.get("turn_history", []),
            missing_fields=data.get("missing_fields", []),
            completeness_score=data.get("completeness_score", 0.0),
        )

    def save(self, redis_client: Any) -> None:
        if redis_client is None:
            return
        try:
            redis_client.set(self.redis_key, json.dumps(self.to_dict()))
        except Exception:  # pragma: no cover - infra failure
            logger.exception("Failed to save requirement session %s", self.session_id)

    @classmethod
    def load(cls, session_id: str, redis_client: Any) -> Optional["RequirementSession"]:
        if redis_client is None:
            return None
        key = f"requirement_session:{session_id}"
        try:
            data = redis_client.get(key)
        except Exception:  # pragma: no cover - infra failure
            logger.exception("Failed to load requirement session %s", session_id)
            return None
        if not data:
            return None
        try:
            return cls.from_dict(json.loads(data))
        except Exception:  # pragma: no cover - corrupt payload
            logger.exception("Corrupt requirement session payload for %s", session_id)
            return None
