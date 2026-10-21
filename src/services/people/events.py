"""Domain events: the catalogue, the envelope, and the transactional outbox write."""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, Optional

#: event type -> (priority, mandatory). Mandatory events ignore the user's per-type opt-out.
CATALOGUE: Dict[str, tuple] = {
    "requisition.created": ("low", False), "requisition.submitted": ("normal", False),
    "approval.requested": ("high", True), "approval.approved": ("normal", False),
    "approval.rejected": ("normal", False), "po.issued": ("normal", False),
    "po.changed": ("normal", False), "invoice.received": ("low", False),
    "invoice.mismatch": ("high", False), "supplier.onboarded": ("normal", False),
    "supplier.risk_flagged": ("high", False), "contract.expiring": ("high", False),
    "contract.signed": ("normal", False), "rfq.published": ("normal", False),
    "rfq.response_received": ("normal", False), "agent.recommendation_ready": ("normal", False),
    "task.assigned": ("normal", False), "comment.mentioned": ("normal", False),
    "message.received": ("normal", False),
}


class UnknownEventType(ValueError):
    pass


@dataclass
class DomainEvent:
    type: str
    actor_id: str
    entity_type: str
    entity_id: str
    summary: str
    tenant_id: str = "default"
    data: Dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    occurred_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    attempts: int = 0

    def __post_init__(self):
        if self.type not in CATALOGUE:
            raise UnknownEventType(self.type)

    @property
    def priority(self) -> str:
        return CATALOGUE[self.type][0]

    @property
    def mandatory(self) -> bool:
        return CATALOGUE[self.type][1]


def emit(conn, event: DomainEvent) -> bool:
    """Write the event to the outbox. Call inside the SAME transaction as the action itself, so
    an action that rolls back never announces itself. Returns False if the id was already seen."""
    with conn.cursor() as cur:
        return emit_on_cursor(cur, event)


def emit_on_cursor(cur, event: DomainEvent) -> bool:
    cur.execute(
        "INSERT INTO proc.bp_domain_event (event_id, event_type, tenant_id, actor_id, entity_type,"
        " entity_id, summary, data, occurred_at) VALUES (%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s)"
        " ON CONFLICT (event_id) DO NOTHING",
        (event.event_id, event.type, event.tenant_id, event.actor_id, event.entity_type,
         event.entity_id, event.summary, json.dumps(event.data, default=_json), event.occurred_at))
    return cur.rowcount == 1


def _json(o):
    if isinstance(o, Decimal):
        return str(o)
    if isinstance(o, datetime):
        return o.isoformat()
    raise TypeError(type(o))
