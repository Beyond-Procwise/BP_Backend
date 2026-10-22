"""Events from a connected product (ERP, requisition tool, procurement suite) into the notification outbox.

Small companies have no requisition system: the approval IS the request, and that path already announces
itself. Larger companies keep requisitions and purchase orders in their ERP; this is how the ERP tells us.
Only a fixed list of event types may arrive from outside, the sentence shown to people is built HERE from
the type and the external reference (nothing the sender wrote is shown), and a retry of the same event is
a no-op because the event id is derived from (type, external id, version).
"""
from __future__ import annotations

import hashlib
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Optional

from services.people.events import DomainEvent

#: what a connected system may announce, and the sentence people see (value-free by construction)
EXTERNAL_EVENTS: Dict[str, str] = {
    "requisition.created": "Requisition {ref} was created",
    "requisition.submitted": "Requisition {ref} was submitted for approval",
    "po.issued": "Purchase order {ref} was issued",
    "po.changed": "Purchase order {ref} was changed",
    "invoice.received": "Invoice {ref} was received",
}
_ENTITY = {"requisition": "requisition", "po": "po", "invoice": "invoice"}


class IntakeError(ValueError):
    pass


def build_event(source: str, event_type: str, external_id: str, requester: Optional[str], *,
                amount: Any = None, version: str = "", link: Optional[str] = None) -> DomainEvent:
    if event_type not in EXTERNAL_EVENTS:
        raise IntakeError(f"'{event_type}' cannot be announced from outside; allowed: {', '.join(sorted(EXTERNAL_EVENTS))}")
    ref = str(external_id or "").strip()
    if not ref or len(ref) > 80:
        raise IntakeError("external_id is required (80 characters at most)")
    src = (source or "").strip()[:40] or "external"
    amt: Optional[Decimal] = None
    if amount is not None:
        try:
            amt = Decimal(str(amount))
        except InvalidOperation:
            raise IntakeError("amount must be a number")
        if amt < 0:
            raise IntakeError("amount cannot be negative")
    who = (requester or "").strip() or f"system:{src}"
    eid = hashlib.sha256(f"{event_type}|{src}|{ref}|{version}".encode()).hexdigest()[:32]
    data: Dict[str, Any] = {"source": src, "external_id": ref}
    if requester:
        data["requester"] = requester.strip()
    if amt is not None:
        data["amount"] = str(amt)
    if link and link.startswith("/") and not link.startswith("//"):
        data["link"] = link
    return DomainEvent(type=event_type, actor_id=who, entity_type=_ENTITY[event_type.split(".")[0]], entity_id=ref,
                       summary=EXTERNAL_EVENTS[event_type].format(ref=ref), event_id=f"ext:{eid}", data=data)
