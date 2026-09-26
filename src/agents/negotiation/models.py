"""The small value types a negotiation is described with.

A counterparty and the thread it is held in, the offer on the table and the two
positions either side of it, and one entry of email history. They carry data and
the rules for reading it; they run nothing and reach nowhere.

Split out of negotiation_agent.py so the helpers that build and read them do not
have to import the agent back. Re-exported from there, because callers ask for
NegotiationIdentifier by that name.
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from agents.email_drafting_agent import DEFAULT_NEGOTIATION_SUBJECT

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UniqueConstraintInfo:
    columns: Tuple[str, ...]
    constraint_name: Optional[str] = None
    predicate: Optional[str] = None
    index_name: Optional[str] = None


@dataclass
class NegotiationContext:
    current_offer: float
    target_price: float
    round_index: int = 1
    currency: Optional[str] = None
    aggressiveness: float = 0.5
    leverage: float = 0.5
    urgency: float = 0.5
    risk_buffer_pct: float = 0.05
    min_abs_buffer: float = 0.0
    step_pct_of_gap: float = 0.1
    min_abs_step: float = 1.0
    max_rounds: int = 3
    walkaway_price: Optional[float] = None
    ask_early_pay_disc: Optional[float] = None
    ask_lead_time_keep: bool = True


@dataclass
class SupplierSignals:
    offer_prev: Optional[float] = None
    offer_new: Optional[float] = None
    message_text: str = ""


@dataclass
class NegotiationIdentifier:
    workflow_id: str
    session_reference: str
    supplier_id: str
    round_number: int = 1
    # True when nothing in the caller's payload named a supplier and the id below
    # was minted here purely to key the lock and the session. It is not a real
    # counterparty, and nothing may be sent to it.
    supplier_synthesised: bool = False

    def __post_init__(self) -> None:
        self.workflow_id = self._normalise(self.workflow_id, fallback_prefix="WF")
        self.session_reference = self._normalise(
            self.session_reference, fallback_prefix="WF"
        )
        self.supplier_id = self._normalise(self.supplier_id, fallback_prefix="SUP")
        try:
            self.round_number = int(self.round_number) if self.round_number else 1
        except Exception:
            self.round_number = 1

    @staticmethod
    def _normalise(value: Optional[str], *, fallback_prefix: str = "") -> str:
        if isinstance(value, str):
            token = value.strip()
        elif value is None:
            token = ""
        else:
            token = str(value).strip()
        if not token:
            return f"{fallback_prefix}-{uuid.uuid4().hex[:12].upper()}" if fallback_prefix else ""
        return token

    @property
    def unique_key(self) -> str:
        return f"{self.workflow_id}:{self.supplier_id}:{self.round_number}"

    @property
    def thread_key(self) -> str:
        return f"{self.workflow_id}:{self.supplier_id}"


@dataclass
class EmailThreadState:
    thread_id: str
    in_reply_to: Optional[str] = None
    references: List[str] = field(default_factory=list)
    subject_base: str = ""

    def to_headers(self, round_number: int) -> Dict[str, Any]:
        message_id = f"<{uuid.uuid4()}@procwise.co.uk>"
        headers: Dict[str, Any] = {"Message-ID": message_id}
        if self.references:
            headers["References"] = " ".join(self.references[-10:])
        if self.in_reply_to:
            headers["In-Reply-To"] = self.in_reply_to
        subject = self.subject_base or DEFAULT_NEGOTIATION_SUBJECT
        if round_number > 1 and subject:
            if subject.lower().startswith("re:"):
                headers["Subject"] = subject
            else:
                headers["Subject"] = f"Re: {subject}".strip()
        elif subject:
            headers["Subject"] = subject
        return headers

    def update_after_send(self, message_id: Optional[str]) -> None:
        token = self._normalise_token(message_id)
        if not token:
            return
        if not self.thread_id:
            self.thread_id = token
        if token not in self.references:
            self.references.append(token)
        self.in_reply_to = token

    def update_after_receive(self, message_id: Optional[str]) -> None:
        token = self._normalise_token(message_id)
        if not token:
            return
        if token not in self.references:
            self.references.append(token)
        self.in_reply_to = token

    def as_dict(self) -> Dict[str, Any]:
        return {
            "thread_id": self.thread_id,
            "in_reply_to": self.in_reply_to,
            "references": list(self.references),
            "subject_base": self.subject_base,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any], *, fallback_subject: str) -> "EmailThreadState":
        thread_id = str(data.get("thread_id") or f"<{uuid.uuid4()}@procwise.co.uk>")
        references = data.get("references") if isinstance(data.get("references"), list) else []
        return EmailThreadState(
            thread_id=thread_id,
            in_reply_to=data.get("in_reply_to"),
            references=[str(item) for item in references if item],
            subject_base=str(data.get("subject_base") or fallback_subject or DEFAULT_NEGOTIATION_SUBJECT),
        )

    @staticmethod
    def _normalise_token(token: Optional[str]) -> Optional[str]:
        if isinstance(token, str):
            value = token.strip()
        elif token is None:
            value = ""
        else:
            value = str(token).strip()
        return value or None


@dataclass
class EmailHistoryEntry:
    email_id: str
    round_number: int
    supplier_id: str
    supplier_name: Optional[str]
    subject: str
    body_text: str
    body_html: str
    sender: str
    recipients: List[str]
    sent_at: datetime
    message_id: Optional[str]
    thread_headers: Dict[str, Any]
    metadata: Dict[str, Any]
    decision: Dict[str, Any]
    negotiation_context: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "email_id": self.email_id,
            "round_number": self.round_number,
            "supplier_id": self.supplier_id,
            "supplier_name": self.supplier_name,
            "subject": self.subject,
            "body_text": self.body_text,
            "body_html": self.body_html,
            "sender": self.sender,
            "recipients": list(self.recipients),
            "sent_at": self.sent_at.isoformat()
            if isinstance(self.sent_at, datetime)
            else self.sent_at,
            "message_id": self.message_id,
            "thread_headers": dict(self.thread_headers),
            "metadata": dict(self.metadata),
            "decision": dict(self.decision),
            "negotiation_context": dict(self.negotiation_context),
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "EmailHistoryEntry":
        sent_at = data.get("sent_at")
        if isinstance(sent_at, str):
            try:
                sent_at = datetime.fromisoformat(sent_at)
            except Exception:
                sent_at = datetime.now(timezone.utc)
        elif not isinstance(sent_at, datetime):
            sent_at = datetime.now(timezone.utc)

        return EmailHistoryEntry(
            email_id=data.get("email_id") or str(uuid.uuid4()),
            round_number=int(data.get("round_number", 1)),
            supplier_id=str(data.get("supplier_id") or ""),
            supplier_name=data.get("supplier_name"),
            subject=data.get("subject", ""),
            body_text=data.get("body_text", ""),
            body_html=data.get("body_html", ""),
            sender=data.get("sender", ""),
            recipients=list(data.get("recipients") or []),
            sent_at=sent_at,
            message_id=data.get("message_id"),
            thread_headers=dict(data.get("thread_headers") or {}),
            metadata=dict(data.get("metadata") or {}),
            decision=dict(data.get("decision") or {}),
            negotiation_context=dict(data.get("negotiation_context") or {}),
        )


@dataclass
class NegotiationPositions:
    start: Optional[float]
    desired: Optional[float]
    no_deal: Optional[float]
    supplier_offer: Optional[float] = None
    history: List[Dict[str, Any]] = field(default_factory=list)

    def serialise(self) -> Dict[str, Any]:
        return {
            "start": self.start,
            "desired": self.desired,
            "no_deal": self.no_deal,
            "supplier_offer": self.supplier_offer,
            "history": list(self.history),
        }

    def snapshot_for_next_round(
        self, counter_price: Optional[float], round_no: int
    ) -> Dict[str, Any]:
        history = list(self.history)

        def _append(entry_type: str, value: Optional[float]) -> None:
            if value is None:
                return
            record = {
                "round": round_no,
                "type": entry_type,
                "value": value,
            }
            if not any(
                existing.get("round") == record["round"]
                and existing.get("type") == record["type"]
                and self._is_close(existing.get("value"), record["value"])
                for existing in history
            ):
                history.append(record)

        _append("supplier_offer", self.supplier_offer)
        _append("counter", counter_price)

        next_start = counter_price if counter_price is not None else self.start

        return {
            "start": next_start,
            "desired": self.desired,
            "no_deal": self.no_deal,
            "supplier_offer": self.supplier_offer,
            "history": history,
            "last_counter": counter_price if counter_price is not None else next_start,
        }

    @staticmethod
    def _is_close(value_a: Any, value_b: Any, *, tolerance: float = 1e-6) -> bool:
        try:
            return abs(float(value_a) - float(value_b)) <= tolerance
        except (TypeError, ValueError):
            return False
