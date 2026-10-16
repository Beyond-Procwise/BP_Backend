"""The MailSource contract.

Deliberately one method. Everything interpretive — parsing, typing, the safety
gate, hashing — happens downstream in shared code, so every adapter gets the
identical treatment. An adapter able to vary the gate would be a hole in it.

Spec §4.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterator, Mapping, Optional, Protocol


@dataclass(frozen=True)
class RawMessage:
    raw: bytes
    provider_message_id: str
    received_at: Optional[datetime]
    source_ref: str


class MailSource(Protocol):
    def iter_messages(
        self, binding: Mapping[str, Any], since: Optional[datetime]
    ) -> Iterator[RawMessage]:
        """Yield every message at or after `since`. Never parse, never skip."""
        ...
