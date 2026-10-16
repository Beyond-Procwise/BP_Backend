"""What every email provider offers the rest of Procwise.

A thin seam: the screens and the sender ask a provider for these four things and
never name Microsoft directly, so a second provider can be added without touching them.
"""
from __future__ import annotations

from typing import Any, List, Protocol


class MailProvider(Protocol):
    name: str      # short code, e.g. "m365"
    label: str     # what a person sees, e.g. "Microsoft 365"
    method: str    # plain-English description of how Procwise connects

    def size_limit(self) -> int:
        """Largest message, in bytes, this provider will accept."""

    def status(self, conn: Any) -> dict:
        """The connection and its mailboxes, for the Email page."""

    def prove(self, conn: Any) -> List[dict]:
        """Evidence rows: {check, answer, result, at}."""

    def send(self, binding: Any, mime: bytes) -> None:
        """Send one message through the binding's mailbox; raises on any failure."""
