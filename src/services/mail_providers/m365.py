"""Microsoft 365 as a mail provider: a thin wrapper over the existing code."""
from __future__ import annotations

from typing import Any, List

from services.mail_m365 import admin, sending
from services.mail_m365.config import load_settings


class M365Provider:
    name = "m365"
    label = "Microsoft 365"
    method = "Your IT team approves Procwise once in Microsoft 365 and limits it to chosen shared mailboxes."

    def size_limit(self) -> int:
        return sending.MAX_MIME_BYTES

    def status(self, conn: Any) -> dict:
        return admin.status(conn, settings=load_settings())

    def prove(self, conn: Any) -> List[dict]:
        return []  # filled in by the evidence task

    def send(self, binding: Any, mime: bytes) -> None:
        sending.send_via_graph(binding, mime)
