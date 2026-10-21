"""Email channel for notifications: a plain message to the person's own work address.

Off unless NOTIFICATION_EMAIL=on (it is the first thing here that leaves the building).
The text is the event's value-free sentence plus the reason they were told; no amounts, no
document contents. The link is included only when APP_BASE_URL says where the app lives.
"""
from __future__ import annotations

import logging
import os
from typing import Callable, Optional

from services.people.models import Person
from services.people.notifications import ChannelSkipped

logger = logging.getLogger(__name__)


def enabled() -> bool:
    return os.environ.get("NOTIFICATION_EMAIL", "off").strip().lower() in ("on", "1", "true")


def compose(payload: dict, base_url: str = "") -> tuple:
    title = str(payload.get("title") or "Notification").strip()
    lines = [title]
    if payload.get("detail"):
        lines += ["", str(payload["detail"]).strip()]
    link = str(payload.get("link") or "")
    if base_url and link.startswith("/"):
        lines += ["", f"Open it: {base_url.rstrip('/')}{link}"]
    lines += ["", "You can change which notifications you get in your notification settings."]
    return f"[ProcWise] {title}", "\n".join(lines)


def make_email_channel(address_for: Callable[[Person], Optional[str]],
                       send: Callable[[str, str, str], bool],
                       base_url: str = "") -> Callable[[Person, dict], None]:
    """address_for -> the person's address or None; send(to, subject, body) -> True when accepted."""
    def channel(person: Person, payload: dict) -> None:
        to = (address_for(person) or "").strip()
        if "@" not in to:
            raise ChannelSkipped("no_address")
        subject, body = compose(payload, base_url)
        if not send(to, subject, body):
            raise RuntimeError("email not accepted by the mail service")
    return channel


def ses_send(to: str, subject: str, body: str) -> bool:
    from types import SimpleNamespace

    from config.settings import Settings
    from services.email_service import EmailService
    settings = Settings()
    sender = getattr(settings, "ses_default_sender", None) or "noreply@procwise.co.uk"
    result = EmailService(SimpleNamespace(settings=settings)).send_email(
        subject=subject, body=body, recipients=[to], sender=sender)
    return bool(getattr(result, "success", result))


def pg_address_for(conn) -> Callable[[Person], Optional[str]]:
    def address_for(person: Person) -> Optional[str]:
        with conn.cursor() as cur:
            cur.execute("SELECT email FROM proc.bp_person WHERE tenant_id=%s AND subject=%s",
                        (person.tenant_id, person.subject))
            r = cur.fetchone()
        return r[0] if r else None
    return address_for
