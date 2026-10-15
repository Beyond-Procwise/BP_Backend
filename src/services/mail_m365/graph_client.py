"""Microsoft Graph for company shared mailboxes: read the Inbox, fetch a message's raw
MIME, send a MIME message. Nothing else is reachable — no move, delete, flag or
mark-read — because we promise the customer we never change their mailbox (spec §5.2).
The allowlist below is checked on every call, and a test tries the paths it refuses.
"""
from __future__ import annotations

import base64
import re
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import quote

from services.style.graph_source import (GRAPH_BASE, GRAPH_SCOPE, TOKEN_URL, GraphCredentials,
                                         MailboxAccessDenied, MailboxUnreachable,
                                         RequestsTransport)

_MBX = r"/users/[^/]+"
ALLOWED_PATH_PATTERNS = {
    "GET": [re.compile(p) for p in (
        rf"^{_MBX}/mailFolders/inbox$",
        rf"^{_MBX}/mailFolders/inbox/messages$",
        rf"^{_MBX}/messages$",
        rf"^{_MBX}/messages/[^/]+/\$value$",
    )],
    "POST": [re.compile(rf"^{_MBX}/sendMail$")],
}


class Throttled(MailboxUnreachable):
    def __init__(self, retry_after: int) -> None:
        super().__init__(f"throttled by Microsoft; retry after {retry_after}s")
        self.retry_after = retry_after


class M365Client:
    def __init__(self, credentials: GraphCredentials, *, transport: Optional[Any] = None,
                 clock: Callable[[], float] = time.time) -> None:
        self.credentials = credentials
        self.transport = transport or RequestsTransport()
        self._clock = clock
        self._token: Optional[str] = None
        self._token_expires = 0.0

    def token(self) -> str:
        if self._token and self._clock() < self._token_expires - 60:
            return self._token
        c = self.credentials
        status, payload = self.transport.post_form(
            TOKEN_URL.format(tenant_id=c.tenant_id),
            {"client_id": c.client_id, "client_secret": c.client_secret,
             "scope": GRAPH_SCOPE, "grant_type": "client_credentials"})
        if status != 200 or not isinstance(payload, dict) or not payload.get("access_token"):
            raise MailboxUnreachable(
                f"token request failed with status {status}: "
                f"{(payload or {}).get('error_description', 'no detail')}"
                if isinstance(payload, dict) else f"token request failed with status {status}")
        self._token = payload["access_token"]
        self._token_expires = self._clock() + float(payload.get("expires_in", 3600))
        return self._token

    def _url(self, path: str, *, method: str) -> str:
        """The one place a Graph URL is built. Refuses anything off the allowlist."""
        bare = path.split("?", 1)[0]
        if ".." in bare or not any(p.match(bare) for p in ALLOWED_PATH_PATTERNS.get(method, [])):
            raise ValueError(f"{method} {path} is not a Graph path this feature may use")
        return f"{GRAPH_BASE}{path}"

    def _headers(self, **extra: str) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.token()}", "Accept": "application/json", **extra}

    @staticmethod
    def _mbx(mailbox: str) -> str:
        return f"/users/{quote(mailbox, safe='@')}"

    @staticmethod
    def _retry_after(payload: Any) -> int:
        try:
            return int((payload or {}).get("retry_after", 30))
        except (TypeError, ValueError, AttributeError):
            return 30

    def can_read(self, mailbox: str) -> Tuple[bool, str]:
        status, _ = self.transport.get(
            self._url(f"{self._mbx(mailbox)}/mailFolders/inbox", method="GET"), self._headers())
        if 200 <= status < 300:
            return True, f"read Inbox with HTTP {status}"
        if status in (401, 403, 404):
            return False, f"denied with HTTP {status}"
        raise MailboxUnreachable(f"Inbox read returned HTTP {status}; inconclusive")

    def probe_control(self, mailbox: str) -> Tuple[bool, str]:
        """Try to read a mailbox this credential must NOT reach. True means the grant is
        too wide. A transport failure is not a denial (same rule as GraphExemplarSource)."""
        status, _ = self.transport.get(
            self._url(f"{self._mbx(mailbox)}/messages", method="GET") + "?$top=1",
            self._headers())
        if status in (401, 403, 404):
            return False, f"denied with HTTP {status}"
        if 200 <= status < 300:
            return True, f"READ SUCCEEDED with HTTP {status}"
        raise MailboxUnreachable(f"control probe returned HTTP {status}; inconclusive")

    def list_inbox(self, mailbox: str, since: datetime, *, page_size: int = 50
                   ) -> List[Dict[str, Any]]:
        stamp = since.strftime("%Y-%m-%dT%H:%M:%SZ")
        url: Optional[str] = (
            self._url(f"{self._mbx(mailbox)}/mailFolders/inbox/messages", method="GET")
            + f"?$top={page_size}&$select=id,receivedDateTime"
            f"&$filter=receivedDateTime ge {stamp}&$orderby=receivedDateTime asc")
        out: List[Dict[str, Any]] = []
        while url:
            status, payload = self.transport.get(url, self._headers())
            if status == 429:
                raise Throttled(self._retry_after(payload))
            if status in (401, 403, 404):
                raise MailboxAccessDenied(f"Inbox listing denied with HTTP {status}")
            if not 200 <= status < 300:
                raise MailboxUnreachable(f"Inbox listing returned HTTP {status}")
            out.extend((payload or {}).get("value", []))
            url = (payload or {}).get("@odata.nextLink")
            if url and not url.startswith(GRAPH_BASE + "/"):
                raise ValueError("nextLink left Microsoft Graph")
        return sorted(out, key=lambda m: m.get("receivedDateTime") or "")

    def get_mime(self, mailbox: str, message_id: str) -> bytes:
        status, body = self.transport.get_bytes(
            self._url(f"{self._mbx(mailbox)}/messages/{quote(message_id, safe='')}/$value",
                      method="GET"), self._headers())
        if status == 429:
            raise Throttled(30)
        if status in (401, 403, 404):
            raise MailboxAccessDenied(f"message fetch denied with HTTP {status}")
        if not 200 <= status < 300:
            raise MailboxUnreachable(f"message fetch returned HTTP {status}")
        return body

    def send_mime(self, mailbox: str, mime: bytes) -> None:
        status, payload = self.transport.post_body(
            self._url(f"{self._mbx(mailbox)}/sendMail", method="POST"),
            self._headers(**{"Content-Type": "text/plain"}), base64.b64encode(mime))
        if 200 <= status < 300:
            return
        if status == 429:
            raise Throttled(self._retry_after(payload))
        if status in (401, 403):
            raise MailboxAccessDenied(f"send denied with HTTP {status}")
        raise MailboxUnreachable(f"send returned HTTP {status}")
