"""Microsoft Graph for company shared mailboxes: read the Inbox, fetch a message's raw
MIME, send a MIME message. Nothing else is reachable — no move, delete, flag or
mark-read — because we promise the customer we never change their mailbox (spec §5.2).
The allowlist below is checked on every call, and a test tries the paths it refuses.
"""
from __future__ import annotations

import base64
import json
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import quote

import jwt

from services.style.graph_source import (GRAPH_BASE, GRAPH_SCOPE, TOKEN_URL,
                                         MailboxAccessDenied, MailboxUnreachable,
                                         RequestsTransport)

ASSERTION_TYPE = "urn:ietf:params:oauth:client-assertion-type:jwt-bearer"
CERT_TTL_SECONDS = 600
_CERT_FIELDS = ("client_id", "private_key_pem", "thumbprint_sha1_b64url")


@dataclass(frozen=True)
class CertCredentials:
    """Our app's identity in one customer's tenant: the app id plus the certificate's
    private key and SHA-1 thumbprint. There is no client secret anywhere in this path.
    The key is kept out of repr so it cannot reach a log line or a traceback."""
    tenant_id: str
    client_id: str
    private_key_pem: str = field(repr=False)
    x5t: str


def client_assertion(client_id: str, tenant_id: str, private_key_pem: str, x5t: str,
                     now: float) -> str:
    """A short-lived JWT, signed with our certificate's private key, that proves to
    Microsoft's token endpoint for `tenant_id` that this request comes from our app."""
    claims = {"aud": f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token",
              "iss": client_id, "sub": client_id, "jti": uuid.uuid4().hex,
              "nbf": int(now) - 5, "exp": int(now) + 300}
    return jwt.encode(claims, private_key_pem, algorithm="RS256", headers={"x5t": x5t})


_cert_lock = threading.Lock()
_cert_cache: Dict[str, Dict[str, Any]] = {}


def _reset_certificate_cache() -> None:
    with _cert_lock:
        _cert_cache.clear()


def load_app_certificate(cert_arn: str, *, tenant_id: str, client: Any = None
                         ) -> CertCredentials:
    """Our certificate from Secrets Manager, as credentials for `tenant_id`.

    The secret is JSON {"client_id", "private_key_pem", "thumbprint_sha1_b64url"}. It is
    read once per process and reused for CERT_TTL_SECONDS, so a rotated certificate is
    picked up within that time without a Secrets Manager call on every token.
    """
    if not str(cert_arn or "").startswith("arn:aws:secretsmanager:"):
        raise ValueError("the certificate reference must be a Secrets Manager ARN")
    now = time.monotonic()
    with _cert_lock:
        hit = _cert_cache.get(cert_arn)
    if hit is None or now - hit["at"] >= CERT_TTL_SECONDS:
        if client is None:  # pragma: no cover - real AWS only
            from src.services import egress
            client = egress.aws_client("secretsmanager", purpose=egress.Purpose.SECRETS,
                                       region_name=cert_arn.split(":")[3])
        raw = client.get_secret_value(SecretId=cert_arn).get("SecretString")
        if not raw:
            raise ValueError(f"secret {cert_arn} has no SecretString payload")
        data = json.loads(raw) if isinstance(raw, str) else dict(raw)
        missing = [k for k in _CERT_FIELDS if not str(data.get(k) or "").strip()]
        if missing:
            raise ValueError(f"secret {cert_arn} is missing {', '.join(missing)}")
        hit = {"at": now, "data": {k: str(data[k]) for k in _CERT_FIELDS}}
        with _cert_lock:
            _cert_cache[cert_arn] = hit
    d = hit["data"]
    return CertCredentials(tenant_id=tenant_id, client_id=d["client_id"],
                           private_key_pem=d["private_key_pem"],
                           x5t=d["thumbprint_sha1_b64url"])

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


class MessageTooLarge(Exception):
    """The message is over what Graph's sendMail accepts. Nothing was sent."""

    def __init__(self, size: int, limit: int = 0) -> None:
        super().__init__(f"message of {size} bytes is over the sendMail limit"
                         + (f" of {limit} bytes" if limit else ""))
        self.size, self.limit = size, limit


class M365Client:
    def __init__(self, credentials: CertCredentials, *, transport: Optional[Any] = None,
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
            {"client_id": c.client_id, "client_assertion_type": ASSERTION_TYPE,
             "client_assertion": client_assertion(c.client_id, c.tenant_id,
                                                  c.private_key_pem, c.x5t, self._clock()),
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
        if ".." in bare or not any(p.fullmatch(bare) for p in ALLOWED_PATH_PATTERNS.get(method, [])):
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
        if status == 401:
            # Microsoft rejected OUR token, which says nothing about this mailbox. Treating it
            # as a denial would read as every customer withdrawing access at once.
            raise MailboxUnreachable("Microsoft rejected our sign-in (HTTP 401)")
        if status in (403, 404):
            return False, f"denied with HTTP {status}"
        raise MailboxUnreachable(f"Inbox read returned HTTP {status}; inconclusive")

    def probe_control(self, mailbox: str) -> Tuple[bool, str]:
        """Try to read a mailbox this credential must NOT reach. True means the grant is
        too wide. A transport failure is not a denial (same rule as GraphExemplarSource)."""
        status, _ = self.transport.get(
            self._url(f"{self._mbx(mailbox)}/messages", method="GET") + "?$top=1",
            self._headers())
        if status == 403:
            return False, "denied with HTTP 403"
        if 200 <= status < 300:
            return True, f"READ SUCCEEDED with HTTP {status}"
        # Only 403 proves the grant is narrow. 404 usually means the control address does
        # not exist (Graph says 404 for unknown users) and 401 means our own token is bad;
        # neither shows the grant is restricted, so neither may pass the proof.
        if status == 404:
            raise MailboxUnreachable("the control mailbox was not found - check the address")
        if status == 401:
            raise MailboxUnreachable("Microsoft rejected our sign-in (HTTP 401)")
        raise MailboxUnreachable(f"control probe returned HTTP {status}; inconclusive")

    def list_inbox(self, mailbox: str, since: datetime, *, page_size: int = 50
                   ) -> List[Dict[str, Any]]:
        """Every Inbox message received at or after `since`, oldest first.

        Pages are fetched with URLs WE build ($top/$skip on the same $filter/$orderby, each
        through the allowlist); the server's @odata.nextLink is never followed, so nothing
        Microsoft returns can steer a request, and a nextLink encoded differently from our
        own path cannot stall the listing at the first page.
        """
        # A naive datetime is taken to be UTC; an aware one is converted, so a +01:00 cursor
        # is not written as local wall-clock time labelled Z (which would skip messages).
        since = since.replace(tzinfo=timezone.utc) if since.tzinfo is None else since
        stamp = since.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        base = self._url(f"{self._mbx(mailbox)}/mailFolders/inbox/messages", method="GET")
        out: List[Dict[str, Any]] = []
        skip = 0
        while True:
            url = (base + f"?$top={page_size}&$skip={skip}&$select=id,receivedDateTime"
                   f"&$filter=receivedDateTime ge {stamp}&$orderby=receivedDateTime asc")
            status, payload = self.transport.get(url, self._headers())
            if status == 429:
                raise Throttled(self._retry_after(payload))
            if status in (401, 403, 404):
                raise MailboxAccessDenied(f"Inbox listing denied with HTTP {status}")
            if not 200 <= status < 300:
                raise MailboxUnreachable(f"Inbox listing returned HTTP {status}")
            page = (payload or {}).get("value", []) or []
            out.extend(page)
            if len(page) < page_size:
                break
            skip += page_size
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
        if status == 413:
            raise MessageTooLarge(len(mime))
        if status in (401, 403):
            raise MailboxAccessDenied(f"send denied with HTTP {status}")
        raise MailboxUnreachable(f"send returned HTTP {status}")
