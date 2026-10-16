"""Microsoft 365 as a mail provider: a thin wrapper over the existing code."""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, List, Optional

import jwt

from services.mail_m365 import admin, sending
from services.mail_m365.config import load_settings, read_integration

logger = logging.getLogger(__name__)

_PERMISSIONS_CHECK = "Mail permissions Microsoft has granted Procwise’s app"

# Microsoft's answer is kept briefly so reading the Email page does not ask Microsoft's
# sign-in every time. A check that could not run is kept only a few seconds, so a passing
# slow spell is not shown as "not proven" for long. The rows keep the time of the actual
# check, so the screen always says when Microsoft was really asked.
PROOF_TTL_SECONDS = 120
UNKNOWN_TTL_SECONDS = 5
_proof_lock = threading.Lock()
_proof_cache: dict = {}   # tenant id -> (expires at, rows)


def _reset_proof_cache() -> None:
    with _proof_lock:
        _proof_cache.clear()


def mail_roles(token: str) -> List[str]:
    """The ``Mail.*`` application permissions in OUR OWN access token's ``roles`` claim.

    The signature is not checked: the token came straight from Microsoft to us, and it is
    only read to report what Microsoft granted us, never to let anyone in."""
    claims = jwt.decode(token, options={"verify_signature": False})
    roles = claims.get("roles") or []
    if isinstance(roles, str):
        roles = [roles]
    return sorted({str(r) for r in roles if str(r).lower().startswith("mail.")})


class M365Provider:
    name = "m365"
    label = "Microsoft 365"
    method = "Your IT team approves Procwise once in Microsoft 365 and limits it to chosen shared mailboxes."

    def __init__(self, client_factory: Optional[Callable[[str], Any]] = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._client_factory = client_factory
        self._clock = clock

    def size_limit(self) -> int:
        return sending.MAX_MIME_BYTES

    def status(self, conn: Any) -> dict:
        return admin.status(conn, settings=load_settings())

    def prove(self, conn: Any) -> List[dict]:
        """Ask Microsoft what our app holds. Never raises: anything that stops the check
        is a row with result ``unknown``, so nothing is shown as proven that was not.

        With ``conn=None`` a short connection of its own is used only to read the
        integration, and is closed before Microsoft is asked."""
        at = datetime.now(timezone.utc).isoformat()

        def row(answer: str, result: str) -> dict:
            return {"restriction": "no_mail_permissions", "check": _PERMISSIONS_CHECK,
                    "answer": answer, "result": result, "at": at}

        try:
            integration = self._integration(conn)
            tenant = integration.get("tenant_id")
            if integration.get("status") != "connected" or not tenant:
                return [row("Microsoft 365 is not connected", "unknown")]
            settings = load_settings()
            if settings is None and self._client_factory is None:
                return [row("Microsoft 365 is not set up on this server", "unknown")]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Microsoft 365 connection could not be read: %s", type(exc).__name__)
            return [row("The connection details could not be read", "unknown")]

        key = str(tenant)
        now = self._clock()
        with _proof_lock:
            hit = _proof_cache.get(key)
        if hit is not None and now < hit[0]:
            return [dict(r) for r in hit[1]]
        rows = self._ask_microsoft(tenant, settings, row)
        ttl = UNKNOWN_TTL_SECONDS if any(r["result"] == "unknown" for r in rows) else PROOF_TTL_SECONDS
        with _proof_lock:
            _proof_cache[key] = (now + ttl, [dict(r) for r in rows])
        return rows

    @staticmethod
    def _integration(conn: Any) -> dict:
        if conn is not None:
            return read_integration(conn)
        from services.db import get_conn
        with get_conn() as own:
            return read_integration(own)

    def _ask_microsoft(self, tenant: str, settings: Any, row: Callable[[str, str], dict]
                       ) -> List[dict]:
        try:
            client = (self._client_factory or admin._default_factory(settings))(tenant)
            token = client.token()
        except Exception as exc:  # noqa: BLE001 - an unanswered check is "not proven"
            logger.warning("Microsoft 365 permission check could not run: %s", type(exc).__name__)
            return [row("Microsoft did not answer, so this could not be checked", "unknown")]
        try:
            found = mail_roles(token)
        except Exception:  # noqa: BLE001 - a token we cannot read proves nothing
            logger.warning("Microsoft 365 access token could not be read")
            return [row("Microsoft’s answer could not be read", "unknown")]
        if found:
            return [row(", ".join(found), "fail")]
        return [row("none", "pass")]

    def send(self, binding: Any, mime: bytes) -> None:
        sending.send_via_graph(binding, mime)
