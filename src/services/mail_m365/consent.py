"""Microsoft's admin-consent round trip.

The callback is the one endpoint Microsoft calls with no bearer token, so it trusts only
what we signed: `state` is HMAC-signed, names the admin who started, lives ten minutes,
and its nonce can be spent once. Consent creates our app's identity in the customer's
tenant and grants no mail access — that is granted in Exchange (spec §2.2).
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional, Tuple
from urllib.parse import urlencode

from services.mail_m365.config import (M365Settings, consume_nonce, read_integration,
                                       redirect_uri, write_integration)

ADMIN_CONSENT_URL = "https://login.microsoftonline.com/organizations/v2.0/adminconsent"
_SCOPE = "https://graph.microsoft.com/.default"
_KEEP_NONCES = 20


class StateRejected(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(f"consent state rejected: {code}")
        self.code = code


class ConsentNotGiven(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(f"Microsoft did not grant approval: {code}")
        self.code = code


def _b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode().rstrip("=")


def _unb64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def sign_state(key: bytes, actor: str, nonce: str, now: float) -> str:
    body = _b64(json.dumps({"a": actor, "n": nonce, "t": int(now)}).encode())
    sig = _b64(hmac.new(key, body.encode(), hashlib.sha256).digest())
    return f"{body}.{sig}"


def verify_state(key: bytes, state: str, now: float, max_age: int = 600) -> Tuple[str, str]:
    try:
        body, sig = state.split(".", 1)
        payload = json.loads(_unb64(body))
    except Exception:
        raise StateRejected("malformed")
    expected = _b64(hmac.new(key, body.encode(), hashlib.sha256).digest())
    if not hmac.compare_digest(expected, sig):
        raise StateRejected("bad_signature")
    if now - float(payload.get("t", 0)) > max_age:
        raise StateRejected("expired")
    return str(payload["a"]), str(payload["n"])


def begin(conn: Any, settings: M365Settings, key: bytes, actor: str, *,
          now: Callable[[], float] = time.time) -> str:
    nonce = secrets.token_urlsafe(16)
    current = read_integration(conn)
    pending = (list(current.get("pending_nonces") or []) + [nonce])[-_KEEP_NONCES:]
    write_integration(conn, {**current, "pending_nonces": pending}, actor=actor)
    return ADMIN_CONSENT_URL + "?" + urlencode({
        "client_id": settings.client_id,
        "redirect_uri": redirect_uri(settings),
        "scope": _SCOPE,
        "state": sign_state(key, actor, nonce, now()),
    })


def complete(conn: Any, settings: M365Settings, key: bytes, *, state: str,
             tenant: Optional[str], admin_consent: Optional[str], error: Optional[str],
             now: Callable[[], float] = time.time) -> Dict[str, Any]:
    actor, nonce = verify_state(key, state, now())
    if not consume_nonce(conn, nonce, actor=actor):
        raise StateRejected("reused")
    current = read_integration(conn)
    if error or str(admin_consent).lower() != "true" or not tenant:
        raise ConsentNotGiven(error or "not_granted")
    updated = {**current, "provider": "m365",
               "tenant_id": tenant, "status": "connected", "consented_by": actor,
               "consented_at": datetime.fromtimestamp(now(), tz=timezone.utc).isoformat()}
    return write_integration(conn, updated, actor=actor)


_KEY_CACHE: Dict[str, bytes] = {}


def load_state_key(settings: M365Settings, *, client: Any = None) -> bytes:
    if settings.state_key_arn in _KEY_CACHE:
        return _KEY_CACHE[settings.state_key_arn]
    if client is None:  # pragma: no cover - real AWS only
        from src.services import egress
        client = egress.aws_client("secretsmanager", purpose=egress.Purpose.SECRETS,
                                   region_name=settings.state_key_arn.split(":")[3])
    raw = client.get_secret_value(SecretId=settings.state_key_arn)["SecretString"]
    key = raw.encode()
    if len(key) < 32:
        raise ValueError("the state signing key must be at least 32 bytes")
    _KEY_CACHE[settings.state_key_arn] = key
    return key
