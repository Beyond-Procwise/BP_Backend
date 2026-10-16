"""Microsoft's admin-consent round trip, then a sign-in that says which organisation it was.

The callback is the one endpoint Microsoft calls with no bearer token, so it trusts only
what we signed: `state` is HMAC-signed, names the admin who started, lives ten minutes,
and its nonce can be spent once. Consent creates our app's identity in the customer's
tenant and grants no mail access — that is granted in Exchange (spec §2.2).

The `tenant` Microsoft appends to the consent return is NOT evidence of anything: it is a
query parameter anyone can edit, and a forged one would connect this deployment to
another customer's tenant (our multitenant app's token for that tenant is valid). So the
consent return only starts step two: the admin signs in to Microsoft (authorization code,
scope `openid` — no mail permission), we redeem the code ourselves with a client assertion
signed by our certificate (there is no client secret), verify the id_token (Microsoft's
signature, our client id, issuer = its tenant, time window, our nonce, and that the person
holds one of the Microsoft administrator roles that may connect an app) and take the
tenant from its verified `tid` claim, never from the URL.
Both steps ride one chain of single-use nonces: the consent nonce is spent to issue the
sign-in nonce, and the sign-in nonce is spent before the code is redeemed.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import re
import secrets
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Mapping, Optional, Tuple
from urllib.parse import urlencode

import jwt

from services.mail_m365.config import (M365Settings, append_nonce, consume_nonce,
                                       merge_integration, redirect_uri)
from services.mail_m365.graph_client import (ASSERTION_TYPE, CertCredentials,
                                             client_assertion, load_app_certificate)

logger = logging.getLogger(__name__)

LOGIN_BASE = "https://login.microsoftonline.com"
ADMIN_CONSENT_URL = f"{LOGIN_BASE}/organizations/v2.0/adminconsent"
AUTHORIZE_URL = LOGIN_BASE + "/{tenant}/oauth2/v2.0/authorize"
SIGNIN_TOKEN_URL = LOGIN_BASE + "/{tenant}/oauth2/v2.0/token"
OPENID_CONFIG_URL = LOGIN_BASE + "/{tenant}/v2.0/.well-known/openid-configuration"
_SCOPE = "https://graph.microsoft.com/.default"
SIGNIN_SCOPE = "openid"
_KEEP_NONCES = 20
_LEEWAY = 120          # seconds of clock skew allowed on exp/nbf
GUID = re.compile(r"[0-9a-fA-F]{8}-([0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}")

# Directory roles (the id_token's `wids`) allowed to connect Procwise: Global Administrator,
# Privileged Role Administrator, Cloud Application Administrator, Application Administrator.
ADMIN_ROLE_IDS = frozenset({"62e90394-69f5-4237-9190-012177145e10",
                            "e8611ab8-c189-46e8-94e1-60213ab1f814",
                            "158c047a-c907-4556-b7ef-446551a6b5f7",
                            "9b895d92-2cd3-44c7-9d02-a6ac2d5ea5c3"})

PHASE_CONSENT = "consent"
PHASE_SIGNIN = "signin"


class StateRejected(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(f"consent state rejected: {code}")
        self.code = code


class ConsentNotGiven(Exception):
    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"Microsoft did not grant approval: {code}"
                         + (f" ({detail})" if detail else ""))
        self.code = code
        self.detail = detail


def _b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode().rstrip("=")


def _unb64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def sign_state(key: bytes, actor: str, nonce: str, now: float, *,
               phase: str = PHASE_CONSENT, tenant: Optional[str] = None) -> str:
    payload: Dict[str, Any] = {"a": actor, "n": nonce, "t": int(now), "p": phase}
    if tenant:
        payload["c"] = tenant
    body = _b64(json.dumps(payload).encode())
    sig = _b64(hmac.new(key, body.encode(), hashlib.sha256).digest())
    return f"{body}.{sig}"


def _verify(key: bytes, state: str, now: float, max_age: int = 600) -> Dict[str, Any]:
    try:
        body, sig = state.split(".", 1)
    except Exception:
        raise StateRejected("malformed")
    expected = _b64(hmac.new(key, body.encode(), hashlib.sha256).digest()).encode()
    if not hmac.compare_digest(expected, sig.encode("utf-8", "replace")):
        raise StateRejected("bad_signature")
    try:
        payload = json.loads(_unb64(body))
        if not isinstance(payload, dict):
            raise ValueError
        out = {"actor": str(payload["a"]), "nonce": str(payload["n"]),
               "issued": float(payload["t"]), "phase": str(payload.get("p") or PHASE_CONSENT),
               "tenant": payload.get("c")}
    except Exception:
        raise StateRejected("malformed")
    if now - out["issued"] > max_age:
        raise StateRejected("expired")
    return out


def verify_state(key: bytes, state: str, now: float, max_age: int = 600) -> Tuple[str, str]:
    v = _verify(key, state, now, max_age)
    return v["actor"], v["nonce"]


def oidc_nonce(key: bytes, nonce: str) -> str:
    """The id_token nonce for a sign-in: derived from (not equal to) our state nonce."""
    return _b64(hmac.new(key, f"oidc:{nonce}".encode(), hashlib.sha256).digest())


def begin(conn: Any, settings: M365Settings, key: bytes, actor: str, *,
          now: Callable[[], float] = time.time) -> str:
    nonce = secrets.token_urlsafe(16)
    append_nonce(conn, nonce, actor=actor, keep=_KEEP_NONCES)
    return ADMIN_CONSENT_URL + "?" + urlencode({
        "client_id": settings.client_id,
        "redirect_uri": redirect_uri(settings),
        "scope": _SCOPE,
        "state": sign_state(key, actor, nonce, now()),
    })


def _spend(conn: Any, v: Mapping[str, Any]) -> None:
    if not consume_nonce(conn, v["nonce"], actor=v["actor"]):
        raise StateRejected("reused")


def complete_consent(conn: Any, settings: M365Settings, key: bytes, *, state: str,
                     tenant: Optional[str], admin_consent: Optional[str],
                     error: Optional[str], now: Callable[[], float] = time.time) -> str:
    """Microsoft's consent page came back. Store NOTHING about the tenant; return the
    sign-in URL that will establish it. The query's `tenant` only picks which
    organisation's sign-in page to show, and must agree with the verified one later."""
    v = _verify(key, state, now())
    if v["phase"] != PHASE_CONSENT:
        raise StateRejected("malformed")
    _spend(conn, v)
    if error or str(admin_consent).lower() != "true" or not tenant:
        raise ConsentNotGiven(error or "not_granted")
    # The tenant arrives on an unauthenticated request and goes into a URL path, so it
    # must be exactly a GUID.
    if not GUID.fullmatch(tenant):
        raise ConsentNotGiven("bad_tenant")
    nonce = secrets.token_urlsafe(16)
    append_nonce(conn, nonce, actor=v["actor"], keep=_KEEP_NONCES)
    return AUTHORIZE_URL.format(tenant=tenant.lower()) + "?" + urlencode({
        "client_id": settings.client_id,
        "response_type": "code",
        "redirect_uri": redirect_uri(settings),
        "response_mode": "query",
        "scope": SIGNIN_SCOPE,
        "state": sign_state(key, v["actor"], nonce, now(), phase=PHASE_SIGNIN,
                            tenant=tenant.lower()),
        "nonce": oidc_nonce(key, nonce),
    })


# ---- the sign-in: redeem the code, verify the id_token ---------------------------------

def _fail(detail: str) -> ConsentNotGiven:
    return ConsentNotGiven("signin_failed", detail)


def verify_id_token(id_token: str, *, client_id: str, expected_nonce: str,
                    transport: Any, now: float) -> Dict[str, Any]:
    """Verify a Microsoft identity-platform v2 id_token and return its claims.

    The tenant to fetch keys for is read from the unverified token only to find its
    issuer's OpenID configuration; nothing is believed until the signature (RS256, a key
    from that issuer's JWKS), audience, issuer, time window and nonce all check out, and
    the issuer must be exactly that tenant's v2 issuer. The signer must also hold one of
    ADMIN_ROLE_IDS (`wids`), else ConsentNotGiven("not_admin").
    """
    try:
        unverified = jwt.decode(id_token, options={"verify_signature": False})
        header = jwt.get_unverified_header(id_token)
    except jwt.PyJWTError as exc:
        raise _fail(f"unreadable id_token: {exc}")
    tid = str(unverified.get("tid") or "")
    if not GUID.fullmatch(tid):
        raise ConsentNotGiven("bad_tenant", "id_token tid is not a tenant id")
    tid = tid.lower()
    status, conf = transport.get(OPENID_CONFIG_URL.format(tenant=tid),
                                 {"Accept": "application/json"})
    if status != 200 or not isinstance(conf, dict) or not conf.get("jwks_uri"):
        raise _fail(f"OpenID configuration returned HTTP {status}")
    issuer = f"{LOGIN_BASE}/{tid}/v2.0"
    if str(conf.get("issuer", "")).lower() != issuer:
        raise _fail("OpenID configuration names a different issuer")
    if not str(conf["jwks_uri"]).startswith(LOGIN_BASE + "/"):
        raise _fail("JWKS is not on Microsoft's sign-in host")
    status, jwks = transport.get(conf["jwks_uri"], {"Accept": "application/json"})
    if status != 200 or not isinstance(jwks, dict):
        raise _fail(f"JWKS returned HTTP {status}")
    kid = header.get("kid")
    match = [k for k in jwks.get("keys", []) if isinstance(k, dict) and k.get("kid") == kid]
    if not kid or not match:
        raise _fail("id_token is signed with a key Microsoft does not publish")
    try:
        key = jwt.PyJWK(match[0], algorithm="RS256").key
        claims = jwt.decode(
            id_token, key, algorithms=["RS256"], audience=client_id, issuer=issuer,
            # exp/nbf are checked below against the injected clock.
            options={"verify_exp": False, "verify_nbf": False, "verify_iat": False,
                     "require": ["exp", "nbf", "aud", "iss", "tid", "nonce"]})
    except jwt.PyJWTError as exc:
        raise _fail(f"id_token rejected: {exc}")
    if str(claims["tid"]).lower() != tid:
        raise _fail("id_token tid does not match its issuer")
    wids = claims.get("wids") or []
    if not isinstance(wids, list) or not (
            {str(w).lower() for w in wids} & ADMIN_ROLE_IDS):
        raise ConsentNotGiven("not_admin",
                              "Only a Microsoft 365 administrator can connect Procwise")
    try:
        exp, nbf = float(claims["exp"]), float(claims["nbf"])
    except (TypeError, ValueError):
        raise _fail("id_token time claims are not numbers")
    if now > exp + _LEEWAY:
        raise _fail("id_token has expired")
    if now < nbf - _LEEWAY:
        raise _fail("id_token is not yet valid")
    if not hmac.compare_digest(str(claims["nonce"]), expected_nonce):
        raise _fail("id_token nonce is not the one this sign-in was given")
    return claims


def _default_app_credentials(settings: M365Settings) -> CertCredentials:
    # The tenant is not used here: the assertion is built for the tenant being redeemed.
    return load_app_certificate(settings.cert_arn, tenant_id="organizations")


def complete_signin(conn: Any, settings: M365Settings, key: bytes, *, state: str,
                    code: Optional[str], error: Optional[str],
                    transport: Any = None,
                    app_credentials: Optional[Callable[[M365Settings], CertCredentials]] = None,
                    now: Callable[[], float] = time.time) -> Dict[str, Any]:
    v = _verify(key, state, now())
    if v["phase"] != PHASE_SIGNIN:
        raise StateRejected("malformed")
    _spend(conn, v)
    if error or not code:
        raise ConsentNotGiven(error or "not_granted")
    claimed = str(v.get("tenant") or "")
    if not GUID.fullmatch(claimed):
        raise ConsentNotGiven("bad_tenant")
    if transport is None:
        from services.style.graph_source import RequestsTransport
        transport = RequestsTransport()
    cert = (app_credentials or _default_app_credentials)(settings)
    status, payload = transport.post_form(SIGNIN_TOKEN_URL.format(tenant=claimed), {
        "client_id": cert.client_id, "client_assertion_type": ASSERTION_TYPE,
        "client_assertion": client_assertion(cert.client_id, claimed, cert.private_key_pem,
                                             cert.x5t, now()),
        "code": code,
        "grant_type": "authorization_code", "redirect_uri": redirect_uri(settings),
        "scope": SIGNIN_SCOPE})
    if status != 200 or not isinstance(payload, dict) or not payload.get("id_token"):
        raise _fail(f"code redemption returned HTTP {status}")
    claims = verify_id_token(payload["id_token"], client_id=settings.client_id,
                             expected_nonce=oidc_nonce(key, v["nonce"]),
                             transport=transport, now=now())
    tenant = str(claims["tid"]).lower()
    if tenant != claimed:
        # Consent was given in one organisation, the admin signed in to another.
        raise ConsentNotGiven("wrong_account")
    merged = merge_integration(conn, {
        "provider": "m365", "tenant_id": tenant, "status": "connected",
        "consented_by": v["actor"],
        "consented_at": datetime.fromtimestamp(now(), tz=timezone.utc).isoformat()},
        actor=v["actor"], unless_other_tenant=tenant)
    if merged is None:   # already connected to a different tenant: Disconnect first
        raise ConsentNotGiven("tenant_mismatch")
    return merged


def handle_callback(conn: Any, settings: M365Settings, key: bytes, *, state: str,
                    tenant: Optional[str] = None, admin_consent: Optional[str] = None,
                    error: Optional[str] = None, code: Optional[str] = None,
                    transport: Any = None,
                    app_credentials: Optional[Callable[[M365Settings], CertCredentials]] = None,
                    now: Callable[[], float] = time.time) -> Optional[str]:
    """One callback URL serves both steps; the signed state says which.

    Returns the URL to send the browser to next (Microsoft's sign-in), or None once the
    tenant is verified and stored."""
    phase = _verify(key, state, now())["phase"]
    if phase == PHASE_SIGNIN:
        complete_signin(conn, settings, key, state=state, code=code, error=error,
                        transport=transport, app_credentials=app_credentials, now=now)
        return None
    return complete_consent(conn, settings, key, state=state, tenant=tenant,
                            admin_consent=admin_consent, error=error, now=now)


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
