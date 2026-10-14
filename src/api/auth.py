"""Who is asking. Cognito ID-token verification for the ask endpoints.

The UI has been sending a Cognito ID token on ``Authorization`` since it was
built (``services/api.js`` sets it on the axios defaults at login) and this
service ignored it: ``/workflows/ask`` declared no dependency, the app added no
authentication middleware, and the ``user_id`` in the request body scoped only
the conversation history and the cache key. Anyone able to reach the port got
answers computed over the whole corpus.

This module verifies that token properly — signature against the pool's JWKS,
then issuer, audience, expiry and ``token_use`` — and hands the route a
principal. PyJWT does the decoding rather than a hand-rolled parser, because the
classic JWT failures (accepting ``alg: none``, or an HMAC token verified against
an RSA public key) come from doing that by hand.

What this does NOT do, so nobody reads more into it than is there:

* It does not scope retrieval. There is no tenant dimension in the corpus to
  scope by — no ``customer_id`` on the vector payloads or the ``bp_`` tables —
  and the ``x-customer-id`` header the UI sends is the constant "001". Every
  authenticated caller still sees the same corpus.
* It consults no policy on whether a given fact may be shown to a given user.
"""

from __future__ import annotations

import os

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import jwt
from fastapi import Header, HTTPException, Request
from jwt import PyJWKClient

logger = logging.getLogger(__name__)

_JWKS_TTL_SECONDS = 3600


class AuthError(Exception):
    """The caller could not be identified. Carries no detail for the client."""


class AuthNotConfigured(AuthError):
    """Enforcement was asked for and cannot be performed.

    Distinct from a bad token: nothing the caller sends can succeed. Both
    refuse — the difference is only in what the transport reports, 503 over
    HTTP against 401.
    """


@dataclass(frozen=True)
class Principal:
    """The authenticated caller."""

    subject: str
    email: Optional[str] = None
    username: Optional[str] = None
    claims: Optional[Dict[str, Any]] = None


class CognitoVerifier:
    """Verify a Cognito ID token against a user pool.

    ``jwks_loader`` exists so the tests can supply a key set directly; in the
    running service it fetches and caches the pool's published keys.
    """

    def __init__(
        self,
        *,
        issuer: str,
        audience: str,
        jwks_loader: Optional[Callable[[], Dict[str, Any]]] = None,
        leeway_seconds: int = 30,
    ) -> None:
        self._issuer = issuer.rstrip("/")
        self._audience = audience
        self._leeway = leeway_seconds
        self._jwks_loader = jwks_loader
        self._jwk_client: Optional[PyJWKClient] = None
        self._cached: Optional[Dict[str, Any]] = None
        self._cached_at = 0.0
        self._lock = threading.Lock()

    # -- key material ----------------------------------------------------
    def _keyset(self) -> Dict[str, Any]:
        with self._lock:
            fresh = self._cached is not None and (time.time() - self._cached_at) < _JWKS_TTL_SECONDS
            if not fresh:
                self._cached = self._jwks_loader()  # type: ignore[misc]
                self._cached_at = time.time()
            return self._cached  # type: ignore[return-value]

    def _signing_key(self, token: str):
        if self._jwks_loader is not None:
            from jwt import PyJWK

            try:
                kid = jwt.get_unverified_header(token).get("kid")
            except Exception as exc:  # noqa: BLE001
                raise AuthError("malformed token header") from exc
            for entry in self._keyset().get("keys", []):
                if entry.get("kid") == kid:
                    return PyJWK.from_dict(entry).key
            raise AuthError("signing key not found for token")

        if self._jwk_client is None:
            self._jwk_client = PyJWKClient(
                f"{self._issuer}/.well-known/jwks.json", cache_keys=True
            )
        try:
            return self._jwk_client.get_signing_key_from_jwt(token).key
        except Exception as exc:  # noqa: BLE001
            raise AuthError("signing key not found for token") from exc

    # -- verification ----------------------------------------------------
    def verify(self, token: str) -> Principal:
        token = (token or "").strip()
        if not token:
            raise AuthError("no token supplied")

        key = self._signing_key(token)
        try:
            claims = jwt.decode(
                token,
                key=key,
                # RS256 only. Naming the algorithm is what stops a token that
                # claims "alg": "none", or an HMAC token signed with the public
                # key as its secret, from being accepted.
                algorithms=["RS256"],
                audience=self._audience,
                issuer=self._issuer,
                leeway=self._leeway,
                options={"require": ["exp", "iss", "aud", "sub"]},
            )
        except Exception as exc:  # noqa: BLE001 - every failure is the same to the caller
            raise AuthError(str(exc)) from exc

        # An access token also validates against this pool but names no user;
        # only an ID token identifies a person.
        if claims.get("token_use") != "id":
            raise AuthError("token_use is not 'id'")

        subject = str(claims.get("sub") or "").strip()
        if not subject:
            raise AuthError("token carries no subject")

        return Principal(
            subject=subject,
            email=claims.get("email"),
            username=claims.get("cognito:username") or claims.get("username"),
            claims=claims,
        )


# ----------------------------------------------------------------------------
# FastAPI wiring
# ----------------------------------------------------------------------------

_verifier: Optional[CognitoVerifier] = None
_mode: str = "unconfigured"


def configure(settings: Any) -> str:
    """Build the verifier from settings. Returns the active mode.

    ``enforce`` is the default. ``off`` is deliberately explicit and logs a
    warning on every startup: an unauthenticated ask endpoint should never be
    something a deployment falls into quietly.
    """

    global _verifier, _mode

    mode = str(getattr(settings, "ask_auth_mode", None) or "enforce").strip().lower()
    if mode not in {"enforce", "off"}:
        mode = "enforce"

    pool = str(getattr(settings, "cognito_user_pool_id", "") or "").strip()
    client = str(getattr(settings, "cognito_app_client_id", "") or "").strip()
    region = str(getattr(settings, "cognito_region", "") or "").strip()

    if mode == "off":
        _verifier, _mode = None, "off"
        logger.warning(
            "ASK AUTH DISABLED (ask_auth_mode=off): /workflows/ask will answer "
            "unauthenticated callers. Set ASK_AUTH_MODE=enforce to require a token."
        )
        return _mode

    if not (pool and client and region):
        # Fail closed. Configured to enforce but unable to, so the endpoint
        # refuses rather than serving the corpus to anyone who asks.
        _verifier, _mode = None, "misconfigured"
        logger.error(
            "ask_auth_mode=enforce but Cognito settings are incomplete "
            "(region=%r pool=%r client=%r) — /workflows/ask will refuse every request",
            region, pool, client,
        )
        return _mode

    _verifier = CognitoVerifier(
        issuer=f"https://cognito-idp.{region}.amazonaws.com/{pool}",
        audience=client,
    )
    _mode = "enforce"
    logger.info("ask auth: enforcing Cognito ID tokens for pool %s", pool)
    return _mode


def auth_mode() -> str:
    return _mode


def _active_verifier() -> Optional[CognitoVerifier]:
    """The verifier to check a token against, or ``None`` when auth is off.

    The single place ASK_AUTH_MODE is interpreted. Every entry point — the HTTP
    dependency below, the WebSocket handshake in ``routers/ws.py`` — reads the
    mode through here, so a surface cannot end up meaning something different
    by the setting than the rest of the product does.

    Raises ``AuthNotConfigured`` when enforcement was asked for but could not be
    set up: that state refuses, it does not fall open.
    """

    if _mode == "off":
        return None
    if _mode != "enforce" or _verifier is None:
        raise AuthNotConfigured("authentication is not configured")
    return _verifier


def principal_from_token(token: Optional[str]) -> Optional[Principal]:
    """Identify a caller from a raw token, for handshakes that carry no header.

    A browser cannot set ``Authorization`` on a WebSocket upgrade, so the token
    arrives as a query parameter instead. The mode semantics are identical to
    ``require_user``; only the transport differs.

    Returns ``None`` only when auth is explicitly switched off. Raises
    ``AuthError`` on every failure to identify the caller.
    """

    verifier = _active_verifier()
    if verifier is None:
        return None

    token = (token or "").strip()
    if not token:
        raise AuthError("no token supplied")
    return verifier.verify(token)


def require_user(request: Request) -> Optional[Principal]:
    """FastAPI dependency for the ask endpoints.

    ``request`` must be annotated ``Request``. Annotated loosely it is read as a
    query parameter instead of being injected, and every call fails validation
    with "query.request: Field required" before authentication is even reached.

    Raises 401 when a caller cannot be identified, 503 when enforcement was
    asked for but could not be configured. Returns ``None`` only when auth is
    explicitly switched off.
    """

    try:
        verifier = _active_verifier()
    except AuthNotConfigured as exc:
        raise HTTPException(status_code=503, detail="authentication is not configured") from exc
    if verifier is None:
        return None

    header = (request.headers.get("authorization") or "").strip()
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise HTTPException(status_code=401, detail="a bearer token is required")

    try:
        return verifier.verify(token)
    except AuthError as exc:
        # Logged in full, returned as a single generic line: a precise reason
        # ("expired", "wrong audience") tells an attacker which knob to turn.
        logger.info("ask auth rejected a token: %s", exc)
        raise HTTPException(status_code=401, detail="invalid or expired token") from exc


# ----------------------------------------------------------------------------
# Pre-existing shared-secret check, kept as it was.
#
# Nothing imports it today, but it is a different mechanism for a different
# caller — a service-to-service key rather than a signed user identity — and
# removing it is not part of authenticating the ask path. Note it is
# open-by-default: with PROCWISE_API_KEY unset it admits everyone, which is why
# the ask endpoints use require_user above and not this.
# ----------------------------------------------------------------------------


def verify_api_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = os.getenv("PROCWISE_API_KEY")
    if not expected:
        return  # auth disabled when no key configured (current default)
    if x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
