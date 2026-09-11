"""The session WebSocket must know who is connecting.

`WS /ws/session/{session_id}` took `session_id` and `websocket` and nothing
else. There was no token parameter and no verification anywhere in the handler,
so anyone who could reach the port and knew — or guessed — a session id was
handed that session's document outcomes: supplier names, deal names, per-file
processing results.

Two comments in the codebase asserted the opposite. `api/main.py` said "Auth
handled inside ws.py via token= query param" and the router guard excluded
`/ws/` as "authenticated on a token= query parameter instead". Neither was
true; both are corrected alongside these tests.

The mode semantics must match `require_user` exactly. A WebSocket that ignored
ASK_AUTH_MODE would be the one surface in the product where the setting means
something different, including the case that matters most: enforcement asked
for but impossible to configure must refuse, not fall open.
"""
from __future__ import annotations

import base64
import time
from contextlib import contextmanager
from unittest.mock import patch

import pytest

pytest.importorskip("fastapi")
jwt = pytest.importorskip("jwt")

from cryptography.hazmat.primitives.asymmetric import rsa  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from starlette.websockets import WebSocketDisconnect  # noqa: E402

from api.auth import CognitoVerifier  # noqa: E402

ISSUER = "https://cognito-idp.eu-west-1.amazonaws.com/eu-west-1_3rDtdvAh1"
AUDIENCE = "2qqm091jr3otl1b7k2f9uo7qeo"
KID = "test-key-1"
SESSION = "SESSION-001"


# ---------------------------------------------------------------------------
# key material — a throwaway RSA pair, so nothing here touches Cognito
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture(scope="module")
def jwks(key):
    numbers = key.public_key().public_numbers()

    def b64(value: int) -> str:
        raw = value.to_bytes((value.bit_length() + 7) // 8, "big")
        return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()

    return {"keys": [{"kty": "RSA", "kid": KID, "use": "sig", "alg": "RS256",
                      "n": b64(numbers.n), "e": b64(numbers.e)}]}


def _token(key, **overrides) -> str:
    now = int(time.time())
    claims = {
        "iss": ISSUER, "aud": AUDIENCE, "token_use": "id",
        "sub": "user-123", "email": "nick@example.com",
        "exp": now + 3600, "iat": now,
    }
    claims.update(overrides)
    return jwt.encode(claims, key, algorithm="RS256", headers={"kid": KID})


# ---------------------------------------------------------------------------
# the app under test — the real router, mounted the way api.main mounts it
# ---------------------------------------------------------------------------
@pytest.fixture()
def client():
    from src.api.routers import ws as ws_router

    app = FastAPI()
    app.include_router(ws_router.router)
    return TestClient(app)


@contextmanager
def _auth(mode: str, verifier=None):
    """Set api.auth's module state directly and put it back.

    The directory conftest forces 'off' for every test here; these tests are
    about the mode, so each one states the mode it needs.
    """
    from api import auth as _mod

    previous = (_mod._mode, _mod._verifier)
    _mod._mode, _mod._verifier = mode, verifier
    try:
        yield
    finally:
        _mod._mode, _mod._verifier = previous


@pytest.fixture()
def enforcing(jwks):
    with _auth("enforce", CognitoVerifier(issuer=ISSUER, audience=AUDIENCE,
                                          jwks_loader=lambda: jwks)):
        yield


@contextmanager
def _resolved(payload):
    """Stand in for the catch-up query so no test needs a database."""
    with patch("src.api.routers.ws._get_resolved_session", return_value=payload):
        yield


def _connect(client, query: str = ""):
    return client.websocket_connect(f"/ws/session/{SESSION}{query}")


# ---------------------------------------------------------------------------
# enforce
# ---------------------------------------------------------------------------
def test_a_connection_with_no_token_is_refused(client, enforcing):
    """The exposure itself: no credential, no connection."""
    with _resolved({"session_id": SESSION}):
        with pytest.raises(WebSocketDisconnect) as refused:
            with _connect(client):
                pass

    assert refused.value.code == 1008


def test_a_connection_with_an_invalid_token_is_refused(client, enforcing):
    with _resolved({"session_id": SESSION}):
        with pytest.raises(WebSocketDisconnect) as refused:
            with _connect(client, "?token=not-a-real-token"):
                pass

    assert refused.value.code == 1008


def test_a_connection_with_an_expired_token_is_refused(client, enforcing, key):
    now = int(time.time())
    stale = _token(key, exp=now - 60, iat=now - 3600)

    with _resolved({"session_id": SESSION}):
        with pytest.raises(WebSocketDisconnect) as refused:
            with _connect(client, f"?token={stale}"):
                pass

    assert refused.value.code == 1008


def test_a_valid_token_connects_and_is_served(client, enforcing, key):
    payload = {"session_id": SESSION, "action_status": "completed"}

    with _resolved(payload):
        with _connect(client, f"?token={_token(key)}") as socket:
            assert socket.receive_json() == payload


# ---------------------------------------------------------------------------
# the other two modes — the WebSocket must not be the surface where
# ASK_AUTH_MODE means something different
# ---------------------------------------------------------------------------
def test_auth_off_admits_a_caller_without_a_token(client):
    """Exactly what require_user does in 'off' mode: returns no principal and
    lets the request through. The local demo runs this way."""
    payload = {"session_id": SESSION, "action_status": "completed"}

    with _auth("off", None), _resolved(payload):
        with _connect(client) as socket:
            assert socket.receive_json() == payload


def test_enforcement_that_could_not_be_configured_refuses(client, key):
    """'misconfigured' is what configure() sets when enforce was asked for and
    the Cognito settings are incomplete. A valid token cannot be checked, so
    the connection is refused rather than admitted."""
    with _auth("misconfigured", None), _resolved({"session_id": SESSION}):
        with pytest.raises(WebSocketDisconnect) as refused:
            with _connect(client, f"?token={_token(key)}"):
                pass

    assert refused.value.code == 1008


# ---------------------------------------------------------------------------
# the handler must not run before the caller is known
# ---------------------------------------------------------------------------
def test_the_session_is_never_read_for_an_unauthenticated_caller(client, enforcing):
    """A refusal that queried the session first would still have done the
    work, and a slow refusal is a probe oracle."""
    with patch("src.api.routers.ws._get_resolved_session") as query:
        with pytest.raises(WebSocketDisconnect):
            with _connect(client):
                pass

    query.assert_not_called()
