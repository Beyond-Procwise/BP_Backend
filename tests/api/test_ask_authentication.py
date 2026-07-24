"""The ask endpoints must know who is asking.

`/workflows/ask` and `/ask/stream` declared no auth dependency, the app added no
authentication middleware, and `user_id` in the request body scoped nothing but
the conversation history and the cache key. Anyone who could reach the port got
answers computed over the whole corpus, and the identity the UI was already
sending — a Cognito ID token on `Authorization` — was ignored.

These tests pin the verifier. They generate a throwaway RSA key and serve its
public half as a JWKS, so nothing here touches the network or Cognito.
"""

import base64
import json
import os
import sys
import time

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

jwt = pytest.importorskip("jwt")
from cryptography.hazmat.primitives.asymmetric import rsa  # noqa: E402

from api.auth import AuthError, CognitoVerifier  # noqa: E402

ISSUER = "https://cognito-idp.eu-west-1.amazonaws.com/eu-west-1_3rDtdvAh1"
AUDIENCE = "2qqm091jr3otl1b7k2f9uo7qeo"
KID = "test-key-1"


@pytest.fixture(scope="module")
def key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture(scope="module")
def jwks(key):
    numbers = key.public_key().public_numbers()

    def b64(value: int) -> str:
        raw = value.to_bytes((value.bit_length() + 7) // 8, "big")
        return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()

    return {
        "keys": [
            {
                "kty": "RSA",
                "kid": KID,
                "use": "sig",
                "alg": "RS256",
                "n": b64(numbers.n),
                "e": b64(numbers.e),
            }
        ]
    }


@pytest.fixture()
def verifier(jwks):
    return CognitoVerifier(
        issuer=ISSUER,
        audience=AUDIENCE,
        jwks_loader=lambda: jwks,
    )


def _token(key, **overrides) -> str:
    now = int(time.time())
    claims = {
        "iss": ISSUER,
        "aud": AUDIENCE,
        "token_use": "id",
        "sub": "user-123",
        "email": "nick@example.com",
        "exp": now + 3600,
        "iat": now,
    }
    claims.update(overrides)
    for empty in [k for k, v in claims.items() if v is None]:
        claims.pop(empty)
    return jwt.encode(claims, key, algorithm="RS256", headers={"kid": KID})


def test_a_valid_token_identifies_the_caller(verifier, key):
    principal = verifier.verify(_token(key))

    assert principal.subject == "user-123"
    assert principal.email == "nick@example.com"


def test_an_expired_token_is_rejected(verifier, key):
    now = int(time.time())
    with pytest.raises(AuthError):
        verifier.verify(_token(key, exp=now - 60, iat=now - 3600))


def test_a_token_from_another_pool_is_rejected(verifier, key):
    with pytest.raises(AuthError):
        verifier.verify(_token(key, iss="https://cognito-idp.eu-west-1.amazonaws.com/eu-west-1_someoneelse"))


def test_a_token_for_another_client_is_rejected(verifier, key):
    with pytest.raises(AuthError):
        verifier.verify(_token(key, aud="a-different-app-client"))


def test_an_access_token_is_not_accepted_where_an_id_token_is_required(verifier, key):
    """token_use separates the two; an access token names no user."""
    with pytest.raises(AuthError):
        verifier.verify(_token(key, token_use="access"))


def test_a_token_signed_by_a_different_key_is_rejected(verifier):
    other = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    with pytest.raises(AuthError):
        verifier.verify(_token(other))


def test_an_unsigned_token_is_rejected(verifier):
    """alg=none is the classic bypass: the signature is simply omitted."""
    header = base64.urlsafe_b64encode(json.dumps({"alg": "none", "kid": KID}).encode()).rstrip(b"=")
    payload = base64.urlsafe_b64encode(
        json.dumps({"iss": ISSUER, "aud": AUDIENCE, "token_use": "id", "sub": "x",
                    "exp": int(time.time()) + 3600}).encode()
    ).rstrip(b"=")
    with pytest.raises(AuthError):
        verifier.verify(f"{header.decode()}.{payload.decode()}.")


def test_garbage_and_empty_tokens_are_rejected(verifier):
    for value in ("", "   ", "not-a-token", "a.b.c"):
        with pytest.raises(AuthError):
            verifier.verify(value)


def test_an_unknown_kid_is_rejected(verifier, key):
    token = jwt.encode({"iss": ISSUER, "aud": AUDIENCE, "token_use": "id", "sub": "x",
                        "exp": int(time.time()) + 3600},
                       key, algorithm="RS256", headers={"kid": "some-other-kid"})
    with pytest.raises(AuthError):
        verifier.verify(token)
