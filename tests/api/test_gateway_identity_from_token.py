"""N8 part 1: the gateway-only routes take the user's identity from their OWN token, not from headers.

The agent-policy and compliance routes are mounted for the gateway alone: a shared key proves the call
came through the gateway, and three headers (x-user-sub, x-user-groups, x-user-email) said who the
user was. Anyone holding the key could therefore be anyone. Now, while sign-in is enforced, the key
still proves the path but the identity comes from the bearer token the gateway forwards, verified by
the same verifier every other route uses; the headers are ignored. Only with sign-in switched off
(local/dev) do the headers still name the caller, as before.
"""
from __future__ import annotations

import json
import os
import sys

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import api.auth as auth
import api.routers.agent_policies as ap
from api.auth import AuthError, Principal

KEY = "k-test"


class _Verifier:
    def verify(self, token):
        if token != "good":
            raise AuthError("bad token")
        return Principal(subject="sub-from-token", email="t@x.io", claims={"cognito:groups": ["bp-buyers"]})


@pytest.fixture()
def app(monkeypatch):
    monkeypatch.setenv("AGENT_POLICY_GATEWAY_KEY", KEY)
    a = FastAPI()

    @a.get("/who")
    def who(p=__import__("fastapi").Depends(ap.gateway_principal)):
        return {"subject": p.subject, "groups": list((p.claims or {}).get("cognito:groups") or [])}
    return TestClient(a)


def _headers(key=KEY, sub="sub-from-header", bearer=None):
    h = {"x-gateway-key": key, "x-user-sub": sub, "x-user-groups": json.dumps(["bp-admins"])}
    if bearer:
        h["authorization"] = f"Bearer {bearer}"
    return h


def test_enforced_identity_comes_from_the_token_and_the_headers_are_ignored(app, monkeypatch):
    monkeypatch.setattr(auth, "_active_verifier", lambda: _Verifier())
    r = app.get("/who", headers=_headers(bearer="good"))
    assert r.status_code == 200, r.text
    assert r.json() == {"subject": "sub-from-token", "groups": ["bp-buyers"]}, "not the header's admin"


def test_enforced_key_alone_is_not_an_identity(app, monkeypatch):
    monkeypatch.setattr(auth, "_active_verifier", lambda: _Verifier())
    assert app.get("/who", headers=_headers()).status_code == 401
    assert app.get("/who", headers=_headers(bearer="forged")).status_code == 401


def test_a_token_without_the_key_is_still_refused(app, monkeypatch):
    monkeypatch.setattr(auth, "_active_verifier", lambda: _Verifier())
    assert app.get("/who", headers=_headers(key="nope", bearer="good")).status_code == 401


def test_sign_in_off_keeps_the_header_path(app, monkeypatch):
    monkeypatch.setattr(auth, "_active_verifier", lambda: None)
    r = app.get("/who", headers=_headers())
    assert r.status_code == 200 and r.json()["subject"] == "sub-from-header"
    assert app.get("/who", headers=_headers(sub="")).status_code == 401


def test_enforced_but_unconfigured_refuses_rather_than_trusting_headers(app, monkeypatch):
    def unconfigured():
        raise auth.AuthNotConfigured("no pool")
    monkeypatch.setattr(auth, "_active_verifier", unconfigured)
    assert app.get("/who", headers=_headers()).status_code == 503
