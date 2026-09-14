"""Internal cost and margin reach only a caller a stated policy permits.

Through the REAL gate (endpoint_gate.require -> guardrail.authorize -> rbac), with
only the policy engine and the audit writer faked. The router tests fake the gate
itself, so they prove the router asks; these prove the answer is no.

The case that motivated this: with ASK_AUTH_MODE=off, require_user returns no
principal, and GET /sales/quotes/{id} asked no gate -- so cost and margin were
served to anyone who could reach the port.
"""
from __future__ import annotations

import contextlib
import datetime as dt
from decimal import Decimal as D

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routers import sales as sr
from src.services import agent_actions, rbac
from tests.guardrails.test_guardrail_gate import engine_with
from tests.guardrails.test_rbac import FakePrincipal

MARGIN_PERMIT = {
    "policyId": "sales_margin_read_authority",
    "policyName": "SalesMarginReadAuthorityPolicy",
    "details": {
        "policy_identifier": "sales_margin_read_authority",
        "applies_to": ["margin.read"],
        "required_role": "Buyer",
        "rules": {"effect": "allow"},
    },
    "raw_row": {"version": 1},
}

MARGIN_PATHS = ["/sales/quotes/9", "/sales/opportunities", "/sales/opportunities/7"]

_QUOTE = {"sales_quote_id": 9, "quote_ref": "SQ-1", "status": "approved", "account_name": "A",
          "contact_name": None, "currency": "GBP", "quote_date": dt.date(2026, 9, 11),
          "valid_until": dt.date(2026, 10, 11), "total_ex_tax": D("56.00"),
          "total_cost": D("40.00"), "total_margin": D("16.00"), "margin_pct": D("0.2857"),
          "lines": [], "justifications": []}


def buyer():
    return FakePrincipal("sub-buyer", {"cognito:groups": ["bp-buyers"]})


def viewer():
    return FakePrincipal("sub-viewer", {"cognito:groups": ["bp-viewers"]})


@pytest.fixture
def serve(monkeypatch):
    fetched, audit = [], []
    monkeypatch.setattr(agent_actions, "record_action_or_fail", lambda **k: audit.append(k))
    monkeypatch.setattr(sr, "get_conn", lambda: contextlib.nullcontext("CONN"))
    monkeypatch.setattr(sr.quotes, "get_quote", lambda c, q: fetched.append(q) or dict(_QUOTE))
    monkeypatch.setattr(sr.opportunities, "get_opportunity",
                        lambda c, o: fetched.append(o) or {"expected_margin": D("16.00")})
    monkeypatch.setattr(sr.opportunities, "list_opportunities",
                        lambda c, **k: fetched.append(k) or [{"expected_margin": D("16.00")}])

    def _serve(principal, *policies):
        monkeypatch.setattr(rbac, "policy_engine", lambda: engine_with(*policies))
        app = FastAPI()
        app.include_router(sr.router)
        app.dependency_overrides[sr.require_user] = lambda: principal
        client = TestClient(app)
        client.fetched, client.audit = fetched, audit
        return client

    return _serve


@pytest.mark.parametrize("path", MARGIN_PATHS)
def test_no_principal_gets_no_margin(serve, path):
    """ASK_AUTH_MODE=off: require_user yields None. This is the live exposure."""
    client = serve(None, MARGIN_PERMIT)
    r = client.get(path)
    assert r.status_code == 403, r.text
    assert client.fetched == []


@pytest.mark.parametrize("path", MARGIN_PATHS)
def test_a_viewer_gets_no_margin(serve, path):
    client = serve(viewer(), MARGIN_PERMIT)
    assert client.get(path).status_code == 403
    assert client.fetched == []


@pytest.mark.parametrize("path", MARGIN_PATHS)
def test_a_buyer_with_the_permit_gets_margin(serve, path):
    client = serve(buyer(), MARGIN_PERMIT)
    r = client.get(path)
    assert r.status_code == 200, r.text
    assert "16.00" in r.text


@pytest.mark.parametrize("path", MARGIN_PATHS)
def test_without_the_permit_row_nobody_gets_margin(serve, path):
    """A missing policy row must withhold margin, not fall open to the read default."""
    client = serve(buyer())
    r = client.get(path)
    assert r.status_code == 403, r.text
    assert client.fetched == []
    assert ("margin.read", "denied") in [(a["action_type"], a["status"]) for a in client.audit]


def test_the_customer_view_stays_open_and_is_audited(serve):
    client = serve(None, MARGIN_PERMIT)
    r = client.get("/sales/quotes/9/customer")
    assert r.status_code == 200, r.text
    assert "16.00" not in r.text
    assert [(a["action_type"], a["status"]) for a in client.audit] == [("quote.read", "allowed")]
