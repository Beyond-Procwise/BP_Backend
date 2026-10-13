"""The Quote builder's reads carry the seller's cost, so they are closed the way the
internal read is: through the REAL gate, with only the policy engine and the audit
writer faked. tests/api/test_quote_builder_router.py fakes the gate and proves the
routes ask; this proves what the answer is.
"""
from __future__ import annotations

import contextlib
from decimal import Decimal as D

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routers import quote_builder as qb
from src.services import agent_actions, rbac
from tests.api.test_sales_margin_gate import MARGIN_PERMIT, buyer, viewer
from tests.guardrails.test_guardrail_gate import engine_with
from tests.guardrails.test_rbac import FakePrincipal

READS = ["/spendiq/quotes/9", "/spendiq/deals/D1/quotes", "/spendiq/quotes/unlinked",
         "/spendiq/quotes/review-queue"]
_QUOTE = {"id": "9", "category": None, "lines": [{"id": "1", "cost": D("125.0000")}]}
_BODY = {"currency": "GBP", "validUntil": "2099-01-01", "lines": []}


def admin():
    return FakePrincipal("sub-admin", {"cognito:groups": ["bp-admins"]})


@pytest.fixture
def serve(monkeypatch):
    fetched, audit = [], []
    monkeypatch.setattr(agent_actions, "record_action_or_fail", lambda **k: audit.append(k))
    monkeypatch.setattr(qb, "get_conn", lambda: contextlib.nullcontext("CONN"))
    monkeypatch.setattr(qb.pricing_checks, "for_quote", lambda category: {})
    monkeypatch.setattr(qb.builder, "get", lambda c, q: fetched.append(q) or dict(_QUOTE))
    for name in ("list_for_deal", "list_unlinked", "review_queue"):
        monkeypatch.setattr(qb.builder, name, lambda c, *a: fetched.append(a) or [dict(_QUOTE)])
    monkeypatch.setattr(qb.builder, "save", lambda c, q, **k: fetched.append(k) or dict(_QUOTE))
    monkeypatch.setattr(qb.entity_profile, "save",
                        lambda c, profile, actor: fetched.append(actor) or profile)

    def _serve(principal, *policies):
        monkeypatch.setattr(rbac, "policy_engine", lambda: engine_with(*policies))
        app = FastAPI()
        app.include_router(qb.router)
        app.dependency_overrides[qb.require_user] = lambda: principal
        client = TestClient(app)
        client.fetched, client.audit = fetched, audit
        return client

    return _serve


@pytest.mark.parametrize("path", READS)
@pytest.mark.parametrize("who", [None, viewer()], ids=["nobody", "viewer"])
def test_cost_does_not_reach_a_caller_without_the_permit(serve, path, who):
    client = serve(who, MARGIN_PERMIT)
    r = client.get(path)
    assert r.status_code == 403, r.text
    assert "125" not in r.text and client.fetched == []


@pytest.mark.parametrize("path", READS)
def test_a_buyer_with_the_permit_reads_the_quote(serve, path):
    r = serve(buyer(), MARGIN_PERMIT).get(path)
    assert r.status_code == 200 and "125.0000" in r.text


@pytest.mark.parametrize("path", READS)
def test_without_the_permit_row_nobody_reads_it(serve, path):
    client = serve(buyer())
    assert client.get(path).status_code == 403
    assert client.fetched == []


def test_a_viewer_cannot_save_a_quote(serve):
    client = serve(viewer(), MARGIN_PERMIT)
    assert client.put("/spendiq/quotes/9", json=_BODY).status_code == 403
    assert client.fetched == []


def test_a_buyer_cannot_approve_without_a_stated_authority(serve):
    """Approval is a transact action: refused unless a policy says who may."""
    client = serve(buyer(), MARGIN_PERMIT)
    assert client.post("/spendiq/quotes/9/approve", json={}).status_code == 403


PROFILE_PERMIT = {
    "policyId": "entity_profile_write_authority",
    "policyName": "EntityProfileWriteAuthorityPolicy",
    "details": {"policy_identifier": "entity_profile_write_authority",
                "applies_to": ["entity_profile.write"], "required_role": "Admin",
                "rules": {"effect": "allow"}},
    "raw_row": {"version": 1},
}


@pytest.mark.parametrize("who, status", [(None, 403), (viewer(), 403), (buyer(), 403),
                                         (admin(), 200)],
                         ids=["nobody", "viewer", "buyer", "admin"])
def test_the_company_profile_is_changed_only_by_an_admin(serve, who, status):
    client = serve(who, PROFILE_PERMIT)
    r = client.put("/entity/profile", json={"legalName": "Northwind Ltd"})
    assert r.status_code == status, r.text
    assert client.fetched == (["sub-admin"] if status == 200 else [])


def test_without_its_permit_row_not_even_an_admin_changes_the_profile(serve):
    client = serve(admin())
    assert client.put("/entity/profile", json={"legalName": "X"}).status_code == 403
    assert client.fetched == []
