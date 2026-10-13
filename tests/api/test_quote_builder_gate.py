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


# --- templates ---------------------------------------------------------------------

TEMPLATE_PERMIT = {
    "policyId": "quote_template_write_authority",
    "policyName": "QuoteTemplateWriteAuthorityPolicy",
    "details": {"policy_identifier": "quote_template_write_authority",
                "applies_to": ["quote_template.write"], "required_role": "Admin",
                "rules": {"effect": "allow"}},
    "raw_row": {"version": 1},
}
_TEMPLATE = {"name": "T", "lines": [{"key": "work", "description": "Work"}]}


@pytest.fixture
def templates(serve, monkeypatch):
    monkeypatch.setattr(qb.quote_templates, "list_templates",
                        lambda c, **k: [{"key": "t", "lines": []}])
    saved = []
    monkeypatch.setattr(qb.quote_templates, "save",
                        lambda c, key, **k: saved.append(k["actor"]) or {"key": key})
    serve.saved = saved
    return serve


def test_a_viewer_can_see_the_templates_because_they_hold_no_figures(templates):
    assert templates(viewer()).get("/spendiq/quote-templates").status_code == 200


@pytest.mark.parametrize("who, status", [(None, 403), (viewer(), 403), (buyer(), 403),
                                         (admin(), 200)],
                         ids=["nobody", "viewer", "buyer", "admin"])
def test_templates_are_changed_only_by_an_admin(templates, who, status):
    r = templates(who, TEMPLATE_PERMIT).put("/spendiq/quote-templates/t", json=_TEMPLATE)
    assert r.status_code == status, r.text
    assert templates.saved == (["sub-admin"] if status == 200 else [])


def test_without_its_permit_row_not_even_an_admin_changes_a_template(templates):
    assert templates(admin()).put("/spendiq/quote-templates/t", json=_TEMPLATE).status_code == 403
    assert templates.saved == []


def test_a_viewer_cannot_start_a_quote_from_a_template(serve, monkeypatch):
    started = []
    monkeypatch.setattr(qb.builder, "create", lambda *a, **k: started.append(k) or {})
    r = serve(viewer(), MARGIN_PERMIT).post(
        "/spendiq/quotes/from-template", json={"templateKey": "t", "currency": "GBP"})
    assert r.status_code == 403 and started == []


def test_starting_a_quote_from_a_template_needs_the_margin_permit_like_any_quote_read(serve, monkeypatch):
    """What comes back is a quote in the builder's shape, cost fields and all. A buyer may
    create quotes by default, so only the permit stands in the way here."""
    started = []
    monkeypatch.setattr(qb.quote_templates, "get",
                        lambda c, key: {"key": key, "status": "active", "lines": []})
    monkeypatch.setattr(qb.builder, "create", lambda *a, **k: started.append(k) or {})
    body = {"templateKey": "t", "currency": "GBP"}
    assert serve(buyer()).post("/spendiq/quotes/from-template", json=body).status_code == 403
    assert started == []


# --- logos -------------------------------------------------------------------------

@pytest.mark.parametrize("who, status", [(None, 403), (viewer(), 403), (buyer(), 403),
                                         (admin(), 200)],
                         ids=["nobody", "viewer", "buyer", "admin"])
def test_a_logo_is_uploaded_and_removed_only_by_an_admin(serve, monkeypatch, who, status):
    done = []
    monkeypatch.setattr(qb.brand_logos, "add", lambda c, **k: done.append("add") or {"id": 1})
    monkeypatch.setattr(qb.brand_logos, "remove", lambda c, i, **k: done.append("remove"))
    monkeypatch.setattr(qb.entity_profile, "load", lambda c: {"logoId": None})
    monkeypatch.setattr(qb, "get_conn", lambda: contextlib.nullcontext(type(
        "C", (), {"rollback": lambda self: None})()))
    client = serve(who, PROFILE_PERMIT)
    assert client.post("/entity/logos", files={"file": ("l.png", b"x", "image/png")}).status_code == status
    assert client.delete("/entity/logos/1").status_code == status
    assert done == (["add", "remove"] if status == 200 else [])
