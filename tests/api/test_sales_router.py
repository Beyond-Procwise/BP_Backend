import contextlib
import datetime as dt
from decimal import Decimal as D

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routers import sales as sr
from src.services import guardrail
from src.services.sell_side import quote_render as qr

CALLER, OTHER = "sub-real-caller", "sub-someone-else"

# What the gate returns when a stated policy permits the action.
_PERMITTED = guardrail.Decision(allowed=True, reason="permitted",
                                policy_id="sales_margin_read_authority",
                                policy_name="SalesMarginReadAuthorityPolicy")
# What it returns when nothing spoke to a read and the reversible-class default let it through.
_DEFAULTED = guardrail.Decision(allowed=True, reason="read is a reversible class",
                                policy_name="RoleDefinitionPolicy")


class _P:
    subject = CALLER


@pytest.fixture
def client(monkeypatch):
    gates = []

    def _gate(action, *a, **k):
        gates.append(action)
        return _PERMITTED

    monkeypatch.setattr(sr, "gate", _gate)
    monkeypatch.setattr(sr, "get_conn", lambda: contextlib.nullcontext("CONN"))
    app = FastAPI()
    app.include_router(sr.router)
    app.dependency_overrides[sr.require_user] = lambda: _P()
    c = TestClient(app)
    c.gates = gates
    return c


def _quote(status="approved"):
    return {"sales_quote_id": 9, "quote_ref": "SQ-1", "status": status, "account_name": "A",
            "contact_name": None, "currency": "GBP", "quote_date": dt.date(2026, 9, 11),
            "valid_until": dt.date(2026, 10, 11), "total_ex_tax": D("56.00"),
            "lines": [{"sales_quote_line_id": 91, "line_no": 1, "item_description": "W",
                       "currency": "GBP",
                       "quantity": D("4"), "unit_price": D("14"), "line_total": D("56.00")}],
            "justifications": []}


def _internal():
    return {"sales_quote_id": 9, "total_cost": D("40.00"), "total_margin": D("16.00"),
            "margin_pct": D("0.2857"),
            "lines": [{"sales_quote_line_id": 91, "line_no": 1, "unit_cost": D("10.37"),
                       "cost_tier_applied": None, "line_margin": D("16.00"),
                       "line_margin_pct": D("0.2857")}],
            "justifications": []}


@pytest.fixture
def both_records(monkeypatch):
    monkeypatch.setattr(sr.quotes, "get_quote", lambda conn, qid: _quote())
    monkeypatch.setattr(sr.quotes, "get_quote_internal", lambda conn, qid: _internal())


def _walk_keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _walk_keys(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_keys(v)


def test_a_quote_is_created_by_the_token_holder(client, monkeypatch):
    seen = {}
    monkeypatch.setattr(sr.quotes, "create_draft", lambda conn, **k: seen.update(k) or _quote("draft"))
    r = client.post("/sales/quotes", json={
        "account_id": "ACC-1", "currency": "GBP", "valid_until": "2026-10-11",
        "created_by": OTHER,
        "lines": [{"catalog_item_id": 1, "quantity": "4", "unit_price": "14"}]})
    assert r.status_code == 200, r.text
    assert seen["created_by"] == CALLER
    assert client.gates == ["sales_quote.create"]


def test_approval_asks_the_transact_gate_and_uses_the_token(client, monkeypatch):
    seen = {}
    monkeypatch.setattr(sr.quotes, "approve",
                        lambda conn, qid, approver: seen.update(approver=approver) or _quote())
    assert client.post("/sales/quotes/9/approve").status_code == 200
    assert seen["approver"] == CALLER
    assert client.gates == ["sales_quote.approve"]


def test_issuing_asks_the_communicate_gate(client, monkeypatch):
    monkeypatch.setattr(sr.quotes, "issue", lambda conn, qid, actor: _quote("issued"))
    r = client.post("/sales/quotes/9/issue")
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "issued"
    assert client.gates == ["sales_quote.issue"]


def test_the_customer_endpoint_carries_no_internal_field(client, both_records):
    r = client.get("/sales/quotes/9/customer")
    assert r.status_code == 200
    assert not set(_walk_keys(r.json())) & qr.INTERNAL_FIELDS


def test_the_customer_html_carries_no_cost(client, both_records):
    r = client.get("/sales/quotes/9/customer.html")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    assert "16.00" not in r.text          # line and total margin
    assert "10.37" not in r.text          # unit cost


def test_a_cost_figure_in_the_customer_record_stops_the_page(client, both_records, monkeypatch):
    """The value check runs on the route, not only in the renderer's own tests."""
    leaky = _quote()
    leaky["lines"][0]["item_description"] = "W, costs us 10.37"
    monkeypatch.setattr(sr.quotes, "get_quote", lambda conn, qid: leaky)
    for path in ("/sales/quotes/9/customer", "/sales/quotes/9/customer.html",
                 "/sales/quotes/9/customer.pdf"):
        with pytest.raises(qr.InternalFieldLeak, match="unit_cost"):
            client.get(path)


def test_a_draft_has_no_customer_view(client, monkeypatch):
    monkeypatch.setattr(sr.quotes, "get_quote", lambda conn, qid: _quote("draft"))
    monkeypatch.setattr(sr.quotes, "get_quote_internal", lambda conn, qid: _internal())
    assert client.get("/sales/quotes/9/customer").status_code == 409


def test_the_internal_view_says_margin_is_front_end_only(client, both_records):
    body = client.get("/sales/quotes/9/internal").json()
    assert body["total_margin"] == "16.00"
    assert body["lines"][0]["sales_quote_line_id"] == 91
    assert "front-end" in body["margin_note"]


def test_the_internal_view_asks_the_margin_gate(client, both_records):
    assert client.get("/sales/quotes/9/internal").status_code == 200
    assert client.gates == ["margin.read"]


def test_the_quote_read_is_the_customer_record_and_never_reads_the_internal_one(client, monkeypatch):
    """Two reads behind two permissions (QDA B1): the quote read asks quote.read, does not
    touch the internal reader at all, and returns no cost or margin."""
    monkeypatch.setattr(sr.quotes, "get_quote", lambda conn, qid: _quote())
    monkeypatch.setattr(sr.quotes, "get_quote_internal",
                        lambda conn, qid: pytest.fail("the quote read reached the internal record"))
    r = client.get("/sales/quotes/9")
    assert r.status_code == 200, r.text
    assert client.gates == ["quote.read"]
    assert not set(_walk_keys(r.json())) & qr.INTERNAL_FIELDS
    assert "margin_note" not in r.json()


def test_reading_an_account_asks_the_gate_as_the_token_holder(client, monkeypatch):
    seen = {}
    monkeypatch.setattr(sr, "gate", lambda action, principal, **k: seen.update(
        action=action, subject=principal.subject, context=k.get("context")))
    monkeypatch.setattr(sr.accounts, "get_account", lambda conn, aid: {"account_id": aid})
    assert client.get("/sales/accounts/ACC-1").status_code == 200
    assert seen == {"action": "account.read", "subject": CALLER,
                    "context": {"account_id": "ACC-1"}}


@pytest.mark.parametrize("path, service, fn", [
    ("/sales/opportunities", "opportunities", "list_opportunities"),
    ("/sales/opportunities/7", "opportunities", "get_opportunity"),
])
def test_opportunity_reads_carry_margin_so_they_ask_the_margin_gate(client, monkeypatch, path, service, fn):
    monkeypatch.setattr(getattr(sr, service), fn,
                        lambda *a, **k: [] if fn.startswith("list") else {"expected_margin": D("1")})
    assert client.get(path).status_code == 200
    assert client.gates == ["margin.read"]


def test_the_customer_pdf_is_the_same_checked_document(client, both_records):
    r = client.get("/sales/quotes/9/customer.pdf")
    assert r.status_code == 200 and r.headers["content-type"] == "application/pdf"
    assert r.content[:5] == b"%PDF-"


def test_the_customer_page_bans_scripts_in_its_header_too(client, both_records):
    r = client.get("/sales/quotes/9/customer.html")
    assert "default-src 'none'" in r.headers["content-security-policy"]


@pytest.mark.parametrize("path", ["/sales/quotes/9/customer.html", "/sales/quotes/9/customer.pdf"])
def test_a_quote_that_fails_its_gate_has_no_page_and_says_why(client, both_records, monkeypatch, path):
    wrong = _quote()
    wrong["total_ex_tax"] = D("50.00")
    monkeypatch.setattr(sr.quotes, "get_quote", lambda conn, qid: wrong)
    r = client.get(path)
    assert r.status_code == 409 and "QUOTE_TOTAL_MISMATCH" in r.json()["detail"]


@pytest.mark.parametrize("path", ["/sales/quotes/9/customer.html", "/sales/quotes/9/customer.pdf"])
def test_a_draft_has_no_page(client, monkeypatch, path):
    monkeypatch.setattr(sr.quotes, "get_quote", lambda conn, qid: _quote("draft"))
    monkeypatch.setattr(sr.quotes, "get_quote_internal", lambda conn, qid: _internal())
    assert client.get(path).status_code == 409


@pytest.mark.parametrize("path", ["/sales/quotes/9/customer", "/sales/quotes/9/customer.html",
                                  "/sales/quotes/9/customer.pdf"])
def test_the_customer_views_ask_the_quote_read_gate(client, both_records, path):
    assert client.get(path).status_code == 200
    assert client.gates == ["quote.read"]


@pytest.mark.parametrize("path", ["/sales/quotes/9/internal", "/sales/opportunities", "/sales/opportunities/7"])
def test_margin_is_withheld_when_only_the_reversible_default_allowed_it(client, monkeypatch, path):
    """A read nobody's policy spoke to is allowed for every role, anonymous included.
    Margin must not ride that default: no stated permit, no margin -- and no query."""
    fetched, audit = [], []
    monkeypatch.setattr(sr, "gate", lambda action, *a, **k: _DEFAULTED)
    monkeypatch.setattr(sr.quotes, "get_quote_internal", lambda *a, **k: fetched.append(a))
    monkeypatch.setattr(sr.opportunities, "get_opportunity", lambda *a, **k: fetched.append(a))
    monkeypatch.setattr(sr.opportunities, "list_opportunities", lambda *a, **k: fetched.append(a))
    monkeypatch.setattr(sr.agent_actions, "record_action_or_fail", lambda **k: audit.append(k))
    r = client.get(path)
    assert r.status_code == 403, r.text
    assert fetched == []
    assert [(a["action_type"], a["status"]) for a in audit] == [("margin.read", "denied")]


def test_an_outcome_is_recorded_by_the_token_holder(client, monkeypatch):
    seen = {}
    monkeypatch.setattr(sr.outcomes, "record_outcome", lambda conn, **k: seen.update(k) or {"outcome": "won"})
    client.post("/sales/quotes/9/outcome", json={"outcome": "won", "outcome_date": "2026-09-11",
                                                 "recorded_by": OTHER})
    assert seen["recorded_by"] == CALLER


def test_calibration_asks_the_compute_gate(client, monkeypatch):
    monkeypatch.setattr(sr.calibration, "calibrate", lambda conn: [])
    assert client.post("/sales/calibrate").status_code == 200
    assert client.gates == ["sales.calibrate"]


def test_money_leaves_as_a_string_not_a_float(client, both_records):
    body = client.get("/sales/quotes/9/customer").json()
    assert body["total_ex_tax"] == "56.00"
    assert body["lines"][0]["unit_price"] == "14"
