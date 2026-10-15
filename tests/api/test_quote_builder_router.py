"""The Quote builder routes: which permission each asks, what reaches the service, and
that who acted is never taken from the body."""
import contextlib

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.endpoint_gate import NotPermitted
from api.routers import quote_builder as qb
from src.services import guardrail

CALLER = "drafter@example.com"
_PERMITTED = guardrail.Decision(allowed=True, reason="stated", policy_name="P", policy_id=7)
QUOTE = {"id": "12", "category": "Platform / Enterprise", "lines": []}
BODY = {"currency": "GBP", "validUntil": "2099-01-01", "lines": [],
        "created_by": "someone.else", "status": "approved", "conditions": ["mine"]}


class _P:
    subject = CALLER


@pytest.fixture
def client(monkeypatch):
    asked, calls = [], []

    monkeypatch.setattr(qb, "gate", lambda action, *a, **k: asked.append(action) or _PERMITTED)
    monkeypatch.setattr(qb, "_require_margin_permit", lambda *a, **k: asked.append("margin.read"))
    monkeypatch.setattr(qb, "get_conn", lambda: contextlib.nullcontext("CONN"))
    monkeypatch.setattr(qb.pricing_checks, "for_quote",
                        lambda category: {"marginFloorPct": None, "for": category})

    def _record(name, result=QUOTE):
        def _f(*a, **k):
            calls.append((name, a, k))
            return result
        return _f

    for name in ("create", "save", "submit", "supersede", "get", "link"):
        monkeypatch.setattr(qb.builder, name, _record(name))
    for name in ("list_for_deal", "list_unlinked", "review_queue"):
        monkeypatch.setattr(qb.builder, name, _record(name, [QUOTE]))
    for name in ("approve", "reject", "withdraw"):
        monkeypatch.setattr(qb.quotes, name, _record(name))

    app = FastAPI()
    app.include_router(qb.router)
    app.dependency_overrides[qb.require_user] = lambda: _P()
    c = TestClient(app)
    c.asked, c.calls = asked, calls
    return c


def _call(client, name):
    return next(c for c in client.calls if c[0] == name)


@pytest.mark.parametrize("method, path, body, asks", [
    ("get", "/spendiq/deals/D1/quotes", None, ["margin.read"]),
    ("get", "/spendiq/quotes/unlinked", None, ["margin.read"]),
    ("get", "/spendiq/quotes/review-queue", None, ["margin.read"]),
    ("get", "/spendiq/quotes/12", None, ["margin.read"]),
    ("post", "/spendiq/deals/D1/quotes", BODY, ["sales_quote.create", "margin.read"]),
    ("post", "/spendiq/quotes", BODY, ["sales_quote.create", "margin.read"]),
    ("put", "/spendiq/quotes/12", BODY, ["sales_quote.price", "margin.read"]),
    ("post", "/spendiq/quotes/12/supersede", None, ["sales_quote.create", "margin.read"]),
    ("post", "/spendiq/quotes/12/link", {"dealId": "D1"}, ["sales.write", "margin.read"]),
    ("post", "/spendiq/quotes/12/review", None, ["sales.write", "margin.read"]),
    ("post", "/spendiq/quotes/12/withdraw", None, ["sales.write", "margin.read"]),
    ("post", "/spendiq/quotes/12/approve", {"conditions": ["x"]}, ["sales_quote.approve", "margin.read"]),
    ("post", "/spendiq/quotes/12/reject", {"reason": "no"}, ["sales_quote.approve", "margin.read"]),
])
def test_every_route_asks_its_permission_and_the_margin_permit(client, method, path, body, asks):
    r = client.request(method, path, json=body)
    assert r.status_code == 200, r.text
    assert client.asked == asks


COSTED = {"id": "12", "category": "Platform / Enterprise", "revision": "r1",
          "lines": [{"id": "7", "tag": "product", "description": "Seats", "unitPrice": "200",
                     "cost": "125", "markupPct": "60", "costSource": "entered"}],
          "totals": {"exTax": "200", "cost": "125", "margin": "75", "marginFraction": "0.375",
                     "costedLines": 1, "uncostedLines": 0}}


def test_a_quote_read_carries_nothing_that_is_ours(client, monkeypatch):
    monkeypatch.setattr(qb.builder, "get", lambda *a, **k: COSTED)
    monkeypatch.setattr(qb.builder, "list_for_deal", lambda *a, **k: [COSTED])
    for body in (client.get("/spendiq/quotes/12").json(),
                 client.get("/spendiq/deals/D1/quotes").json()[0]):
        assert "checks" not in body
        assert body["lines"] == [{"id": "7", "description": "Seats", "unitPrice": "200"}]
        assert body["totals"] == {"exTax": "200"}


def test_the_cost_record_is_its_own_read_keyed_by_line_id_with_the_limits(client, monkeypatch):
    monkeypatch.setattr(qb.builder, "get", lambda *a, **k: COSTED)
    r = client.get("/spendiq/quotes/12/internal")
    assert r.status_code == 200 and client.asked == ["margin.read"]
    assert r.json() == {
        "quoteId": "12", "revision": "r1",
        "lines": {"7": {"tag": "product", "cost": "125", "markupPct": "60",
                        "costSource": "entered"}},
        "totals": {"cost": "125", "margin": "75", "marginFraction": "0.375",
                   "costedLines": 1, "uncostedLines": 0},
        "checks": {"marginFloorPct": None, "for": "Platform / Enterprise"}}


def test_the_cost_record_is_written_behind_the_price_permit(client, monkeypatch):
    seen = {}
    monkeypatch.setattr(qb.builder, "save_internal",
                        lambda c, qid, **kw: seen.update(kw, qid=qid) or COSTED)
    r = client.put("/spendiq/quotes/12/internal",
                   json={"lines": {"7": {"cost": 130}}, "revision": "r1"})
    assert r.status_code == 200, r.text
    assert client.asked == ["sales_quote.price", "margin.read"]
    assert seen == {"qid": 12, "lines": {"7": {"cost": 130}}, "actor": CALLER, "revision": "r1"}
    assert set(r.json()) == {"quote", "internal"} and "cost" not in r.json()["quote"]["lines"][0]


def test_the_settings_are_read_by_anyone_signed_in(client, monkeypatch):
    monkeypatch.setattr(qb.pricing_checks, "settings", lambda: {"quote.tax.rate_pct": {
        "value": None, "source": "not_set", "locked": True}})
    assert client.get("/settings/quote").json()["quote.tax.rate_pct"]["value"] is None


def test_a_review_refused_by_unset_policy_names_its_codes(client, monkeypatch):
    def _refuse(*a, **k):
        raise qb.builder.PolicyUnset(12, [{"code": "TAX_POLICY_UNSET", "message": "No tax."}])
    monkeypatch.setattr(qb.builder, "submit", _refuse)
    r = client.post("/spendiq/quotes/12/review")
    assert r.status_code == 409 and r.json()["detail"]["codes"] == ["TAX_POLICY_UNSET"]


def test_who_drafted_is_the_token_and_what_the_drafter_may_not_write_is_dropped(client):
    client.post("/spendiq/deals/D1/quotes", json=BODY)
    _, _, kw = _call(client, "create")
    assert kw["created_by"] == CALLER and kw["deal_id"] == "D1"
    assert not {"created_by", "status", "conditions", "dealId", "revision"} & set(kw["draft"])


def test_a_quote_may_be_created_with_no_deal(client):
    client.post("/spendiq/quotes", json={**BODY, "dealId": None})
    assert _call(client, "create")[2]["deal_id"] is None


def test_a_save_that_does_not_name_a_deal_does_not_touch_the_link(client):
    client.put("/spendiq/quotes/12", json={**BODY, "revision": "r1"})
    kw = _call(client, "save")[2]
    assert kw["deal_id"] is qb.builder.UNSET and kw["revision"] == "r1" and kw["actor"] == CALLER
    client.calls.clear()
    client.put("/spendiq/quotes/12", json={**BODY, "dealId": "D9"})
    assert _call(client, "save")[2]["deal_id"] == "D9"


def test_the_approver_is_the_token_and_the_conditions_are_theirs(client):
    client.post("/spendiq/quotes/12/approve", json={"conditions": ["Held to year end"],
                                                    "approver": "someone.else"})
    assert _call(client, "approve")[2] == {"approver": CALLER, "conditions": ["Held to year end"]}
    client.post("/spendiq/quotes/12/reject", json={"reason": "too deep"})
    assert _call(client, "reject")[2] == {"approver": CALLER, "reason": "too deep"}


def test_a_refused_permission_stops_the_route_before_the_service(client, monkeypatch):
    def _refuse(*a, **k):
        raise NotPermitted("no")
    monkeypatch.setattr(qb, "_require_margin_permit", _refuse)
    assert client.get("/spendiq/quotes/12").status_code == 403
    assert client.put("/spendiq/quotes/12", json=BODY).status_code == 403
    assert client.calls == []


def test_the_entity_profile_is_read_by_anyone_signed_in_and_written_behind_a_gate(client, monkeypatch):
    stored = {}
    monkeypatch.setattr(qb.entity_profile, "load", lambda c: {"legalName": stored.get("legalName")})
    monkeypatch.setattr(qb.entity_profile, "save",
                        lambda c, profile, actor: stored.update(profile, by=actor) or profile)
    assert client.get("/entity/profile").json() == {"legalName": None}
    assert client.asked == []
    client.put("/entity/profile", json={"legalName": "Northwind Ltd", "bank": {"name": "B"}})
    assert client.asked == ["entity_profile.write"]
    assert stored["legalName"] == "Northwind Ltd" and stored["by"] == CALLER


# --- reading a seller's quote into a proposed template ------------------------------

def test_reading_a_quote_asks_quote_read_writes_nothing_and_returns_the_reading(client, monkeypatch):
    seen = {}

    def _read(data, filename):
        seen.update(data=data, filename=filename)
        return {"name": "Acme", "lines": [], "pricesLeftOut": 0}

    monkeypatch.setattr(qb.template_reader, "read", _read)
    r = client.post("/spendiq/quote-templates/read",
                    files={"file": ("acme.csv", b"Description,Qty\n", "text/csv")})
    assert r.status_code == 200 and r.json()["name"] == "Acme"
    assert client.asked == ["quote.read"]
    assert seen == {"data": b"Description,Qty\n", "filename": "acme.csv"}
    assert client.calls == []                       # no quote service touched


@pytest.mark.parametrize("raised, status", [
    (lambda: qb.template_reader.Unsupported("I can read PDF ... not “.exe”."), 400),
    (lambda: qb.template_reader.Unreadable("I could not find a pricing table"), 422),
    (lambda: qb.quote_agent.AgentUnavailable("the model is busy"), 503),
])
def test_reading_a_quote_says_why_it_could_not(client, monkeypatch, raised, status):
    def _read(*a, **k):
        raise raised()

    monkeypatch.setattr(qb.template_reader, "read", _read)
    r = client.post("/spendiq/quote-templates/read",
                    files={"file": ("q.exe", b"x", "application/octet-stream")})
    assert r.status_code == status
    assert r.json()["detail"] == str(raised())


def test_reading_a_quote_reads_at_most_one_byte_past_the_limit(client, monkeypatch):
    got = {}
    monkeypatch.setattr(qb.template_reader, "MAX_BYTES", 4)
    monkeypatch.setattr(qb.template_reader, "read",
                        lambda data, filename: got.setdefault("n", len(data)) and {})
    client.post("/spendiq/quote-templates/read", files={"file": ("q.txt", b"x" * 100, "text/plain")})
    assert got["n"] == 5
