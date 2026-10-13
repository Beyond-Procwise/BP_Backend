"""The Quote builder over HTTP against a real database: the journey the screen makes,
with only the permission gate stood in for (it has its own tests)."""
import datetime as dt
import json
import uuid

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routers import quote_builder as qb
from src.services import guardrail
from src.services.sell_side import quote_agent
from tests.sell_side.conftest import SENTINEL, live, live_db  # noqa: F401  (fixture)

pytestmark = live
TODAY = dt.date.today()
_OK = guardrail.Decision(allowed=True, reason="stated", policy_name="P", policy_id=1)


class _Who:
    def __init__(self, subject):
        self.subject = subject


@pytest.fixture
def http(live_db, monkeypatch):
    conn, _ = live_db
    monkeypatch.setattr(qb, "gate", lambda *a, **k: _OK)
    monkeypatch.setattr(qb, "_require_margin_permit", lambda *a, **k: None)
    who = {"now": f"{SENTINEL}-drafter"}
    app = FastAPI()
    app.include_router(qb.router)
    app.dependency_overrides[qb.require_user] = lambda: _Who(who["now"])
    client = TestClient(app)
    client.who = who
    deal = f"{SENTINEL}-DEAL-{uuid.uuid4().hex[:8]}"
    with conn.cursor() as cur:
        cur.execute("INSERT INTO proc.bp_deal (deal_id) VALUES (%s)", (deal,))
    conn.commit()
    client.deal, client.conn = deal, conn
    return client


def _body(**over):
    return {"id": "q_local_1", "status": "draft", "number": None, "version": 1,
            "currency": "GBP", "category": "Platform / Enterprise",
            "issuedAt": TODAY.isoformat(),
            "validUntil": (TODAY + dt.timedelta(days=30)).isoformat(),
            "preparedBy": "A. Okafor", "customer": {"name": "Kestrel Freight", "contact": "A. Reyes"},
            "scopeHtml": "<p>Scope</p>", "assumptionsHtml": "",
            "lines": [{"id": "l_local_1", "tag": "product", "description": "Seats", "qty": 340,
                       "unit": "seat", "cost": 125, "markupPct": 60, "unitPrice": 200,
                       "discountPct": 4}],
            "terms": {"payment": "Net 30", "start": "", "lengthMonths": 12, "renewal": "", "sla": ""},
            "conditions": [], "checks": {"marginFloorPct": 30}, **over}


def test_the_journey_from_a_blank_quote_to_an_approved_one(http):
    # started with no deal, saved, and given the server's own id
    r = http.post("/spendiq/quotes", json=_body(dealId=None))
    assert r.status_code == 200, r.text
    q = r.json()
    assert q["id"].isdigit() and q["dealId"] is None and q["number"] is None
    assert q["lines"][0]["unitPrice"] == "200.0000", "money leaves as a string, never a float"
    assert q["checks"]["marginFloorPct"] is None, "the screen's own 30 was not taken as policy"
    assert q["checks"]["sources"]["marginFloorPct"] == "not_set"

    # linked to its deal, and then it is the deal's quote
    r = http.put(f"/spendiq/quotes/{q['id']}", json=_body(dealId=http.deal, revision=q["revision"]))
    assert r.status_code == 200, r.text
    listed = http.get(f"/spendiq/deals/{http.deal}/quotes").json()
    assert [x["id"] for x in listed] == [q["id"]] and listed[0]["dealId"] == http.deal

    # a second quote on the same deal is refused in words
    r = http.post(f"/spendiq/deals/{http.deal}/quotes", json=_body())
    assert r.status_code == 409 and "already has quote" in r.json()["detail"]

    # a stale save is refused; a bad line is a 422 naming the field
    assert http.put(f"/spendiq/quotes/{q['id']}",
                    json=_body(revision=q["revision"])).status_code == 409
    r = http.put(f"/spendiq/quotes/{q['id']}", json=_body(lines=[{"qty": 0, "unitPrice": 1}]))
    assert r.status_code == 422 and "qty" in r.json()["detail"]

    # review: the drafter cannot approve their own; a reviewer can, with conditions
    assert http.post(f"/spendiq/quotes/{q['id']}/review").json()["status"] == "in-review"
    assert q["id"] in [x["id"] for x in http.get("/spendiq/quotes/review-queue").json()]
    r = http.post(f"/spendiq/quotes/{q['id']}/approve", json={"conditions": ["Mine"]})
    assert r.status_code == 409 and "their own" in r.json()["detail"]
    http.who["now"] = f"{SENTINEL}-reviewer"
    done = http.post(f"/spendiq/quotes/{q['id']}/approve",
                     json={"conditions": ["Price held to year end"]}).json()
    assert done["status"] == "approved" and done["conditions"] == ["Price held to year end"]
    assert done["approvedBy"] == f"{SENTINEL}-reviewer"
    assert http.get("/spendiq/quotes/999999999").status_code == 404


def test_the_agent_proposes_over_http_and_the_quote_is_untouched(http, monkeypatch):
    q = http.post(f"/spendiq/deals/{http.deal}/quotes", json=_body()).json()
    monkeypatch.setattr(quote_agent, "_default_prompt", lambda name: "{quote} {message}")
    monkeypatch.setattr(quote_agent, "_default_busy", lambda: False)
    monkeypatch.setattr(quote_agent, "ollama_generate", lambda prompt, **kw: json.dumps(
        {"refused": "", "intents": [{"op": "markup_all", "percent": 40}]}))

    r = http.post(f"/spendiq/quotes/{q['id']}/agent",
                  json={"message": "apply 40% markup to all items", "created_by": "x"})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["change"] == {"lines": [{"id": q["lines"][0]["id"], "markupPct": "40",
                                        "unitPrice": "175.0000"}]}
    assert out["refused"] is None and out["session_id"] == q["id"]
    after = http.get(f"/spendiq/quotes/{q['id']}").json()
    assert after["lines"] == q["lines"] and after["revision"] == q["revision"]

    # turned down: kept on the quote's trail
    r = http.post(f"/spendiq/quotes/{q['id']}/agent/dismiss",
                  json={"summary": out["summary"], "message": "apply 40% markup to all items"})
    assert r.status_code == 200
    with http.conn.cursor() as cur:
        cur.execute("SELECT details->>'proposal' FROM proc.bp_agent_actions WHERE doc_pk = %s "
                    "AND action_type = 'sales_quote.proposal_dismissed'", (q["id"],))
        assert cur.fetchone()[0] == out["summary"]
    http.conn.rollback()


def test_with_no_stored_prompt_the_agent_is_unavailable_rather_than_improvising(http, monkeypatch):
    q = http.post(f"/spendiq/deals/{http.deal}/quotes", json=_body()).json()

    def _missing(name):
        raise quote_agent.AgentUnavailable("prompt unavailable: sales_quote_drafting")
    monkeypatch.setattr(quote_agent, "_default_prompt", _missing)
    r = http.post(f"/spendiq/quotes/{q['id']}/agent", json={"message": "40% markup"})
    assert r.status_code == 503 and "prompt unavailable" in r.json()["detail"]


def test_the_company_profile_round_trips_and_starts_empty(http, monkeypatch):
    """Under a LIVETEST key: the real profile is on every quote that goes out, and a test
    must not blank it even for a moment."""
    key = f"{SENTINEL}_entity_profile"
    monkeypatch.setattr(qb.entity_profile, "CONFIG_KEY", key)

    def _clear():
        with http.conn.cursor() as cur:
            cur.execute("DELETE FROM proc.bp_admin_config WHERE config_key = %s", (key,))
        http.conn.commit()

    _clear()
    try:
        empty = http.get("/entity/profile").json()
        assert empty["legalName"] is None and empty["bank"]["iban"] is None
        saved = http.put("/entity/profile", json={
            "legalName": f"{SENTINEL} Trading Ltd", "vatNumber": "GB000",
            "bank": {"name": "Bank", "sortCode": "00-00-00", "accountNumber": "00000000"}}).json()
        assert saved["legalName"] == f"{SENTINEL} Trading Ltd" and saved["tradingName"] is None
        assert saved["updatedBy"] == f"{SENTINEL}-drafter"
        assert http.get("/entity/profile").json()["bank"]["sortCode"] == "00-00-00"
    finally:
        _clear()


# --- templates ---------------------------------------------------------------------

def test_a_quote_is_started_from_a_template_and_filled_from_what_the_seller_says(http, monkeypatch):
    listed = http.get("/spendiq/quote-templates").json()
    assert "software_subscription" in [t["key"] for t in listed]
    chosen = next(t for t in listed if t["key"] == "software_subscription")
    assert not any({"unitPrice", "cost", "discountPct"} & set(l) for l in chosen["lines"])

    r = http.post("/spendiq/quotes/from-template",
                  json={"templateKey": "software_subscription", "currency": "GBP",
                        "dealId": http.deal})
    assert r.status_code == 200, r.text
    q = r.json()
    assert q["templateKey"] == "software_subscription" and q["dealId"] == http.deal
    assert [l["description"] for l in q["lines"]] == [
        "Subscription licences", "Onboarding and set-up", "Training"]
    assert all(l["unitPrice"] == "0.0000" and l["cost"] is None for l in q["lines"])
    assert q["totals"]["exTax"] == "0.00" and q["totals"]["margin"] is None
    assert [m["field"] for m in q["missing"]][:2] == ["customer.name", "customer.contact"]
    assert len(q["missing"]) == 2 + 3 + 5 + 1          # ... and the Scope section
    assert q["missing"][-1] == {"field": "sections.scope", "label": "the “Scope” section"}
    assert [x["kind"] for x in q["sections"]][:4] == ["letterhead", "customer", "scope", "lines"]

    seats, onboarding, _training = [l["id"] for l in q["lines"]]
    asked_with = {}

    def _model(prompt, **kw):
        asked_with.update(prompt=prompt, **kw)
        return json.dumps({"refused": "", "intents": [
            {"op": "set_customer", "name": "kestrel freight", "contact": "A. Reyes"},
            {"op": "set_line", "line_id": seats, "qty": 340, "unit_price": 200, "cost": 125},
            {"op": "set_line", "line_id": onboarding, "unit_price": 18000},
            {"op": "set_term", "field": "payment", "text": "Net 30"},
            {"op": "set_term", "field": "lengthMonths", "number": 12}]})
    monkeypatch.setattr(quote_agent, "_default_prompt",
                        lambda name: name + " {template} {quote} {message}")
    monkeypatch.setattr(quote_agent, "_default_busy", lambda: False)
    monkeypatch.setattr(quote_agent, "ollama_generate", _model)

    said = ("It's for Kestrel Freight, contact A. Reyes. 340 seats at £200 each, they cost us "
            "125. Onboarding is 18k. Net 30, 12 months.")
    out = http.post(f"/spendiq/quotes/{q['id']}/agent", json={"message": said}).json()
    assert asked_with["prompt"].startswith("sales_quote_template_fill")
    assert "Software subscription" in asked_with["prompt"]
    assert asked_with["format"] == quote_agent.FILL_SCHEMA
    assert out["refused"] is None, out
    assert out["change"] == {
        "customer": {"name": "Kestrel Freight", "contact": "A. Reyes"},
        "terms": {"payment": "Net 30", "lengthMonths": 12},
        "lines": [{"id": seats, "qty": "340", "unitPrice": "200", "cost": "125", "markupPct": "60.0"},
                  {"id": onboarding, "unitPrice": "18000"}]}
    assert [m["field"] for m in out["missing"]] == [
        f"lines.{_training}.unitPrice", "terms.start", "terms.renewal", "terms.sla",
        "sections.scope"]
    assert "Still needed: a price for “Training”; a start date" in out["summary"]

    # proposed, not written
    after = http.get(f"/spendiq/quotes/{q['id']}").json()
    assert after["lines"] == q["lines"] and after["customer"]["name"] is None


def test_a_retired_or_unknown_template_starts_nothing(http):
    assert http.post("/spendiq/quotes/from-template",
                     json={"templateKey": "no_such_template", "currency": "GBP"}).status_code == 404
    key = f"livetest_{uuid.uuid4().hex[:8]}"
    body = {"name": f"{SENTINEL} template", "status": "retired",
            "lines": [{"key": "work", "tag": "service", "description": "Work", "unit": "day",
                       "unitPrice": 999}]}
    try:
        saved = http.put(f"/spendiq/quote-templates/{key}", json=body).json()
        assert saved["status"] == "retired" and "unitPrice" not in saved["lines"][0]
        assert key not in [t["key"] for t in http.get("/spendiq/quote-templates").json()]
        assert key in [t["key"] for t in
                       http.get("/spendiq/quote-templates?include_retired=true").json()]
        r = http.post("/spendiq/quotes/from-template", json={"templateKey": key, "currency": "GBP"})
        assert r.status_code == 409 and "retired" in r.json()["detail"]
        http.put(f"/spendiq/quote-templates/{key}", json={**body, "status": "active"})
        started = http.post("/spendiq/quotes/from-template",
                            json={"templateKey": key, "currency": "GBP"}).json()
        assert started["dealId"] is None and started["lines"][0]["unitPrice"] == "0.0000"
    finally:
        with http.conn.cursor() as cur:
            cur.execute("DELETE FROM proc.bp_sales_quote_template WHERE template_key = %s", (key,))
        http.conn.commit()


# --- logos and layout --------------------------------------------------------------

def _png(colour):
    import io

    from PIL import Image
    out = io.BytesIO()
    Image.new("RGB", (160, 48), colour).save(out, format="PNG")
    return out.getvalue()


def test_a_logo_is_uploaded_swapped_on_a_quote_and_the_default_cannot_be_removed(http, monkeypatch):
    key = f"{SENTINEL}_entity_profile"
    monkeypatch.setattr(qb.entity_profile, "CONFIG_KEY", key)
    try:
        blue, red = _png((20, 60, 160)), _png((180, 30, 30))
        first = http.post("/entity/logos", files={"file": ("blue.png", blue, "image/png")},
                          data={"name": "Blue mark"}).json()
        second = http.post("/entity/logos", files={"file": ("red.png", red, "image/png")}).json()
        assert first["name"] == "Blue mark" and second["name"] == "red.png"
        assert (first["width"], first["height"], first["contentType"]) == (160, 48, "image/png")
        assert {first["id"], second["id"]} <= {x["id"] for x in http.get("/entity/logos").json()}

        served = http.get(first["imagePath"])
        assert served.content == blue and served.headers["content-type"] == "image/png"
        assert served.headers["x-content-type-options"] == "nosniff"

        # not a picture, whatever it says it is
        r = http.post("/entity/logos", files={"file": ("logo.png", b"<svg><script>x</script></svg>",
                                                       "image/png")})
        assert r.status_code == 422 and "PNG or JPEG" in r.json()["detail"]

        # the company's default, then a different one on this quote's letterhead
        assert http.put("/entity/profile", json={"legalName": f"{SENTINEL} Ltd",
                                                 "logoId": first["id"]}).json()["logoId"] == first["id"]
        assert http.put("/entity/profile", json={"logoId": 999_999_999}).status_code == 422
        q = http.post(f"/spendiq/deals/{http.deal}/quotes", json=_body()).json()
        layout = q["sections"]
        assert layout[0]["kind"] == "letterhead" and layout[0]["logoId"] is None
        layout[0]["logoId"] = second["id"]
        layout[2]["heading"] = "Our understanding"
        layout.insert(3, {"key": "deliverables", "kind": "text", "heading": "Deliverables",
                          "html": "<p>Platform</p>"})
        saved = http.put(f"/spendiq/quotes/{q['id']}", json=_body(sections=layout)).json()
        assert saved["sections"][0]["logoId"] == second["id"]
        assert [(x["key"], x["heading"]) for x in saved["sections"]][2:4] == [
            ("scope", "Our understanding"), ("deliverables", "Deliverables")]
        assert saved["sections"][3]["html"] == "<p>Platform</p>"

        # a save that says nothing about the layout leaves it alone
        assert http.put(f"/spendiq/quotes/{q['id']}", json=_body()).json()["sections"] == saved["sections"]
        # a logo nobody uploaded cannot be put on a quote
        layout[0]["logoId"] = 999_999_999
        assert http.put(f"/spendiq/quotes/{q['id']}", json=_body(sections=layout)).status_code == 422

        # the default stays until another is chosen; a removed logo leaves the list but
        # is still served, because quotes already wear it
        r = http.delete(f"/entity/logos/{first['id']}")
        assert r.status_code == 409 and "choose another" in r.json()["detail"]
        assert http.delete(f"/entity/logos/{second['id']}").status_code == 200
        assert second["id"] not in {x["id"] for x in http.get("/entity/logos").json()}
        assert http.get(second["imagePath"]).content == red
        assert http.delete(f"/entity/logos/{second['id']}").status_code == 404
    finally:
        with http.conn.cursor() as cur:
            cur.execute("DELETE FROM proc.bp_admin_config WHERE config_key = %s", (key,))
        http.conn.commit()


def test_a_template_carries_its_layout_onto_the_quote_and_the_agent_fills_a_section(http, monkeypatch):
    key = f"livetest_{uuid.uuid4().hex[:8]}"
    try:
        http.put(f"/spendiq/quote-templates/{key}", json={
            "name": f"{SENTINEL} proposal", "lines": [{"key": "work", "description": "Work", "unit": "day"}],
            "sections": [{"key": "intro", "kind": "text", "heading": "Why us", "html": "not kept"},
                         {"key": "pricing", "kind": "lines", "heading": "Investment"},
                         {"key": "timeline", "kind": "text", "heading": "Timeline"}]})
        q = http.post("/spendiq/quotes/from-template",
                      json={"templateKey": key, "currency": "GBP"}).json()
        assert [(x["heading"], x.get("html")) for x in q["sections"]] == [
            ("Why us", ""), ("Investment", None), ("Timeline", "")]
        assert [m["label"] for m in q["missing"]][-2:] == [
            "the “Why us” section", "the “Timeline” section"]

        monkeypatch.setattr(quote_agent, "_default_prompt", lambda name: "{template} {quote} {message}")
        monkeypatch.setattr(quote_agent, "_default_busy", lambda: False)
        monkeypatch.setattr(quote_agent, "ollama_generate", lambda prompt, **kw: json.dumps(
            {"expression": "20 * 850", "label": "Delivery", "line_id": "", "into": ""}
            if kw["format"] == quote_agent.SUM_SCHEMA else {"refused": "", "intents": []}))
        out = http.post(f"/spendiq/quotes/{q['id']}/agent", json={
            "message": "Timeline: kick-off in January, live by March. What is 20 days at 850?"}).json()
        assert out["change"] == {"sections": [
            {"key": "timeline", "html": "<p>kick-off in January, live by March.</p>"}]}
        assert out["calculations"] == [{"label": "Delivery", "working": "20 × 850 = 17000",
                                        "result": "17000.0000"}]
        assert http.get(f"/spendiq/quotes/{q['id']}").json()["sections"] == q["sections"]
    finally:
        with http.conn.cursor() as cur:
            cur.execute("DELETE FROM proc.bp_sales_quote_template WHERE template_key = %s", (key,))
        http.conn.commit()
