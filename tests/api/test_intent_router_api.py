"""POST /intent/route: a request in, a proposal out. It reads suppliers to resolve a name and runs nothing."""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routers import intent as ir_router


class _P:
    subject = "sub-1"


@pytest.fixture
def client(monkeypatch):
    seen = []

    def lookup(text):
        seen.append(text)
        return [("SUP-AcmeLtd", "Acme Ltd")] if "acme" in text.lower() else []

    monkeypatch.setattr(ir_router, "_supplier_lookup", lookup)
    app = FastAPI()
    app.include_router(ir_router.router)
    app.dependency_overrides[ir_router.require_user] = lambda: _P()
    c = TestClient(app)
    c.lookups = seen
    return c


def test_a_request_is_routed_and_the_supplier_resolved(client):
    r = client.post("/intent/route", json={"text": "Prepare a QBR for Acme Ltd for Q3 2026"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["decision"] == "route" and body["proposal_only"] is True
    step = body["steps"][0]
    assert (step["kind"], step["id"]) == ("report", "supplier_qbr")
    assert step["inputs"]["supplier_id"] == "SUP-AcmeLtd" and step["ready"] is True
    assert client.lookups == ["Prepare a QBR for Acme Ltd for Q3 2026"]


def test_nothing_recognised_says_so_and_offers_the_options(client):
    body = client.post("/intent/route", json={"text": "what is the weather"}).json()
    assert body["decision"] == "none" and body["steps"] == [] and body["options"]


@pytest.mark.parametrize("payload", [{}, {"text": ""}, {"text": "x" * 2001}, {"text": 5}])
def test_a_bad_body_is_refused(client, payload):
    assert client.post("/intent/route", json=payload).status_code == 422


def test_a_failing_lookup_does_not_fail_the_request(client, monkeypatch):
    def boom(text):
        raise RuntimeError("db down")
    monkeypatch.setattr(ir_router, "_supplier_lookup", boom)
    body = client.post("/intent/route", json={"text": "QBR for Acme Ltd for Q3 2026"}).json()
    assert "supplier_id" in body["steps"][0]["missing"]


def test_the_reply_names_no_route_or_table():
    """The output-safety boundary withholds a field that names an internal route or table."""
    import json
    from src.services import intent_router as ir
    text = json.dumps(ir.route("QBR for Acme Ltd for Q3 2026, then rank the suppliers"))
    for leak in ("/reports", "/agents", "proc.", "bp_"):
        assert leak not in text


def test_the_router_is_registered_behind_sign_in():
    import api.main as main
    assert ir_router.router in main._AUTHENTICATED_ROUTERS      # every router there takes require_user
    assert "/intent/route" in main.app.openapi()["paths"]
