"""Advice endpoints, and the dashboard's swap from constants to ranked plays.

Route paths: the plan's interface list says "/negotiate/{deal_id}/advice", but
src/api/routers/negotiate.py mounts its router with prefix="/deals" — the live
dashboard route is /deals/{deal_id}/negotiate, confirmed against the running
API's openapi. Task 7's file list does not include main.py, so these routes go on
the existing deal-scoped router rather than a second top-level prefix, and the
paths below are what the app actually serves.
"""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import src.api.routers.negotiate as rt


_ADVICE = {"advice_id": "A-1", "deal_id": "D-1", "quadrant": "Leverage",
           "quadrant_source": "computed", "quadrant_reasons": ["because"],
           "quadrant_confidence": 0.8, "style": "Competitive",
           "style_source": "computed", "style_reasons": [],
           "indeterminate": False, "signals": {}, "stated_facts": {},
           "plays": [{"lever": "Commercial", "play": "Benchmark it",
                      "state": "ready", "evidence": [], "score": 1.0}]}


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(rt.router)
    return TestClient(app)


def test_get_advice_returns_the_payload(client, monkeypatch):
    monkeypatch.setattr(rt, "build_advice", lambda deal_id, **kw: dict(_ADVICE))
    r = client.get("/deals/D-1/advice")
    assert r.status_code == 200
    assert r.json()["quadrant"] == "Leverage"
    assert r.json()["plays"][0]["state"] == "ready"


def test_get_advice_404s_for_unknown_deal(client, monkeypatch):
    monkeypatch.setattr(rt, "build_advice", lambda deal_id, **kw: None)
    assert client.get("/deals/NOPE/advice").status_code == 404


def test_post_message_applies_a_turn(client, monkeypatch):
    seen = {}

    def _apply(deal_id, message, **kw):
        seen["message"] = message
        return dict(_ADVICE, style="Principled")

    monkeypatch.setattr(rt, "apply_turn", _apply)
    r = client.post("/deals/D-1/advice/message",
                    json={"action": "override", "style": "Principled"})
    assert r.status_code == 200
    assert r.json()["style"] == "Principled"
    assert seen["message"]["action"] == "override"


def test_post_message_404s_for_unknown_deal(client, monkeypatch):
    monkeypatch.setattr(rt, "apply_turn", lambda deal_id, message, **kw: None)
    r = client.post("/deals/NOPE/advice/message", json={"action": "more_plays"})
    assert r.status_code == 404


def test_delete_fact_withdraws_and_returns_refreshed_advice(client, monkeypatch):
    seen = {}

    def _apply(deal_id, message, **kw):
        seen["message"] = message
        return dict(_ADVICE)

    monkeypatch.setattr(rt, "apply_turn", _apply)
    r = client.delete("/deals/D-1/advice/fact/alternative_supplier_count")
    assert r.status_code == 200
    assert seen["message"]["action"] == "withdraw_fact"
    assert seen["message"]["fact_key"] == "alternative_supplier_count"


# --- the dashboard swap -------------------------------------------------------
import src.services.negotiate_dashboard as nd


class _Cur:
    description = ()

    def execute(self, *a, **k):
        pass

    def fetchall(self):
        return []


def test_strategy_no_longer_returns_hardcoded_leverage_points(monkeypatch):
    monkeypatch.setattr(nd, "_deal", lambda cur, deal_id: {
        "deal_id": "D-1", "supplier_id": "SUP-1", "currency": "GBP",
        "quote_total": 100.0, "po_total": 90.0, "invoice_total": 95.0,
        "last_activity_date": None})
    monkeypatch.setattr(nd, "_supplier_insights",
                        lambda cur, d: ("p", "k", "r"))
    monkeypatch.setattr(nd, "_advice_plays", lambda deal_id: [])
    out = nd.negotiation_strategy(_Cur(), "D-1")[0]
    assert "leveragePoints" not in out
    assert "counterStrategy" not in out
    assert "plays" in out
    # the computed parts survive
    assert "currentStandpoint" in out
    assert "preferredOutcome" in out


def test_strategy_degrades_to_empty_plays_when_advice_fails(monkeypatch):
    monkeypatch.setattr(nd, "_deal", lambda cur, deal_id: {
        "deal_id": "D-1", "supplier_id": "SUP-1", "currency": "GBP",
        "quote_total": 100.0, "po_total": 90.0, "invoice_total": 95.0,
        "last_activity_date": None})
    monkeypatch.setattr(nd, "_supplier_insights",
                        lambda cur, d: ("p", "k", "r"))
    monkeypatch.setattr(nd, "_advice_plays",
                        lambda deal_id: (_ for _ in ()).throw(RuntimeError("x")))
    out = nd.negotiation_strategy(_Cur(), "D-1")[0]
    assert out["plays"] == []
