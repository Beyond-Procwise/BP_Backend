import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import src.api.routers.deal_summary as router_mod


def _client():
    app = FastAPI()
    app.include_router(router_mod.router)
    return TestClient(app)


def test_get_summary_ok(monkeypatch):
    monkeypatch.setattr(
        router_mod, "summarize_deal",
        lambda deal_id, conn=None: {
            "deal_id": deal_id, "deal_name": "Acme Deal",
            "summary": "One invoice for ACME.",
            "sources": {"invoices": 1, "actions": 2},
        },
    )
    resp = _client().get("/deals/D-9/summary")
    assert resp.status_code == 200
    body = resp.json()
    assert body["deal_id"] == "D-9"
    assert body["summary"] == "One invoice for ACME."
    assert "generated_at" in body


def test_get_summary_unknown_deal_404(monkeypatch):
    monkeypatch.setattr(router_mod, "summarize_deal", lambda deal_id, conn=None: None)
    resp = _client().get("/deals/NOPE/summary")
    assert resp.status_code == 404


def test_get_summary_llm_failure_502(monkeypatch):
    def boom(deal_id, conn=None):
        raise router_mod.SummarizationError("empty")
    monkeypatch.setattr(router_mod, "summarize_deal", boom)
    resp = _client().get("/deals/D-9/summary")
    assert resp.status_code == 502
