from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.auth import require_user
from api.routers import triage as triage_router
from src.services.governed_limits import LimitUnavailable

VIEW = {"deal_id": "D1", "verdict": "Blocked", "summary": "Blocked · 1 finding needs action",
        "counts": {"s1": 1, "s2": 0, "notes": 0}, "exposure_gbp": "450.00", "findings": []}


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(triage_router.router)
    app.dependency_overrides[require_user] = lambda: None
    return TestClient(app)


def test_get_returns_the_view(client, monkeypatch):
    monkeypatch.setattr(triage_router.engine, "triage_deal_view", lambda d: VIEW)
    r = client.get("/triage/deals/D1")
    assert r.status_code == 200 and r.json()["verdict"] == "Blocked"


def test_get_unknown_deal_is_404(client, monkeypatch):
    monkeypatch.setattr(triage_router.engine, "triage_deal_view", lambda d: None)
    assert client.get("/triage/deals/NOPE").status_code == 404


def test_missing_policy_is_503(client, monkeypatch):
    def refuse(d):
        raise LimitUnavailable("no triage_tolerances")
    monkeypatch.setattr(triage_router.engine, "triage_deal_view", refuse)
    assert client.get("/triage/deals/D1").status_code == 503


def test_post_runs_and_returns_writes(client, monkeypatch):
    report = SimpleNamespace(run_id="RUN-1", failed={}, deals_done=1,
                             write_counts={"inserted": 1})
    monkeypatch.setattr(triage_router.engine, "run_triage", lambda ids, mode: report)
    monkeypatch.setattr(triage_router.engine, "triage_deal_view", lambda d: VIEW)
    r = client.post("/triage/deals/D1/run")
    assert r.status_code == 200
    assert r.json()["run_id"] == "RUN-1" and r.json()["writes"] == {"inserted": 1}
