"""POST /intent/route asks the gate before it reads a single supplier name.

The route only PROPOSES a report or workflow from free text, but to do so it looks supplier names up in
proc.bp_deal_overview. It used to need only a signed-in user, and the endpoint-gate ratchet
(tests/test_endpoint_gate_ratchet.py) went red on it. It now asks `report.read`, the same question as
reading a report, first: a refusal reads nothing.
"""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.endpoint_gate import NotPermitted
from api.routers import intent as ir_router


@pytest.fixture
def rig(monkeypatch):
    asked, looked_up = [], []
    monkeypatch.setattr(ir_router, "gate", lambda action, *a, **k: asked.append(action))
    monkeypatch.setattr(ir_router, "_supplier_lookup", lambda text: looked_up.append(text) or [])
    app = FastAPI()
    app.include_router(ir_router.router)
    app.dependency_overrides[ir_router.require_user] = lambda: None
    return TestClient(app), asked, looked_up, monkeypatch


def test_route_asks_to_read_a_report_first(rig):
    http, asked, _looked, _mp = rig
    assert http.post("/intent/route", json={"text": "QBR for Acme Ltd for Q3 2026"}).status_code == 200
    assert asked == ["report.read"]


def test_route_refused_by_the_gate_reads_no_supplier_name(rig):
    http, _asked, looked_up, mp = rig

    def refuse(*a, **k):
        raise NotPermitted("no")

    mp.setattr(ir_router, "gate", refuse)
    assert http.post("/intent/route", json={"text": "QBR for Acme Ltd for Q3 2026"}).status_code == 403
    assert looked_up == []
