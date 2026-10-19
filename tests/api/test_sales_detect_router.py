import contextlib

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.endpoint_gate import NotPermitted
from api.routers import sales as sr


class _P:
    subject = "sub-caller"


def _client(monkeypatch, gate):
    monkeypatch.setattr(sr, "gate", gate)
    monkeypatch.setattr(sr, "get_conn", lambda: contextlib.nullcontext("CONN"))
    app = FastAPI()
    app.include_router(sr.router)
    app.dependency_overrides[sr.require_user] = lambda: _P()
    return TestClient(app)


def test_detect_runs_the_detector_behind_its_permit(monkeypatch):
    gates, calls = [], []
    monkeypatch.setattr(sr.detector, "detect",
                        lambda c, *, account_id=None: calls.append((c, account_id))
                        or [{"sales_opportunity_id": 1}])
    client = _client(monkeypatch, lambda action, *a, **k: gates.append(action))
    r = client.post("/sales/opportunities/detect?account_id=A1")
    assert r.status_code == 200, r.text
    assert r.json() == {"created": [{"sales_opportunity_id": 1}]}
    assert gates == ["sales.detect"]
    assert calls == [("CONN", "A1")]


def test_detect_refused_when_the_gate_refuses(monkeypatch):
    calls = []

    def deny(action, *a, **k):
        raise NotPermitted("no")

    monkeypatch.setattr(sr.detector, "detect", lambda *a, **k: calls.append(1) or [])
    client = _client(monkeypatch, deny)
    assert client.post("/sales/opportunities/detect").status_code == 403
    assert calls == []


def test_sales_detect_is_a_registered_action():
    from src.services import actions
    assert actions.ACTIONS["sales.detect"] == "compute"
