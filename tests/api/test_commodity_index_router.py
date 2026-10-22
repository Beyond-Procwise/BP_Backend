"""The commodity index routes: a load needs the permission setting's role AND the action gate; reads ask too."""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import api.routers.commodity_index as r
from api.endpoint_gate import NotPermitted
from src.services import commodity_index as CI
from src.services.contract_tools import store

BODY = {"publisher": "LME", "points": [{"period_date": "2026-01-01", "value": 2000}], "unit": "USD/tonne"}


@pytest.fixture()
def rig(monkeypatch):
    asked, did = [], []
    monkeypatch.setattr(r, "gate", lambda action, principal, **k: asked.append((action, k.get("context"))))
    monkeypatch.setattr(r.permission_settings, "minimum_role", lambda key, fallback="Admin": "Admin")
    monkeypatch.setattr(r.rbac, "effective_role", lambda principal: "Admin")
    monkeypatch.setattr(r.CI, "load_series", lambda code, pub, pts, by, **k: did.append(("load", code, pub, len(pts), by, k.get("unit")))
                        or {"series_code": code, "points_loaded": len(pts), "recorded_by": by})
    monkeypatch.setattr(r.CI, "list_series", lambda: did.append("list") or {"found": False, "series": []})
    monkeypatch.setattr(r.CI, "lookup", lambda code, start, end: did.append(("lookup", code, start, end)) or {"found": True})
    app = FastAPI()
    app.include_router(r.router)
    app.dependency_overrides[r.require_user] = lambda: SimpleNamespace(subject="carol")
    return TestClient(app), asked, did, monkeypatch


def test_load_checks_the_role_asks_the_gate_and_signs_with_the_caller(rig):
    c, asked, did, _ = rig
    out = c.post("/commodity-index/LME-AL-CASH", json=BODY)
    assert out.status_code == 200 and out.json()["recorded_by"] == "carol"
    assert asked == [("index_series.record", {"series_code": "LME-AL-CASH", "publisher": "LME", "points": 1})]
    assert did == [("load", "LME-AL-CASH", "LME", 1, "carol", "USD/tonne")]


def test_a_lower_role_is_refused_before_the_gate_or_any_write(rig):
    c, asked, did, mp = rig
    mp.setattr(r.rbac, "effective_role", lambda principal: "Approver")
    out = c.post("/commodity-index/LME-AL-CASH", json=BODY)
    assert out.status_code == 403 and "Admin" in out.json()["detail"] and asked == [] and did == []


def test_the_setting_can_lower_the_bar_but_unreadable_means_admin(rig):
    c, asked, did, mp = rig
    mp.setattr(r.rbac, "effective_role", lambda principal: "Buyer")
    mp.setattr(r.permission_settings, "minimum_role", lambda key, fallback="Admin": "Buyer")
    assert c.post("/commodity-index/X", json=BODY).status_code == 200

    def boom(key, fallback="Admin"):
        raise RuntimeError("db")
    mp.setattr(r.permission_settings, "minimum_role", boom)
    assert c.post("/commodity-index/X", json=BODY).status_code == 403


def test_a_gate_refusal_an_invalid_series_and_an_outage(rig):
    c, asked, did, mp = rig

    def refuse(*a, **k):
        raise NotPermitted("no")
    mp.setattr(r, "gate", refuse)
    assert c.post("/commodity-index/X", json=BODY).status_code == 403 and did == []
    mp.setattr(r, "gate", lambda *a, **k: None)

    def invalid(*a, **k):
        raise CI.InvalidSeries("point 1 value must be a number")
    mp.setattr(r.CI, "load_series", invalid)
    out = c.post("/commodity-index/X", json=BODY)
    assert out.status_code == 422 and "point 1" in out.json()["detail"]

    def down(*a, **k):
        raise store.StoreUnavailable("db down")
    mp.setattr(r.CI, "load_series", down)
    assert c.post("/commodity-index/X", json=BODY).status_code == 503


def test_an_empty_body_is_refused_by_the_schema(rig):
    c, asked, did, _ = rig
    assert c.post("/commodity-index/X", json={"publisher": "LME", "points": []}).status_code == 422 and did == []


def test_reads_ask_to_read_contracts(rig):
    c, asked, did, _ = rig
    assert c.get("/commodity-index").status_code == 200
    assert c.get("/commodity-index/LME-AL-CASH?start=2026-01-01").json() == {"found": True}
    assert asked == [("contract.read", None), ("contract.read", {"series_code": "LME-AL-CASH"})]
    assert did == ["list", ("lookup", "LME-AL-CASH", "2026-01-01", None)]


def test_the_setting_and_the_action_exist():
    from src.services import actions, permission_settings
    assert actions.ACTIONS["index_series.record"] == "write"
    s = permission_settings.setting("index_series_record")
    assert s.default_role == "Admin" and s.applies_to == ("index_series.record",)
