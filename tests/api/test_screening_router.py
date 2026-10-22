"""The screening register routes: a write needs the permission setting's role AND the action gate; reads ask too."""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import api.routers.screening as r
from api.endpoint_gate import NotPermitted
from src.services import screening_register as SR
from src.services.contract_tools import store

BODY = {"supplier_id": "S1", "check_type": "sanctions", "result": "clear", "provider": "Dow Jones",
        "checked_on": "2026-10-01", "recorded_by": "mallory"}


@pytest.fixture()
def rig(monkeypatch):
    asked, did = [], []
    monkeypatch.setattr(r, "gate", lambda action, principal, **k: asked.append((action, k.get("context"))))
    monkeypatch.setattr(r.permission_settings, "minimum_role", lambda key, fallback="Admin": "Admin")
    monkeypatch.setattr(r.rbac, "effective_role", lambda principal: "Admin")
    monkeypatch.setattr(r.SR, "record", lambda entry, by, **k: did.append(("record", entry["supplier_id"], by)) or
                        {"screening_id": 1, "recorded_by": by})
    monkeypatch.setattr(r.SR, "history", lambda sid, **k: did.append(("history", sid)) or [])
    app = FastAPI()
    app.include_router(r.router)
    app.dependency_overrides[r.require_user] = lambda: SimpleNamespace(subject="carol")
    return TestClient(app), asked, did, monkeypatch


def test_record_checks_the_role_asks_the_gate_and_signs_with_the_caller(rig):
    c, asked, did, _ = rig
    out = c.post("/screening", json=BODY)
    assert out.status_code == 200 and out.json()["recorded_by"] == "carol"
    assert asked == [("supplier.screening.record", {"supplier_id": "S1", "check_type": "sanctions", "result": "clear"})]
    assert did == [("record", "S1", "carol")]


def test_a_lower_role_than_the_setting_is_refused_before_the_gate_or_any_write(rig):
    c, asked, did, mp = rig
    mp.setattr(r.rbac, "effective_role", lambda principal: "Buyer")
    out = c.post("/screening", json=BODY)
    assert out.status_code == 403 and "Admin" in out.json()["detail"]
    assert asked == [] and did == []


def test_the_setting_can_lower_the_bar_but_unreadable_or_unknown_means_admin(rig):
    c, asked, did, mp = rig
    mp.setattr(r.rbac, "effective_role", lambda principal: "Approver")
    mp.setattr(r.permission_settings, "minimum_role", lambda key, fallback="Admin": "Approver")
    assert c.post("/screening", json=BODY).status_code == 200

    def boom(key, fallback="Admin"):
        raise RuntimeError("db")
    mp.setattr(r.permission_settings, "minimum_role", boom)
    assert c.post("/screening", json=BODY).status_code == 403
    mp.setattr(r.permission_settings, "minimum_role", lambda key, fallback="Admin": "Superuser")
    assert c.post("/screening", json=BODY).status_code == 403


def test_a_gate_refusal_writes_nothing(rig):
    c, asked, did, mp = rig

    def refuse(*a, **k):
        raise NotPermitted("no")
    mp.setattr(r, "gate", refuse)
    assert c.post("/screening", json=BODY).status_code == 403 and did == []


def test_an_invalid_entry_is_422_and_an_outage_is_503(rig):
    c, asked, did, mp = rig

    def invalid(entry, by, **k):
        raise SR.InvalidEntry("checked_on cannot be in the future")
    mp.setattr(r.SR, "record", invalid)
    out = c.post("/screening", json=BODY)
    assert out.status_code == 422 and "future" in out.json()["detail"]

    def down(entry, by, **k):
        raise store.StoreUnavailable("db down")
    mp.setattr(r.SR, "record", down)
    assert c.post("/screening", json=BODY).status_code == 503


def test_history_asks_to_read_suppliers(rig):
    c, asked, did, _ = rig
    out = c.get("/screening/S1")
    assert out.status_code == 200 and out.json() == {"supplier_id": "S1", "count": 0, "checks": []}
    assert asked == [("supplier.read", {"supplier_id": "S1"})] and did == [("history", "S1")]


def test_types_are_listed_for_the_screen(rig):
    c, *_ = rig
    out = c.get("/screening/types").json()
    assert out["check_types"]["sanctions"] == "sanctions screening" and "not_performed" in out["results"]


def test_the_setting_and_the_action_exist():
    from src.services import actions, permission_settings
    assert actions.ACTIONS["supplier.screening.record"] == "write"
    s = permission_settings.setting("screening_record")
    assert s.default_role == "Admin" and s.applies_to == ("supplier.screening.record",)
