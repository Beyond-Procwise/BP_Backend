"""The permission-settings admin routes: every one asks the gate first; a refusal changes nothing."""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import api.routers.permission_settings as r
from api.endpoint_gate import NotPermitted


@pytest.fixture()
def rig(monkeypatch):
    asked, did = [], []
    monkeypatch.setattr(r, "gate", lambda action, *a, **k: asked.append(action))
    monkeypatch.setattr(r.ps, "current", lambda: did.append("current") or [{"key": "supplier_master"}])
    monkeypatch.setattr(r.ps, "seed_defaults",
                        lambda by: did.append(("seed", by)) or {"created": ["supplier_master"]})
    monkeypatch.setattr(r.ps, "set_role",
                        lambda key, role, by: did.append(("set", key, role, by)) or
                        {"key": key, "role": role, "changed": True, "version": 2})
    app = FastAPI()
    app.include_router(r.router)
    app.dependency_overrides[r.require_user] = lambda: SimpleNamespace(subject="alice")
    return TestClient(app), asked, did, monkeypatch


def _refuse(*a, **k):
    raise NotPermitted("no")


def test_list_asks_to_read_policy(rig):
    c, asked, did, _ = rig
    assert c.get("/admin/permissions").json()["settings"] == [{"key": "supplier_master"}]
    assert asked == ["policy.read"]


def test_change_asks_to_write_policy_and_passes_the_caller(rig):
    c, asked, did, _ = rig
    out = c.put("/admin/permissions/supplier_master", json={"role": "Approver"})
    assert out.status_code == 200 and out.json()["role"] == "Approver"
    assert asked == ["policy.write"] and did == [("set", "supplier_master", "Approver", "alice")]


def test_refused_change_does_nothing(rig):
    c, asked, did, mp = rig
    mp.setattr(r, "gate", _refuse)
    assert c.put("/admin/permissions/supplier_master", json={"role": "Buyer"}).status_code == 403
    assert did == []


def test_refused_seed_does_nothing(rig):
    c, asked, did, mp = rig
    mp.setattr(r, "gate", _refuse)
    assert c.post("/admin/permissions/seed").status_code == 403
    assert did == []


def test_unknown_setting_is_404_and_bad_role_is_422(rig):
    c, asked, did, mp = rig
    mp.setattr(r.ps, "set_role", lambda k, role, by: (_ for _ in ()).throw(r.ps.UnknownSetting(k)))
    assert c.put("/admin/permissions/nope", json={"role": "Admin"}).status_code == 404
    mp.setattr(r.ps, "set_role", lambda k, role, by: (_ for _ in ()).throw(r.ps.InvalidRole(role)))
    assert c.put("/admin/permissions/supplier_master", json={"role": "Wizard"}).status_code == 422


def test_a_stale_save_is_409(rig):
    c, asked, did, mp = rig
    mp.setattr(r.ps, "set_role", lambda k, role, by: (_ for _ in ()).throw(r.ps.SettingConflict("x")))
    assert c.put("/admin/permissions/supplier_master", json={"role": "Buyer"}).status_code == 409
