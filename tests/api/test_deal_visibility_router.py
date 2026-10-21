"""The deal-visibility admin routes: every one asks the gate first; a refusal changes nothing."""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import api.routers.deal_visibility as r
from api.endpoint_gate import NotPermitted


@pytest.fixture()
def rig(monkeypatch):
    asked, did = [], []
    monkeypatch.setattr(r, "gate", lambda action, *a, **k: asked.append(action))
    monkeypatch.setattr(r.dv, "current", lambda: [{"role": "Buyer", "read": "mine", "write": "mine"}])
    monkeypatch.setattr(r.dv, "seed_defaults", lambda by: did.append(("seed", by)) or {"created": True})
    monkeypatch.setattr(r.dv, "set_scope", lambda role, by, read=None, write=None:
                        did.append(("set", role, read, write, by)) or
                        {"role": role, "read": read or "mine", "write": write or "mine",
                         "changed": True, "version": 2})
    app = FastAPI()
    app.include_router(r.router)
    app.dependency_overrides[r.require_user] = lambda: SimpleNamespace(subject="alice")
    return TestClient(app), asked, did, monkeypatch


def _refuse(*a, **k):
    raise NotPermitted("no")


def test_list_asks_to_read_policy_and_names_the_scopes(rig):
    c, asked, did, _ = rig
    body = c.get("/admin/visibility").json()
    assert body["roles"][0]["role"] == "Buyer" and body["readScopes"] == ["mine", "all"]
    assert asked == ["policy.read"]


def test_change_asks_to_write_policy_and_passes_the_caller(rig):
    c, asked, did, _ = rig
    out = c.put("/admin/visibility/Approver", json={"read": "all"})
    assert out.status_code == 200 and out.json()["read"] == "all"
    assert asked == ["policy.write"] and did == [("set", "Approver", "all", None, "alice")]


def test_refused_change_and_seed_do_nothing(rig):
    c, asked, did, mp = rig
    mp.setattr(r, "gate", _refuse)
    assert c.put("/admin/visibility/Buyer", json={"read": "all"}).status_code == 403
    assert c.post("/admin/visibility/seed").status_code == 403
    assert did == []


def test_bad_input_maps_to_404_422_409(rig):
    c, _, _, mp = rig
    for exc, code in ((r.ps.InvalidRole("x"), 404), (r.dv.InvalidScope("x"), 422),
                      (r.dv.AdminCannotBeNarrowed("x"), 422), (r.ps.SettingConflict("x"), 409)):
        def boom(*a, _e=exc, **k):
            raise _e
        mp.setattr(r.dv, "set_scope", boom)
        assert c.put("/admin/visibility/Buyer", json={"read": "all"}).status_code == code
