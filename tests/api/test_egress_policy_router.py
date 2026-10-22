"""The egress-policy admin routes: every one asks the gate first; bad rules and stale saves are refused."""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import api.routers.egress_policy as r
from api.endpoint_gate import NotPermitted


@pytest.fixture()
def rig(monkeypatch):
    asked, did = [], []
    monkeypatch.setattr(r, "gate", lambda action, *a, **k: asked.append(action))
    monkeypatch.setattr(r.ep, "current", lambda: did.append("current") or {"configured": True, "version": 3})
    monkeypatch.setattr(r.ep, "seed", lambda by: did.append(("seed", by)) or {"changed": True, "version": 1})

    def _replace(rules, by, expected_version):
        did.append(("replace", by, expected_version))
        if expected_version != 3:
            raise r.ep.RulesConflict("moved")
        if rules.get("allow") == "bad":
            raise r.ep.InvalidRules("bad")
        return {"version": 4, "changed": True}
    monkeypatch.setattr(r.ep, "replace_rules", _replace)
    app = FastAPI()
    app.include_router(r.router)
    app.dependency_overrides[r.require_user] = lambda: SimpleNamespace(subject="alice")
    return TestClient(app), asked, did, monkeypatch


def _refuse(*a, **k):
    raise NotPermitted("no")


def test_read_asks_to_read_policy(rig):
    c, asked, did, _ = rig
    assert c.get("/admin/egress-policy").json()["version"] == 3
    assert asked == ["policy.read"]


def test_replace_asks_to_write_and_passes_who_and_version(rig):
    c, asked, did, _ = rig
    res = c.put("/admin/egress-policy", json={"rules": {"allow": [], "deny": []}, "version": 3})
    assert res.status_code == 200 and res.json() == {"version": 4, "changed": True}
    assert asked == ["policy.write"] and did == [("replace", "alice", 3)]


def test_bad_rules_are_400_and_a_stale_version_409(rig):
    c, _, _, _ = rig
    assert c.put("/admin/egress-policy", json={"rules": {"allow": "bad"}, "version": 3}).status_code == 400
    assert c.put("/admin/egress-policy", json={"rules": {"allow": []}, "version": 2}).status_code == 409


def test_a_refused_gate_changes_nothing(rig):
    c, asked, did, mp = rig
    mp.setattr(r, "gate", _refuse)
    assert c.put("/admin/egress-policy", json={"rules": {"allow": []}, "version": 3}).status_code == 403
    assert c.post("/admin/egress-policy/seed").status_code == 403
    assert c.get("/admin/egress-policy").status_code == 403
    assert did == []


def test_seed_asks_to_write(rig):
    c, asked, did, _ = rig
    assert c.post("/admin/egress-policy/seed").json()["changed"] is True
    assert asked == ["policy.write"] and did == [("seed", "alice")]
