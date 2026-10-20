"""The extraction-feedback write routes ask the gate before they act (audit RT-5).

Approving a proposal writes a versioned hint into proc.bp_prompt -- the governance table whose prompts
override the code -- so it is asked as promoting a prompt. Rejecting is the same reviewer's decision.
Running the proposer only creates drafts, so it is asked as authoring one. A refusal does nothing.
"""
from __future__ import annotations

import os
import sys

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import api.routers.extraction_feedback as ef
from api.endpoint_gate import NotPermitted


@pytest.fixture()
def rig(monkeypatch):
    asked, did = [], []
    monkeypatch.setattr(ef, "gate", lambda action, *a, **k: asked.append(action))
    monkeypatch.setattr(ef.apply, "approve", lambda pid, who: did.append(("approve", pid)) or {"ok": True})
    monkeypatch.setattr(ef.apply, "reject", lambda pid, who, reason: did.append(("reject", pid)))
    monkeypatch.setattr(ef.proposer, "propose_all", lambda **k: did.append(("run", k)) or [1, 2])
    app = FastAPI()
    app.include_router(ef.router)
    app.dependency_overrides[ef.require_user] = lambda: None
    return TestClient(app), asked, did, monkeypatch


def _refuse(*a, **k):
    raise NotPermitted("no")


def test_approve_asks_to_promote_a_prompt(rig):
    http, asked, did, _mp = rig
    assert http.post("/extraction/proposals/7/approve", json={}).status_code == 200
    assert asked == ["prompt.promote"]
    assert did == [("approve", 7)]


def test_approve_refused_applies_nothing(rig):
    http, _asked, did, mp = rig
    mp.setattr(ef, "gate", _refuse)
    assert http.post("/extraction/proposals/7/approve", json={}).status_code == 403
    assert did == []


def test_reject_asks_to_promote_a_prompt(rig):
    http, asked, did, _mp = rig
    assert http.post("/extraction/proposals/7/reject", json={"reason": "wrong vendor"}).status_code == 200
    assert asked == ["prompt.promote"]
    assert did == [("reject", 7)]


def test_reject_refused_records_nothing(rig):
    http, _asked, did, mp = rig
    mp.setattr(ef, "gate", _refuse)
    assert http.post("/extraction/proposals/7/reject", json={}).status_code == 403
    assert did == []


def test_run_asks_to_author_a_prompt(rig):
    http, asked, did, _mp = rig
    resp = http.post("/extraction/proposals/run")
    assert resp.status_code == 200 and resp.json()["count"] == 2
    assert asked == ["prompt.author"]


def test_run_refused_runs_nothing(rig):
    http, _asked, did, mp = rig
    mp.setattr(ef, "gate", _refuse)
    assert http.post("/extraction/proposals/run").status_code == 403
    assert did == []
