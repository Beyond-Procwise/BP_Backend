"""The opportunity sync and deal-link routes ask the gate before they act (audit RT-6).

POST /opportunities/sync upserts a findings file into proc.bp_opportunity -- and the caller may name
the file -- and POST /opportunities/link-deals rewrites deal_id on every opportunity. Both are the
miner's bulk write-back, so they are asked as the miner's own action. A refusal does nothing, and a
refused sync never opens the named file.
"""
from __future__ import annotations

import os
import sys

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import api.routers.opportunities as opp
from api.endpoint_gate import NotPermitted


@pytest.fixture()
def rig(monkeypatch):
    asked, did = [], []
    monkeypatch.setattr(opp, "gate", lambda action, *a, **k: asked.append(action))
    monkeypatch.setattr(opp, "sync_findings_from_json",
                        lambda path: did.append(("sync", path)) or {"synced": 1, "rejected": 0})
    monkeypatch.setattr(opp, "link_opportunities_to_deals", lambda: did.append(("link",)) or 3)
    app = FastAPI()
    app.include_router(opp.router)
    app.dependency_overrides[opp.require_user] = lambda: None
    return TestClient(app), asked, did, monkeypatch


def _refuse(*a, **k):
    raise NotPermitted("no")


def test_sync_asks_to_mine_opportunities(rig):
    http, asked, did, _mp = rig
    assert http.post("/opportunities/sync", params={"path": "/x.json"}).status_code == 200
    assert asked == ["opportunity.mine"]
    assert did == [("sync", "/x.json")]


def test_sync_refused_reads_no_file_and_writes_nothing(rig):
    http, _asked, did, mp = rig
    mp.setattr(opp, "gate", _refuse)
    assert http.post("/opportunities/sync", params={"path": "/etc/passwd"}).status_code == 403
    assert did == []


def test_link_deals_asks_to_mine_opportunities(rig):
    http, asked, did, _mp = rig
    assert http.post("/opportunities/link-deals").status_code == 200
    assert asked == ["opportunity.mine"]
    assert did == [("link",)]


def test_link_deals_refused_links_nothing(rig):
    http, _asked, did, mp = rig
    mp.setattr(opp, "gate", _refuse)
    assert http.post("/opportunities/link-deals").status_code == 403
    assert did == []
