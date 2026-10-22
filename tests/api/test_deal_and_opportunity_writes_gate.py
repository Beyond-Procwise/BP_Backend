"""Six more writes ask the gate before touching anything (audit RT-12 and the deal-summary gaps).

Advancing an opportunity's stage is asked as reviewing an opportunity (``opportunity.review``). Naming,
saving, promoting and reconciling a deal, and the bulk summary backfill, are asked as writing a deal
(``deal.write``). Both are 'write'-class actions: with no policy row Buyers and above are allowed, so no
screen loses a button; an Admin can narrow either with a policy row. A refusal changes nothing.
"""
from __future__ import annotations

import os
import sys

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import api.routers.deal_summary as ds
import api.routers.opportunities as op
from api.endpoint_gate import NotPermitted


@pytest.fixture()
def rig(monkeypatch):
    asked, did = [], []
    monkeypatch.setattr(ds, "gate", lambda action, *a, **k: asked.append(action), raising=False)
    monkeypatch.setattr(op, "gate", lambda action, *a, **k: asked.append(action))
    monkeypatch.setattr(ds, "save_deal", lambda deal_id, name, actor=None: did.append(("save", deal_id, actor)))
    monkeypatch.setattr(ds, "save_reference", lambda deal_id: did.append(("reference", deal_id)))
    monkeypatch.setattr(ds, "promote_deal", lambda deal_id: did.append(("promote", deal_id)))
    monkeypatch.setattr(ds, "reconcile_deal", lambda deal_id: did.append(("reconcile", deal_id)) or {"deal_id": deal_id})
    monkeypatch.setattr(ds, "deal_view", lambda deal_id: {"deal_id": deal_id}, raising=False)
    import src.services.deal_analysis_service as das
    monkeypatch.setattr(das, "sync_deal_summaries", lambda: did.append(("sync",)) or {"synced": 0})
    monkeypatch.setattr(op, "set_stage", lambda oid, stage, savings: did.append(("stage", oid, stage)))
    app = FastAPI()
    app.include_router(ds.router)
    app.include_router(op.router)
    for dep in (ds.require_user, op.require_user):
        app.dependency_overrides[dep] = lambda: type("P", (), {"subject": "alice"})()
    return TestClient(app), asked, did, monkeypatch


CALLS = [
    ("post", "/deals/D1/save", {"name": "Laptops Q4"}, "deal.write"),
    ("post", "/deals/D1/save-reference", None, "deal.write"),
    ("post", "/deals/D1/promote", None, "deal.write"),
    ("post", "/deals/D1/reconcile", None, "deal.write"),
    ("post", "/deals/analysis-summary/sync", None, "deal.write"),
    ("post", "/opportunities/O1/stage", {"stage": "identified"}, "opportunity.review"),
]


@pytest.mark.parametrize("method,path,body,action", CALLS)
def test_each_write_asks_the_gate_first(rig, method, path, body, action):
    http, asked, did, _ = rig
    res = getattr(http, method)(path, json=body) if body is not None else getattr(http, method)(path)
    assert res.status_code == 200, res.text
    assert asked == [action] and len(did) == 1


@pytest.mark.parametrize("method,path,body,action", CALLS)
def test_a_refusal_changes_nothing(rig, method, path, body, action):
    http, asked, did, mp = rig

    def _no(*a, **k):
        raise NotPermitted("no")
    mp.setattr(ds, "gate", _no, raising=False)
    mp.setattr(op, "gate", _no)
    # "before any database work" is asserted, not inherited: opening a connection is itself a failure here.
    for mod in (ds, op):
        if hasattr(mod, "get_conn"):
            mp.setattr(mod, "get_conn", lambda: (_ for _ in ()).throw(AssertionError("database opened after a refusal")))
    res = getattr(http, method)(path, json=body) if body is not None else getattr(http, method)(path)
    assert res.status_code == 403
    assert did == []
