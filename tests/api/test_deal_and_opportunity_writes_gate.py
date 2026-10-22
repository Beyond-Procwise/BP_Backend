"""Six more writes ask the gate before touching anything (audit RT-12 and the deal-summary gaps).

Moving an opportunity's stage is its own act (``opportunity.stage``), except a move to "rejected", which is
the second half of the UI's Reject button and is asked as ``opportunity.reject`` so one Admin row governs
every rejection. Naming, saving, promoting and reconciling a deal, and the bulk summary backfill, are asked
as writing a deal (``deal.write``). All are 'write'-class actions: with no policy row Buyers and above are
allowed, so no screen loses a button; an Admin can narrow any of them with a policy row. A refusal writes
nothing to a deal or an opportunity (the gate's own audit row is written, as always).
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
    ("post", "/opportunities/O1/stage", {"stage": "negotiation"}, "opportunity.stage"),
    ("post", "/opportunities/O1/stage", {"stage": "rejected"}, "opportunity.reject"),
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
    res = getattr(http, method)(path, json=body) if body is not None else getattr(http, method)(path)
    assert res.status_code == 403
    assert did == []


def test_the_stage_action_is_its_own_write():
    from src.services import actions
    assert actions.action_class("opportunity.stage") == "write"
    assert actions.action_class("opportunity.reject") == "write"
