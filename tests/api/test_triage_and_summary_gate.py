"""Re-triaging a deal and generating persona summaries ask the gate first (audit RT-14).

A manual re-triage recomputes a deal's findings and writes them to the Action Centre: asked as
``finding.triage`` (write class: Buyer and above with no policy row). Generating or pre-computing a
persona summary runs the model and stores the result: asked as ``report.generate`` (compute class:
Buyer and above). Reading a stored summary stays ungated. A refusal runs nothing.
"""
from __future__ import annotations

import os
import sys

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import api.routers.summary as sm
import api.routers.triage as tr
from api.endpoint_gate import NotPermitted
from src.services import actions


def test_the_triage_action_is_a_write():
    assert actions.action_class("finding.triage") == "write"
    assert actions.action_class("report.generate") == "compute"


class _Report:
    run_id, failed, deals_done, write_counts = "R1", {}, 1, {"findings": 2}


@pytest.fixture()
def rig(monkeypatch):
    asked, did = [], []
    monkeypatch.setattr(tr, "gate", lambda action, *a, **k: asked.append(action), raising=False)
    monkeypatch.setattr(sm, "gate", lambda action, *a, **k: asked.append(action), raising=False)
    monkeypatch.setattr(tr.engine, "run_triage", lambda ids, mode: did.append(("triage", tuple(ids))) or _Report())
    monkeypatch.setattr(tr.engine, "triage_deal_view", lambda deal_id: {"deal_id": deal_id})
    monkeypatch.setattr(sm.summary_agent, "generate_summary",
                        lambda persona, deal_id=None, as_of=None: did.append(("summary", persona)) or {"summary_id": 1, "text": "x"})
    monkeypatch.setattr(sm.summary_agent, "precompute_summaries",
                        lambda personas=None, deal_ids=None: did.append(("precompute",)) or {"generated": 0})
    app = FastAPI()
    app.include_router(tr.router)
    app.include_router(sm.router)
    for dep in (tr.require_user, sm.require_user):
        app.dependency_overrides[dep] = lambda: type("P", (), {"subject": "alice"})()
    return TestClient(app), asked, did, monkeypatch


CALLS = [
    ("/triage/deals/D1/run", None, "finding.triage"),
    ("/summary", {"persona": "cpo"}, "report.generate"),
    ("/summary/precompute", {"personas": ["cpo"]}, "report.generate"),
]


@pytest.mark.parametrize("path,body,action", CALLS)
def test_each_write_asks_the_gate_first(rig, path, body, action):
    http, asked, did, _ = rig
    res = http.post(path, json=body) if body is not None else http.post(path)
    assert res.status_code == 200, res.text
    assert asked == [action] and len(did) == 1


@pytest.mark.parametrize("path,body,action", CALLS)
def test_a_refusal_runs_nothing(rig, path, body, action):
    http, asked, did, mp = rig

    def _no(*a, **k):
        raise NotPermitted("no")
    mp.setattr(tr, "gate", _no, raising=False)
    mp.setattr(sm, "gate", _no, raising=False)
    res = http.post(path, json=body) if body is not None else http.post(path)
    assert res.status_code == 403 and did == []
