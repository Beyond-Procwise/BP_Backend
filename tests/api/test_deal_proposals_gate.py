"""The four deal-proposal writes ask the gate before touching anything (audit RT-11).

Generating proposals clusters a batch into deals; confirming mints a deal; rejecting bars a grouping for
good; moving members rewrites a proposal. All four are asked as writing a deal (``deal.write``), the
same class as the other deal writes. A refusal changes nothing.
"""
from __future__ import annotations

import os
import sys

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import api.routers.deal_proposals as dp
from api.endpoint_gate import NotPermitted


class _Cur:
    def execute(self, *a, **k):
        raise AssertionError("the database must not be touched when the gate refuses")


class _Conn:
    autocommit = True

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def cursor(self):
        return _Cur()

    def commit(self):
        pass

    def rollback(self):
        pass


@pytest.fixture()
def rig(monkeypatch):
    asked, did = [], []
    monkeypatch.setattr(dp, "gate", lambda action, *a, **k: asked.append(action), raising=False)
    monkeypatch.setattr(dp, "_generate", lambda batch, session: did.append(("generate", batch)) or {"proposals": []})
    monkeypatch.setattr(dp, "_confirm", lambda pid, by, expected: did.append(("confirm", pid, by)) or {"deal_id": "D1"})
    monkeypatch.setattr(dp.proposal_store, "reject_proposal", lambda cur, pid, by: did.append(("reject", pid, by)))
    monkeypatch.setattr(dp.proposal_store, "update_members", lambda cur, pid, remove: did.append(("members", pid)))
    monkeypatch.setattr(dp, "get_conn", lambda: _Conn())
    app = FastAPI()
    app.include_router(dp.router)
    app.dependency_overrides[dp.require_user] = lambda: type("P", (), {"subject": "alice"})()
    return TestClient(app), asked, did, monkeypatch


CALLS = [
    ("post", "/deals/proposals/generate", {"batch_deal_id": "B1"}),
    ("post", "/deals/proposals/7/confirm", {"confirmed_by": "ignored: the token names the confirmer"}),
    ("post", "/deals/proposals/7/reject", {"rejected_by": "ignored: the token names the rejecter"}),
    ("patch", "/deals/proposals/7/members", {"remove": []}),
]


@pytest.mark.parametrize("method,path,body", CALLS)
def test_each_write_asks_to_write_a_deal_first(rig, method, path, body):
    http, asked, did, _ = rig
    res = getattr(http, method)(path, json=body)
    assert res.status_code == 200, res.text
    assert asked == ["deal.write"] and len(did) == 1


@pytest.mark.parametrize("method,path,body", CALLS)
def test_a_refusal_changes_nothing(rig, method, path, body):
    http, asked, did, mp = rig

    def _no(*a, **k):
        raise NotPermitted("no")
    mp.setattr(dp, "gate", _no, raising=False)
    assert getattr(http, method)(path, json=body).status_code == 403
    assert did == []


def test_listing_proposals_is_not_a_write_and_is_not_gated(rig, monkeypatch):
    http, asked, did, _ = rig
    monkeypatch.setattr(dp.proposal_store, "list_proposals", lambda cur, batch: [])
    monkeypatch.setattr(dp, "get_conn", lambda: type("C", (), {"__enter__": lambda s: s, "__exit__": lambda s, *a: False,
                                                               "cursor": lambda s: None})())
    assert http.get("/deals/proposals?batch=B1").status_code == 200
    assert asked == []


# ---- through the REAL gate: what 'write' means with no policy row ------------------------------------

def test_with_no_policy_row_a_viewer_is_refused_and_a_buyer_is_allowed(monkeypatch):
    from src.services import agent_actions, guardrail, rbac
    from tests.guardrails.test_guardrail_gate import engine_with
    from tests.guardrails.test_rbac import FakePrincipal

    monkeypatch.setattr(guardrail.policy_observation, "record", lambda **k: True)
    monkeypatch.setattr(guardrail, "_raise_for_a_human", lambda **k: 1)
    audit = []
    monkeypatch.setattr(agent_actions, "record_action_or_fail", lambda **k: audit.append(k))
    monkeypatch.setattr(rbac, "policy_engine", lambda: engine_with())
    did = []
    monkeypatch.setattr(dp, "_confirm", lambda pid, by, expected: did.append(("confirm", pid, by)) or {"deal_id": "D1"})

    def serve(principal):
        app = FastAPI()
        app.include_router(dp.router)
        app.dependency_overrides[dp.require_user] = lambda: principal
        return TestClient(app, raise_server_exceptions=False)

    viewer = FakePrincipal("sub-viewer", {"cognito:groups": ["bp-viewers"]})
    buyer = FakePrincipal("sub-buyer", {"cognito:groups": ["bp-buyers"]})
    body = {"confirmed_by": "ignored"}
    assert serve(viewer).post("/deals/proposals/7/confirm", json=body).status_code == 403
    assert did == []
    assert serve(buyer).post("/deals/proposals/7/confirm", json=body).status_code == 200
    assert did == [("confirm", 7, "sub-buyer")]
    assert [a["status"] for a in audit] == ["denied", "allowed"], "both attempts are audited"
