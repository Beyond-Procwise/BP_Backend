"""The four value-ledger writes ask the gate first (audit RT-15).

Closing a money finding, settling its claim, marking an opportunity's saving realised and correcting a
recorded figure all append to the value ledger (proc.bp_value_outcome), which the Value Found headline
and the savings reports are built from. They are asked as ``value.record``, a write-class action: with
no policy row Buyers and above are allowed, so the Action Centre keeps its buttons; an Admin can narrow
it (for example to Approvers) with one policy row. A refusal appends nothing.
"""
from __future__ import annotations

import os
import sys

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import api.routers.value_ledger as vl
from api.endpoint_gate import NotPermitted
from src.services import actions


def test_the_ledger_action_is_a_write():
    assert actions.action_class("value.record") == "write"


@pytest.fixture()
def rig(monkeypatch):
    asked, did = [], []
    monkeypatch.setattr(vl, "gate", lambda action, *a, **k: asked.append(action), raising=False)
    for name in ("record_finding_outcome", "settle_claim", "realise_opportunity", "correct_outcome"):
        monkeypatch.setattr(vl.value_ledger, name, (lambda n: lambda *a, **k: did.append((n, k.get("actor"))) or {"outcome_id": 1})(name))
    app = FastAPI()
    app.include_router(vl.router)
    app.dependency_overrides[vl.require_user] = lambda: type("P", (), {"subject": "alice"})()
    return TestClient(app), asked, did, monkeypatch


CALLS = [
    ("/value/findings/7/outcome", {"outcome": "recovered", "amount": "5", "currency": "GBP"}, "record_finding_outcome"),
    ("/value/findings/7/settle", {"outcome": "credit_received", "amount": "5", "currency": "GBP"}, "settle_claim"),
    ("/value/opportunities/O1/realise", {"amount": "5", "currency": "GBP"}, "realise_opportunity"),
    ("/value/outcomes/9/correct", {"amount": "4", "currency": "GBP", "note": "typo"}, "correct_outcome"),
]


@pytest.mark.parametrize("path,body,fn", CALLS)
def test_each_write_asks_the_gate_first_and_records_the_token_as_actor(rig, path, body, fn):
    http, asked, did, _ = rig
    res = http.post(path, json=body)
    assert res.status_code == 200, res.text
    assert asked == ["value.record"] and did == [(fn, "alice")]


@pytest.mark.parametrize("path,body,fn", CALLS)
def test_a_refusal_appends_nothing(rig, path, body, fn):
    http, asked, did, mp = rig

    def _no(*a, **k):
        raise NotPermitted("no")
    mp.setattr(vl, "gate", _no, raising=False)
    assert http.post(path, json=body).status_code == 403 and did == []
