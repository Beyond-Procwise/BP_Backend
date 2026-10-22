"""Four more writes ask the gate first (audit RT-16b).

Confirming or rejecting a proposed order link writes (or bars) a reference on a final document: asked as
promoting a document (``document.promote``, write: Buyer and above with no policy row). Stating or
withdrawing a negotiation fact changes the advice recorded for a deal: asked as writing a deal
(``deal.write``). A refusal changes nothing.
"""
from __future__ import annotations

import os
import sys

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import api.routers.negotiate as ng
import api.routers.promotion as pr
from api.endpoint_gate import NotPermitted


@pytest.fixture()
def rig(monkeypatch):
    asked, did = [], []
    monkeypatch.setattr(pr, "gate", lambda action, *a, **k: asked.append(action))
    monkeypatch.setattr(ng, "gate", lambda action, *a, **k: asked.append(action))
    # The real callees, patched WITHOUT raising=False: a wrong name must fail loudly, not create an
    # unused attribute while the real service runs (review of 16e5c3ec).
    monkeypatch.setattr(pr, "confirm_parent_link", lambda *a, **k: did.append("confirm") or {"status": "linked"})
    monkeypatch.setattr(pr, "reject_parent_link", lambda *a, **k: did.append("reject") or {"status": "rejected"})
    monkeypatch.setattr(ng, "apply_turn", lambda deal_id, body, **k: did.append(("turn", deal_id, body.get("action"))) or {"advice": []})
    app = FastAPI()
    app.include_router(pr.router)
    app.include_router(ng.router)
    for dep in (pr.require_user, ng.require_user):
        app.dependency_overrides[dep] = lambda: type("P", (), {"subject": "alice"})()
    return TestClient(app, raise_server_exceptions=False), asked, did, monkeypatch


def test_advice_writes_ask_to_write_a_deal_first(rig):
    http, asked, did, _ = rig
    assert http.post("/deals/D1/advice/message", json={"action": "state_fact", "text": "x"}).status_code == 200
    assert http.delete("/deals/D1/advice/fact/price").status_code == 200
    assert asked == ["deal.write", "deal.write"] and [d[0] for d in did] == ["turn", "turn"]


def test_link_decisions_ask_to_promote_a_document_first(rig):
    http, asked, did, _ = rig
    r1 = http.post("/promotion/link-proposals/invoice/INV1/confirm", json={"po_id": "PO1"})
    r2 = http.post("/promotion/link-proposals/invoice/INV1/reject", json={"po_id": "PO1"})
    assert r1.status_code == 200 and r2.status_code == 200, (r1.text[:120], r2.text[:120])
    assert asked == ["document.promote", "document.promote"]
    assert did == ["confirm", "reject"], "the gate is followed by the write"


def test_a_refusal_changes_nothing(rig):
    http, asked, did, mp = rig

    def _no(*a, **k):
        raise NotPermitted("no")
    mp.setattr(pr, "gate", _no)
    mp.setattr(ng, "gate", _no, raising=False)
    assert http.post("/deals/D1/advice/message", json={"action": "state_fact"}).status_code == 403
    assert http.delete("/deals/D1/advice/fact/price").status_code == 403
    assert http.post("/promotion/link-proposals/invoice/INV1/confirm", json={"po_id": "PO1"}).status_code == 403
    assert http.post("/promotion/link-proposals/invoice/INV1/reject", json={"po_id": "PO1"}).status_code == 403
    assert did == []
