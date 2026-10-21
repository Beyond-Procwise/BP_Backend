"""POST /promotion/canonicalize-po asks the gate before it rewrites po_id across _trgt (audit RT-7).

It edits final tables in bulk, so it is asked as promoting documents -- the same action as /promotion/run.
A refusal rewrites nothing.
"""
from __future__ import annotations

import os
import sys

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import api.routers.promotion as promo
from api.endpoint_gate import NotPermitted


@pytest.fixture()
def rig(monkeypatch):
    asked, did = [], []
    monkeypatch.setattr(promo, "gate", lambda action, *a, **k: asked.append(action))
    monkeypatch.setattr(promo, "canonicalize_po_references", lambda: did.append("canon") or {"updated": 0})
    app = FastAPI()
    app.include_router(promo.router)
    app.dependency_overrides[promo.require_user] = lambda: None
    return TestClient(app), asked, did, monkeypatch


def test_canonicalize_asks_to_promote_documents(rig):
    http, asked, did, _mp = rig
    assert http.post("/promotion/canonicalize-po").status_code == 200
    assert asked == ["document.renumber"]
    assert did == ["canon"]


def test_canonicalize_refused_rewrites_nothing(rig):
    http, _asked, did, mp = rig

    def _no(*a, **k):
        raise NotPermitted("no")

    mp.setattr(promo, "gate", _no)
    assert http.post("/promotion/canonicalize-po").status_code == 403
    assert did == []
