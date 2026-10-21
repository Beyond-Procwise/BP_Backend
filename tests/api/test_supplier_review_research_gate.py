"""Supplier-master write routes ask the gate before they act (audit RT-8).

Confirming a review merges a name into the supplier master, rejecting it splits one, the duplicate
sweep flags suppliers for review, and rejecting an enrichment closes a proposal. All are the same
write the neighbouring /enrichment/{id}/apply already asks as supplier.write. A refusal changes nothing.
"""
from __future__ import annotations

import os
import sys

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import api.routers.supplier_research as sres
import api.routers.supplier_review as srev
from api.endpoint_gate import NotPermitted


class _Conn:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@pytest.fixture()
def rig(monkeypatch):
    asked, did = [], []

    def gate(action, *a, **k):
        asked.append(action)

    for m in (sres, srev):
        monkeypatch.setattr(m, "gate", gate, raising=False)
        monkeypatch.setattr(m, "get_conn", lambda: _Conn())
    monkeypatch.setattr(srev.SR, "sweep_supplier_duplicates", lambda c, min_score=None: did.append("sweep") or {})
    monkeypatch.setattr(srev.SR, "confirm_review", lambda i, who, c: did.append(("confirm", i)) or {})
    monkeypatch.setattr(srev.SR, "reject_review", lambda i, who, c: did.append(("reject", i)) or {})
    monkeypatch.setattr(sres.R, "reject_enrichment", lambda i, who, c: did.append(("enrich-reject", i)) or {})
    app = FastAPI()
    app.include_router(srev.router)
    app.include_router(sres.router)
    app.dependency_overrides[srev.require_user] = lambda: None
    return TestClient(app), asked, did, monkeypatch


CASES = [
    ("/reviews/sweep", "sweep", "/reviews/sweep", {}),
    ("/reviews/{review_id}/confirm", ("confirm", 5), "/reviews/5/confirm", {}),
    ("/reviews/{review_id}/reject", ("reject", 5), "/reviews/5/reject", {}),
    ("/enrichment/{enrichment_id}/reject", ("enrich-reject", 5), "/enrichment/5/reject", {}),
]


@pytest.mark.parametrize("suffix,effect,url,body", CASES, ids=[c[0] for c in CASES])
def test_asks_supplier_write_then_acts(rig, suffix, effect, url, body):
    http, asked, did, _mp = rig
    assert http.post("/suppliers" + url, json=body).status_code == 200
    assert asked == ["supplier.write"]
    assert did == [effect]


@pytest.mark.parametrize("suffix,effect,url,body", CASES, ids=[c[0] for c in CASES])
def test_refused_changes_nothing(rig, suffix, effect, url, body):
    http, _asked, did, mp = rig

    def _no(*a, **k):
        raise NotPermitted("no")

    mp.setattr(srev, "gate", _no, raising=False)
    mp.setattr(sres, "gate", _no)
    assert http.post("/suppliers" + url, json=body).status_code == 403
    assert did == []
