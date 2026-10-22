"""The vendor-onboarding writes ask the gate first (audit RT-13).

Uploading a document runs extraction on it (``document.extract``, compute: Buyer and above). Correcting
and saving teach the system a vendor TEMPLATE that changes how everyone's documents of that layout are
read from then on, so they are asked as ``extraction_template.write``, a configure-class action: Admin
only, refused-and-asked for anyone else, like the quote-template and entity-profile writes. No screen
calls these routes (the legacy HTML form at GET /vendors/onboard is the only client). A refusal changes
nothing.
"""
from __future__ import annotations

import os
import sys

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import api.routers.vendors as v
from api.endpoint_gate import NotPermitted
from src.services import actions


def test_the_template_action_is_a_configure():
    assert actions.action_class("extraction_template.write") == "configure"
    assert actions.action_class("document.extract") == "compute"


class _Store:
    def __init__(self):
        self.recorded = []

    def record_correction(self, **kw):
        self.recorded.append(kw)


@pytest.fixture()
def rig(monkeypatch):
    asked = []
    monkeypatch.setattr(v, "gate", lambda action, *a, **k: asked.append(action), raising=False)
    monkeypatch.setattr(v, "_run_pipeline", lambda body, fn, dt, store: {"fingerprint": "fp1", "fields": {}})
    store = _Store()
    monkeypatch.setattr(v, "_SESSIONS", {})
    app = FastAPI()
    app.include_router(v.build_router(store))
    app.dependency_overrides[v.require_user] = lambda: type("P", (), {"subject": "alice"})()
    return TestClient(app), asked, store


def _upload(http):
    return http.post("/vendors/onboard/upload", files={"file": ("a.pdf", b"%PDF", "application/pdf")},
                     data={"doc_type": "invoice"})


def test_upload_asks_to_extract_and_correct_and_save_ask_to_write_a_template(rig):
    http, asked, store = rig
    res = _upload(http)
    assert res.status_code == 200, res.text
    sid = res.json()["session_id"]
    assert asked == ["document.extract"]
    assert http.post(f"/vendors/onboard/{sid}/correct", json={"field": "total", "value": "5", "label": "x"}).status_code == 200
    assert http.post(f"/vendors/onboard/{sid}/save", json={"vendor_name": "Borcelle"}).status_code == 200
    assert asked == ["document.extract", "extraction_template.write", "extraction_template.write"]
    assert len(store.recorded) == 1 and store.recorded[0]["vendor_name"] == "Borcelle"


def test_a_refusal_changes_nothing(rig, monkeypatch):
    http, asked, store = rig
    sid = _upload(http).json()["session_id"]
    sess_before = dict(v._SESSIONS[sid].corrections)

    def _no(*a, **k):
        raise NotPermitted("no")
    monkeypatch.setattr(v, "gate", _no, raising=False)
    assert http.post(f"/vendors/onboard/{sid}/correct", json={"field": "total", "value": "5", "label": "x"}).status_code == 403
    assert http.post(f"/vendors/onboard/{sid}/save", json={"vendor_name": "Borcelle"}).status_code == 403
    assert _upload(http).status_code == 403
    assert v._SESSIONS[sid].corrections == sess_before and store.recorded == []
    assert len(v._SESSIONS) == 1, "a refused upload opens no session"
