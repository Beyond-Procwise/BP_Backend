"""The exports routes: a file goes only to the person it was built for (or an Admin); others see 404."""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import api.routers.exports as r
from src.services.contract_tools import store

RECORD = {"export_id": "abc", "kind": "xlsx", "filename": "spend.xlsx", "media_type": r.export_store.MEDIA["xlsx"],
          "title": "Spend", "content": b"PK..", "size_bytes": 4, "source": "agent", "created_by": "carol", "created_date": "2026-10-22T18:00:00"}


@pytest.fixture(autouse=True)
def _no_gate(monkeypatch):
    """The gate on delete (RT-24) has its own suite; here every call is allowed."""
    monkeypatch.setattr(r, "gate", lambda *a, **k: None)


@pytest.fixture()
def rig(monkeypatch):
    calls = []
    monkeypatch.setattr(r.export_store, "get", lambda export_id, **k: calls.append(("get", export_id)) or (dict(RECORD) if export_id == "abc" else None))
    monkeypatch.setattr(r.export_store, "recent", lambda who, limit=20, **k: calls.append(("recent", who, limit)) or [{k_: v for k_, v in RECORD.items() if k_ != "content"}])
    monkeypatch.setattr(r.export_store, "delete", lambda export_id, created_by, **k: calls.append(("delete", export_id, created_by)) or (export_id == "abc" and created_by == "carol"))
    monkeypatch.setattr(r.rbac, "effective_role", lambda principal: "Buyer")
    app = FastAPI()
    app.include_router(r.router)
    who = {"subject": "carol"}
    app.dependency_overrides[r.require_user] = lambda: SimpleNamespace(subject=who["subject"])
    return TestClient(app), calls, who, monkeypatch


def test_the_owner_downloads_the_file_as_an_attachment(rig):
    c, calls, _, _ = rig
    out = c.get("/exports/abc")
    assert out.status_code == 200 and out.content == b"PK.." and out.headers["content-type"].startswith(r.export_store.MEDIA["xlsx"])
    assert out.headers["content-disposition"] == 'attachment; filename="spend.xlsx"' and out.headers["x-content-type-options"] == "nosniff"


def test_someone_else_gets_404_and_an_admin_gets_the_file(rig):
    c, _, who, mp = rig
    who["subject"] = "mallory"
    assert c.get("/exports/abc").status_code == 404
    mp.setattr(r.rbac, "effective_role", lambda principal: "Admin")
    assert c.get("/exports/abc").status_code == 200


def test_an_unknown_id_is_404_and_the_list_is_the_callers_own(rig):
    c, calls, _, _ = rig
    assert c.get("/exports/zzz").status_code == 404
    out = c.get("/exports?limit=5")
    assert out.status_code == 200 and out.json()["exports"][0]["export_id"] == "abc" and "content" not in out.json()["exports"][0]
    assert ("recent", "carol", 5) in calls


def test_delete_is_the_owners_act(rig):
    c, calls, who, _ = rig
    assert c.delete("/exports/abc").json() == {"deleted": "abc"}
    who["subject"] = "mallory"
    assert c.delete("/exports/abc").status_code == 404


def test_an_unreadable_store_is_503_not_404(rig):
    c, _, _, mp = rig

    def boom(*a, **k):
        raise store.StoreUnavailable("down")
    mp.setattr(r.export_store, "get", boom)
    assert c.get("/exports/abc").status_code == 503
