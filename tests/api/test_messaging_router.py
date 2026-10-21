"""Messaging routes: signed-in only, always the caller's own subject; the live signal never reaches the client body."""
from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import api.routers.messaging as r
from services.people.admin import PeopleError


@pytest.fixture()
def rig(monkeypatch):
    calls, pushed = [], []
    monkeypatch.setattr(r, "get_conn", contextmanager(lambda: (yield "CONN")))
    for name in ("open_direct", "open_group", "open_entity_thread", "list_conversations", "suggest_recipients",
                 "get_conversation", "list_messages", "mark_read", "set_muted", "add_participant", "leave", "edit", "delete"):
        monkeypatch.setattr(r.svc, name, (lambda n: lambda conn, *a, **k: calls.append((n, a)) or {"ok": n})(name))
    monkeypatch.setattr(r.svc, "send", lambda conn, *a, **k: calls.append(("send", a)) or
                        {"id": 7, "body": "hi", "signal": ["bob", "carol"]})
    monkeypatch.setattr(r.realtime, "push", lambda subjects, payload: pushed.append((sorted(subjects), payload)))
    app = FastAPI(); app.include_router(r.router)
    user = {"p": SimpleNamespace(subject="alice")}
    app.dependency_overrides[r.require_user] = lambda: user["p"]
    return TestClient(app), calls, pushed, user, monkeypatch


def test_everything_needs_a_signed_in_person(rig):
    c, calls, _, user, _ = rig
    user["p"] = None
    for resp in (c.post("/messages/direct", json={"to": "b"}), c.post("/messages/groups", json={"title": "t", "members": ["b"]}),
                 c.post("/messages/entity-thread", json={"entity_type": "po", "entity_id": "1"}),
                 c.get("/messages/conversations"), c.get("/messages/suggest-recipients"), c.get("/messages/conversations/1"),
                 c.get("/messages/conversations/1/messages"), c.post("/messages/conversations/1/messages", json={"body": "x"}),
                 c.post("/messages/conversations/1/read", json={}), c.put("/messages/conversations/1/mute", json={"muted": True}),
                 c.post("/messages/conversations/1/participants", json={"subject": "b"}), c.post("/messages/conversations/1/leave"),
                 c.put("/messages/1", json={"body": "x"}), c.delete("/messages/1")):
        assert resp.status_code == 401, resp.request.url
    assert calls == []


def test_the_caller_is_always_the_acting_subject(rig):
    c, calls, *_ = rig
    c.post("/messages/direct", json={"to": "bob"}); c.get("/messages/conversations")
    c.post("/messages/conversations/3/read", json={}); c.delete("/messages/9")
    assert all(a[0] == "alice" for _, a in calls)


def test_send_signals_participants_but_hides_the_signal_list(rig):
    c, calls, pushed, *_ = rig
    out = c.post("/messages/conversations/3/messages", json={"body": "hi"}).json()
    assert "signal" not in out and out["id"] == 7
    assert pushed == [(["bob", "carol"], {"type": "message", "conversation_id": 3, "message_id": 7})]
    assert "hi" not in str(pushed)                                      # signals never carry text


def test_service_errors_become_http_errors(rig):
    c, _, pushed, _, mp = rig
    def boom(conn, *a, **k): raise PeopleError(404, "no such conversation")
    mp.setattr(r.svc, "send", boom)
    resp = c.post("/messages/conversations/3/messages", json={"body": "hi"})
    assert resp.status_code == 404 and resp.json()["detail"] == "no such conversation" and pushed == []
