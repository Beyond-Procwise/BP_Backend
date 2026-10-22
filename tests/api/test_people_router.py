"""People / notification routes: admin routes ask the gate first and a refusal changes nothing;
own-data routes need a signed-in user and only ever touch that user's rows."""
from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import api.routers.people as r
from api.endpoint_gate import NotPermitted
from services.people import admin as svc
from services.people.models import Person
from services.people.stores import InMemoryPeopleStore


@pytest.fixture()
def rig(monkeypatch):
    asked, calls = [], []
    store = InMemoryPeopleStore([Person("alice"), Person("bob", manager_subject="alice")])
    monkeypatch.setattr(r, "gate", lambda action, *a, **k: asked.append(action))
    monkeypatch.setattr(r, "get_conn", contextmanager(lambda: (yield object())))
    monkeypatch.setattr(r, "PgPeopleStore", lambda conn: store)
    for name in ("list_people", "get_person", "upsert_person", "list_rules", "put_rule", "delete_rule",
                 "list_notifications", "unread_count", "mark_read", "mark_all_read", "get_prefs", "put_prefs"):
        monkeypatch.setattr(svc, name, (lambda n: lambda *a, **k: calls.append((n, a[1:], k)) or _RET.get(n))(name))
    app = FastAPI(); app.include_router(r.router)
    user = {"p": SimpleNamespace(subject="alice")}
    app.dependency_overrides[r.require_user] = lambda: user["p"]
    return TestClient(app), asked, calls, user, monkeypatch


_RET = {"list_people": [], "get_person": {"subject": "alice"}, "upsert_person": {"subject": "bob"},
        "list_rules": [], "list_notifications": {"items": [], "unread": 0, "next_before_id": None},
        "unread_count": 3, "mark_read": True, "mark_all_read": 2, "get_prefs": {}, "put_prefs": {"channels": ["email"]},
        "delete_rule": 1}


def refuse(*a, **k):
    raise NotPermitted("no")


def test_admin_reads_ask_policy_read_and_writes_ask_policy_write(rig):
    c, asked, calls, *_ = rig
    c.get("/admin/people"); c.get("/admin/people/alice"); c.get("/admin/routing-rules")
    assert asked == ["policy.read"] * 3
    asked.clear()
    c.put("/admin/people/bob", json={"approval_limit": 100})
    c.put("/admin/routing-rules/po.issued", json={"targets": ["entity_owner"]})
    c.delete("/admin/routing-rules/po.issued")
    assert asked == ["policy.write"] * 3


def test_refused_admin_write_changes_nothing(rig):
    c, asked, calls, user, mp = rig
    mp.setattr(r, "gate", refuse)
    for resp in (c.put("/admin/people/bob", json={"is_active": False}),
                 c.put("/admin/routing-rules/po.issued", json={"targets": ["manager"]}),
                 c.delete("/admin/routing-rules/po.issued")):
        assert resp.status_code == 403
    assert calls == []


def test_only_fields_sent_are_written(rig):
    c, asked, calls, *_ = rig
    c.put("/admin/people/bob", json={"approval_limit": 100})
    up = [x for x in calls if x[0] == "upsert_person"][0]
    assert list(up[1][1]) == ["approval_limit"] and up[1][2] == "alice"       # (subject, clean, by)


def test_validation_errors_are_422_and_write_nothing(rig):
    c, _, calls, *_ = rig
    assert c.put("/admin/people/bob", json={"manager_subject": "bob"}).status_code == 422
    assert c.put("/admin/people/alice", json={"manager_subject": "bob"}).status_code == 422     # alice <- bob loop
    assert c.put("/admin/routing-rules/po.issued", json={"targets": ["everyone"]}).status_code == 422
    assert c.put("/admin/routing-rules/nope.x", json={"targets": ["manager"]}).status_code == 404
    assert not any(x[0] in ("upsert_person", "put_rule") for x in calls)


def test_own_routes_need_a_signed_in_user(rig):
    c, _, calls, user, _ = rig
    user["p"] = None
    for resp in (c.get("/notifications"), c.get("/notifications/unread-count"), c.post("/notifications/read-all"),
                 c.post("/notifications/5/read"), c.get("/notifications/preferences"),
                 c.put("/notifications/preferences", json={}), c.get("/people/me"),
                 c.put("/people/me/delegate", json={})):
        assert resp.status_code == 401
    assert calls == []


def test_own_routes_are_scoped_to_the_caller(rig):
    c, _, calls, user, _ = rig
    user["p"] = SimpleNamespace(subject="bob")
    c.get("/notifications"); c.post("/notifications/9/read"); c.post("/notifications/read-all")
    c.put("/notifications/preferences", json={"channels": ["email"]})
    assert all(x[1][0] == "bob" for x in calls if x[0] in ("list_notifications", "mark_read", "mark_all_read", "put_prefs"))


def test_marking_someone_elses_notification_is_404(rig):
    c, _, calls, user, mp = rig
    mp.setattr(svc, "mark_read", lambda *a, **k: False)
    assert c.post("/notifications/123/read").status_code == 404


def test_preferences_cannot_mute_mandatory(rig):
    c, *_ = rig
    assert c.put("/notifications/preferences", json={"muted_events": ["approval.requested"]}).status_code == 422


def test_my_delegate_requires_being_set_up(rig):
    c, _, calls, user, mp = rig
    mp.setattr(svc, "get_person", lambda *a, **k: None)
    assert c.put("/people/me/delegate", json={"delegate_subject": "bob", "delegate_until": "2999-01-01T00:00:00Z"}).status_code == 404


def test_my_delegate_can_only_change_my_own_row(rig):
    c, _, calls, user, mp = rig
    user["p"] = SimpleNamespace(subject="alice")
    r_ = c.put("/people/me/delegate", json={"delegate_subject": "bob", "delegate_until": "2999-01-01T00:00:00Z"})
    assert r_.status_code == 200
    up = [x for x in calls if x[0] == "upsert_person"][0]
    assert up[1][0] == "alice" and set(up[1][1]) == {"delegate_subject", "delegate_until"} and up[1][2] == "alice"


def test_import_and_connected_events_are_admin_writes_and_starter_is_a_read(rig):
    c, asked, calls, user, mp = rig
    from services.people import orgchart
    mp.setattr(orgchart, "import_people", lambda *a, **k: {"created": [], "updated": [], "deactivated": [], "dry_run": True})
    from services.people import events
    mp.setattr(events, "emit", lambda conn, ev: True)
    assert c.get("/admin/people/starter-chart").status_code == 200
    assert c.post("/admin/people/import", json={"people": [], "dry_run": True}).status_code == 200
    ok = c.post("/integrations/events", json={"event_type": "requisition.submitted", "external_id": "PR-1"})
    assert ok.status_code == 200 and ok.json()["accepted"] is True
    assert asked == ["policy.read", "policy.write", "policy.write"]


def test_connected_events_refuse_types_that_are_ours(rig):
    c, *_ = rig
    r_ = c.post("/integrations/events", json={"event_type": "approval.approved", "external_id": "1"})
    assert r_.status_code == 422


def test_refused_import_and_event_write_nothing(rig):
    c, asked, calls, user, mp = rig
    from services.people import orgchart, events
    mp.setattr(r, "gate", refuse)
    mp.setattr(orgchart, "import_people", lambda *a, **k: pytest.fail("must not run"))
    mp.setattr(events, "emit", lambda *a, **k: pytest.fail("must not run"))
    assert c.post("/admin/people/import", json={"people": []}).status_code in (401, 403)
    assert c.post("/integrations/events", json={"event_type": "po.issued", "external_id": "1"}).status_code in (401, 403)
