"""Who may change a saved agent group (decision D-61, answer B).

Groups are SHARED: everyone lists them. Only the CREATOR or an ADMIN may edit or delete one.
A legacy group whose creator is blank, missing or "system" belongs to nobody, so only an Admin
may change it. Creating stays open to any signed-in user. With sign-in switched off there is no
principal at all and nothing is refused (as before).

No database: the repository is replaced by an in-memory fake and the caller's role by a table.
"""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api import auth
from api.auth import Principal
from api.routers import agent_groups as router_mod

_MEMBERS = [{"agent_slug": "data_extraction", "display_name": "D", "dx": 0, "dy": 0}]


class FakeRepo:
    def __init__(self):
        self.rows = {}
        self.next_id = 1
        self.writes = []

    def add(self, created_by):
        gid = self.next_id
        self.next_id += 1
        self.rows[gid] = {"group_id": gid, "name": f"g{gid}", "members": list(_MEMBERS),
                          "links": [], "created_by": created_by, "active": True}
        return gid

    def list_active(self):
        return [r for r in self.rows.values() if r["active"]]

    def get(self, gid):
        r = self.rows.get(gid)
        return r if r and r["active"] else None

    def create(self, name, members, links, created_by=None):
        gid = self.add(created_by)
        self.rows[gid]["name"] = name
        return gid

    def update(self, gid, **kw):
        self.writes.append(("update", gid))
        if kw.get("name") is not None:
            self.rows[gid]["name"] = kw["name"]

    def soft_delete(self, gid):
        self.writes.append(("delete", gid))
        if gid in self.rows:
            self.rows[gid]["active"] = False


ROLES = {"admin-1": "Admin", "alice": "Buyer", "bob": "Buyer"}


@pytest.fixture
def env(monkeypatch):
    repo = FakeRepo()
    monkeypatch.setattr(router_mod, "repo", repo)
    # The write gate on create (RT-23) has its own suite; these tests are about ownership of edits/deletes.
    monkeypatch.setattr(router_mod, "gate", lambda *a, **k: None)
    monkeypatch.setattr(router_mod.rbac, "effective_role",
                        lambda p, policy_engine=None: ROLES.get(getattr(p, "subject", None), "Viewer"))
    app = FastAPI()
    app.include_router(router_mod.router)
    state = {"who": None}
    app.dependency_overrides[auth.require_user] = lambda: (
        Principal(subject=state["who"]) if state["who"] else None)
    client = TestClient(app)

    def as_(subject):
        state["who"] = subject
        return client

    return repo, as_


def test_creator_may_edit_and_delete_own_group(env):
    repo, as_ = env
    gid = repo.add("alice")
    assert as_("alice").put(f"/agent-groups/{gid}", json={"name": "renamed"}).status_code == 200
    assert repo.rows[gid]["name"] == "renamed"
    assert as_("alice").delete(f"/agent-groups/{gid}").status_code == 200
    assert repo.rows[gid]["active"] is False


def test_other_user_refused_on_put_and_delete_and_nothing_written(env):
    repo, as_ = env
    gid = repo.add("alice")
    r = as_("bob").put(f"/agent-groups/{gid}", json={"name": "hijack"})
    assert r.status_code == 403
    assert "created" in r.json()["detail"].lower() and "admin" in r.json()["detail"].lower()
    assert as_("bob").delete(f"/agent-groups/{gid}").status_code == 403
    assert repo.writes == [] and repo.rows[gid]["name"] == f"g{gid}" and repo.rows[gid]["active"]


def test_admin_may_edit_and_delete_anyones_group(env):
    repo, as_ = env
    gid = repo.add("alice")
    assert as_("admin-1").put(f"/agent-groups/{gid}", json={"name": "x"}).status_code == 200
    assert as_("admin-1").delete(f"/agent-groups/{gid}").status_code == 200


@pytest.mark.parametrize("legacy", [None, "", "   ", "system", "SYSTEM"])
def test_legacy_group_is_admin_only(env, legacy):
    repo, as_ = env
    gid = repo.add(legacy)
    assert as_("bob").put(f"/agent-groups/{gid}", json={"name": "x"}).status_code == 403
    assert as_("bob").delete(f"/agent-groups/{gid}").status_code == 403
    assert repo.writes == []
    assert as_("admin-1").put(f"/agent-groups/{gid}", json={"name": "x"}).status_code == 200
    assert as_("admin-1").delete(f"/agent-groups/{gid}").status_code == 200


def test_missing_group_is_404_for_everyone_before_any_permission_talk(env):
    _, as_ = env
    for who in ("alice", "admin-1"):
        assert as_(who).put("/agent-groups/999", json={"name": "x"}).status_code == 404
        assert as_(who).delete("/agent-groups/999").status_code == 404


def test_deleted_group_is_404_and_hidden_from_list(env):
    repo, as_ = env
    gid = repo.add("alice")
    as_("alice").delete(f"/agent-groups/{gid}")
    assert as_("alice").delete(f"/agent-groups/{gid}").status_code == 404
    assert all(g["group_id"] != gid for g in as_("alice").get("/agent-groups").json()["groups"])


def test_list_is_shared_everyone_sees_everything(env):
    repo, as_ = env
    a, legacy = repo.add("alice"), repo.add(None)
    seen = {g["group_id"] for g in as_("bob").get("/agent-groups").json()["groups"]}
    assert seen == {a, legacy}


def test_any_signed_in_user_may_create_and_is_recorded_as_owner(env):
    repo, as_ = env
    body = {"name": "mine", "members": _MEMBERS, "links": []}
    r = as_("bob").post("/agent-groups", json=body)
    assert r.status_code == 200
    assert repo.rows[r.json()["group_id"]]["created_by"] == "bob"


def test_sign_in_off_nothing_is_refused(env):
    repo, as_ = env
    gid = repo.add("alice")
    assert as_(None).put(f"/agent-groups/{gid}", json={"name": "x"}).status_code == 200
    assert as_(None).delete(f"/agent-groups/{gid}").status_code == 200
