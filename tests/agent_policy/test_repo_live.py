"""Repository against bp_testdb. Needs PROCWISE_TEST_LIVE_DB=1 and DB_NAME=bp_testdb.
Every test uses a fresh policy under the Unassigned (GEN) prefix; versions are immutable,
so nothing is cleaned up. That is the point of the table."""
import copy
import os

import pytest

from repositories import agent_policy_repo as repo
from tests.agent_policy.fixtures import FORM_EXAMPLE

pytestmark = pytest.mark.skipif(os.getenv("PROCWISE_TEST_LIVE_DB") != "1", reason="live DB required")


@pytest.fixture
def conn():
    from services.db import get_conn
    with get_conn() as c:
        yield c


def test_draft_saves_with_only_a_name_and_gets_a_stable_id(conn):
    out = repo.create_draft(conn, {"name": "Name only"}, actor="test")
    assert out["version"] == 1 and out["policyKey"].startswith("GEN-")
    again = repo.save_version(conn, out["policyKey"], {"name": "Name only, renamed"},
                              base_version=1, intent="draft", actor="test", change_note="rename")
    assert again == {"policyKey": out["policyKey"], "version": 2}
    got = repo.get_policy(conn, out["policyKey"])
    assert [v["version"] for v in got["versions"]] == [1, 2] and got["status"] == "draft"


def test_ids_are_never_reused(conn):
    a = repo.create_draft(conn, {"name": "A"}, actor="test")["policyKey"]
    b = repo.create_draft(conn, {"name": "B"}, actor="test")["policyKey"]
    assert int(b.split("-")[1]) > int(a.split("-")[1])


def test_stale_base_version_is_refused(conn):
    key = repo.create_draft(conn, {"name": "Race"}, actor="test")["policyKey"]
    repo.save_version(conn, key, {"name": "Race 2"}, base_version=1, intent="draft", actor="a", change_note="")
    with pytest.raises(repo.StaleVersion):
        repo.save_version(conn, key, {"name": "Race 2b"}, base_version=1, intent="draft", actor="b", change_note="")


def test_activate_refused_with_every_problem(conn):
    key = repo.create_draft(conn, {"name": "Not ready"}, actor="test")["policyKey"]
    with pytest.raises(repo.NotReady) as err:
        repo.save_version(conn, key, {"name": "Not ready"}, base_version=1, intent="activate", actor="t", change_note="")
    assert len(err.value.problems) > 3
    assert repo.get_policy(conn, key)["latestVersion"] == 1   # nothing written on refusal


def test_live_stays_live_while_a_new_draft_exists(conn, monkeypatch):
    form = copy.deepcopy(FORM_EXAMPLE)
    form["businessArea"] = None  # keep it in GEN for tests; activation needs an area, so patch readiness
    monkeypatch.setattr(repo, "_activation_problems", lambda f, r, s: [])
    monkeypatch.setattr(repo, "_contract_problems", lambda d, r: [])
    key = repo.create_draft(conn, form, actor="test")["policyKey"]
    repo.save_version(conn, key, form, base_version=1, intent="activate", actor="t", change_note="go")
    repo.save_version(conn, key, {**form, "name": "edited"}, base_version=2, intent="draft", actor="t", change_note="edit")
    got = repo.get_policy(conn, key)
    assert got["status"] == "live" and got["liveVersion"] == 2 and got["latestVersion"] == 3
    assert key in {d["id"] for d in repo.live_documents(conn)}
    repo.retire(conn, key, base_version=3, actor="t", change_note="done")
    got = repo.get_policy(conn, key)
    assert got["status"] == "retired" and got["liveVersion"] is None and got["latestVersion"] == 4
    assert key not in {d["id"] for d in repo.live_documents(conn)}
    vs = {v["version"]: v for v in got["versions"]}
    assert vs[4]["form"] == vs[2]["form"] and vs[4]["form"] != vs[3]["form"]   # the form that was live, not the newer draft
    with pytest.raises(repo.InvalidTransition):
        repo.retire(conn, key, base_version=4, actor="t", change_note="again")
    after = repo.get_policy(conn, key)
    assert after["latestVersion"] == 4 and len(after["versions"]) == 4


def test_unknown_business_area_does_not_crash_and_leaves_area_unchanged(conn):
    key = repo.create_draft(conn, {"name": "Area probe"}, actor="test")["policyKey"]
    before = repo.get_policy(conn, key)["areaName"]
    out = repo.save_version(conn, key, {"name": "Area probe", "businessArea": "No such area"},
                           base_version=1, intent="draft", actor="test", change_note="")
    assert out["version"] == 2
    assert repo.get_policy(conn, key)["areaName"] == before
    bad = repo.create_draft(conn, {"name": "Area probe 2", "businessArea": "No such area"}, actor="test")
    assert bad["policyKey"].startswith("GEN-")
    assert repo.get_policy(conn, bad["policyKey"])["areaName"] is None


def _count(conn, key):
    cur = conn.cursor()
    cur.execute("SELECT count(*) FROM proc.bp_agent_policy_version WHERE policy_key=%s", (key,))
    return cur.fetchone()[0]


def test_activate_refused_with_readiness_and_contract_problems_in_one_list(conn, monkeypatch):
    key = repo.create_draft(conn, {"name": "Both"}, actor="test")["policyKey"]
    monkeypatch.setattr(repo, "_activation_problems", lambda f, r, s: [{"field": "outcome", "message": "r1", "routeTo": "author"}])
    monkeypatch.setattr(repo, "_contract_problems", lambda d, r: ["c1", "c2", "c1"])
    with pytest.raises(repo.NotReady) as err:
        repo.save_version(conn, key, {"name": "Both"}, base_version=1, intent="activate", actor="t", change_note="")
    msgs = [p["message"] for p in err.value.problems]
    assert msgs == ["r1", "c1", "c2"]
    assert err.value.problems[0]["field"] == "outcome" and err.value.problems[1]["field"] == "registry"
    assert repo.get_policy(conn, key)["latestVersion"] == 1 and _count(conn, key) == 1


def test_refused_activation_writes_nothing(conn, monkeypatch):
    key = repo.create_draft(conn, {"name": "Nothing written"}, actor="test")["policyKey"]
    monkeypatch.setattr(repo, "_activation_problems", lambda f, r, s: [])
    monkeypatch.setattr(repo, "_contract_problems", lambda d, r: ["x"])
    with pytest.raises(repo.NotReady):
        repo.save_version(conn, key, {"name": "Nothing written"}, base_version=1, intent="activate", actor="t", change_note="")
    got = repo.get_policy(conn, key)
    assert got["latestVersion"] == 1 and got["status"] == "draft" and _count(conn, key) == 1


def test_retiring_a_retired_policy_is_refused_and_writes_nothing(conn):
    key = repo.create_draft(conn, {"name": "Retire me"}, actor="test")["policyKey"]
    assert repo.retire(conn, key, base_version=1, actor="t", change_note="")["version"] == 2   # draft may be retired
    got = repo.get_policy(conn, key)
    assert got["versions"][1]["form"] == got["versions"][0]["form"]   # draft-only: latest form
    with pytest.raises(repo.InvalidTransition):
        repo.retire(conn, key, base_version=2, actor="t", change_note="")
    assert repo.get_policy(conn, key)["latestVersion"] == 2 and _count(conn, key) == 2


def test_create_draft_ignores_a_forged_checked_by(conn):
    key = repo.create_draft(conn, {"name": "Forged", "checked": {"by": "someone-else", "at": "1999-01-01T00:00:00Z"}},
                            actor="real-actor")["policyKey"]
    checked = repo.get_policy(conn, key)["versions"][0]["form"]["checked"]
    assert checked["by"] == "real-actor" and not checked["at"].startswith("1999")


def test_save_version_keeps_a_carried_confirmation_and_clears_after_change(conn):
    key = repo.create_draft(conn, {"name": "Carry", "situation": "s", "checked": {"by": "x", "at": "y"}},
                            actor="alice")["policyKey"]
    first = repo.get_policy(conn, key)["versions"][0]["form"]
    repo.save_version(conn, key, {**first, "owner": "Ops"}, base_version=1, intent="draft", actor="bob", change_note="")
    v2 = repo.get_policy(conn, key)["versions"][1]["form"]
    assert v2["checked"] == first["checked"] and v2["checked"]["by"] == "alice"
    repo.save_version(conn, key, {**v2, "situation": "new"}, base_version=2, intent="draft", actor="bob", change_note="")
    assert repo.get_policy(conn, key)["versions"][2]["form"]["checked"] is None   # stale confirmation cleared by the edit
