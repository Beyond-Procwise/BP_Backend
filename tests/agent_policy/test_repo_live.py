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
