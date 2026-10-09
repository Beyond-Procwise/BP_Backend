"""Repeat-5: the same live conflict decided the same way N times proposes a standing rule (brief test 18).

Needs PROCWISE_TEST_LIVE_DB=1 and DB_NAME=bp_testdb. The two policies are REAL live policies (the
proposal is raised against the policies as they now are) with tool tst_<hex>, so they can never meet
anyone else's; they are RETIRED at teardown. The gate is fed those same documents through
gate._load_policies. Every case, action, conflict and notification row made here is removed
afterwards; firing rows are append-only BY DESIGN and stay.
"""
import os
from datetime import datetime, timedelta, timezone

import pytest

from repositories import agent_policy_repo as repo
from services.agent_policy import approvals as A
from services.agent_policy import conflict_live as CL
from services.agent_policy import settings as S
from tests.agent_policy import test_conflict_live_gate as T

pytestmark = pytest.mark.skipif(os.getenv("PROCWISE_TEST_LIVE_DB") != "1", reason="live DB required")


@pytest.fixture
def conn():
    assert os.getenv("DB_NAME") == "bp_testdb", "conflict live tests run on bp_testdb only"
    from services.db import get_conn
    with get_conn() as c:
        yield c


@pytest.fixture
def world(conn, monkeypatch):
    monkeypatch.setenv("AGENT_POLICY_ENFORCEMENT", "on")
    w = T.new_world(conn)
    w.n = 0
    monkeypatch.setattr(repo, "_activation_problems", lambda f, r, s: [])   # tst_ tools are unknown
    monkeypatch.setattr(repo, "_contract_problems", lambda d, r: [])        # to the real registry
    forms = {"A": T.make_form(tool=w.tool, outcome="approve", deciders=[w.la1, w.la2], source=f"TST Finance {w.tag}"),
             "B": T.make_form(tool=w.tool, outcome="approve", deciders=[w.lb], source=f"TST Customer {w.tag}")}
    w.docs = {}
    try:
        for letter, form in forms.items():
            form["name"] = f"TST live conflict {letter} {w.tag}"
            form["businessArea"], form["subArea"] = None, None
            form["owner"] = w.la2 if letter == "A" else w.lb
            key = repo.create_draft(conn, form, actor="test")["policyKey"]
            w.keys.append(key)
            repo.save_version(conn, key, form, base_version=1, intent="activate", actor="test", change_note="go")
            got = repo.get_policy(conn, key)
            [v] = [v for v in got["versions"] if v["version"] == got["liveVersion"]]
            w.docs[letter] = v["compiled"]
        yield w
    finally:
        T.cleanup(conn, w)
        for key in w.keys:
            got = repo.get_policy(conn, key)
            if got["status"] != "retired":
                repo.retire(conn, key, base_version=got["latestVersion"], actor="test", change_note="test teardown")


def once(conn, w, monkeypatch, verb="approve"):
    """One gate call that pauses on the A/B conflict, then the members decided (or timed out)."""
    w.n += 1
    wf = f"{w.wf}-{w.n}"
    w.wfs.append(wf)
    T.use(monkeypatch, [w.docs["A"], w.docs["B"]])
    res, _ = T.run(monkeypatch, T.stub_tools(w, []), [T._round(w.tool), T.FINAL], workflow_id=wf)
    assert res.calls[0].result["result"] == "paused_for_approval"
    ma, mb = T.rows(conn, "SELECT decision_id FROM proc.bp_decision WHERE subject_type = %s AND workflow_id = %s "
                          "AND decision = 'approve_or_reject' ORDER BY decision_id", (A.SUBJECT_TYPE, wf))
    if verb == "timeout":
        later = datetime.now(timezone.utc) + timedelta(hours=5)
        A.sweep(conn, later, decision_ids=[ma["decision_id"], mb["decision_id"]])
    elif verb == "approve":
        T.act(conn, w, ma["decision_id"], w.la2)
        T.act(conn, w, mb["decision_id"], w.lb)
    else:
        T.act(conn, w, ma["decision_id"], w.la2, verb="reject")
    [lv] = T.rows(conn, "SELECT c.outcome, c.by_person FROM proc.bp_decision d JOIN proc.bp_agent_policy_conflict c "
                        "USING (decision_id) WHERE d.subject_type = %s AND d.workflow_id = %s",
                  (CL.SUBJECT_LIVE, wf))
    return lv


def repeat_cases(conn, w):
    return [p for p in T.policy_cases(conn, w) if p["raised_by"] == "repeat"]


def test_same_conflict_decided_same_way_5_times_raises_policy_case(conn, world, monkeypatch):
    for i in range(4):
        assert once(conn, world, monkeypatch) == {"outcome": "approve", "by_person": True}
        assert repeat_cases(conn, world) == [], f"no proposal after {i + 1}"
    once(conn, world, monkeypatch)
    [pc] = repeat_cases(conn, world)
    assert pc["subject_id"] == "|".join(sorted(world.keys)) and pc["is_open"]
    assert pc["facts"]["proposal"] == {"from": "repeat", "count": 5, "outcome": "approve"}
    once(conn, world, monkeypatch)
    assert len(repeat_cases(conn, world)) == 1, "a 6th raises no second case while one is open"


def test_mixed_outcomes_reset_the_count(conn, world, monkeypatch):
    for _ in range(4):
        once(conn, world, monkeypatch)
    assert once(conn, world, monkeypatch, "reject") == {"outcome": "reject", "by_person": True}
    for _ in range(4):
        once(conn, world, monkeypatch)
    assert repeat_cases(conn, world) == [], "the reject reset the count"
    once(conn, world, monkeypatch)
    assert [p["facts"]["proposal"]["outcome"] for p in repeat_cases(conn, world)] == ["approve"]


def test_timeouts_do_not_count(conn, world, monkeypatch):
    for _ in range(4):
        once(conn, world, monkeypatch)
    assert once(conn, world, monkeypatch, "timeout") == {"outcome": "reject", "by_person": False}
    assert repeat_cases(conn, world) == [], "a timeout is not a fifth decision"
    once(conn, world, monkeypatch)
    assert [p["facts"]["proposal"]["count"] for p in repeat_cases(conn, world)] == [5], \
        "a timeout neither counts nor resets"


def test_threshold_comes_from_company_setting(conn, world, monkeypatch):
    monkeypatch.setattr(S, "load_settings", lambda conn=None: S.merge({"live_conflict_repeat": 2}))
    once(conn, world, monkeypatch)
    assert repeat_cases(conn, world) == []
    once(conn, world, monkeypatch)
    [pc] = repeat_cases(conn, world)
    assert pc["facts"]["proposal"] == {"from": "repeat", "count": 2, "outcome": "approve"}
