"""Conflict endpoints and the approval-card conflict block through the WHOLE app against bp_testdb.

Needs PROCWISE_TEST_LIVE_DB=1 and DB_NAME=bp_testdb. Policies are DRAFTS saved with tools named
tst_<tag> (they can never contradict anyone else's) and RETIRED at teardown; detection passes
among=. Where a live conflict is needed, the same documents (same key, version 1) are injected
into the gate, so no live policy is ever written to the shared database, while the masking still
reads each policy version's sensitive inputs from the database as it does in production.
Every case, action, rule, conflict, notification and decider-map row made here is removed
afterwards, in FK order; firing rows are append-only BY DESIGN and stay.
"""
import copy
import json
import os
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from api.routers import agent_policies as R
from repositories import agent_policy_repo as repo
from services.agent_policy import conflict_cases as CC
from services.agent_policy import conflict_live as CL
from services.agent_policy.compiler import compile_policy
from services.agent_policy.enforcement import MASK
from tests.agent_policy import fixtures as F
from tests.agent_policy import test_conflict_live_gate as LG

pytestmark = pytest.mark.skipif(os.getenv("PROCWISE_TEST_LIVE_DB") != "1", reason="live DB required")

NOW = datetime(2026, 10, 9, 9, 0, tzinfo=timezone.utc)
ADMIN_SUB = "tst-admin-unlinked"


@pytest.fixture(autouse=True)
def enforcement_on(monkeypatch):
    monkeypatch.setenv("AGENT_POLICY_ENFORCEMENT", "on")


@pytest.fixture
def conn():
    assert os.getenv("DB_NAME") == "bp_testdb", "conflict live tests run on bp_testdb only"
    from services.db import get_conn
    with get_conn() as c:
        yield c


@pytest.fixture
def world(conn, monkeypatch):
    w = LG.new_world(conn)
    w.oa, w.ob = f"TST Owner A {w.tag}", f"TST Owner B {w.tag}"
    w.email_a, w.email_b = f"oa-{w.tag}@example.test", f"ob-{w.tag}@example.test"
    with conn.cursor() as cur:
        cur.execute("INSERT INTO proc.bp_policy_decider_map (decider_name, groups, emails, last_modified_by) "
                    "VALUES (%s, '{}', %s, 'test'), (%s, '{}', %s, 'test')",
                    (w.oa, [w.email_a], w.ob, [w.email_b]))
    w.people.update({w.oa: w.email_a, w.ob: w.email_b})
    w.audits = []
    monkeypatch.setattr(R.agent_actions, "record_action_or_fail", lambda **kw: w.audits.append(kw))
    monkeypatch.setattr(R, "_role_of", lambda p: "Admin" if p.subject == ADMIN_SUB else "Viewer")
    monkeypatch.setattr(R, "_replay_later", lambda did: None)
    yield w
    pairs = w.pairs()
    with conn.cursor() as cur:
        cur.execute("SELECT decision_id FROM proc.bp_decision WHERE subject_type = %s AND subject_id = ANY(%s)",
                    (CC.SUBJECT_POLICY, pairs))
        pids = [r[0] for r in cur.fetchall()]
        cur.execute("DELETE FROM proc.bp_policy_notification WHERE firing_id IS NULL AND link = ANY(%s)",
                    ([f"conflict:{i}" for i in pids],))
        cur.execute("DELETE FROM proc.bp_agent_policy_conflict_rule WHERE pair_key = ANY(%s)", (pairs,))
    LG.cleanup(conn, w)    # gate rows by workflow id, policy cases by pair, decider-map rows
    for key in w.keys:
        got = repo.get_policy(conn, key)
        if got["status"] != "retired":
            repo.retire(conn, key, base_version=got["latestVersion"], actor="test", change_note="test teardown")


@pytest.fixture
def client(monkeypatch):
    from api.main import app
    monkeypatch.setenv("AGENT_POLICY_GATEWAY_KEY", "k1")
    return TestClient(app)


def _hdr(sub, email=""):
    return {"X-Gateway-Key": "k1", "X-User-Sub": sub, "X-User-Email": email, "X-User-Groups": "[]"}


def _as(w, name):
    return _hdr(f"sub-{w.people[name]}", w.people[name])


STRANGER = _hdr("tst-stranger", "stranger@example.test")
ADMIN = _hdr(ADMIN_SUB, "admin@example.test")


def _form(w, *, outcome, gt, doc, owner, deciders=()):
    form = copy.deepcopy(F.FORM_EXAMPLE)
    form["name"] = f"TST {outcome} over {gt} {w.tag}"
    form["businessArea"], form["subArea"] = None, None
    form["outcome"] = outcome
    form["deciders"] = list(deciders) if outcome == "approve" else []
    form["notify"] = []
    form["owner"] = owner
    form["checked"] = {"by": "test", "at": "2026-10-09T08:00:00Z"}
    form["source"] = {"document": doc, "documentVersion": 1, "reference": "1.1",
                      "excerpt": f"Refunds above ${gt} need a decision ({outcome})."}
    form["hidden"]["actions"]["tools"] = [w.tool]
    form["hidden"]["condition"] = {"all": [{"field": "tool.name", "op": "in", "value": [w.tool]},
                                           {"field": "args.amount", "op": "gt", "value": gt}]}
    form["hidden"]["inputs"][0]["sensitive"] = True          # args.amount, as make_form(sensitive=True)
    assert form["hidden"]["inputs"][0]["field"] == "args.amount"
    form["examples"] = [{"input": {"tool.name": w.tool, "args.amount": gt + 1}, "agentExpected": outcome,
                         "flipped": False}]
    return form


def _draft(conn, w, **kw):
    """A sensitive DRAFT in the database and the same document (key, version 1) for the gate."""
    form = _form(w, **kw)
    key = repo.create_draft(conn, form, actor="test")["policyKey"]
    w.keys.append(key)
    doc = compile_policy(form, policy_key=key, version=1, status="live", settings=F.SETTINGS, never_suggest=False)
    return key, doc


def _a(conn, w):
    return _draft(conn, w, outcome="approve", gt=500, doc=f"TST Finance {w.tag}", owner=w.oa,
                  deciders=[w.la1, w.la2])


def _b(conn, w):
    return _draft(conn, w, outcome="approve", gt=500, doc=f"TST Customer {w.tag}", owner=w.ob, deciders=[w.lb])


def _c(conn, w):
    return _draft(conn, w, outcome="block", gt=10000, doc=f"TST Legal {w.tag}", owner=w.ob)


def _design_case(conn, w):
    (a, _), (c, _) = _a(conn, w), _c(conn, w)
    [did] = CC.detect_for(conn, c, now=NOW, among=[a])
    return a, c, did


def _gate(monkeypatch, w, docs, amount=900):
    LG.use(monkeypatch, docs)
    res, _ = LG.run(monkeypatch, LG.stub_tools(w, []), [LG._round(w.tool, {**LG.ARGS, "amount": amount}), LG.FINAL],
                    workflow_id=w.wf)
    return res.calls[0].result


# ------------------------------------------------------------------ the three routes
def test_conflict_view_masks_sensitive_witness(client, conn, world):
    a, c, did = _design_case(conn, world)
    owner = client.get("/agent-policies/conflicts", headers=_as(world, world.oa))
    assert owner.status_code == 200, owner.text
    [mine] = [v for v in owner.json()["conflicts"] if v["decisionId"] == did]
    assert mine["caseId"] == f"pc_{did}" and mine["status"] == "open" and mine["canDecide"] is True
    assert mine["example"] == {"tool.name": world.tool, "args.amount": 10001}
    assert [p["id"] for p in mine["policies"]] == sorted([a, c])
    assert {p["latestVersion"] for p in mine["policies"]} == {1} and {p["status"] for p in mine["policies"]} == {"draft"}
    assert mine["raisedBy"] == "save" and mine["respondWithin"] is None and mine["decision"] is None
    assert mine["optionLabels"][f"keep_both:{c}"] == f"Keep both: {c} takes priority"
    assert set(mine["options"]) == set(mine["optionLabels"])
    assert mine["prior"] == {"sameConflict": 0, "lastOutcome": None} and mine["why"].startswith("One policy needs")

    for hdr in (STRANGER, ADMIN):
        r = client.get("/agent-policies/conflicts", headers=hdr)
        [theirs] = [v for v in r.json()["conflicts"] if v["decisionId"] == did]
        assert theirs["canDecide"] is False
        assert theirs["example"] == {"tool.name": world.tool, "args.amount": MASK}
        assert "10001" not in json.dumps(theirs)
        one = client.get(f"/agent-policies/conflicts/{did}", headers=hdr).json()
        assert one["example"]["args.amount"] == MASK and "10001" not in json.dumps(one) and one["history"] == []

    one = client.get(f"/agent-policies/conflicts/{did}", headers=_as(world, world.ob)).json()
    assert one["canDecide"] is True and one["example"]["args.amount"] == 10001


def test_decide_through_the_app_is_audited_and_closes_the_case(client, conn, world):
    a, c, did = _design_case(conn, world)
    body = {"option": f"keep_both:{c}", "reason": "Customer terms win"}
    r = client.post(f"/agent-policies/conflicts/{did}/decide", json=body, headers=STRANGER)
    assert r.status_code == 403
    assert [(x["phase"], x["status"]) for x in world.audits] == [("authorize", "allowed"), ("decide", "refused")]
    assert world.audits[-1]["details"]["code"] == "not_eligible"
    assert {x["action_type"] for x in world.audits} == {"agent_policy.conflict_decide"}

    r = client.post(f"/agent-policies/conflicts/{did}/decide", json={"option": f"keep_both:{c}"},
                    headers=_as(world, world.ob))
    assert r.status_code == 422 and r.json()["problems"][0]["field"] == "reason"

    world.audits.clear()
    r = client.post(f"/agent-policies/conflicts/{did}/decide", json=body, headers=_as(world, world.ob))
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["caseId"] == f"pc_{did}" and out["decision"] == f"keep_both:{c}" and out["applied"] == "standing_rule"
    assert [(x["phase"], x["status"]) for x in world.audits] == [("authorize", "allowed"), ("decide", "done")]
    assert world.audits[-1]["details"]["actionId"] == out["actionId"]

    one = client.get(f"/agent-policies/conflicts/{did}", headers=_as(world, world.oa)).json()
    assert one["status"] == "actioned" and one["canDecide"] is False
    assert one["decision"]["decision"] == f"keep_both:{c}" and one["decision"]["reason"] == "Customer terms win"
    assert one["decision"]["caseId"] == f"pc_{did}" and one["decision"]["decidedBy"] == f"sub-{world.email_b}"
    assert [h["decision"] for h in one["history"]] == [f"keep_both:{c}"]
    assert one["example"]["args.amount"] == 10001         # an owner still reads it after the decision
    ids = lambda s: {v["decisionId"] for v in client.get(f"/agent-policies/conflicts?status={s}",  # noqa: E731
                                                           headers=STRANGER).json()["conflicts"]}
    assert did not in ids("open") and did in ids("closed") and did in ids("all")

    r = client.post(f"/agent-policies/conflicts/{did}/decide", json=body, headers=_as(world, world.oa))
    assert r.status_code == 409


def test_newest_first(client, conn, world):
    (a, _), (c, _) = _a(conn, world), _c(conn, world)
    (b, _) = _b(conn, world)
    first = CC.detect_for(conn, c, now=NOW, among=[a])
    second = CC.detect_for(conn, c, now=NOW, among=[b])
    shown = [v["decisionId"] for v in client.get("/agent-policies/conflicts", headers=STRANGER).json()["conflicts"]]
    assert shown.index(second[0]) < shown.index(first[0])


def test_live_and_unknown_ids_are_404(client, conn, world, monkeypatch):
    (a, da), (b, db) = _a(conn, world), _b(conn, world)
    _gate(monkeypatch, world, [da, db])
    [lv] = LG.lives(conn, world)
    for did in (lv["decision_id"], 999999999999):
        assert client.get(f"/agent-policies/conflicts/{did}", headers=_as(world, world.oa)).status_code == 404
        r = client.post(f"/agent-policies/conflicts/{did}/decide", json={"option": f"change:{a}", "reason": "x"},
                        headers=_as(world, world.oa))
        assert r.status_code == 404
    listed = {v["decisionId"] for v in client.get("/agent-policies/conflicts?status=all",
                                                   headers=STRANGER).json()["conflicts"]}
    assert lv["decision_id"] not in listed, "live cases are never listed as policy cases"


def test_an_action_row_is_not_a_case(client, conn, world):
    a, c, did = _design_case(conn, world)
    out = client.post(f"/agent-policies/conflicts/{did}/decide", json={"option": f"change:{a}", "reason": "x"},
                      headers=_as(world, world.oa)).json()
    assert client.get(f"/agent-policies/conflicts/{out['actionId']}", headers=STRANGER).status_code == 404
    listed = [v["decisionId"] for v in client.get("/agent-policies/conflicts?status=all",
                                                   headers=STRANGER).json()["conflicts"]]
    assert out["actionId"] not in listed and listed.count(did) == 1


# ------------------------------------------------------------------ approval-card conflict block
def test_live_member_view_masks_for_non_approver(client, conn, world, monkeypatch):
    (a, da), (b, db) = _a(conn, world), _b(conn, world)
    out = _gate(monkeypatch, world, [da, db])
    assert out["result"] == "paused_for_approval"
    [lv] = LG.lives(conn, world)
    ma, mb = LG.members(conn, world)
    assert ma["policy_name"] == a and ma["facts"]["liveConflict"] == lv["decision_id"]

    # la2 decides A's member (last level only): the block shows the action's values
    mine = client.get(f"/agent-policies/approvals/{ma['decision_id']}", headers=_as(world, world.la2)).json()
    assert mine["canDecide"] is True
    blk = mine["conflict"]
    assert blk["caseId"] == f"pc_{lv['decision_id']}" and blk["options"] == ["approve", "reject"]
    assert blk["args"] == {"amount": 900} and blk["example"] == {"tool.name": world.tool, "args.amount": 900}
    assert sorted(p["id"] for p in blk["policies"]) == sorted([a, b])
    assert blk["respondWithin"] == "PT4H" and blk["prior"] == {"sameConflict": 0, "lastOutcome": None}
    assert blk["why"] and blk["actionPlain"]
    listed = [c for c in client.get("/agent-policies/approvals", headers=_as(world, world.la2)).json()["approvals"]
              if c["id"] == ma["decision_id"]]
    assert listed[0]["conflict"]["args"] == {"amount": 900}

    # an Admin who is not linked reads it, masked, in the list and on its own
    theirs = client.get(f"/agent-policies/approvals/{ma['decision_id']}", headers=ADMIN).json()
    assert theirs["canDecide"] is False
    assert theirs["conflict"]["args"] == {"amount": MASK}
    assert theirs["conflict"]["example"] == {"tool.name": world.tool, "args.amount": MASK}
    assert "900" not in json.dumps(theirs["conflict"])
    assert {i["field"]: i["value"] for i in theirs["inputs"]}["args.amount"] == MASK
    [row] = [c for c in client.get("/agent-policies/approvals", headers=ADMIN).json()["approvals"]
             if c["id"] == ma["decision_id"]]
    assert row["conflict"]["args"] == {"amount": MASK} and row["conflict"]["example"]["args.amount"] == MASK
    # lb approves B's member, not A's: A's member is not theirs to read
    assert client.get(f"/agent-policies/approvals/{ma['decision_id']}", headers=_as(world, world.lb)).status_code == 404
    # lb decides B's member and sees its block unmasked
    own = client.get(f"/agent-policies/approvals/{mb['decision_id']}", headers=_as(world, world.lb)).json()
    assert own["canDecide"] is True and own["conflict"]["args"] == {"amount": 900}


def test_policy_case_from_a_live_block_masks_its_witness(client, conn, world, monkeypatch):
    (a, da), (c, dc) = _a(conn, world), _c(conn, world)
    out = _gate(monkeypatch, world, [da, dc], amount=12000)     # over both A's 500 and C's 10000
    assert out["result"] == "blocked"
    [pc] = LG.policy_cases(conn, world)
    assert pc["raised_by"] == "live"
    did = pc["decision_id"]
    owner = client.get(f"/agent-policies/conflicts/{did}", headers=_as(world, world.oa)).json()
    assert owner["raisedBy"] == "live" and owner["canDecide"] is True
    assert owner["example"] == {"tool.name": world.tool, "args.amount": 12000}
    for hdr in (STRANGER, ADMIN):
        one = client.get(f"/agent-policies/conflicts/{did}", headers=hdr).json()
        assert one["example"] == {"tool.name": world.tool, "args.amount": MASK}
        assert "12000" not in json.dumps(one)
        [v] = [v for v in client.get("/agent-policies/conflicts", headers=hdr).json()["conflicts"]
               if v["decisionId"] == did]
        assert v["example"]["args.amount"] == MASK


def test_policy_case_from_repeat_proposal_masks_its_witness(client, conn, world, monkeypatch):
    """maybe_propose copies the live witness into a policy case: the view masks it there too."""
    (a, da), (b, db) = _a(conn, world), _b(conn, world)
    _gate(monkeypatch, world, [da, db])
    [lv] = LG.lives(conn, world)
    monkeypatch.setattr(CL, "_live_docs", lambda cur, keys: {a: da, b: db})
    with conn.cursor() as cur:
        cur.execute("UPDATE proc.bp_agent_policy_conflict SET is_open = false, outcome = 'approve', by_person = true, "
                    "decided_by = 'test', decided_at = now() WHERE decision_id = %s", (lv["decision_id"],))
        [pid] = CL.maybe_propose(cur, lv["decision_id"], now=NOW, threshold=1)
    for hdr, shown in ((STRANGER, MASK), (_as(world, world.oa), 900)):
        one = client.get(f"/agent-policies/conflicts/{pid}", headers=hdr).json()
        assert one["raisedBy"] == "repeat" and one["proposal"]["from"] == "repeat"
        assert one["example"]["args.amount"] == shown


# ------------------------------------------------------------------ notifications
def test_notification_with_a_conflict_link_returns_conflict_id(client, conn, world):
    a, c, did = _design_case(conn, world)
    notes = client.get("/agent-policies/notifications?mine=1", headers=_as(world, world.oa)).json()["notifications"]
    [n] = [n for n in notes if n.get("conflictId") == did]
    assert n["recipient"] == world.oa and n["decisionId"] is None


# ------------------------------------------------------------------ a policy's conflict history (final review I2)
def _live_entry(client, w, key, live_id, headers=STRANGER):
    got = client.get(f"/agent-policies/{key}", headers=headers)
    assert got.status_code == 200, got.text
    [entry] = [c for c in got.json()["conflicts"] if c["caseId"] == f"pc_{live_id}"]
    return entry, got.text


def test_history_shows_a_settled_live_conflict_as_the_approvers_decision(client, conn, world, monkeypatch):
    (a, da), (b, db) = _a(conn, world), _b(conn, world)
    _gate(monkeypatch, world, [da, db])
    [lv] = LG.lives(conn, world)
    ma, mb = LG.members(conn, world)
    LG.act(conn, world, ma["decision_id"], world.la2)
    LG.act(conn, world, mb["decision_id"], world.lb, reason="Paying the 900 refund is fine.")
    entry, text = _live_entry(client, world, a, lv["decision_id"])
    assert entry["kind"] == "live" and entry["isOpen"] is False
    d = entry["decision"]
    assert (d["decision"], d["decidedBy"]) == ("approve", f"sub-{world.people[world.lb]}")
    assert d["decidedAt"]
    # the decider's free text quotes the sensitive amount: never shown on the policy page
    assert d.get("reason") is None
    assert "Paying the" not in text


def test_history_shows_a_timed_out_live_conflict_as_a_system_reject(client, conn, world, monkeypatch):
    from services.agent_policy import approvals as A
    (a, da), (b, db) = _a(conn, world), _b(conn, world)
    _gate(monkeypatch, world, [da, db])
    [lv] = LG.lives(conn, world)
    ms = LG.members(conn, world)
    A.sweep(conn, lv["respond_by"] + LG.timedelta(seconds=1), decision_ids=[m["decision_id"] for m in ms])
    entry, _ = _live_entry(client, world, b, lv["decision_id"])
    assert (entry["decision"]["decision"], entry["decision"]["decidedBy"]) == ("reject", "system:timeout")
