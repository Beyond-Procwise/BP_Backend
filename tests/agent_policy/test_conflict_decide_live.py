"""Owners decide policy conflict cases against bp_testdb; standing rules reach conflicts[].

Needs PROCWISE_TEST_LIVE_DB=1 and DB_NAME=bp_testdb. Policies are made with tools named
tst_<tag> (they can never contradict anyone else's) and RETIRED at teardown; every detection
passes among=. Cases, action rows, standing rules, conflict rows, notifications and decider-map
rows made here are removed afterwards, in FK order.
"""
import copy
import json
import os
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from repositories import agent_policy_repo as repo
from services.agent_policy import approvals, conflict_cases as CC, contract, registry as REG
from tests.agent_policy import fixtures as F

pytestmark = pytest.mark.skipif(os.getenv("PROCWISE_TEST_LIVE_DB") != "1", reason="live DB required")

NOW = datetime(2026, 10, 9, 9, 0, tzinfo=timezone.utc)
LATER = NOW + timedelta(hours=1)


@pytest.fixture
def conn():
    assert os.getenv("DB_NAME") == "bp_testdb", "conflict live tests run on bp_testdb only"
    from services.db import get_conn
    with get_conn() as c:
        yield c


@pytest.fixture
def world(conn):
    tag = uuid.uuid4().hex[:8]
    w = SimpleNamespace(tag=tag, tool=f"tst_{tag}", owner_a=f"TST Owner A {tag}", owner_b=f"TST Owner B {tag}",
                        email_a=f"a-{tag}@example.test", email_b=f"b-{tag}@example.test", keys=[])
    w.registry = REG.snapshot_from_rows([
        {"kind": "checkpoint", "name": "tool.call.before", "checkpoint": None, "plain": "before a tool runs",
         "status": "live"},
        F._a(w.tool), F._i("tool.name", "tool name", "string"), F._i("agent.reason", "the agent's reason", "string"),
        F._i("args.amount", "amount", "number")])
    with conn.cursor() as cur:
        cur.execute("INSERT INTO proc.bp_policy_decider_map (decider_name, groups, emails, last_modified_by) "
                    "VALUES (%s, '{}', %s, 'test'), (%s, '{}', %s, 'test')",
                    (w.owner_a, [w.email_a], w.owner_b, [w.email_b]))
    yield w
    pairs = sorted({f"{a}|{b}" for a in w.keys for b in w.keys if a < b})
    with conn.cursor() as cur:
        cur.execute("SELECT decision_id FROM proc.bp_decision WHERE subject_type = %s AND subject_id = ANY(%s)",
                    (CC.SUBJECT_POLICY, pairs))
        ids = [r[0] for r in cur.fetchall()]
        cur.execute("DELETE FROM proc.bp_policy_notification WHERE firing_id IS NULL AND link = ANY(%s)",
                    ([f"conflict:{i}" for i in ids],))
        cur.execute("DELETE FROM proc.bp_agent_policy_conflict_rule WHERE pair_key = ANY(%s)", (pairs,))
        cur.execute("DELETE FROM proc.bp_agent_policy_conflict WHERE decision_id = ANY(%s)", (ids,))
        cur.execute("DELETE FROM proc.bp_decision WHERE decision_id = ANY(%s)", (ids,))
        cur.execute("DELETE FROM proc.bp_policy_decider_map WHERE decider_name = ANY(%s)", ([w.owner_a, w.owner_b],))
    for key in w.keys:
        got = repo.get_policy(conn, key)
        if got["status"] != "retired":
            repo.retire(conn, key, base_version=got["latestVersion"], actor="test", change_note="test teardown")


def _form(w, *, outcome, gt, doc, owner):
    form = copy.deepcopy(F.FORM_EXAMPLE)
    form["name"] = f"TST {outcome} over {gt} {w.tag}"
    form["businessArea"], form["subArea"] = None, None
    form["outcome"] = outcome
    form["deciders"] = ["Finance Manager", "CFO"] if outcome == "approve" else []
    form["notify"] = []
    form["owner"] = owner
    form["checked"] = {"by": "test", "at": "2026-10-09T08:00:00Z"}
    form["source"] = {"document": doc, "documentVersion": 1, "reference": "1.1",
                      "excerpt": f"Refunds above ${gt} need a decision ({outcome})."}
    form["hidden"]["actions"]["tools"] = [w.tool]
    form["hidden"]["condition"] = {"all": [{"field": "tool.name", "op": "in", "value": [w.tool]},
                                           {"field": "args.amount", "op": "gt", "value": gt}]}
    form["examples"] = [{"input": {"tool.name": w.tool, "args.amount": gt + 1}, "agentExpected": outcome,
                         "flipped": False}]
    return form


def _make(conn, w, *, live=False, monkeypatch=None, **kw):
    form = _form(w, **kw)
    key = repo.create_draft(conn, form, actor="test")["policyKey"]
    w.keys.append(key)
    if live:
        # tst_ tools are unknown to the real registry: readiness and the final check are bypassed here,
        # and contract.validate is asserted below against a registry that knows the tool
        monkeypatch.setattr(repo, "_activation_problems", lambda f, r, s: [])
        monkeypatch.setattr(repo, "_contract_problems", lambda d, r: [])
        repo.save_version(conn, key, form, base_version=1, intent="activate", actor="test", change_note="go")
    return key


def _pair(conn, w, *, live=False, monkeypatch=None):
    """A (approve over 500, owner A) and B (block over 10000, owner B), and their open case."""
    a = _make(conn, w, live=live, monkeypatch=monkeypatch, outcome="approve", gt=500,
              doc=f"TST Finance {w.tag}", owner=w.owner_a)
    b = _make(conn, w, live=live, monkeypatch=monkeypatch, outcome="block", gt=10000,
              doc=f"TST Customer {w.tag}", owner=w.owner_b)
    [did] = CC.detect_for(conn, b, now=NOW, among=[a])
    return a, b, did


def _who(email, groups=()):
    return SimpleNamespace(subject=f"sub-{email}", email=email, claims={"cognito:groups": list(groups)})


def _decide(conn, did, who, option, reason="Agreed by both owners", limit_text=None, now=LATER):
    return CC.decide_policy(conn, did, principal=who, option=option, reason=reason, limit_text=limit_text, now=now)


def _refused(conn, did, who, option, **kw):
    with pytest.raises(CC.ConflictRefused) as err:
        _decide(conn, did, who, option, **kw)
    return err.value


def _row(conn, sql, params):
    with conn.cursor() as cur:
        cur.execute(sql, params)
        r = cur.fetchone()
        return dict(zip([d[0] for d in cur.description], r)) if r else None


def _rows(conn, sql, params):
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return [dict(zip([d[0] for d in cur.description], r)) for r in cur.fetchall()]


def _versions(conn, key):
    return len(repo.get_policy(conn, key)["versions"])


def _notes(conn, did):
    with conn.cursor() as cur:
        cur.execute("SELECT recipient, message FROM proc.bp_policy_notification WHERE link = %s "
                    "ORDER BY notification_id", (f"conflict:{did}",))
        return cur.fetchall()


def _feed_client(monkeypatch, registry):
    from fastapi.testclient import TestClient
    from api.main import app
    from api.routers import agent_policies as R
    monkeypatch.setenv("AGENT_POLICY_GATEWAY_KEY", "k1")
    monkeypatch.setenv("AGENT_POLICY_ORCHESTRATOR_KEY", "o1")
    monkeypatch.setattr(R, "load_registry", lambda conn=None: registry)
    return TestClient(app)


# ---------------------------------------------------------------------------- keep both
def test_standing_rule_written_into_conflicts_of_both(conn, world, monkeypatch):
    a, b, did = _pair(conn, world, live=True, monkeypatch=monkeypatch)
    out = _decide(conn, did, _who(world.email_a), f"keep_both:{b}")
    assert out["applied"] == "standing_rule" and out["caseId"] == f"pc_{did}"
    assert out["decision"] == f"keep_both:{b}" and out["scope"] == "standing_rule"
    rule = f"{b} takes priority over {a}"
    live = {d["id"]: d for d in repo.live_documents(conn) if d.get("id") in (a, b)}
    assert set(live) == {a, b}
    for key, other in ((a, b), (b, a)):
        [entry] = live[key]["conflicts"]
        assert entry["with"] == other and entry["rule"] == rule and entry["caseId"] == f"pc_{did}"
        assert entry["prevails"] == b and entry["decidedAt"]
        assert contract.validate(live[key], world.registry) == []
    # the orchestrator feed, through the whole app (output-safety middleware included)
    body = _feed_client(monkeypatch, world.registry).get("/orchestrator/agent-policies/v2/live",
                                                         headers={"X-Orchestrator-Key": "o1"}).json()
    fed = {p["id"]: p for p in body["policies"] if p["id"] in (a, b)}
    assert set(fed) == {a, b}
    assert [e["rule"] for e in fed[a]["conflicts"]] == [rule] == [e["rule"] for e in fed[b]["conflicts"]]
    # the enforcement cache sees it too (invalidated after the commit)
    from services.agent_policy import live_policies
    monkeypatch.setattr(live_policies, "load_registry", lambda conn=None: world.registry)
    live_policies.invalidate()
    enforced = {d["id"]: d for d in live_policies.load(conn) if d["id"] in (a, b)}
    assert [e["rule"] for e in enforced[a]["conflicts"]] == [rule]
    # Admin's live-version compiled is overlaid too; the stored version row is not
    got = repo.get_policy(conn, a)
    [lv] = [v for v in got["versions"] if v["version"] == got["liveVersion"]]
    assert [e["rule"] for e in lv["compiled"]["conflicts"]] == [rule]
    stored = _row(conn, "SELECT compiled FROM proc.bp_agent_policy_version WHERE policy_key = %s AND version = %s",
                  (a, got["liveVersion"]))["compiled"]
    assert stored["conflicts"] == []


def test_decision_stored_on_both_histories(conn, world):
    a, b, did = _pair(conn, world)
    _decide(conn, did, _who(world.email_b), f"keep_both:{b}", reason="Customer terms win")
    for key, other in ((a, b), (b, a)):
        got = repo.get_policy(conn, key)
        [c] = [c for c in got["conflicts"] if c["caseId"] == f"pc_{did}"]
        assert c["kind"] == "policy" and c["isOpen"] is False and c["otherPolicies"] == [other]
        assert c["raisedAt"]
        d = c["decision"]
        assert d["option"] == f"keep_both:{b}" and d["scope"] == "standing_rule"
        assert d["decidedBy"] == {"kind": "person", "name": f"sub-{world.email_b}"}
        assert d["reason"] == "Customer terms win"
        assert d["decidedAt"] == LATER.isoformat()
        assert got["pendingConflictAction"] is None
    rows = repo.list_policies(conn)
    assert {r["policyKey"]: r["openConflicts"] for r in rows if r["policyKey"] in (a, b)} == {a: [], b: []}
    # the records: action row, original closed, conflict row closed by a person, owners notified
    act = _row(conn, "SELECT * FROM proc.bp_decision WHERE subject_type = %s AND subject_id = %s "
                     "AND decision_id <> %s", (CC.SUBJECT_POLICY, "|".join(sorted([a, b])), did))
    assert act["decision"] == f"keep_both:{b}" and act["decision_scope"] == "standing_rule"
    assert act["status"] == "actioned" and act["actioned_by"] == f"sub-{world.email_b}"
    assert act["override_reason"] == "Customer terms win"
    assert act["facts"]["versionsAtDecision"] == {a: 1, b: 1} and act["facts"]["limitText"] is None
    assert act["facts"]["policies"] == _row(conn, "SELECT facts FROM proc.bp_decision WHERE decision_id = %s",
                                            (did,))["facts"]["policies"]
    assert _row(conn, "SELECT status FROM proc.bp_decision WHERE decision_id = %s", (did,))["status"] == "actioned"
    cr = _row(conn, "SELECT * FROM proc.bp_agent_policy_conflict WHERE decision_id = %s", (did,))
    assert cr["is_open"] is False and cr["outcome"] == f"keep_both:{b}" and cr["by_person"] is True
    assert cr["decided_by"] == f"sub-{world.email_b}" and cr["decided_at"] == LATER
    decided = [n for n in _notes(conn, did) if "decided" in n[1]]
    assert sorted(n[0] for n in decided) == sorted([world.owner_a, world.owner_b])


def test_list_shows_open_conflicts(conn, world):
    a, b, did = _pair(conn, world)
    rows = {r["policyKey"]: r for r in repo.list_policies(conn) if r["policyKey"] in (a, b)}
    assert rows[a]["openConflicts"] == [f"pc_{did}"] and rows[b]["openConflicts"] == [f"pc_{did}"]
    [c] = repo.get_policy(conn, a)["conflicts"]
    assert c["isOpen"] is True and c["decision"] is None


def test_second_keep_both_supersedes_first_rule(conn, world, monkeypatch):
    a, b, first = _pair(conn, world)
    one = _decide(conn, first, _who(world.email_a), f"keep_both:{b}")
    # an in-force rule stops a new raise; force a second case for the same pair (a later proposal)
    monkeypatch.setattr(CC, "_covered", lambda cur, key, versions: False)
    [second] = CC.detect_for(conn, a, now=NOW, among=[b])
    assert second != first
    two = _decide(conn, second, _who(world.email_b), f"keep_both:{b}", reason="Still agreed", now=LATER + timedelta(1))
    rules = _rows(conn, "SELECT * FROM proc.bp_agent_policy_conflict_rule WHERE pair_key = %s ORDER BY rule_id",
                  ("|".join(sorted([a, b])),))
    assert len(rules) == 2
    old, new = rules
    assert old["superseded_at"] == LATER + timedelta(1) and old["superseded_by"] is not None
    act2 = _row(conn, "SELECT decision_id FROM proc.bp_decision WHERE subject_type = %s AND decision = %s "
                      "AND actioned_at = %s", (CC.SUBJECT_POLICY, f"keep_both:{b}", LATER + timedelta(1)))
    assert old["superseded_by"] == act2["decision_id"]
    assert new["superseded_at"] is None and new["superseded_by"] is None and new["decision_id"] == second
    assert new["prevails"] == b and new["yields"] == a and new["rule_text"] == f"{b} takes priority over {a}"
    with conn.cursor() as cur:
        in_force = CC.rules_for(cur, [a])
    assert [r["decision_id"] for r in in_force] == [second]
    assert one["caseId"] != two["caseId"]


# ---------------------------------------------------------------------------- change / limit / retire
def test_change_decision_opens_pending_draft_and_writes_no_version(conn, world):
    a, b, did = _pair(conn, world)
    before = (_versions(conn, a), _versions(conn, b))
    out = _decide(conn, did, _who(world.email_a), f"change:{a}", reason="Raise the threshold")
    assert out["applied"] == "draft_pending" and out["scope"] == "this_action"
    assert (_versions(conn, a), _versions(conn, b)) == before          # never applied silently
    p = repo.get_policy(conn, a)["pendingConflictAction"]
    assert p == {"caseId": f"pc_{did}", "action": "change",
                 "changeNote": f"Conflict decision pc_{did}: Change {a} — Raise the threshold",
                 "limitText": None, "decidedAt": LATER.isoformat()}
    assert repo.get_policy(conn, b)["pendingConflictAction"] is None
    # the owner's own save applies it; the pending action then goes away
    repo.save_version(conn, a, _form(world, outcome="approve", gt=600, doc=f"TST Finance {world.tag}",
                                     owner=world.owner_a), base_version=1, intent="draft", actor="test",
                      change_note=p["changeNote"])
    assert repo.get_policy(conn, a)["pendingConflictAction"] is None


def test_limit_decision_requires_limit_text_and_writes_no_version(conn, world):
    a, b, did = _pair(conn, world)
    before = _versions(conn, a)
    for blank in (None, "", "   "):
        e = _refused(conn, did, _who(world.email_a), f"limit:{a}", limit_text=blank)
        assert (e.code, e.status) == ("limit_required", 422)
    out = _decide(conn, did, _who(world.email_a), f"limit:{a}", reason="Only below 10000",
                  limit_text="  Only for refunds up to $10,000  ")
    assert out["applied"] == "draft_pending"
    assert _versions(conn, a) == before
    p = repo.get_policy(conn, a)["pendingConflictAction"]
    assert p["action"] == "limit" and p["limitText"] == "Only for refunds up to $10,000"
    assert p["changeNote"] == f"Conflict decision pc_{did}: Limit {a} — Only below 10000"


def test_retire_decision_is_pending_until_two_step_retire(conn, world):
    a, b, did = _pair(conn, world)
    before = _versions(conn, b)
    out = _decide(conn, did, _who(world.email_b), f"retire:{b}", reason="Superseded by finance policy")
    assert out["applied"] == "retire_pending"
    got = repo.get_policy(conn, b)
    assert len(got["versions"]) == before and got["status"] == "draft"
    assert got["pendingConflictAction"]["action"] == "retire"
    # a save is not a retire: still pending
    repo.save_version(conn, b, _form(world, outcome="block", gt=10000, doc=f"TST Customer {world.tag}",
                                     owner=world.owner_b), base_version=1, intent="draft", actor="test",
                      change_note="edit")
    assert repo.get_policy(conn, b)["pendingConflictAction"]["action"] == "retire"
    repo.retire(conn, b, base_version=2, actor="test", change_note="retired per conflict decision")
    assert repo.get_policy(conn, b)["pendingConflictAction"] is None


# ---------------------------------------------------------------------------- refusals
def test_not_eligible_without_owner_link(conn, world):
    a, b, did = _pair(conn, world)
    for who in (_who(f"stranger-{world.tag}@example.test"),
                _who(f"admin-{world.tag}@example.test", groups=["PROCWISE_ADMIN"])):   # Admin is not automatic
        e = _refused(conn, did, who, f"change:{a}")
        assert (e.code, e.status) == ("not_eligible", 403)
    cr = _row(conn, "SELECT is_open FROM proc.bp_agent_policy_conflict WHERE decision_id = %s", (did,))
    assert cr["is_open"] is True


def test_reason_required(conn, world):
    a, b, did = _pair(conn, world)
    for blank in (None, "", "  "):
        e = _refused(conn, did, _who(world.email_a), f"change:{a}", reason=blank)
        assert (e.code, e.status) == ("reason_required", 422)


def test_unknown_option_refused(conn, world):
    a, b, did = _pair(conn, world)
    # Q1: a block pair offers only "<block> takes priority"; anything else is not an option
    for bad in (f"keep_both:{a}", "approve", f"change:GEN-0000", ""):
        e = _refused(conn, did, _who(world.email_a), bad)
        assert (e.code, e.status) == ("unknown_option", 422)


def test_unknown_case_is_404(conn, world):
    e = _refused(conn, 2 ** 62, _who(world.email_a), "change:GEN-0000")
    assert (e.code, e.status) == ("not_found", 404)


def test_decided_case_is_409(conn, world):
    a, b, did = _pair(conn, world)
    _decide(conn, did, _who(world.email_a), f"change:{a}")
    e = _refused(conn, did, _who(world.email_b), f"change:{b}")
    assert (e.code, e.status) == ("not_open", 409)


# ---------------------------------------------------------------------------- moot
def _router_client(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from api.routers import agent_policies as R
    monkeypatch.setenv("AGENT_POLICY_GATEWAY_KEY", "k1")
    monkeypatch.setattr(R, "_role_of", lambda principal: "Admin")
    monkeypatch.setattr(R.agent_actions, "record_action_or_fail", lambda **kw: None)
    app = FastAPI()
    app.include_router(R.router)
    return TestClient(app)


def test_retire_closes_open_case_as_moot(conn, world, monkeypatch):
    a, b, did = _pair(conn, world)
    hdr = {"X-Gateway-Key": "k1", "X-User-Sub": "u1", "X-User-Email": "u1@x",
           "X-User-Groups": json.dumps(["PROCWISE_ADMIN"])}
    r = _router_client(monkeypatch).post(f"/agent-policies/{a}/retire",
                                         json={"baseVersion": 1, "changeNote": "no longer needed"}, headers=hdr)
    assert r.status_code == 200
    cr = _row(conn, "SELECT * FROM proc.bp_agent_policy_conflict WHERE decision_id = %s", (did,))
    assert cr["is_open"] is False and cr["outcome"] == "moot" and cr["by_person"] is False
    assert cr["decided_by"] == "system:retired"
    act = _row(conn, "SELECT * FROM proc.bp_decision WHERE subject_type = %s AND subject_id = %s AND decision = 'moot'",
               (CC.SUBJECT_POLICY, "|".join(sorted([a, b]))))
    assert act["actioned_by"] == "system:retired" and act["override_reason"] == f"{a} was retired"
    assert act["decision_scope"] == "this_action" and act["status"] == "actioned"
    assert _row(conn, "SELECT status FROM proc.bp_decision WHERE decision_id = %s", (did,))["status"] == "actioned"
    moot = [n for n in _notes(conn, did) if "retired" in n[1]]
    assert sorted(n[0] for n in moot) == sorted([world.owner_a, world.owner_b])
    # a later decide is refused
    e = _refused(conn, did, _who(world.email_b), f"change:{b}")
    assert (e.code, e.status) == ("not_open", 409)
    # B's history shows the moot close, and nothing is pending
    got = repo.get_policy(conn, b)
    [c] = got["conflicts"]
    assert c["isOpen"] is False and c["decision"]["option"] == "moot" and got["pendingConflictAction"] is None
    with conn.cursor() as cur:
        assert CC.open_cases_by_policy(cur).get(b) is None


def test_close_moot_only_closes_cases_naming_the_key(conn, world):
    a, b, did = _pair(conn, world)
    c = _make(conn, world, outcome="block", gt=20000, doc=f"TST Other {world.tag}", owner=world.owner_b)
    [other] = CC.detect_for(conn, c, now=NOW, among=[a])
    assert CC.close_moot(conn, b, now=LATER) == 1
    assert CC.close_moot(conn, b, now=LATER) == 0                 # nothing left to close
    open_ = {r["decision_id"]: r["is_open"] for r in
             _rows(conn, "SELECT decision_id, is_open FROM proc.bp_agent_policy_conflict WHERE decision_id = ANY(%s)",
                   ([did, other],))}
    assert open_ == {did: False, other: True}
    with conn.cursor() as cur:
        assert CC.open_cases_by_policy(cur)[a] == [f"pc_{other}"]


def test_scan_closes_open_case_of_policy_whose_moot_close_failed(conn, world, monkeypatch):
    a, b, did = _pair(conn, world)

    def broken(*args, **kw):
        raise RuntimeError("moot close down")
    monkeypatch.setattr(CC, "close_moot", broken)
    hdr = {"X-Gateway-Key": "k1", "X-User-Sub": "u1", "X-User-Email": "u1@x",
           "X-User-Groups": json.dumps(["PROCWISE_ADMIN"])}
    r = _router_client(monkeypatch).post(f"/agent-policies/{a}/retire",
                                         json={"baseVersion": 1, "changeNote": "no longer needed"}, headers=hdr)
    assert r.status_code == 200                                   # the retire is never undone
    assert _row(conn, "SELECT is_open FROM proc.bp_agent_policy_conflict WHERE decision_id = %s",
                (did,))["is_open"] is True
    monkeypatch.undo()
    stats = CC.detect_all(conn, now=LATER, among=[a, b])           # the scan is the backstop
    assert stats["mooted"] == 1
    cr = _row(conn, "SELECT * FROM proc.bp_agent_policy_conflict WHERE decision_id = %s", (did,))
    assert cr["is_open"] is False and cr["outcome"] == "moot" and cr["by_person"] is False
    act = _row(conn, "SELECT * FROM proc.bp_decision WHERE subject_type = %s AND subject_id = %s AND decision = 'moot'",
               (CC.SUBJECT_POLICY, "|".join(sorted([a, b]))))
    assert act["actioned_by"] == "system:retired" and act["override_reason"] == f"{a} was retired"
    assert CC.detect_all(conn, now=LATER, among=[a, b])["mooted"] == 0


def test_decide_on_case_naming_a_retired_policy_closes_it_moot(conn, world):
    a, b, did = _pair(conn, world)
    repo.retire(conn, b, base_version=1, actor="test", change_note="retired behind the case's back")  # no moot close
    e = _refused(conn, did, _who(world.email_a), f"keep_both:{b}")
    assert (e.code, e.status) == ("not_open", 409) and e.message == f"{b} was retired; this conflict is closed."
    cr = _row(conn, "SELECT * FROM proc.bp_agent_policy_conflict WHERE decision_id = %s", (did,))
    assert cr["is_open"] is False and cr["outcome"] == "moot" and cr["by_person"] is False
    assert cr["decided_by"] == "system:retired"
    acts = _rows(conn, "SELECT decision, override_reason FROM proc.bp_decision WHERE subject_type = %s "
                       "AND subject_id = %s AND decision_id <> %s", (CC.SUBJECT_POLICY, "|".join(sorted([a, b])), did))
    assert acts == [{"decision": "moot", "override_reason": f"{b} was retired"}]
    assert _rows(conn, "SELECT 1 FROM proc.bp_agent_policy_conflict_rule WHERE pair_key = %s",
                 ("|".join(sorted([a, b])),)) == []
    assert _row(conn, "SELECT status FROM proc.bp_decision WHERE decision_id = %s", (did,))["status"] == "actioned"
