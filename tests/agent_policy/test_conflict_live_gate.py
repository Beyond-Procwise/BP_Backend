"""Live conflicts in the gate (stage 4 Task 7): a scripted chat stand-in drives run_tools, no model.

Live tests (PROCWISE_TEST_LIVE_DB=1, DB_NAME=bp_testdb) inject the live policies through
gate._load_policies (and replay._load_policies), so no live policy is ever written to the shared
database. Their keys are TST-<hex><letter> and their tool is tst_<hex>, so nothing here can meet
anyone else's policies. Every case, action, replay, conflict and notification row made here is
removed afterwards, in FK order; firing rows are append-only BY DESIGN and stay.
"""
import copy
import json
import os
import subprocess
import sys
import types
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from services import tool_runtime as TR
from services.agent_policy import approvals as A
from services.agent_policy import conflict_engine as CE
from services.agent_policy import conflict_live as CL
from services.agent_policy import gate as G
from services.agent_policy import replay as R
from services.agent_policy.compiler import compile_policy
from tests.agent_policy.fixtures import FORM_EXAMPLE, SETTINGS

live = pytest.mark.skipif(os.getenv("PROCWISE_TEST_LIVE_DB") != "1", reason="live DB required")

REASON = "The customer asked for a refund of the duplicate charge."
ARGS = {"amount": 900, "order": "A-1", "note": "duplicate charge"}
BASE_COMMIT = "78ac566f"          # stage 4 Task 6: the gate and approvals exactly as stage 3 left them
ROOT = Path(__file__).resolve().parents[2]


# ------------------------------------------------------------------ scripted chat stand-in
def _round(tool, args=None, content=REASON):
    return {"role": "assistant", "content": content,
            "tool_calls": [{"function": {"name": tool, "arguments": dict(args or ARGS)}}]}


FINAL = {"role": "assistant", "content": "Done."}


class Script:
    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []

    def chat(self, messages, schemas, model, timeout):
        self.seen.append(copy.deepcopy(messages))
        return copy.deepcopy(self.replies.pop(0))


def run(monkeypatch, tools, replies, *, workflow_id, user_id="req@example.test"):
    script = Script(replies)
    monkeypatch.setattr(TR, "_chat", script.chat)
    res = TR.run_tools("task", tools, "system", max_rounds=4, agent="overcharge_hunter",
                       workflow_id=workflow_id, user_id=user_id)
    return res, script


@pytest.fixture(autouse=True)
def enforcement_on(monkeypatch):
    monkeypatch.setenv("AGENT_POLICY_ENFORCEMENT", "on")


# ------------------------------------------------------------------ policies
def make_doc(key, *, conflicts=None, **kw):
    return compile_policy(make_form(**kw), policy_key=key, version=1, status="live", settings=SETTINGS,
                          never_suggest=False, conflicts=conflicts)


def make_form(*, tool, outcome, deciders=(), source, notify=(), gt=500, response_time=None, sensitive=False):
    form = copy.deepcopy(FORM_EXAMPLE)
    form["checked"] = {"by": "user_8841", "at": "2026-10-08T09:14:00Z"}
    form["outcome"] = outcome
    form["deciders"] = list(deciders) if outcome == "approve" else []
    form["notify"] = list(notify)
    form["responseTime"] = response_time
    form["source"] = {"document": source, "documentVersion": 1, "reference": "1.1",
                      "excerpt": f"Refunds above ${gt} ({outcome})."}
    form["hidden"]["actions"]["tools"] = [tool]
    form["hidden"]["condition"] = {"all": [{"field": "tool.name", "op": "in", "value": [tool]},
                                           {"field": "args.amount", "op": "gt", "value": gt}]}
    if sensitive:
        form["hidden"]["inputs"][0]["sensitive"] = True
    form["examples"] = [{"input": {"tool.name": tool, "args.amount": gt + 1}, "agentExpected": outcome or "none",
                         "flipped": False}]
    return form


# ------------------------------------------------------------------ no database: lc None is stage 3
class _Cur:
    def __init__(self, log, ids):
        self.log, self.ids, self.description = log, ids, None
        self._last = ""

    def execute(self, sql, params=None):
        self.log.append((" ".join(sql.split()), json.dumps(params, default=str)))
        self._last = sql

    def fetchone(self):
        if "RETURNING" in self._last:
            return (next(self.ids),)
        return None

    def fetchall(self):
        return []

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _Conn:
    def __init__(self, log):
        self.log, self.autocommit = log, True
        self.ids = iter(range(1000, 2000))

    def cursor(self):
        return _Cur(self.log, self.ids)

    def commit(self):
        self.log.append(("COMMIT", ""))

    def rollback(self):
        self.log.append(("ROLLBACK", ""))

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _stage3_module(name, path):
    try:
        src = subprocess.run(["git", "show", f"{BASE_COMMIT}:{path}"], cwd=ROOT, capture_output=True,
                             text=True, check=True).stdout
    except Exception:  # noqa: BLE001 - a shallow clone without the base commit cannot compare
        pytest.skip(f"stage 3 source {BASE_COMMIT}:{path} is not available")
    mod = types.ModuleType(name)
    mod.__file__ = f"<{BASE_COMMIT}:{path}>"
    sys.modules[name] = mod
    exec(compile(src, mod.__file__, "exec"), mod.__dict__)   # noqa: S102 - our own committed source
    return mod


class _FixedNow(datetime):
    @classmethod
    def now(cls, tz=None):
        return datetime(2026, 10, 9, 12, 0, tzinfo=timezone.utc)


def _no_conflict_scenarios():
    t = "refund.issue"
    tiered = [make_doc("TSB-0001", tool=t, outcome="approve", deciders=["FM", "CFO"], source="Finance"),
              make_doc("TSB-0002", tool=t, outcome="approve", deciders=["CFO"], source="finance ", gt=800)]
    same_deciders = [make_doc("TSB-0003", tool=t, outcome="approve", deciders=["FM"], source="Finance"),
                     make_doc("TSB-0004", tool=t, outcome="approve", deciders=["FM"], source="Customer")]
    notify = make_doc("TSB-0005", tool=t, outcome="notify", notify=["Ops"], source="Ops")
    block = make_doc("TSB-0006", tool=t, outcome="block", source="Finance", notify=["Ops"])
    unreadable = make_doc("TSB-0007", tool=t, outcome="block", source="Legal")
    unreadable["trigger"]["condition"] = {"all": [{"field": "args.amount", "op": "between", "value": "x"}]}
    return {"single": [tiered[0]], "tiered": tiered, "same_deciders": same_deciders + [notify],
            "block_same_source": [block, tiered[0]], "unreadable_block": [unreadable, same_deciders[1]],
            "notify_only": [notify]}


@pytest.mark.parametrize("name", sorted(_no_conflict_scenarios()))
def test_no_live_conflict_is_stage3_byte_for_byte(monkeypatch, name):
    """lc.kind None: every statement, parameter, commit and answer equals the stage 3 gate's."""
    docs = _no_conflict_scenarios()[name]
    assert CE.classify(G.enforcement.check(G._ctx("refund.issue", ARGS, "a", REASON), docs)).kind is None
    old_gate = _stage3_module("stage3_gate_snapshot", "src/services/agent_policy/gate.py")
    for name_ in ("insert_live", "settle_for_group", "maybe_propose"):
        monkeypatch.setattr(CL, name_, lambda *a, **k: pytest.fail("conflict_live used without a conflict"))
    monkeypatch.setattr(uuid, "uuid4", lambda: SimpleNamespace(hex="g" * 32))
    out = {}
    for label, mod in (("stage3", old_gate), ("now", G)):
        log = []
        monkeypatch.setattr(mod, "_load_policies", lambda: copy.deepcopy(docs))
        monkeypatch.setattr(mod, "_connect", lambda log=log: _Conn(log))
        monkeypatch.setattr(mod, "_default_response_time", lambda: "PT4H")
        monkeypatch.setattr(mod, "datetime", _FixedNow)
        monkeypatch.setattr(mod.deciders, "load_map", lambda conn: {"FM": {"groups": ["g"], "emails": []}})
        res = mod.before_tool(tool_name="refund.issue", args=dict(ARGS), agent="a", reason=REASON,
                              workflow_id="wf", user_id="u")
        out[label] = (log, res.allow, res.to_agent, res.firing_ids, res.case_ids)
    assert out["now"][0], "the scenario wrote nothing; it proves nothing"
    assert out["now"] == out["stage3"]


def test_insert_case_default_is_stage3_byte_for_byte(monkeypatch):
    old = _stage3_module("stage3_approvals_snapshot", "src/services/agent_policy/approvals.py")
    doc = make_doc("TSB-0010", tool="refund.issue", outcome="approve", deciders=["FM", "CFO"], source="F")
    logs = []
    for mod in (old, A):
        log = []
        mod._insert_case(_Cur(log, iter(range(5, 9))), policy_doc=doc, firing_id=3,
                         action={"tool": "refund.issue", "args": ARGS}, requested_by="u",
                         now=_FixedNow.now(), mapping={}, extra_facts={"x": 1})
        logs.append(log)
    assert logs[0] == logs[1]


def test_last_level_only_keeps_one_level_that_rejects_on_timeout():
    doc = make_doc("TSB-0011", tool="refund.issue", outcome="approve", deciders=["FM", "CFO"], source="F")
    log = []
    A._insert_case(_Cur(log, iter(range(5, 9))), policy_doc=doc, firing_id=3,
                   action={"tool": "refund.issue", "args": ARGS}, requested_by="u", now=_FixedNow.now(),
                   mapping={"CFO": {"groups": ["g"], "emails": []}}, last_level_only=True)
    params = json.loads(log[0][1])
    assert json.loads(params[-1]) == [{"name": "CFO", "respondWithin": "PT4H"}]
    assert params[-3] == "reject"
    assert "unroutable" not in json.loads(params[4]), "only the last level's name must be linked"


# ------------------------------------------------------------------ live fixtures
@pytest.fixture
def conn():
    assert os.getenv("DB_NAME") == "bp_testdb", "conflict live tests run on bp_testdb only"
    from services.db import get_conn
    with get_conn() as c:
        yield c


def cleanup(conn, w):
    """Remove every row this test made (FK order). Firing rows are append-only and stay."""
    with conn.cursor() as cur:
        cur.execute("SELECT firing_id FROM proc.bp_policy_firing WHERE workflow_id = ANY(%s)", (w.wfs,))
        fids = [r[0] for r in cur.fetchall()]
        cur.execute("SELECT decision_id FROM proc.bp_decision WHERE workflow_id = ANY(%s) AND subject_type IN "
                    "(%s, %s, %s)", (w.wfs, A.SUBJECT_TYPE, CL.SUBJECT_LIVE, R.SUBJECT_TYPE))
        ids = [r[0] for r in cur.fetchall()]
        cur.execute("SELECT decision_id FROM proc.bp_decision WHERE subject_type = %s AND subject_id = ANY(%s)",
                    (CL.SUBJECT_POLICY, w.pairs()))
        pids = [r[0] for r in cur.fetchall()]
        cur.execute("DELETE FROM proc.bp_policy_notification WHERE firing_id = ANY(%s) OR link = ANY(%s)",
                    (fids, [f"decision:{i}" for i in ids] + [f"conflict:{i}" for i in pids]))
        cur.execute("DELETE FROM proc.bp_agent_policy_conflict WHERE decision_id = ANY(%s)", (ids + pids,))
        cur.execute("DELETE FROM proc.bp_decision WHERE decision_id = ANY(%s)", (ids + pids,))
        cur.execute("DELETE FROM proc.bp_policy_decider_map WHERE decider_name = ANY(%s)", (list(w.people),))


def new_world(conn, prefix="TST"):
    tag = uuid.uuid4().hex[:8]
    w = SimpleNamespace(tag=tag, tool=f"tst_{tag}", wf=f"wf-lc-{tag}", wfs=[f"wf-lc-{tag}"], keys=[],
                        la1=f"TST LA1 {tag}", la2=f"TST LA2 {tag}", lb=f"TST LB {tag}", ntf=f"TST LN {tag}")
    w.key = lambda letter: f"{prefix}-{tag}{letter}"
    w.pairs = lambda: sorted({f"{a}|{b}" for a in w.keys for b in w.keys if a < b})
    w.people = {w.la1: f"la1-{tag}@example.test", w.la2: f"la2-{tag}@example.test",
                w.lb: f"lb-{tag}@example.test"}
    with conn.cursor() as cur:
        for name, email in w.people.items():
            cur.execute("INSERT INTO proc.bp_policy_decider_map (decider_name, groups, emails, last_modified_by) "
                        "VALUES (%s, '{}', %s, 'test')", (name, [email]))
    return w


@pytest.fixture
def world(conn):
    w = new_world(conn)
    yield w
    cleanup(conn, w)


def _known(w, key):
    if key not in w.keys:
        w.keys.append(key)
    return key


def doc_a(w, **kw):
    _known(w, w.key("A"))
    return make_doc(w.key("A"), tool=w.tool, outcome="approve", deciders=[w.la1, w.la2],
                    source=f"TST Finance {w.tag}", **kw)


def doc_b(w, outcome="approve", **kw):
    _known(w, w.key("B"))
    return make_doc(w.key("B"), tool=w.tool, outcome=outcome,
                    deciders=[w.lb] if outcome == "approve" else [], source=f"TST Customer {w.tag}", **kw)


def doc_n(w):
    return make_doc(w.key("N"), tool=w.tool, outcome="notify", notify=[w.ntf], source=f"TST Ops {w.tag}")


def who(w, name):
    return SimpleNamespace(subject=f"sub-{w.people[name]}", email=w.people[name], claims={"cognito:groups": []})


@pytest.fixture
def ran():
    return []


def stub_tools(w, ran):
    def handler(**kw):
        ran.append(kw)
        return {"refunded": kw.get("amount")}
    return [TR.Tool(name=w.tool, description="Issue a refund", parameters={"type": "object", "properties": {}},
                    handler=handler)]


def use(monkeypatch, docs, w=None, ran=None):
    monkeypatch.setattr(G, "_load_policies", lambda: copy.deepcopy(docs))
    monkeypatch.setattr(R, "_load_policies", lambda: copy.deepcopy(docs))
    if w is not None:
        monkeypatch.setattr(R, "_build_tools", lambda nick, **kw: stub_tools(w, ran))


def rows(conn, sql, params):
    with conn.cursor() as cur:
        cur.execute(sql, params)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]


def members(conn, w):
    return rows(conn, "SELECT decision_id, policy_name, status, levels, current_level, on_timeout, facts "
                      "FROM proc.bp_decision WHERE subject_type = %s AND workflow_id = ANY(%s) "
                      "AND decision = 'approve_or_reject' ORDER BY decision_id", (A.SUBJECT_TYPE, w.wfs))


def actions(conn, w, subject_type=A.SUBJECT_TYPE):
    return rows(conn, "SELECT decision_id, subject_id, decision, actioned_by, decision_scope, override_reason, "
                      "facts FROM proc.bp_decision WHERE subject_type = %s AND workflow_id = ANY(%s) "
                      "AND status = 'actioned' AND actioned_by IS NOT NULL ORDER BY decision_id",
                (subject_type, w.wfs))


def lives(conn, w):
    return rows(conn, "SELECT d.decision_id, d.subject_id, d.decision, d.status, d.actioned_by, d.decision_scope, "
                      "d.override_reason, d.options, d.respond_by, d.on_timeout, d.facts, d.evidence, "
                      "c.kind, c.raised_by, c.is_open, c.outcome, c.by_person, c.decided_by, c.policy_keys "
                      "FROM proc.bp_decision d JOIN proc.bp_agent_policy_conflict c USING (decision_id) "
                      "WHERE d.subject_type = %s AND d.workflow_id = ANY(%s) ORDER BY d.decision_id",
                (CL.SUBJECT_LIVE, w.wfs))


def policy_cases(conn, w):
    return rows(conn, "SELECT d.decision_id, d.subject_id, d.facts, c.raised_by, c.is_open "
                      "FROM proc.bp_decision d JOIN proc.bp_agent_policy_conflict c USING (decision_id) "
                      "WHERE d.subject_type = %s AND d.subject_id = ANY(%s) ORDER BY d.decision_id",
                (CL.SUBJECT_POLICY, w.pairs()))


def firings(conn, w):
    return rows(conn, "SELECT firing_id, policy_key, outcome, result, decision_id, decided_by "
                      "FROM proc.bp_policy_firing WHERE workflow_id = ANY(%s) ORDER BY firing_id", (w.wfs,))


def notes(conn, fids):
    return rows(conn, "SELECT recipient, message, link FROM proc.bp_policy_notification "
                      "WHERE firing_id = ANY(%s) ORDER BY notification_id", (list(fids),))


def act(conn, w, did, name, verb="approve", reason=None, replay=lambda _d: None):
    return A.act(conn, did, principal=who(w, name), verb=verb, reason=reason or ("No." if verb == "reject" else None),
                 now=datetime.now(timezone.utc), replay=replay)


# ------------------------------------------------------------------ live tests
@live
def test_live_multi_match_pauses_and_sends_live_case_with_condition_fields_only(conn, world, monkeypatch, ran):
    a, b = doc_a(world), doc_b(world)
    use(monkeypatch, [a, b])
    res, _ = run(monkeypatch, stub_tools(world, ran), [_round(world.tool), FINAL], workflow_id=world.wf)
    assert ran == [], "the tool must not run while the conflict is open"
    out = res.calls[0].result
    assert out["result"] == "paused_for_approval"
    [lv] = lives(conn, world)
    assert out["conflictCaseId"] == f"pc_{lv['decision_id']}"
    assert (lv["status"], lv["decision"], lv["kind"], lv["raised_by"], lv["is_open"]) == \
           ("open", "approve_or_reject", "live", "live", True)
    assert lv["subject_id"] == f"{a['id']}|{b['id']}" and sorted(lv["policy_keys"]) == [a["id"], b["id"]]
    assert lv["options"] == ["approve", "reject"] and lv["on_timeout"] == "reject" and lv["respond_by"]
    assert lv["facts"]["action"]["args"] == {"amount": 900}, "only condition fields are stored"
    assert lv["evidence"][0]["example"] == {"tool.name": world.tool, "args.amount": 900}
    assert lv["facts"]["priorDecisions"] == {"sameConflict": 0, "lastOutcome": None}
    assert lv["facts"]["pairs"] == [[a["id"], b["id"]]]
    ms = members(conn, world)
    assert [m["policy_name"] for m in ms] == [a["id"], b["id"]]
    assert out["requestIds"] == [m["decision_id"] for m in ms]
    assert [m["levels"] for m in ms] == [[{"name": world.la2, "respondWithin": "PT4H"}],
                                         [{"name": world.lb, "respondWithin": "PT4H"}]]
    assert {m["on_timeout"] for m in ms} == {"reject"}
    assert {m["facts"]["liveConflict"] for m in ms} == {lv["decision_id"]}
    assert ms[0]["facts"]["firing_group"] == ms[1]["facts"]["firing_group"]
    # each member's (last-level) decider hears a decision is waiting; the first level does not
    recips = [n["recipient"] for n in notes(conn, [f["firing_id"] for f in firings(conn, world)])]
    assert sorted(recips) == sorted([world.la2, world.lb])


@live
def test_no_precedence_between_approve_policies(conn, world, monkeypatch, ran):
    a, b = doc_a(world), doc_b(world)
    use(monkeypatch, [a, b], world, ran)
    run(monkeypatch, stub_tools(world, ran), [_round(world.tool), FINAL], workflow_id=world.wf)
    ma, mb = members(conn, world)
    nick = object()
    replay = lambda d: R.run(d, agent_nick=nick)   # noqa: E731
    act(conn, world, ma["decision_id"], world.la2, replay=replay)
    assert ran == [], "one approval of two never runs the action"
    assert lives(conn, world)[0]["status"] == "open"
    act(conn, world, mb["decision_id"], world.lb, replay=replay)
    assert ran == [ARGS], "the action runs once every member approved"
    [lv] = lives(conn, world)
    assert (lv["status"], lv["is_open"], lv["outcome"], lv["by_person"]) == ("actioned", False, "approve", True)
    [act_row] = actions(conn, world, CL.SUBJECT_LIVE)
    assert (act_row["decision"], act_row["decision_scope"], act_row["actioned_by"]) == \
           ("approve", "this_action", f"sub-{world.people[world.lb]}")
    assert act_row["facts"]["caseId"] == f"pc_{lv['decision_id']}"


@live
def test_block_in_conflict_still_blocks_and_is_recorded(conn, world, monkeypatch, ran):
    a, b = doc_a(world), doc_b(world, outcome="block")
    use(monkeypatch, [a, b])
    res, _ = run(monkeypatch, stub_tools(world, ran), [_round(world.tool), FINAL], workflow_id=world.wf)
    out = res.calls[0].result
    assert ran == [] and out["result"] == "blocked" and out["policies"] == [b["id"]]
    assert members(conn, world) == [], "a block is never put to a person"
    [lv] = lives(conn, world)
    assert out["conflictCaseId"] == f"pc_{lv['decision_id']}"
    assert (lv["status"], lv["decision"], lv["actioned_by"], lv["decision_scope"]) == \
           ("actioned", "block", "system:not_allowed", "this_action")
    assert (lv["is_open"], lv["outcome"], lv["by_person"]) == (False, "block", False)
    [pc] = policy_cases(conn, world)
    assert pc["subject_id"] == f"{a['id']}|{b['id']}" and pc["raised_by"] == "live" and pc["is_open"]
    assert [f["result"] for f in firings(conn, world)] == ["blocked", "blocked"]


@live
def test_standing_rule_decides_automatically(conn, world, monkeypatch, ran):
    rule = {"with": world.key("B"), "prevails": world.key("A"),
            "rule": f"{world.key('A')} takes priority over {world.key('B')}"}
    a, b = doc_a(world, conflicts=[rule]), doc_b(world)
    use(monkeypatch, [a, b], world, ran)
    res, _ = run(monkeypatch, stub_tools(world, ran), [_round(world.tool), FINAL], workflow_id=world.wf)
    out = res.calls[0].result
    [m] = members(conn, world)
    assert m["policy_name"] == a["id"] and out["requestIds"] == [m["decision_id"]]
    assert [lv["name"] for lv in m["levels"]] == [world.la1, world.la2], "normal levels"
    [lv] = lives(conn, world)
    assert out["conflictCaseId"] == f"pc_{lv['decision_id']}"
    assert (lv["status"], lv["decision"], lv["actioned_by"], lv["decision_scope"]) == \
           ("actioned", "standing_rule", "system:standing_rule", "standing_rule")
    assert rule["rule"] in lv["override_reason"]
    assert (lv["is_open"], lv["by_person"]) == (False, False)
    # the rule's winner decides alone: its approval runs the action, the loser is never asked
    act(conn, world, m["decision_id"], world.la1, replay=lambda d: R.run(d, agent_nick=object()))
    assert ran == [ARGS]
    assert len(members(conn, world)) == 1
    assert {f["policy_key"]: f["result"] for f in firings(conn, world)} == {a["id"]: "approved", b["id"]: "approved"}


@live
def test_live_conflict_timeout_rejects(conn, world, monkeypatch, ran):
    a, b = doc_a(world), doc_b(world)
    use(monkeypatch, [a, b], world, ran)
    run(monkeypatch, stub_tools(world, ran), [_round(world.tool), FINAL], workflow_id=world.wf)
    ms = members(conn, world)
    [lv] = lives(conn, world)
    later = lv["respond_by"] + timedelta(seconds=1)
    counts = A.sweep(conn, later, decision_ids=[m["decision_id"] for m in ms])
    assert counts["rejected"] == 1 and counts["escalated"] == 0
    acts = actions(conn, world)
    assert sorted(x["actioned_by"] for x in acts) == ["system:group", "system:timeout"]
    assert {x["decision"] for x in acts} == {"reject"}
    [lv] = lives(conn, world)
    assert (lv["status"], lv["outcome"], lv["by_person"], lv["decided_by"]) == \
           ("actioned", "reject", False, "system:timeout")
    [la] = actions(conn, world, CL.SUBJECT_LIVE)
    assert la["decision"] == "reject" and la["actioned_by"] == "system:timeout"
    assert ran == [] and R.run(ms[0]["decision_id"], agent_nick=object())["status"] == "rejected"
    assert ran == []


@live
def test_live_conflict_record_failure_refuses(conn, world, monkeypatch, ran):
    a, b = doc_a(world), doc_b(world)
    use(monkeypatch, [a, b, doc_n(world)])

    def boom(*a_, **k):
        raise RuntimeError("conflict store down")
    monkeypatch.setattr(CL, "insert_live", boom)
    res, _ = run(monkeypatch, stub_tools(world, ran), [_round(world.tool), FINAL], workflow_id=world.wf)
    assert ran == [] and res.calls[0].result == G.UNAVAILABLE
    assert members(conn, world) == [] and lives(conn, world) == []
    fs = firings(conn, world)
    assert [f["policy_key"] for f in fs] == ["*"], "only the refusal is logged"
    assert notes(conn, [f["firing_id"] for f in fs]) == []


@live
def test_repeat_call_reuses_live_conflict(conn, world, monkeypatch, ran):
    a, b = doc_a(world), doc_b(world)
    use(monkeypatch, [a, b])
    res, _ = run(monkeypatch, stub_tools(world, ran), [_round(world.tool), _round(world.tool), FINAL],
                 workflow_id=world.wf)
    first, second = res.calls[0].result, res.calls[1].result
    assert ran == [] and first == second
    assert len(lives(conn, world)) == 1, "a repeat writes no second live record"
    assert len(members(conn, world)) == 2, "a repeat reuses the member cases"
    fs = firings(conn, world)
    assert len(fs) == 4 and {f["decision_id"] for f in fs} == set(first["requestIds"])


@live
def test_notify_still_sent_in_live_conflict(conn, world, monkeypatch, ran):
    a, b, n = doc_a(world), doc_b(world), doc_n(world)
    use(monkeypatch, [a, b, n])
    run(monkeypatch, stub_tools(world, ran), [_round(world.tool), FINAL], workflow_id=world.wf)
    fs = {f["policy_key"]: f for f in firings(conn, world)}
    assert fs[n["id"]]["result"] == "paused_for_approval" and fs[n["id"]]["decision_id"]
    sent = [x for x in notes(conn, [fs[n["id"]]["firing_id"]]) if x["recipient"] == world.ntf]
    assert [x["message"] for x in sent] == [f"Issuing a refund or credit (policy {n['id']}) is waiting for approval."]
    assert len(lives(conn, world)) == 1


@live
def test_timeout_raced_by_a_persons_approval_settles_as_the_timeout(conn, world, monkeypatch, ran):
    """Review I1: the sweep times A out while a person holds B (its SKIP LOCKED pass leaves B open,
    so nothing settles); the person then approves B and closes the group. The live case must
    settle as the TIMEOUT -- not as that person's reject -- and never count toward repeat-N."""
    a, b = doc_a(world), doc_b(world)
    use(monkeypatch, [a, b])
    run(monkeypatch, stub_tools(world, ran), [_round(world.tool), FINAL], workflow_id=world.wf)
    ma, mb = members(conn, world)
    proposed = []
    monkeypatch.setattr(CL, "maybe_propose", lambda *a_, **k: proposed.append(a_) or [])
    now = datetime.now(timezone.utc)
    with A._tx(conn):                      # what _sweep_one wrote for A before skipping the locked B
        with conn.cursor() as cur:
            case = A._load_locked(cur, ma["decision_id"])
            A._record(cur, case, verb="reject", actor=A.TIMEOUT_ACTOR, reason=A.TIMEOUT_REASON, now=now,
                      level=0, level_name=world.la2)
            A._update_firing(cur, case["facts"].get("firingId"), result="timed_out", level=0,
                             actor=A.TIMEOUT_ACTOR, now=now, reason=A.TIMEOUT_REASON,
                             decision_id=ma["decision_id"])
    act(conn, world, mb["decision_id"], world.lb)          # the person's approval closes the group
    [lv] = lives(conn, world)
    assert (lv["status"], lv["outcome"], lv["decided_by"], lv["by_person"]) == \
           ("actioned", "reject", A.TIMEOUT_ACTOR, False)
    [la] = actions(conn, world, CL.SUBJECT_LIVE)
    assert (la["decision"], la["actioned_by"], la["override_reason"]) == ("reject", A.TIMEOUT_ACTOR, A.TIMEOUT_REASON)
    assert proposed == [], "a timeout never counts toward repeat-N"
    assert ran == []
