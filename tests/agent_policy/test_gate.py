"""The agent-policy gate in the tool loop (Task 6): a scripted chat stand-in, no model, no Ollama.

Stub tools record whether they ran. Live tests (PROCWISE_TEST_LIVE_DB=1, DB_NAME=bp_testdb) write
real firing / notification / case rows under TST-<hex> keys; the live policies themselves are
injected through gate._load_policies, so no live policy is ever written to the shared database.
Cases and notifications made here are removed afterwards; firing rows are append-only BY DESIGN
and stay.
"""
import copy
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from services import tool_runtime as TR
from services.agent_policy import approvals as A
from services.agent_policy import gate as G
from services.agent_policy import live_policies
from services.agent_policy.compiler import compile_policy
from tests.agent_policy.fixtures import FORM_EXAMPLE, SETTINGS

live = pytest.mark.skipif(os.getenv("PROCWISE_TEST_LIVE_DB") != "1", reason="live DB required")

REASON = "The customer asked for a refund of the duplicate charge."
ARGS = {"amount": 900, "order": "A-1"}


# ------------------------------------------------------------------ scripted chat stand-in
def _round(tool="refund.issue", args=None, content=REASON):
    return {"role": "assistant", "content": content,
            "tool_calls": [{"function": {"name": tool, "arguments": dict(args or ARGS)}}]}


FINAL = {"role": "assistant", "content": "Done."}


class Script:
    """Replaces the model: hands back scripted messages, records what it was sent."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []

    def chat(self, messages, schemas, model, timeout):
        self.seen.append(copy.deepcopy(messages))
        return copy.deepcopy(self.replies.pop(0))

    def egress(self):
        script = self

        class _Resp:
            def __init__(self, reply):
                self.reply = reply

            def raise_for_status(self):
                return None

            def iter_lines(self):
                msg = dict(self.reply)
                calls = msg.pop("tool_calls", None)
                content = msg.get("content") or ""
                half = len(content) // 2
                yield json.dumps({"message": {"role": "assistant", "content": content[:half]}})
                yield json.dumps({"message": {"role": "assistant", "content": content[half:],
                                              **({"tool_calls": calls} if calls else {})}})

        def post(url, *, json=None, **_kw):
            script.seen.append(copy.deepcopy(json["messages"]))
            return _Resp(script.replies.pop(0))

        return SimpleNamespace(post=post, Purpose=SimpleNamespace(MODEL_INFERENCE="model"))


@pytest.fixture(autouse=True)
def enforcement_on(monkeypatch):
    """Every test states the switch it depends on; the kill-switch tests override it."""
    monkeypatch.setenv("AGENT_POLICY_ENFORCEMENT", "on")


@pytest.fixture
def ran():
    return []


@pytest.fixture
def tools(ran):
    def refund(**kw):
        ran.append(("refund.issue", kw))
        return {"refunded": kw.get("amount")}

    return [TR.Tool(name="refund.issue", description="Issue a refund",
                    parameters={"type": "object", "properties": {}}, handler=refund)]


def _run(monkeypatch, tools, replies, *, stream=False, agent="overcharge_hunter", workflow_id=None,
         user_id="req@example.test"):
    script = Script(replies)
    if stream:
        monkeypatch.setattr(TR, "egress", script.egress())
        res = TR.run_tools_stream("task", tools, "system", max_rounds=4, agent=agent,
                                  workflow_id=workflow_id, user_id=user_id)
    else:
        monkeypatch.setattr(TR, "_chat", script.chat)
        res = TR.run_tools("task", tools, "system", max_rounds=4, agent=agent,
                           workflow_id=workflow_id, user_id=user_id)
    return res, script


def _tool_messages(script):
    """Every tool message the model was sent, in the final transcript."""
    return [m for m in script.seen[-1] if m.get("role") == "tool"]


# ------------------------------------------------------------------ no database
@pytest.mark.parametrize("stream", [False, True])
def test_no_live_policies_is_identical_to_today(monkeypatch, tools, ran, stream):
    monkeypatch.setattr(G, "_load_policies", lambda: [])
    gated, s1 = _run(monkeypatch, tools, [_round(), FINAL], stream=stream)

    monkeypatch.setattr(TR, "_policy_refusal", lambda *a, **k: None)   # the gate patched out
    plain, s2 = _run(monkeypatch, tools, [_round(), FINAL], stream=stream)

    assert s1.seen == s2.seen
    assert [c.to_dict() | {"duration_ms": 0} for c in gated.calls] == \
           [c.to_dict() | {"duration_ms": 0} for c in plain.calls]
    assert gated.answer == plain.answer == "Done."
    assert len(ran) == 2


@pytest.mark.parametrize("value", ["off", "OFF", " Off "])
def test_kill_switch_allows_without_loading_anything(monkeypatch, tools, ran, value):
    monkeypatch.setenv("AGENT_POLICY_ENFORCEMENT", value)

    def boom():
        raise AssertionError("the kill switch must not load policies")
    monkeypatch.setattr(G, "_load_policies", boom)
    monkeypatch.setattr(G, "_connect", boom)
    res, _ = _run(monkeypatch, tools, [_round(), FINAL])
    assert ran == [("refund.issue", ARGS)] and res.calls[0].ok


@pytest.mark.parametrize("stream", [False, True])
def test_store_unavailable_refuses(monkeypatch, tools, ran, stream):
    monkeypatch.setenv("AGENT_POLICY_ENFORCEMENT", "on")

    def down():
        raise live_policies.PolicyStoreUnavailable("live agent policies could not be loaded")

    def no_db():
        raise RuntimeError("db down")
    monkeypatch.setattr(G, "_load_policies", down)
    monkeypatch.setattr(G, "_connect", no_db)       # the best-effort log fails too; refusal stands
    res, script = _run(monkeypatch, tools, [_round(), FINAL], stream=stream)
    assert ran == []
    call = res.calls[0]
    assert call.ok is False and call.error is None and call.result == G.UNAVAILABLE
    tool_msgs = _tool_messages(script)
    assert tool_msgs == [{"role": "tool", "name": "refund.issue", "content": json.dumps(G.UNAVAILABLE)}]


def _gate_unimportable(monkeypatch):
    import services.agent_policy as pkg
    monkeypatch.delattr(pkg, "gate", raising=False)
    monkeypatch.setitem(sys.modules, "services.agent_policy.gate", None)   # import -> ImportError


def test_gate_import_failure_refuses_when_enforcement_is_on(monkeypatch, tools, ran):
    _gate_unimportable(monkeypatch)
    res, script = _run(monkeypatch, tools, [_round(), FINAL])
    assert ran == [] and res.calls[0].result["reasonCode"] == "policy_check_unavailable"
    assert json.loads(_tool_messages(script)[0]["content"])["reasonCode"] == "policy_check_unavailable"


def test_gate_import_failure_still_honours_the_kill_switch(monkeypatch, tools, ran):
    monkeypatch.setenv("AGENT_POLICY_ENFORCEMENT", "Off")
    _gate_unimportable(monkeypatch)
    res, _ = _run(monkeypatch, tools, [_round(), FINAL])
    assert ran == [("refund.issue", ARGS)] and res.calls[0].ok


def test_check_raising_refuses(monkeypatch, tools, ran):
    monkeypatch.setattr(G, "_load_policies", lambda: [{"id": "X"}])

    def bad(*a, **k):
        raise ValueError("malformed")
    monkeypatch.setattr(G.enforcement, "check", bad)
    monkeypatch.setattr(G, "_connect", lambda: (_ for _ in ()).throw(RuntimeError("db down")))
    res, _ = _run(monkeypatch, tools, [_round(), FINAL])
    assert ran == [] and res.calls[0].result["reasonCode"] == "policy_check_unavailable"


def test_reason_is_the_round_text_stripped_and_capped(monkeypatch, tools):
    seen = []

    def spy(**kw):
        seen.append(kw)
        return G.GateResult(allow=True)
    monkeypatch.setattr(G, "before_tool", spy)
    _run(monkeypatch, tools, [_round(content="  " + "x" * 900 + "  "), _round(content=""), FINAL],
         workflow_id="wf-1")
    assert seen[0]["agent"] == "overcharge_hunter" and seen[0]["workflow_id"] == "wf-1"
    assert seen[0]["user_id"] == "req@example.test" and seen[0]["args"] == ARGS
    assert G.clean_reason(seen[0]["reason"]) == "x" * 500
    assert G.clean_reason(seen[1]["reason"]) is None


def test_unknown_tool_is_not_gated(monkeypatch, tools):
    monkeypatch.setattr(G, "before_tool", lambda **kw: pytest.fail("gate called for an unknown tool"))
    res, _ = _run(monkeypatch, tools, [_round(tool="nope"), FINAL])
    assert res.calls[0].error == "unknown tool 'nope'"


def test_callers_pass_the_agent(monkeypatch):
    from orchestration import agentnick_control as AC
    got = {}
    monkeypatch.setattr(AC, "build_tools", lambda *a, **k: [])
    monkeypatch.setattr(AC, "run_tools", lambda task, tools, system, **kw: got.update(kw) or TR.ToolRunResult())
    AC.reason(object(), "t", agent="overcharge_hunter", workflow_id="wf", user_id="u")
    assert (got["agent"], got["workflow_id"], got["user_id"]) == ("overcharge_hunter", "wf", "u")


# ------------------------------------------------------------------ live
@pytest.fixture
def conn():
    assert os.getenv("DB_NAME") == "bp_testdb", "gate live tests run on bp_testdb only"
    from services.db import get_conn
    with get_conn() as c:
        yield c


@pytest.fixture
def world(conn, monkeypatch):
    monkeypatch.setenv("AGENT_POLICY_ENFORCEMENT", "on")
    tag = uuid.uuid4().hex[:8]
    w = SimpleNamespace(tag=tag, wf=f"wf-gate-{tag}", decider=f"TST GT {tag}", notify=f"TST GN {tag}",
                        email=f"gt-{tag}@example.test")
    with conn.cursor() as cur:
        cur.execute("INSERT INTO proc.bp_policy_decider_map (decider_name, groups, emails, last_modified_by) "
                    "VALUES (%s, '{}', %s, 'test')", (w.decider, [w.email]))
    yield w
    with conn.cursor() as cur:
        cur.execute("SELECT firing_id FROM proc.bp_policy_firing WHERE workflow_id = %s", (w.wf,))
        fids = [r[0] for r in cur.fetchall()]
        cur.execute("DELETE FROM proc.bp_policy_notification WHERE firing_id = ANY(%s)", (fids,))
        cur.execute("DELETE FROM proc.bp_decision WHERE subject_type = %s AND workflow_id = %s",
                    (A.SUBJECT_TYPE, w.wf))
        cur.execute("DELETE FROM proc.bp_policy_decider_map WHERE decider_name = %s", (w.decider,))


def _doc(w, key, outcome):
    form = copy.deepcopy(FORM_EXAMPLE)
    form["checked"] = {"by": "user_8841", "at": "2026-10-08T09:14:00Z"}
    form["outcome"] = outcome
    form["deciders"] = [w.decider] if outcome == "approve" else []
    form["notify"] = [w.notify] if outcome == "notify" else []
    return compile_policy(form, policy_key=key, version=1, status="live",
                          settings=SETTINGS, never_suggest=False)


def _firings(conn, w):
    with conn.cursor() as cur:
        cur.execute("SELECT firing_id, policy_key, outcome, result, agent, decision_id, matched_values, decided_by, "
                    "duration_ms, requested_by FROM proc.bp_policy_firing WHERE workflow_id = %s "
                    "ORDER BY firing_id", (w.wf,))
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]


def _cases(conn, w):
    with conn.cursor() as cur:
        cur.execute("SELECT decision_id, facts FROM proc.bp_decision WHERE subject_type = %s "
                    "AND workflow_id = %s ORDER BY decision_id", (A.SUBJECT_TYPE, w.wf))
        return [(r[0], r[1] if isinstance(r[1], dict) else json.loads(r[1])) for r in cur.fetchall()]


def _notes(conn, fids):
    with conn.cursor() as cur:
        cur.execute("SELECT recipient, message, link FROM proc.bp_policy_notification "
                    "WHERE firing_id = ANY(%s) ORDER BY notification_id", (fids,))
        return cur.fetchall()


@live
@pytest.mark.parametrize("stream", [False, True])
def test_blocked_tool_is_never_executed(conn, world, monkeypatch, tools, ran, stream):
    doc = _doc(world, f"TST-{world.tag}B", "block")
    monkeypatch.setattr(G, "_load_policies", lambda: [doc])
    res, script = _run(monkeypatch, tools, [_round(), FINAL], stream=stream, workflow_id=world.wf)
    assert ran == []
    expected = {"result": "blocked", "reasonCode": f"{doc['id']}.over_limit",
                "reason": FORM_EXAMPLE["messageForAgent"],
                "messageForPerson": FORM_EXAMPLE["messageForPerson"], "policies": [doc["id"]]}
    assert res.calls[0].ok is False and res.calls[0].result == expected and res.calls[0].error is None
    assert _tool_messages(script) == [{"role": "tool", "name": "refund.issue", "content": json.dumps(expected)}]
    rows = _firings(conn, world)
    assert [(r["policy_key"], r["outcome"], r["result"], r["agent"]) for r in rows] == \
           [(doc["id"], "block", "blocked", "overcharge_hunter")]
    assert rows[0]["matched_values"] == {"args.amount": 900, "tool.name": "refund.issue"}
    assert rows[0]["duration_ms"] is not None and rows[0]["requested_by"] == "req@example.test"


@live
@pytest.mark.parametrize("stream", [False, True])
def test_repeat_call_while_paused_reuses_the_case(conn, world, monkeypatch, tools, ran, stream):
    doc = _doc(world, f"TST-{world.tag}A", "approve")
    monkeypatch.setattr(G, "_load_policies", lambda: [doc])
    res, script = _run(monkeypatch, tools, [_round(), _round(), FINAL], stream=stream, workflow_id=world.wf)
    assert ran == []
    first, second = res.calls[0].result, res.calls[1].result
    assert first["result"] == "paused_for_approval" and first["respondWithin"] == "PT4H"
    cases = _cases(conn, world)
    assert len(cases) == 1, "a repeat while paused must not open a second case"
    did, facts = cases[0]
    assert first["requestIds"] == second["requestIds"] == [did]
    msgs = _tool_messages(script)
    assert [json.loads(m["content"]) for m in msgs] == [first, second]
    # what the replay needs: the group, the exact ctx, the full args, and who asked
    assert facts["firing_group"] and facts["argsDigest"] == G.args_digest(ARGS)
    assert facts["ctx"] == {"checkpoint": "tool.call.before", "tool.name": "refund.issue",
                            "agent.name": "overcharge_hunter", "agent.reason": REASON, "args": ARGS}
    assert facts["action"]["args"] == ARGS and facts["requestedBy"] == "req@example.test"
    rows = _firings(conn, world)
    assert [r["decision_id"] for r in rows] == [did, did]
    # the first-level decider hears a decision is waiting -- once, not per repeat
    assert _notes(conn, [r["firing_id"] for r in rows]) == [
        (world.decider, f"Issuing a refund or credit (policy {doc['id']}) needs your decision.",
         f"decision:{did}")]
    # one decision settles the call: the case's own row AND the repeat's row
    _approve(conn, world, did)
    assert [(r["result"], r["decided_by"]) for r in _firings(conn, world)] == \
           [("approved", f"sub-{world.tag}")] * 2


def _approve(conn, w, did):
    principal = SimpleNamespace(subject=f"sub-{w.tag}", email=w.email, claims={"cognito:groups": []})
    return A.act(conn, did, principal=principal, verb="approve", reason=None,
                 now=datetime.now(timezone.utc), replay=lambda _d: None)


@live
def test_another_requesters_identical_call_does_not_reuse_the_case(conn, world, monkeypatch, tools, ran):
    doc = _doc(world, f"TST-{world.tag}A", "approve")
    monkeypatch.setattr(G, "_load_policies", lambda: [doc])
    a, _ = _run(monkeypatch, tools, [_round(), FINAL], workflow_id=world.wf, user_id="a@example.test")
    b, _ = _run(monkeypatch, tools, [_round(), FINAL], workflow_id=world.wf, user_id="b@example.test")
    cases = _cases(conn, world)
    assert len(cases) == 2 and ran == []
    assert a.calls[0].result["requestIds"] == [cases[0][0]]
    assert b.calls[0].result["requestIds"] == [cases[1][0]]
    assert [c[1]["requestedBy"] for c in cases] == ["a@example.test", "b@example.test"]


@live
def test_a_failed_write_leaves_nothing_behind(conn, world, monkeypatch, tools, ran):
    d1, d2 = _doc(world, f"TST-{world.tag}A", "approve"), _doc(world, f"TST-{world.tag}C", "approve")
    monkeypatch.setattr(G, "_load_policies", lambda: [d1, d2])
    real, seen = A._insert_case, []

    def second_fails(cur, **kw):
        seen.append(kw["policy_doc"]["id"])
        if len(seen) == 2:
            raise RuntimeError("insert failed")
        return real(cur, **kw)
    monkeypatch.setattr(G.approvals, "_insert_case", second_fails)
    res, _ = _run(monkeypatch, tools, [_round(), FINAL], workflow_id=world.wf)
    assert seen == [d1["id"], d2["id"]]
    assert ran == [] and res.calls[0].result == G.UNAVAILABLE
    assert _cases(conn, world) == [], "no orphan open case for a refused call"
    assert [r["policy_key"] for r in _firings(conn, world)] == ["*"], "only the refusal is logged"


@live
def test_notify_row_of_a_paused_call_is_settled_by_its_case(conn, world, monkeypatch, tools, ran):
    apv, ntf = _doc(world, f"TST-{world.tag}A", "approve"), _doc(world, f"TST-{world.tag}N", "notify")
    monkeypatch.setattr(G, "_load_policies", lambda: [apv, ntf])
    _run(monkeypatch, tools, [_round(), FINAL], workflow_id=world.wf)
    (did, _), = _cases(conn, world)
    rows = {r["outcome"]: r for r in _firings(conn, world)}
    assert (rows["notify"]["result"], rows["notify"]["decision_id"]) == ("paused_for_approval", did)
    _approve(conn, world, did)
    assert sorted(r["result"] for r in _firings(conn, world)) == ["approved", "approved"]


@live
def test_different_args_open_a_new_case(conn, world, monkeypatch, tools, ran):
    doc = _doc(world, f"TST-{world.tag}A", "approve")
    monkeypatch.setattr(G, "_load_policies", lambda: [doc])
    res, _ = _run(monkeypatch, tools, [_round(), _round(args={"amount": 901, "order": "A-1"}), FINAL],
                  workflow_id=world.wf)
    assert len(_cases(conn, world)) == 2 and ran == []
    assert res.calls[0].result["requestIds"] != res.calls[1].result["requestIds"]


@live
def test_two_approve_policies_share_one_firing_group(conn, world, monkeypatch, tools, ran):
    d1, d2 = _doc(world, f"TST-{world.tag}A", "approve"), _doc(world, f"TST-{world.tag}C", "approve")
    monkeypatch.setattr(G, "_load_policies", lambda: [d1, d2])
    res, _ = _run(monkeypatch, tools, [_round(), FINAL], workflow_id=world.wf)
    cases = _cases(conn, world)
    assert len(cases) == 2 and res.calls[0].result["requestIds"] == [c[0] for c in cases]
    assert cases[0][1]["firing_group"] == cases[1][1]["firing_group"]


@live
@pytest.mark.parametrize("stream", [False, True])
def test_notify_only_runs_the_tool_and_writes_notifications(conn, world, monkeypatch, tools, ran, stream):
    doc = _doc(world, f"TST-{world.tag}N", "notify")
    monkeypatch.setattr(G, "_load_policies", lambda: [doc])
    res, script = _run(monkeypatch, tools, [_round(), FINAL], stream=stream, workflow_id=world.wf)
    assert ran == [("refund.issue", ARGS)] and res.calls[0].ok and res.calls[0].result == {"refunded": 900}
    rows = _firings(conn, world)
    assert [(r["outcome"], r["result"]) for r in rows] == [("notify", "allowed")]
    notes = _notes(conn, [rows[0]["firing_id"]])
    assert notes == [(world.notify, f"Issuing a refund or credit (policy {doc['id']}) was allowed to run.",
                      f"agent-policy:{doc['id']}")]
    assert "900" not in notes[0][1]


@live
def test_notify_is_written_even_when_blocked(conn, world, monkeypatch, tools, ran):
    blk, ntf = _doc(world, f"TST-{world.tag}B", "block"), _doc(world, f"TST-{world.tag}N", "notify")
    monkeypatch.setattr(G, "_load_policies", lambda: [blk, ntf])
    _run(monkeypatch, tools, [_round(), FINAL], workflow_id=world.wf)
    assert ran == []
    rows = _firings(conn, world)
    assert sorted((r["outcome"], r["result"]) for r in rows) == [("block", "blocked"), ("notify", "blocked")]
    notes = _notes(conn, [r["firing_id"] for r in rows])
    assert [n[1] for n in notes] == [f"Issuing a refund or credit (policy {ntf['id']}) was blocked."]


@live
def test_store_unavailable_is_logged_best_effort(conn, world, monkeypatch, tools, ran):
    def down():
        raise live_policies.PolicyStoreUnavailable("down")
    monkeypatch.setattr(G, "_load_policies", down)
    res, _ = _run(monkeypatch, tools, [_round(), FINAL], workflow_id=world.wf)
    assert ran == [] and res.calls[0].result == G.UNAVAILABLE
    rows = _firings(conn, world)
    assert [(r["policy_key"], r["outcome"], r["result"]) for r in rows] == [("*", "block", "error")]
