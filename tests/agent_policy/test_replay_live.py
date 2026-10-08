"""Replay on approval (Task 5) against bp_testdb, with a stub tool that records calls (no model).

Needs PROCWISE_TEST_LIVE_DB=1 and DB_NAME=bp_testdb. The "current live policies" are injected
through replay._load_policies so no live policy is ever written to the shared database; every
case, action, replay row and decider-map row made here is removed afterwards. Firing rows are
append-only BY DESIGN and stay (policy_key 'TST-<hex>' / 'TSN-<hex>').
"""
import copy
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from services.agent_policy import approvals as A
from services.agent_policy import replay as R
from services.agent_policy.compiler import compile_policy
from tests.agent_policy.fixtures import FORM_EXAMPLE, SETTINGS

live = pytest.mark.skipif(os.getenv("PROCWISE_TEST_LIVE_DB") != "1", reason="live DB required")

NOW = datetime.now(timezone.utc)
ARGS = {"amount": 900, "iban": "GB00SECRET"}


def _NO_REPLAY(_decision_id):
    return None


def _P(subject, email):
    return SimpleNamespace(subject=subject, email=email, claims={"cognito:groups": []})


# ------------------------------------------------------------------ pure
def test_run_never_raises_when_the_database_is_unreachable(monkeypatch):
    def boom():
        raise RuntimeError("db down")
    monkeypatch.setattr(R, "_connect", boom)
    out = R.run(123)
    assert out["status"] == "error" and "db down" in out["error"]


def test_summary_masks_sensitive_values_and_is_capped():
    s = R.summarise({"iban": "GB00SECRET", "note": "paid to GB00SECRET", "pad": "x" * 5000},
                    ARGS, {"iban"})
    assert "GB00SECRET" not in s and len(s) == R.SUMMARY_LIMIT


# ------------------------------------------------------------------ live fixtures
@pytest.fixture
def conn():
    assert os.getenv("DB_NAME") == "bp_testdb", "replay live tests run on bp_testdb only"
    from services.db import get_conn
    with get_conn() as c:
        yield c


@pytest.fixture
def world(conn):
    tag = uuid.uuid4().hex[:8]
    w = SimpleNamespace(tag=tag, decider=f"TST RP {tag}", email=f"rp-{tag}@example.test",
                        requester=f"req-{tag}@example.test")
    with conn.cursor() as cur:
        cur.execute("INSERT INTO proc.bp_policy_decider_map (decider_name, groups, emails, last_modified_by) "
                    "VALUES (%s, '{}', %s, 'test')", (w.decider, [w.email]))
    yield w
    with conn.cursor() as cur:
        pats = [f"TST-{tag}%", f"TSN-{tag}%"]
        cur.execute("SELECT decision_id FROM proc.bp_decision WHERE subject_type IN (%s, %s) "
                    "AND (subject_id LIKE %s OR subject_id LIKE %s)", (A.SUBJECT_TYPE, R.SUBJECT_TYPE, *pats))
        ids = [r[0] for r in cur.fetchall()]
        cur.execute("DELETE FROM proc.bp_policy_notification WHERE link = ANY(%s)",
                    ([f"decision:{i}" for i in ids],))
        cur.execute("DELETE FROM proc.bp_decision WHERE decision_id = ANY(%s)", (ids,))
        cur.execute("DELETE FROM proc.bp_policy_decider_map WHERE decider_name = %s", (w.decider,))


def _doc(w, key, outcome="approve"):
    form = copy.deepcopy(FORM_EXAMPLE)
    form["checked"] = {"by": "user_8841", "at": "2026-10-08T09:14:00Z"}
    form["outcome"] = outcome
    form["deciders"] = [w.decider] if outcome == "approve" else []
    form["hidden"]["inputs"].append({"name": "IBAN", "field": "args.iban", "type": "string",
                                     "from": "action", "showApprover": False, "sensitive": True})
    return compile_policy(form, policy_key=key, version=1, status="live",
                          settings=SETTINGS, never_suggest=False)


def _firing(conn, key):
    with conn.cursor() as cur:
        cur.execute("INSERT INTO proc.bp_policy_firing (policy_key, policy_version, checkpoint, action_name, "
                    "agent, outcome, result) VALUES (%s, 1, 'tool.call.before', 'refund.issue', 'agent_nick', "
                    "'approve', 'paused_for_approval') RETURNING firing_id", (key,))
        return cur.fetchone()[0]


def _open(conn, w, doc, group=None):
    fid = _firing(conn, doc["id"])
    did = A.open_case(conn, policy_doc=doc, firing_id=fid,
                      action={"tool": "refund.issue", "args": dict(ARGS), "agent": "agent_nick",
                              "workflowId": f"wf-{w.tag}", "userId": "u-1", "reason": "customer asked"},
                      requested_by=w.requester, now=NOW,
                      extra_facts={"firing_group": group} if group else None)
    return did, fid


def _approve(conn, w, did, replay=_NO_REPLAY):
    return A.act(conn, did, principal=_P(f"sub-{w.tag}", w.email), verb="approve", reason=None,
                 now=datetime.now(timezone.utc), replay=replay)


def _replays(conn, w):
    with conn.cursor() as cur:
        cur.execute("SELECT decision_id, subject_id, status, facts FROM proc.bp_decision "
                    "WHERE subject_type = %s AND subject_id LIKE %s ORDER BY decision_id",
                    (R.SUBJECT_TYPE, f"TST-{w.tag}%"))
        return [dict(zip(("id", "subject_id", "status", "facts"), r)) for r in cur.fetchall()]


@pytest.fixture
def stub(monkeypatch):
    """A tool that records its calls, reachable only through replay._build_tools."""
    s = SimpleNamespace(calls=[], built=[], fail=None, delay=0.0, lock=threading.Lock())

    def handler(**kwargs):
        if s.delay:
            time.sleep(s.delay)
        with s.lock:
            s.calls.append(kwargs)
        if s.fail:
            raise s.fail
        return {"refund": "ok", "iban": kwargs.get("iban"), "echo": f"paid to {kwargs.get('iban')}"}

    def build(agent_nick, *, workflow_id, user_id):
        s.built.append((agent_nick, workflow_id, user_id))
        return [SimpleNamespace(name="other.tool", handler=lambda **_: None),
                SimpleNamespace(name="refund.issue", handler=handler)]

    s.nick = object()
    monkeypatch.setattr(R, "_build_tools", build)
    return s


def _policies(monkeypatch, docs):
    monkeypatch.setattr(R, "_load_policies", lambda: list(docs))


# ------------------------------------------------------------------ live tests
@live
def test_replay_runs_the_stored_action_once_approved(conn, world, stub, monkeypatch):
    doc = _doc(world, f"TST-{world.tag}")
    _policies(monkeypatch, [doc])
    did, _ = _open(conn, world, doc)
    _approve(conn, world, did, replay=lambda d: R.run(d, agent_nick=stub.nick))
    assert stub.calls == [ARGS]
    assert stub.built == [(stub.nick, f"wf-{world.tag}", "u-1")]
    rows = _replays(conn, world)
    assert len(rows) == 1 and rows[0]["status"] == "actioned"
    assert rows[0]["subject_id"].startswith(f"TST-{world.tag}:")
    f = rows[0]["facts"]
    assert f["ok"] is True and f["error"] is None and f["caseIds"] == [did]
    assert "GB00SECRET" not in f["resultSummary"] and "refund" in f["resultSummary"]
    # a second run (late duplicate) never runs it again
    assert R.run(did, agent_nick=stub.nick)["status"] == "already_replayed"
    assert len(stub.calls) == 1


@live
def test_replay_rechecks_current_policies(conn, world, stub, monkeypatch):
    approve_doc = _doc(world, f"TST-{world.tag}")
    did, fid = _open(conn, world, approve_doc)
    # after the case opened, a block policy went live
    _policies(monkeypatch, [approve_doc, _doc(world, f"TSN-{world.tag}", outcome="block")])
    _approve(conn, world, did)
    out = R.run(did, agent_nick=stub.nick)
    assert out["status"] == "blocked"
    assert stub.calls == []
    rows = _replays(conn, world)
    assert len(rows) == 1
    assert rows[0]["facts"]["ok"] is False and rows[0]["facts"]["error"] == R.BLOCKED_REASON
    assert rows[0]["facts"]["blockedBy"] == [f"TSN-{world.tag}"]


@live
def test_replay_opens_a_case_for_a_new_approve_policy_and_waits(conn, world, stub, monkeypatch):
    first = _doc(world, f"TST-{world.tag}")
    did, _ = _open(conn, world, first)
    newer = _doc(world, f"TSN-{world.tag}")
    _policies(monkeypatch, [first, newer])
    _approve(conn, world, did)
    out = R.run(did, agent_nick=stub.nick)
    assert out["status"] == "new_approval_required" and len(out["caseIds"]) == 1
    assert stub.calls == [] and _replays(conn, world) == []
    new_id = out["caseIds"][0]
    with conn.cursor() as cur:
        cur.execute("SELECT status, policy_name, facts FROM proc.bp_decision WHERE decision_id = %s", (new_id,))
        status, name, facts = cur.fetchone()
    assert status == "open" and name == f"TSN-{world.tag}" and facts["replayOf"] == [did]
    assert facts["action"]["args"] == ARGS
    # re-running before the new case is decided waits, opens nothing more
    assert R.run(did, agent_nick=stub.nick)["status"] == "waiting"
    # approving the new case runs the action, once
    _approve(conn, world, new_id, replay=lambda d: R.run(d, agent_nick=stub.nick))
    assert stub.calls == [ARGS]


@live
def test_two_case_group_runs_only_after_both_approve(conn, world, stub, monkeypatch):
    d1, d2 = _doc(world, f"TST-{world.tag}"), _doc(world, f"TSN-{world.tag}")
    _policies(monkeypatch, [d1, d2])
    group = f"grp-{world.tag}"
    c1, _ = _open(conn, world, d1, group)
    c2, _ = _open(conn, world, d2, group)
    run = lambda d: R.run(d, agent_nick=stub.nick)   # noqa: E731
    _approve(conn, world, c1, replay=run)
    assert stub.calls == [] and _replays(conn, world) == []
    _approve(conn, world, c2, replay=run)
    assert stub.calls == [ARGS]
    rows = _replays(conn, world)
    assert len(rows) == 1 and sorted(rows[0]["facts"]["caseIds"]) == sorted([c1, c2])


@live
def test_group_with_a_rejection_never_runs(conn, world, stub, monkeypatch):
    d1, d2 = _doc(world, f"TST-{world.tag}"), _doc(world, f"TSN-{world.tag}")
    _policies(monkeypatch, [d1, d2])
    group = f"grp-{world.tag}"
    c1, _ = _open(conn, world, d1, group)
    c2, _ = _open(conn, world, d2, group)
    A.act(conn, c2, principal=_P(f"sub-{world.tag}", world.email), verb="reject", reason="no",
          now=datetime.now(timezone.utc), replay=_NO_REPLAY)
    _approve(conn, world, c1)
    assert R.run(c1, agent_nick=stub.nick)["status"] == "rejected"
    assert stub.calls == [] and _replays(conn, world) == []


@live
def test_exactly_once_under_two_concurrent_approvals(world, stub, monkeypatch):
    from services.db import get_conn
    d1, d2 = _doc(world, f"TST-{world.tag}"), _doc(world, f"TSN-{world.tag}")
    _policies(monkeypatch, [d1, d2])
    group = f"grp-{world.tag}"
    with get_conn() as c:
        c1, _ = _open(c, world, d1, group)
        c2, _ = _open(c, world, d2, group)
    # both approvals commit first, then both replays race for the same group
    barrier = threading.Barrier(2)
    stub.delay = 0.3
    errors = []

    def racing_replay(d):
        barrier.wait(timeout=10)
        return R.run(d, agent_nick=stub.nick)

    approved = threading.Barrier(2)

    def worker(did):
        try:
            with get_conn() as c:
                A.act(c, did, principal=_P(f"sub-{world.tag}", world.email), verb="approve", reason=None,
                      now=datetime.now(timezone.utc),
                      replay=lambda d: (approved.wait(timeout=10), racing_replay(d)))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    ts = [threading.Thread(target=worker, args=(c,)) for c in (c1, c2)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=30)
    assert errors == []
    assert stub.calls == [ARGS]
    with get_conn() as c:
        assert len(_replays(c, world)) == 1


@live
def test_replay_error_is_recorded_and_never_raises(conn, world, stub, monkeypatch):
    doc = _doc(world, f"TST-{world.tag}")
    _policies(monkeypatch, [doc])
    did, _ = _open(conn, world, doc)
    _approve(conn, world, did)
    stub.fail = ValueError("bank refused GB00SECRET")
    out = R.run(did, agent_nick=stub.nick)
    assert out["status"] == "error" and out["ok"] is False
    rows = _replays(conn, world)
    assert len(rows) == 1
    f = rows[0]["facts"]
    assert f["ok"] is False and "ValueError" in f["error"] and "GB00SECRET" not in f["error"]


@live
def test_no_agent_runtime_is_recorded_and_not_run(conn, world, stub, monkeypatch):
    doc = _doc(world, f"TST-{world.tag}")
    _policies(monkeypatch, [doc])
    monkeypatch.setattr(R, "_resolve_agent_nick", lambda _n: None)
    did, _ = _open(conn, world, doc)
    _approve(conn, world, did)
    out = R.run(did)
    assert out["status"] == "error" and out["error"] == R.NO_RUNTIME and stub.calls == []
    assert _replays(conn, world)[0]["facts"]["outcome"] == "not_run"


@live
def test_policy_check_unavailable_does_not_run(conn, world, stub, monkeypatch):
    doc = _doc(world, f"TST-{world.tag}")
    did, _ = _open(conn, world, doc)
    _approve(conn, world, did)

    def down():
        raise RuntimeError("store down")
    monkeypatch.setattr(R, "_load_policies", down)
    assert R.run(did, agent_nick=stub.nick)["status"] == "check_unavailable"
    assert stub.calls == []
