"""Approval cases, decisions and the timeout sweeper against bp_testdb.

Needs PROCWISE_TEST_LIVE_DB=1 and DB_NAME=bp_testdb. The decider-map rows, decisions and
notifications made here are removed afterwards. proc.bp_policy_firing is append-only BY DESIGN,
so every run leaves its firing rows (policy_key 'TST-<hex>') in bp_testdb; dashboards and counts
over the firing log must exclude policy_key LIKE 'TST-%'.
"""
import copy
import os
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from services.agent_policy import approvals as A
from services.agent_policy.compiler import compile_policy
from tests.agent_policy.fixtures import FORM_EXAMPLE, SETTINGS

live = pytest.mark.skipif(os.getenv("PROCWISE_TEST_LIVE_DB") != "1", reason="live DB required")

NOW = datetime(2026, 10, 8, 9, 0, tzinfo=timezone.utc)


def _NO_REPLAY(_decision_id):
    return None


# ------------------------------------------------------------------ pure
def test_reject_without_reason_refused_before_any_db_work():
    with pytest.raises(A.ApprovalRefused) as e:
        A.act(None, 1, principal=SimpleNamespace(subject="s"), verb="reject", reason="  ", now=NOW)
    assert e.value.code == "reason_required" and e.value.status == 422


# ------------------------------------------------------------------ live
def _P(subject, email=None, groups=()):
    return SimpleNamespace(subject=subject, email=email, claims={"cognito:groups": list(groups)})


@pytest.fixture
def conn():
    assert os.getenv("DB_NAME") == "bp_testdb", "approval live tests run on bp_testdb only"
    from services.db import get_conn
    with get_conn() as c:
        yield c


@pytest.fixture
def world(conn):
    """Two linked deciders (L1 by email, L2 by group) plus cleanup of everything made here."""
    tag = uuid.uuid4().hex[:8]
    l1, l2 = f"TST L1 {tag}", f"TST L2 {tag}"
    w = SimpleNamespace(tag=tag, l1=l1, l2=l2, l1_email=f"l1-{tag}@example.test",
                        l2_group=f"TST_GROUP_{tag}", requester=f"req-{tag}@example.test",
                        decisions=[])
    with conn.cursor() as cur:
        cur.execute("INSERT INTO proc.bp_policy_decider_map (decider_name, groups, emails, last_modified_by) "
                    "VALUES (%s, '{}', %s, 'test'), (%s, %s, '{}', 'test')",
                    (l1, [w.l1_email], l2, [w.l2_group]))
    yield w
    with conn.cursor() as cur:
        subj = [f"TST-{tag}:%"]
        cur.execute("SELECT decision_id FROM proc.bp_decision WHERE subject_type = %s AND subject_id LIKE %s",
                    (A.SUBJECT_TYPE, subj[0]))
        ids = [r[0] for r in cur.fetchall()]
        cur.execute("DELETE FROM proc.bp_policy_notification WHERE link = ANY(%s)",
                    ([f"decision:{i}" for i in ids],))
        cur.execute("DELETE FROM proc.bp_decision WHERE decision_id = ANY(%s)", (ids,))
        cur.execute("DELETE FROM proc.bp_policy_decider_map WHERE decider_name = ANY(%s)", ([l1, l2],))


def _doc(w, levels):
    form = copy.deepcopy(FORM_EXAMPLE)
    form["checked"] = {"by": "user_8841", "at": "2026-10-08T09:14:00Z"}
    form["deciders"] = levels
    form["responseTime"] = "PT2H"
    return compile_policy(form, policy_key=f"TST-{w.tag}", version=3, status="live",
                          settings=SETTINGS, never_suggest=False)


def _firing(conn, key):
    with conn.cursor() as cur:
        cur.execute("INSERT INTO proc.bp_policy_firing (policy_key, policy_version, checkpoint, action_name, "
                    "agent, outcome, result) VALUES (%s, 3, 'tool.call.before', 'refund.issue', 'agent_nick', "
                    "'approve', 'paused_for_approval') RETURNING firing_id", (key,))
        return cur.fetchone()[0]


def _open(conn, w, levels=None, requested_by="__default__"):
    levels = levels if levels is not None else [w.l1, w.l2]
    doc = _doc(w, levels)
    fid = _firing(conn, doc["id"])
    did = A.open_case(conn, policy_doc=doc, firing_id=fid,
                      action={"tool": "refund.issue", "args": {"amount": 900, "iban": "GB00SECRET"},
                              "agent": "agent_nick", "workflowId": "wf-1", "userId": "u-1",
                              "reason": "customer asked"},
                      requested_by=w.requester if requested_by == "__default__" else requested_by, now=NOW)
    return did, fid


def _row(conn, table, key, val):
    with conn.cursor() as cur:
        cur.execute(f"SELECT * FROM proc.{table} WHERE {key} = %s", (val,))
        r = cur.fetchone()
        return dict(zip([d[0] for d in cur.description], r))


def _notes(conn, did):
    with conn.cursor() as cur:
        cur.execute("SELECT recipient, message FROM proc.bp_policy_notification WHERE link = %s "
                    "ORDER BY notification_id", (f"decision:{did}",))
        return cur.fetchall()


def _actions(conn, w):
    with conn.cursor() as cur:
        cur.execute("SELECT decision, actioned_by, actioned_at, override_reason, status, current_level "
                    "FROM proc.bp_decision WHERE subject_type = %s AND subject_id LIKE %s "
                    "AND status = 'actioned' AND actioned_by IS NOT NULL",
                    (A.SUBJECT_TYPE, f"TST-{w.tag}:%"))
        return cur.fetchall()


@live
def test_open_case_shape(conn, world):
    did, fid = _open(conn, world)
    r = _row(conn, "bp_decision", "decision_id", did)
    assert r["subject_id"] == f"TST-{world.tag}:{fid}" and r["status"] == "open"
    assert r["decision"] == "approve_or_reject" and r["resolution"] == "escalated"
    assert r["policy_name"] == f"TST-{world.tag}" and r["policy_id"] is None
    assert r["levels"] == [{"name": world.l1, "respondWithin": "PT2H"}, {"name": world.l2, "respondWithin": "PT2H"}]
    assert r["current_level"] == 0 and r["on_timeout"] == "escalate_next" and r["options"] == ["approve", "reject"]
    assert r["respond_by"] == NOW + timedelta(hours=2) and r["created_by"] == world.requester
    f = r["facts"]
    assert f["action"]["args"] == {"amount": 900, "iban": "GB00SECRET"}      # full args, for the replay
    assert f["policy"]["excerpt"] == FORM_EXAMPLE["source"]["excerpt"] and f["policy"]["version"] == 3
    assert f["approvalInputs"] == ["args.amount", "agent.reason"] and f["requestedBy"] == world.requester
    assert "unroutable" not in f
    assert _row(conn, "bp_policy_firing", "firing_id", fid)["decision_id"] == did


@live
def test_unmapped_level_still_opens_and_is_marked_unroutable(conn, world):
    did, _ = _open(conn, world, levels=[world.l1, f"Nobody {world.tag}"], requested_by=None)
    r = _row(conn, "bp_decision", "decision_id", did)
    assert r["status"] == "open" and r["facts"]["unroutable"] == [f"Nobody {world.tag}"]
    assert r["created_by"] == "agent:agent_nick"


@live
def test_levels_keep_their_order_and_escalate_in_order(conn, world):
    did, _ = _open(conn, world, levels=[world.l2, world.l1])
    assert [lv["name"] for lv in _row(conn, "bp_decision", "decision_id", did)["levels"]] == [world.l2, world.l1]
    later = NOW + timedelta(hours=2, seconds=1)
    assert A.sweep(conn, later, decision_ids=[did])["escalated"] == 1
    r = _row(conn, "bp_decision", "decision_id", did)
    assert r["current_level"] == 1 and r["status"] == "open" and r["respond_by"] == later + timedelta(hours=2)
    notes = _notes(conn, did)
    assert [n[0] for n in notes] == [world.l1]
    assert "previous approver did not answer in time" in notes[0][1]
    assert "GB00SECRET" not in notes[0][1] and "900" not in notes[0][1]


@live
def test_not_yet_due_is_not_swept(conn, world):
    did, _ = _open(conn, world)
    assert A.sweep(conn, NOW + timedelta(hours=1), decision_ids=[did]) == \
        {"escalated": 0, "rejected": 0, "skipped": 0, "errors": 0}


@live
def test_last_level_timeout_rejects_and_never_approves(conn, world):
    did, fid = _open(conn, world)
    t1 = NOW + timedelta(hours=2, seconds=1)
    A.sweep(conn, t1, decision_ids=[did])
    t2 = t1 + timedelta(hours=2, seconds=1)
    assert A.sweep(conn, t2, decision_ids=[did])["rejected"] == 1
    assert _row(conn, "bp_decision", "decision_id", did)["status"] == "actioned"
    acts = _actions(conn, world)
    assert len(acts) == 1
    decision, by, at, reason, status, level = acts[0]
    assert decision == "reject" and by == "system:timeout" and level == 1
    assert reason == "No decision in time; a timeout never approves"
    f = _row(conn, "bp_policy_firing", "firing_id", fid)
    assert f["result"] == "timed_out" and f["decided_by"] == "system:timeout" and f["decided_level"] == 1
    assert [n[0] for n in _notes(conn, did)] == [world.l2, world.l1, world.requester]
    # sweeping again does nothing: the case is closed
    assert A.sweep(conn, t2 + timedelta(days=1), decision_ids=[did])["rejected"] == 0


@live
def test_single_level_timeout_rejects(conn, world):
    did, fid = _open(conn, world, levels=[world.l1])
    assert _row(conn, "bp_decision", "decision_id", did)["on_timeout"] == "reject"
    assert A.sweep(conn, NOW + timedelta(hours=3), decision_ids=[did])["rejected"] == 1
    assert _row(conn, "bp_policy_firing", "firing_id", fid)["result"] == "timed_out"
    assert [a[0] for a in _actions(conn, world)] == ["reject"]


@live
def test_concurrent_sweeps_escalate_once(conn, world):
    """Two sweepers race on one due case. The first holds the row lock (slowed by the seam);
    the second must skip it, so the case escalates exactly once and one notification is sent."""
    from services.db import get_conn
    did, _ = _open(conn, world)
    later = NOW + timedelta(hours=2, seconds=1)
    results, errors = [], []
    barrier = threading.Barrier(2)

    def slow(_did):
        time.sleep(1.0)

    A._after_lock = slow
    try:
        def run():
            try:
                with get_conn() as c:
                    barrier.wait()
                    results.append(A.sweep(c, later, decision_ids=[did]))
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)
        threads = [threading.Thread(target=run) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(30)
    finally:
        A._after_lock = None
    assert not errors
    assert sum(r["escalated"] for r in results) == 1
    assert _row(conn, "bp_decision", "decision_id", did)["current_level"] == 1
    assert len(_notes(conn, did)) == 1


@live
def test_act_approve_records_actor_level_and_time_then_replays_after_commit(conn, world):
    did, fid = _open(conn, world)
    seen = []

    def replay(d):
        # runs after commit: a fresh connection already sees the closed case
        from services.db import get_conn
        with get_conn() as c2, c2.cursor() as cur:
            cur.execute("SELECT status FROM proc.bp_decision WHERE decision_id = %s", (d,))
            seen.append((d, cur.fetchone()[0]))

    when = NOW + timedelta(minutes=5)
    out = A.act(conn, did, principal=_P("sub-l1", world.l1_email.upper()), verb="approve",
                reason=None, now=when, replay=replay)
    assert out["result"] == "approved" and out["level"] == 0 and out["levelName"] == world.l1
    assert out["decidedBy"] == "sub-l1"
    assert seen == [(did, "actioned")]
    acts = _actions(conn, world)
    assert len(acts) == 1 and acts[0][:3] == ("approve", "sub-l1", when) and acts[0][5] == 0
    f = _row(conn, "bp_policy_firing", "firing_id", fid)
    assert (f["result"], f["decided_by"], f["decided_level"], f["decided_at"]) == ("approved", "sub-l1", 0, when)


@live
def test_act_reject_with_reason_at_escalated_level(conn, world):
    did, fid = _open(conn, world)
    A.sweep(conn, NOW + timedelta(hours=3), decision_ids=[did])
    # L1 is no longer the current level
    with pytest.raises(A.ApprovalRefused) as e:
        A.act(conn, did, principal=_P("sub-l1", world.l1_email), verb="reject", reason="no", now=NOW)
    assert e.value.code == "not_eligible"
    out = A.act(conn, did, principal=_P("sub-l2", groups=[world.l2_group]), verb="reject",
                reason="Over budget", now=NOW + timedelta(hours=4),
                replay=lambda d: pytest.fail("a rejection never replays"))
    assert out["result"] == "rejected" and out["level"] == 1
    f = _row(conn, "bp_policy_firing", "firing_id", fid)
    assert f["result"] == "rejected" and f["reason"] == "Over budget"


@live
def test_act_refuses_ineligible(conn, world):
    did, _ = _open(conn, world)
    for p in (_P("admin", "admin@example.test", ["PROCWISE_ADMIN"]), _P("sub-l2", groups=[world.l2_group])):
        with pytest.raises(A.ApprovalRefused) as e:
            A.act(conn, did, principal=p, verb="approve", reason=None, now=NOW,
              replay=_NO_REPLAY)
        assert e.value.code == "not_eligible" and e.value.status == 403
    assert _row(conn, "bp_decision", "decision_id", did)["status"] == "open" and not _actions(conn, world)


@live
def test_act_refuses_self_approval(conn, world):
    # the requester IS linked to level 1, and is still refused
    did, _ = _open(conn, world, requested_by=world.l1_email)
    with pytest.raises(A.ApprovalRefused) as e:
        A.act(conn, did, principal=_P("someone", world.l1_email), verb="approve", reason=None, now=NOW,
              replay=_NO_REPLAY)
    assert e.value.code == "self_approval"
    did2, _ = _open(conn, world, requested_by="sub-l1")
    with pytest.raises(A.ApprovalRefused) as e:
        A.act(conn, did2, principal=_P("sub-l1", world.l1_email), verb="approve", reason=None, now=NOW,
              replay=_NO_REPLAY)
    assert e.value.code == "self_approval"
    assert not _actions(conn, world)


@live
def test_act_refuses_reject_without_reason_and_already_closed(conn, world):
    did, _ = _open(conn, world)
    p = _P("sub-l1", world.l1_email)
    with pytest.raises(A.ApprovalRefused) as e:
        A.act(conn, did, principal=p, verb="reject", reason="", now=NOW)
    assert e.value.code == "reason_required"
    A.act(conn, did, principal=p, verb="approve", reason=None, now=NOW,
              replay=_NO_REPLAY)
    with pytest.raises(A.ApprovalRefused) as e:
        A.act(conn, did, principal=p, verb="reject", reason="changed my mind", now=NOW)
    assert e.value.code == "not_open" and e.value.status == 409
    assert len(_actions(conn, world)) == 1


@live
def test_replay_failure_does_not_undo_the_approval(conn, world):
    did, _ = _open(conn, world)

    def boom(_d):
        raise RuntimeError("replay broke")

    out = A.act(conn, did, principal=_P("sub-l1", world.l1_email), verb="approve", reason=None,
                now=NOW, replay=boom)
    assert out["result"] == "approved"
    assert _row(conn, "bp_decision", "decision_id", did)["status"] == "actioned"


@live
@pytest.mark.parametrize("sla,expect", [("P1W", timedelta(weeks=1)), ("PT1.5H", timedelta(minutes=90)),
                                        ("soon", timedelta(hours=6)), (None, timedelta(hours=6))])
def test_timer_and_agent_are_told_the_same_wait(conn, world, sla, expect):
    from services.agent_policy import enforcement
    doc = _doc(world, [world.l1])
    doc["enforcement"]["intervention"]["sla"]["respondWithin"] = sla
    fid = _firing(conn, doc["id"])
    did = A.open_case(conn, policy_doc=doc, firing_id=fid, action={"tool": "refund.issue", "args": {}},
                      requested_by=None, now=NOW, default_response_time="PT6H")
    r = _row(conn, "bp_decision", "decision_id", did)
    assert r["respond_by"] == NOW + expect
    told = enforcement.check({"checkpoint": "tool.call.before", "tool.name": "refund.issue",
                              "args": {"amount": 900}}, [doc], default_response_time="PT6H")
    assert told.to_agent["respondWithin"] == r["levels"][0]["respondWithin"]


@live
def test_approve_without_injected_replay_runs_replay_module(conn, world, monkeypatch):
    import sys
    called = []
    stub = SimpleNamespace(run=lambda d: called.append(d))
    import services.agent_policy as pkg
    monkeypatch.setitem(sys.modules, "services.agent_policy.replay", stub)
    monkeypatch.setattr(pkg, "replay", stub, raising=False)
    did, _ = _open(conn, world)
    A.act(conn, did, principal=_P("sub-l1", world.l1_email), verb="approve", reason=None, now=NOW)
    assert called == [did]


@live
def test_missing_replay_module_is_an_error_not_silence(conn, world, monkeypatch, caplog):
    import logging
    import sys
    import services.agent_policy as pkg
    monkeypatch.setitem(sys.modules, "services.agent_policy.replay", None)   # import -> ImportError
    monkeypatch.delattr(pkg, "replay", raising=False)
    did, _ = _open(conn, world)
    with caplog.at_level(logging.ERROR, logger=A.__name__):
        out = A.act(conn, did, principal=_P("sub-l1", world.l1_email), verb="approve", reason=None, now=NOW)
    assert out["result"] == "approved"
    assert any(r.levelno == logging.ERROR and "approved action has nowhere to run" in r.getMessage()
               for r in caplog.records)


@live
def test_multi_level_escalates_even_if_sla_says_reject(conn, world):
    doc = _doc(world, [world.l1, world.l2])
    doc["enforcement"]["intervention"]["sla"]["onTimeout"] = "reject"
    fid = _firing(conn, doc["id"])
    did = A.open_case(conn, policy_doc=doc, firing_id=fid, action={"tool": "refund.issue", "args": {}},
                      requested_by=None, now=NOW)
    assert _row(conn, "bp_decision", "decision_id", did)["on_timeout"] == "escalate_next"
    assert A.sweep(conn, NOW + timedelta(hours=3), decision_ids=[did])["escalated"] == 1
    # even a stored 'reject' on a multi-level row escalates: only the last level rejects
    did2, _ = _open(conn, world)
    with conn.cursor() as cur:
        cur.execute("UPDATE proc.bp_decision SET on_timeout = 'reject' WHERE decision_id = %s", (did2,))
    assert A.sweep(conn, NOW + timedelta(hours=3), decision_ids=[did2])["escalated"] == 1
    assert _row(conn, "bp_decision", "decision_id", did2)["current_level"] == 1


@live
def test_unroutable_level_refuses_admin_and_times_out_to_rejected(conn, world):
    nobody = f"Nobody {world.tag}"
    did, fid = _open(conn, world, levels=[nobody])
    assert _row(conn, "bp_decision", "decision_id", did)["facts"]["unroutable"] == [nobody]
    with pytest.raises(A.ApprovalRefused) as e:
        A.act(conn, did, principal=_P("admin", "admin@example.test", ["PROCWISE_ADMIN"]),
              verb="approve", reason=None, now=NOW, replay=_NO_REPLAY)
    assert e.value.code == "not_eligible"
    assert A.sweep(conn, NOW + timedelta(hours=3), decision_ids=[did])["rejected"] == 1
    assert _row(conn, "bp_policy_firing", "firing_id", fid)["result"] == "timed_out"
    assert [a[:2] for a in _actions(conn, world)] == [("reject", "system:timeout")]


# ------------------------------------------------------------------ stage 4 task 0, m2: lock after permission
@live
def test_a_refused_caller_never_takes_group_locks(conn, world, monkeypatch):
    """Eligibility, self-approval and not-open are decided BEFORE the group lock, so a refused
    caller never blocks (or is blocked by) the people deciding the group."""
    locked = []
    real = A._member_ids

    def spy(cur, decision_id, facts, *, lock=False):
        locked.append(lock)
        return real(cur, decision_id, facts, lock=lock)
    monkeypatch.setattr(A, "_member_ids", spy)

    did, _ = _open(conn, world)
    with pytest.raises(A.ApprovalRefused) as e:
        A.act(conn, did, principal=_P("sub-l2", groups=[world.l2_group]), verb="approve", reason=None, now=NOW,
              replay=_NO_REPLAY)
    assert e.value.code == "not_eligible"
    did2, _ = _open(conn, world, requested_by="sub-l1")
    with pytest.raises(A.ApprovalRefused) as e:
        A.act(conn, did2, principal=_P("sub-l1", world.l1_email), verb="approve", reason=None, now=NOW,
              replay=_NO_REPLAY)
    assert e.value.code == "self_approval"
    assert True not in locked, "a refused caller took a group lock"

    # an eligible caller still locks the group, once, and the decision is recorded
    A.act(conn, did, principal=_P("sub-l1", world.l1_email), verb="approve", reason=None, now=NOW,
          replay=_NO_REPLAY)
    assert locked.count(True) == 1
    # and a second decision on the now-closed case is refused without locks
    n = locked.count(True)
    with pytest.raises(A.ApprovalRefused) as e:
        A.act(conn, did, principal=_P("sub-l1", world.l1_email), verb="approve", reason=None, now=NOW,
              replay=_NO_REPLAY)
    assert e.value.code == "not_open" and locked.count(True) == n
