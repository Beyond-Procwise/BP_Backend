"""Stage 3 final fix wave against bp_testdb: I2, I3, I4, T6, sibling closing, D2/D3, unroutable notice.

Needs PROCWISE_TEST_LIVE_DB=1 and DB_NAME=bp_testdb. No model: the tool is the recording stub of
test_replay_live and live policies are injected through replay._load_policies. Every case,
notification and decider-map row made here is removed afterwards; firing rows are append-only BY
DESIGN and stay (policy_key 'TST-<hex>' / 'TSN-<hex>').
"""
from datetime import datetime, timedelta, timezone

import pytest

from services.agent_policy import approval_views as V
from services.agent_policy import approvals as A
from services.agent_policy import gate as G
from services.agent_policy import replay as R
from services.agent_policy import replay_retry
from tests.agent_policy.test_replay_live import (  # noqa: F401 - fixtures
    ARGS, _approve, _doc, _NO_REPLAY, _open, _P, _policies, _replays, conn, live, stub, world)


def _firings(conn, key):
    with conn.cursor() as cur:
        cur.execute("SELECT firing_id, outcome, result, reason, decision_id FROM proc.bp_policy_firing "
                    "WHERE policy_key = %s ORDER BY firing_id", (key,))
        return [dict(zip(("id", "outcome", "result", "reason", "decision_id"), r)) for r in cur.fetchall()]


def _notes(conn, *, firing_ids=None, link=None):
    with conn.cursor() as cur:
        if firing_ids is not None:
            cur.execute("SELECT recipient, message, link FROM proc.bp_policy_notification "
                        "WHERE firing_id = ANY(%s) ORDER BY notification_id", (list(firing_ids),))
        else:
            cur.execute("SELECT recipient, message, link FROM proc.bp_policy_notification "
                        "WHERE link = %s ORDER BY notification_id", (link,))
        return [dict(zip(("recipient", "message", "link"), r)) for r in cur.fetchall()]


def _status(conn, did):
    with conn.cursor() as cur:
        cur.execute("SELECT status FROM proc.bp_decision WHERE decision_id = %s", (did,))
        return cur.fetchone()[0]


def _notify_row(conn, key, case_id):
    """A paused notify row linked to a case, as the gate writes for a paused call."""
    with conn.cursor() as cur:
        cur.execute("INSERT INTO proc.bp_policy_firing (policy_key, policy_version, checkpoint, action_name, "
                    "outcome, result, decision_id) VALUES (%s, 1, 'tool.call.before', 'refund.issue', 'notify', "
                    "'paused_for_approval', %s) RETURNING firing_id", (key, case_id))
        return cur.fetchone()[0]


def _row(conn, fid):
    with conn.cursor() as cur:
        cur.execute("SELECT result, decided_by, reason FROM proc.bp_policy_firing WHERE firing_id = %s", (fid,))
        return dict(zip(("result", "decided_by", "reason"), cur.fetchone()))


# ------------------------------------------------------------------ I2: firings at the re-check
@live
def test_recheck_block_writes_firing_rows_and_notifications(conn, world, stub, monkeypatch):
    told = f"TST told {world.tag}"
    approve_doc = _doc(world, f"TST-{world.tag}")
    did, _ = _open(conn, world, approve_doc)
    block_doc = _doc(world, f"TSN-{world.tag}", outcome="block", notify=[told])
    _policies(monkeypatch, [approve_doc, block_doc])
    _approve(conn, world, did)
    assert R.run(did, agent_nick=stub.nick)["status"] == "blocked"
    rows = _firings(conn, f"TSN-{world.tag}")
    assert [(r["outcome"], r["result"], r["reason"]) for r in rows] == [("block", "blocked", R.RECHECK_REASON)]
    notes = _notes(conn, firing_ids=[rows[0]["id"]])
    assert [n["recipient"] for n in notes] == [told] and "was blocked" in notes[0]["message"]
    assert "GB00SECRET" not in notes[0]["message"]
    # the approved policy, already decided by a person, gets no new row
    assert [r["result"] for r in _firings(conn, f"TST-{world.tag}")] == ["approved"]
    assert stub.calls == []


@live
def test_recheck_notify_writes_its_row_and_the_action_runs(conn, world, stub, monkeypatch):
    told = f"TST told {world.tag}"
    approve_doc = _doc(world, f"TST-{world.tag}")
    did, _ = _open(conn, world, approve_doc)
    _policies(monkeypatch, [approve_doc, _doc(world, f"TSN-{world.tag}", outcome="notify", notify=[told])])
    _approve(conn, world, did)
    assert R.run(did, agent_nick=stub.nick)["status"] == "ran"
    rows = _firings(conn, f"TSN-{world.tag}")
    assert [(r["outcome"], r["result"]) for r in rows] == [("notify", "allowed")]
    notes = _notes(conn, firing_ids=[rows[0]["id"]])
    assert [n["recipient"] for n in notes] == [told] and "was allowed to run" in notes[0]["message"]
    assert stub.calls == [ARGS]


@live
def test_recheck_rows_roll_back_with_the_claim(conn, world, stub, monkeypatch):
    """Inside _claim's transaction: a failure after the rows were written leaves none behind."""
    approve_doc = _doc(world, f"TST-{world.tag}")
    did, _ = _open(conn, world, approve_doc)
    _policies(monkeypatch, [approve_doc, _doc(world, f"TSN-{world.tag}", outcome="block")])
    _approve(conn, world, did)

    def boom(*a, **kw):
        raise RuntimeError("insert failed")
    monkeypatch.setattr(R, "_insert_replay", boom)
    assert R.run(did, agent_nick=stub.nick)["status"] == "error"
    assert _firings(conn, f"TSN-{world.tag}") == []


# ------------------------------------------------------------------ I3: an outage never consumes the claim
@live
def test_recheck_outage_writes_nothing_so_the_retry_can_run_it(conn, world, stub, monkeypatch):
    doc = _doc(world, f"TST-{world.tag}")
    did, _ = _open(conn, world, doc)
    _approve(conn, world, did)

    def down():
        raise RuntimeError("store down")
    monkeypatch.setattr(R, "_load_policies", down)
    out = R.run(did, agent_nick=stub.nick)
    assert out["status"] == "check_unavailable" and out.get("retry") is True
    assert _replays(conn, world) == [] and stub.calls == []
    # the sweeper's retry finds it (no replay row) and, with the store back, it runs
    _policies(monkeypatch, [doc])
    later = datetime.now(timezone.utc) + replay_retry.RETRY_AFTER + timedelta(seconds=5)
    counts = replay_retry.retry_lost_replays(conn, later, run=lambda d: R.run(d, agent_nick=stub.nick),
                                             decision_ids=[did])
    assert counts["retried"] == 1 and stub.calls == [ARGS]


@live
def test_recheck_outage_on_the_last_retry_records_not_run_and_tells_people(conn, world, stub, monkeypatch):
    doc = _doc(world, f"TST-{world.tag}")
    did, fid = _open(conn, world, doc)
    _approve(conn, world, did)

    def down():
        raise RuntimeError("store down")
    monkeypatch.setattr(R, "_load_policies", down)
    later = datetime.now(timezone.utc) + replay_retry.RETRY_AFTER + timedelta(seconds=5)
    run = lambda d: R.run(d, agent_nick=stub.nick)   # noqa: E731
    for attempt in range(1, replay_retry.MAX_ATTEMPTS + 1):
        assert replay_retry.retry_lost_replays(conn, later, run=run, decision_ids=[did])["retried"] == 1
        if attempt < replay_retry.MAX_ATTEMPTS:
            assert _replays(conn, world) == [], f"attempt {attempt} consumed the claim"
    rows = _replays(conn, world)
    assert len(rows) == 1 and rows[0]["facts"]["outcome"] == "not_run"
    assert rows[0]["facts"]["error"] == "policy_check_unavailable: RuntimeError"
    notes = [n for n in _notes(conn, link=f"decision:{did}") if "could not be run" in n["message"]]
    assert sorted(n["recipient"] for n in notes) == sorted([f"sub-{world.tag}", world.requester])
    assert all("approved but could not be run" in n["message"] for n in notes)
    assert stub.calls == []
    # nothing further to retry
    assert replay_retry.retry_lost_replays(conn, later, run=run, decision_ids=[did])["retried"] == 0


# ------------------------------------------------------------------ I4: the re-check's new case
@live
def test_recheck_new_case_notifies_its_first_level_and_is_found_by_reuse(conn, world, stub, monkeypatch):
    first, newer = _doc(world, f"TST-{world.tag}"), _doc(world, f"TSN-{world.tag}")
    did, _ = _open(conn, world, first)
    _policies(monkeypatch, [first, newer])
    _approve(conn, world, did)
    out = R.run(did, agent_nick=stub.nick)
    assert out["status"] == "new_approval_required"
    new_id = out["caseIds"][0]
    notes = _notes(conn, link=f"decision:{new_id}")
    assert [n["recipient"] for n in notes] == [world.decider] and "needs your decision" in notes[0]["message"]
    with conn.cursor() as cur:
        cur.execute("SELECT facts FROM proc.bp_decision WHERE decision_id = %s", (new_id,))
        facts = cur.fetchone()[0]
    assert facts["argsDigest"] == G.args_digest(ARGS) and facts["requestedBy"] == world.requester
    # the gate's repeat-call reuse finds it exactly as it finds a case the gate opened
    with conn.cursor() as cur:
        found = G._open_case_for(cur, key=f"TSN-{world.tag}", tool_name="refund.issue", digest=G.args_digest(ARGS),
                                 workflow_id=f"wf-{world.tag}", requested_by=world.requester)
    assert found and found["decision_id"] == new_id


# ------------------------------------------------------------------ T6 + siblings + D2
@live
def test_notify_row_of_a_two_approval_group_waits_for_the_whole_group(conn, world):
    d1, d2 = _doc(world, f"TST-{world.tag}"), _doc(world, f"TSN-{world.tag}")
    group = f"grp-{world.tag}"
    c1, f1 = _open(conn, world, d1, group)
    c2, f2 = _open(conn, world, d2, group)
    note = _notify_row(conn, f"TST-{world.tag}", c1)
    out = _approve(conn, world, c1)
    assert out["group"] == {"open": 1, "approved": 1, "rejected": 0, "total": 2}
    assert _row(conn, note)["result"] == "paused_for_approval", "settled while another approval is open"
    assert _row(conn, f1)["result"] == "approved"
    out = _approve(conn, world, c2)
    assert out["group"] == {"open": 0, "approved": 2, "rejected": 0, "total": 2}
    assert _row(conn, note)["result"] == "approved"


@live
def test_a_rejection_closes_the_open_siblings_and_settles_the_group(conn, world):
    d1, d2 = _doc(world, f"TST-{world.tag}"), _doc(world, f"TSN-{world.tag}")
    group = f"grp-{world.tag}"
    c1, f1 = _open(conn, world, d1, group)
    c2, f2 = _open(conn, world, d2, group)
    note = _notify_row(conn, f"TST-{world.tag}", c1)
    out = A.act(conn, c1, principal=_P(f"sub-{world.tag}", world.email), verb="reject", reason="duplicate",
                now=datetime.now(timezone.utc), replay=_NO_REPLAY)
    assert out["group"] == {"open": 0, "approved": 0, "rejected": 2, "total": 2}
    assert _status(conn, c2) == "actioned"
    sib = _row(conn, f2)
    assert sib == {"result": "rejected", "decided_by": A.GROUP_ACTOR, "reason": A.GROUP_REFUSED_REASON}
    assert _row(conn, note)["result"] == "rejected"
    told = _notes(conn, link=f"decision:{c2}")
    assert any(n["recipient"] == world.decider and "no longer needs your decision" in n["message"] for n in told)
    # D3: both read as rejected, never as a bare "actioned"
    principal = _P(f"sub-{world.tag}", world.email)
    for did in (c1, c2):
        view = V.get_case(conn, did, principal, is_admin=True)
        assert view["outcome"] == "rejected" and view["history"]["group"]["rejected"] == 2


@live
def test_a_last_level_timeout_closes_the_siblings_and_reads_timed_out(conn, world):
    d1, d2 = _doc(world, f"TST-{world.tag}"), _doc(world, f"TSN-{world.tag}")
    group = f"grp-{world.tag}"
    c1, f1 = _open(conn, world, d1, group)
    c2, f2 = _open(conn, world, d2, group)
    note = _notify_row(conn, f"TST-{world.tag}", c1)
    # only c1 is past its deadline
    with conn.cursor() as cur:
        cur.execute("UPDATE proc.bp_decision SET respond_by = now() - interval '1 minute' WHERE decision_id = %s",
                    (c1,))
    counts = A.sweep(conn, datetime.now(timezone.utc), decision_ids=[c1, c2])
    assert counts["rejected"] == 1 and counts["errors"] == 0
    assert _status(conn, c2) == "actioned" and _row(conn, f2)["reason"] == A.GROUP_REFUSED_REASON
    assert _row(conn, note)["result"] == "timed_out"
    principal = _P(f"sub-{world.tag}", world.email)
    assert V.get_case(conn, c1, principal, is_admin=True)["outcome"] == "timed_out"
    assert V.get_case(conn, c2, principal, is_admin=True)["outcome"] == "rejected"
    # the list view carries the same outcome
    listed = {c["id"]: c for c in V.list_cases(conn, principal, is_admin=True, status="closed")}
    assert listed[c1]["outcome"] == "timed_out" and listed[c2]["outcome"] == "rejected"


@live
def test_a_single_case_still_settles_its_notify_row_and_reads_approved(conn, world):
    doc = _doc(world, f"TST-{world.tag}")
    did, fid = _open(conn, world, doc)
    note = _notify_row(conn, f"TST-{world.tag}", did)
    out = _approve(conn, world, did)
    assert out["group"] == {"open": 0, "approved": 1, "rejected": 0, "total": 1}
    assert _row(conn, note)["result"] == "approved" and _row(conn, fid)["result"] == "approved"
    view = V.get_case(conn, did, _P(f"sub-{world.tag}", world.email), is_admin=True)
    assert view["outcome"] == "approved" and view["history"]["group"]["open"] == 0
    assert view["history"]["replay"] is None


# ------------------------------------------------------------------ unroutable: the administrators hear
@live
def test_an_unroutable_case_tells_the_administrators(conn, world):
    doc = _doc(world, f"TST-{world.tag}")
    doc["enforcement"]["intervention"]["escalateTo"] = [{"name": f"TST nobody {world.tag}"}]
    did, _ = _open(conn, world, doc)
    notes = _notes(conn, link=f"decision:{did}")
    assert [n["recipient"] for n in notes] == [A.ADMIN_RECIPIENT]
    assert notes[0]["message"].endswith(f"cannot be routed: link TST nobody {world.tag}")

    class Admin:
        subject, email = f"admin-{world.tag}", f"admin-{world.tag}@example.test"
    mine = V.my_notifications(conn, Admin(), 200, is_admin=True)
    assert any(n["decisionId"] == did for n in mine)
    assert not any(n["decisionId"] == did for n in V.my_notifications(conn, Admin(), 200, is_admin=False))
