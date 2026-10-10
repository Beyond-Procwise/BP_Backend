"""An Approver records that an approved request was later reversed. Needs PROCWISE_TEST_LIVE_DB=1.

A reversal row is append-only BY DESIGN and stays in the shared database, so the direct tests use
only TST-<hex>a keys (never a real policy) and the two tests that need the scan or the activity
list endpoint use a real "TST …" policy on a tst_<tag> tool, retired at teardown.

The one worked number (the banding rule is in test_learning_scan_live's docstring): 33 approvals
at 510..830 under "args.amount gt 500" raise 500 -> 800 on band 1 = 510..800 (30 of 30). The
request at 600 is inside that band. Once it is marked reversed the band holds a reversal and
fails, nothing is proposed, and the card stays about the same change as "Not enough yet": the
released set (500, 800] is still 30 decisions, 30 approved, now 1 of them reversed.
"""
import json
import os
import uuid
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from api.routers import agent_policies as R
from repositories import agent_policy_repo as repo
from services.agent_policy import approval_views as V
from services.agent_policy import learning as L
from tests.agent_policy import test_learning_endpoints_live as E
from tests.agent_policy.test_learning_scan_live import BASE, NOW, _case_ids, _reverse, fire, world  # noqa: F401

pytestmark = pytest.mark.skipif(os.getenv("PROCWISE_TEST_LIVE_DB") != "1", reason="live DB required")

WHY = "TST the refund was clawed back"


def _stored(conn, firing_ids):
    cur = conn.cursor()
    cur.execute("SELECT firing_id, reversed_by, reversed_at, reason FROM proc.bp_policy_firing_reversal "
                "WHERE firing_id = ANY(%s) ORDER BY firing_id", (list(firing_ids),))
    return cur.fetchall()


def _firing(conn, key, outcome, result):
    cur = conn.cursor()
    cur.execute("INSERT INTO proc.bp_policy_firing (policy_key, policy_version, checkpoint, action_name, outcome, "
                "result, matched_values, decided_by, decided_at, created_at) "
                "VALUES (%s, 1, 'tool.call.before', 'tst', %s, %s, %s, 'person-0', %s, %s) RETURNING firing_id",
                (key, outcome, result, json.dumps({"args.amount": 600}), NOW, NOW))
    return cur.fetchone()[0]


def _refusal(conn, firing_id, reason=WHY):
    with pytest.raises(L.Refused) as refused:
        L.record_reversal(conn, firing_id, actor="tst-approver", reason=reason, now=NOW)
    assert conn.autocommit is True, "the connection is handed back as it came"
    return refused.value.code


# ------------------------------------------------------------------ record_reversal
def test_an_approver_marks_an_approved_firing_reversed(world):
    conn, t = world
    [fid] = fire(conn, t.key, [600])
    out = L.record_reversal(conn, fid, actor="tst-approver", reason=f"  {WHY}  ", now=NOW)
    assert out == {"firingId": fid, "reversedBy": "tst-approver", "reversedAt": NOW.isoformat(), "reason": WHY}
    assert _stored(conn, [fid]) == [(fid, "tst-approver", NOW, WHY)]
    [decided] = L.decisions(conn.cursor(), t.key, [1])
    assert decided.reversed is True, "the scan reads it"


@pytest.mark.parametrize("outcome,result", [("approve", "rejected"), ("approve", "paused_for_approval"),
                                            ("approve", "timed_out"), ("notify", "allowed"),
                                            ("notify", "approved"), ("block", "blocked")])
def test_a_rejected_or_paused_or_notify_firing_cannot_be_reversed(world, outcome, result):
    conn, t = world
    fid = _firing(conn, t.key, outcome, result)
    with pytest.raises(L.Refused) as refused:
        L.record_reversal(conn, fid, actor="tst-approver", reason=WHY, now=NOW)
    assert (refused.value.code, refused.value.message) == \
        ("not_approved", "Only an approved request can be marked as reversed.")
    assert _stored(conn, [fid]) == []


def test_a_second_reversal_of_the_same_decision_is_refused_even_from_a_sibling_row(world):
    conn, t = world
    [case] = _case_ids(1)
    first, second, third = fire(conn, t.key, [600, 600, 600], decision_ids=[case] * 3)
    out = L.record_reversal(conn, second, actor="tst-approver", reason=WHY, now=NOW)
    assert out["firingId"] == first, "stored on the case's earliest row, whichever linked row was clicked"
    for clicked in (first, second, third):
        assert _refusal(conn, clicked, "TST again") == "already_reversed"
    assert _stored(conn, [first, second, third]) == [(first, "tst-approver", NOW, WHY)]


def test_a_reversal_already_stored_on_a_later_row_of_the_case_still_counts_as_the_one(world):
    conn, t = world
    [case] = _case_ids(1)
    first, second = fire(conn, t.key, [600, 600], decision_ids=[case] * 2)
    _reverse(conn, second)                       # written straight to the table, on the later row
    assert _refusal(conn, first) == "already_reversed"
    assert [r[0] for r in _stored(conn, [first, second])] == [second]


def test_two_unlinked_firings_are_two_decisions(world):
    conn, t = world
    a, b = fire(conn, t.key, [600, 610])
    assert L.record_reversal(conn, a, actor="tst-approver", reason=WHY, now=NOW)["firingId"] == a
    assert L.record_reversal(conn, b, actor="tst-approver", reason=WHY, now=NOW)["firingId"] == b


@pytest.mark.parametrize("reason", ["", "   ", "\n\t", None])
def test_a_blank_reason_is_refused(world, reason):
    conn, t = world
    [fid] = fire(conn, t.key, [600])
    assert _refusal(conn, fid, reason) == "reason_required"
    assert _stored(conn, [fid]) == []


def test_an_unknown_firing_is_not_found(world):
    conn, _t = world
    assert _refusal(conn, 2 ** 62) == "not_found"


def test_the_policy_activity_list_shows_the_reversal(world):
    conn, t = world
    [case] = _case_ids(1)
    first, second, other = fire(conn, t.key, [600, 600, 610], decision_ids=[case, case, None])
    before = V.policy_firings(conn, t.key, 50)
    assert [f["reversed"] for f in before] == [None, None, None]
    L.record_reversal(conn, second, actor="tst-approver", reason=WHY, now=NOW)
    after = V.policy_firings(conn, t.key, 50)
    shown = {f["id"]: f["reversed"] for f in after}
    mark = {"by": "tst-approver", "at": NOW.isoformat(), "reason": WHY}
    assert shown == {first: mark, second: mark, other: None}, "the row and its linked sibling; not the other request"
    # the health line reads the same rows: nothing else about them changes, and there are no more of them
    assert [{k: v for k, v in f.items() if k != "reversed"} for f in after] == \
        [{k: v for k, v in f.items() if k != "reversed"} for f in before]
    assert len(V.policy_firings(conn, t.key, 2)) == 2


# ------------------------------------------------------------------ through the whole app
@pytest.fixture
def api(monkeypatch):
    from api.main import app
    audits = []
    monkeypatch.setenv("AGENT_POLICY_GATEWAY_KEY", "k1")
    monkeypatch.setattr(R.agent_actions, "record_action_or_fail", lambda **kw: audits.append(kw))
    monkeypatch.setattr(R, "_role_of", lambda p: E.ROLES.get(p.subject, "Nobody"))
    return SimpleNamespace(client=TestClient(app), audits=audits)


def _post(api, firing_id, reason=WHY, headers=E.APPROVER):
    return api.client.post(f"/agent-policies/firings/{firing_id}/reversal", json={"reason": reason}, headers=headers)


def test_the_endpoint_needs_the_approver_role(api, world):
    conn, t = world
    [fid] = fire(conn, t.key, [600])
    for low in (E.VIEWER, E.BUYER, E._hdr("tst-nobody")):
        assert _post(api, fid, headers=low).status_code == 403
    assert _stored(conn, [fid]) == [], "a refused caller records nothing"
    assert [(a["action_type"], a["phase"], a["status"]) for a in api.audits] == \
        [("agent_policy.reversal", "authorize", "denied")] * 3
    del api.audits[:]
    r = _post(api, fid)
    assert r.status_code == 201, r.text
    [(stored_id, by, at, reason)] = _stored(conn, [fid])
    assert (stored_id, by, reason) == (fid, "tst-approver", WHY)
    # the answer after the output filter (this POST is not exempt): a plain reason arrives readable
    assert r.json() == {"firingId": fid, "reversedBy": "tst-approver", "reversedAt": at.isoformat(), "reason": WHY}
    assert [(a["action_type"], a["phase"], a["status"]) for a in api.audits] == [
        ("agent_policy.reversal", "authorize", "allowed"), ("agent_policy.reversal", "act", "done")]
    assert api.audits[0]["details"]["firing"] == fid and api.audits[0]["details"]["minimum"] == "Approver"


def test_the_endpoints_refusals_arrive_readable(api, world):
    conn, t = world
    [fid] = fire(conn, t.key, [600])
    rejected = _firing(conn, t.key, "approve", "rejected")
    r = _post(api, rejected)
    assert (r.status_code, r.json()) == (409, {"code": "not_approved",
                                               "message": "Only an approved request can be marked as reversed."})
    assert _post(api, 2 ** 62).status_code == 404
    assert _post(api, fid, reason="   ").json()["code"] == "reason_required"
    assert _post(api, fid, reason="").status_code == 422 and _post(api, fid, reason="x" * 2001).status_code == 422
    assert api.client.post(f"/agent-policies/firings/{fid}/reversal", json={}, headers=E.APPROVER).status_code == 422
    assert _post(api, 0).status_code == 422 and _post(api, 10 ** 30).status_code == 422
    assert _post(api, fid, reason="x" * 2000).status_code == 201
    again = _post(api, fid)
    assert again.status_code == 409 and again.json()["code"] == "already_reversed"
    assert [(a["phase"], a["status"], a["details"].get("code")) for a in api.audits[-2:]] == [
        ("authorize", "allowed", None), ("act", "refused", "already_reversed")]


def test_a_reason_the_output_filter_withholds_is_still_stored_whole(api, world):
    conn, t = world
    [fid] = fire(conn, t.key, [600])
    dated = "TST clawed back on 01/04/2026, see deal DL/2024/001"
    r = _post(api, fid, reason=dated)
    assert r.status_code == 201, r.text
    # the 201 body is scrubbed like any non-exempt answer: the ids survive, this reason is replaced
    # by the filter's own sentence. The screens re-read the activity list (an exempt read) instead.
    assert (r.json()["firingId"], r.json()["reversedBy"]) == (fid, "tst-approver")
    assert r.json()["reason"] == "I couldn't retrieve that. I've raised it with the team."
    assert "01/04/2026" not in r.text
    assert _stored(conn, [fid])[0][3] == dated


@pytest.fixture
def policy(api, monkeypatch):
    """A live approve policy "amount over 500" with the base evidence and its one open suggestion
    (as test_learning_endpoints_live builds it). `ids` are the 33 firings, in BASE order."""
    assert os.getenv("DB_NAME") == "bp_testdb", "these tests run on bp_testdb only"
    from services.db import get_conn
    tag = uuid.uuid4().hex[:8]
    w = SimpleNamespace(tag=tag, key=None, live=2)
    monkeypatch.setattr(repo, "_activation_problems", lambda f, r, s: [])
    monkeypatch.setattr(repo, "_contract_problems", lambda d, r: [])
    form = E._form(tag)
    with get_conn() as conn:
        w.conn = conn
        try:
            w.key = repo.create_draft(conn, form, actor="test")["policyKey"]
            repo.save_version(conn, w.key, form, base_version=1, intent="activate", actor="test", change_note="go")
            w.ids = fire(conn, w.key, BASE, version=w.live)
            assert E._scan(conn, w)["raised"] == 1
            yield w
        finally:
            if w.key:
                conn.cursor().execute("DELETE FROM proc.bp_agent_policy_suggestion WHERE policy_key = %s", (w.key,))
                got = repo.get_policy(conn, w.key)
                if got["status"] != "retired":
                    repo.retire(conn, w.key, base_version=got["latestVersion"], actor="test",
                                change_note="test teardown")


def _suggestion(conn, key):
    cur = conn.cursor()
    cur.execute("SELECT status, confident, proposal, stats FROM proc.bp_agent_policy_suggestion "
                "WHERE policy_key = %s", (key,))
    return cur.fetchall()


def test_a_reversal_reaches_the_next_scan(api, policy):
    conn = policy.conn
    [(status, confident, proposal, stats)] = _suggestion(conn, policy.key)
    assert (status, confident, proposal["to"], stats["reversed"]) == ("open", True, 800, 0)
    at_600 = policy.ids[BASE.index(600)]                     # inside band 1 (510..800)
    assert _post(api, at_600).status_code == 201
    counts = E._scan(conn, policy)
    assert (counts["weakened"], counts["updated"], counts["raised"]) == (1, 0, 0)
    [(status, confident, proposal, stats)] = _suggestion(conn, policy.key)
    assert (status, confident, proposal["to"]) == ("open", False, 800)
    assert (stats["reversed"], stats["decisions"], stats["approved"]) == (1, 30, 30)
    [card] = [c for c in api.client.get("/agent-policies/suggestions", headers=E.VIEWER).json()["suggestions"]
              if c["policyKey"] == policy.key]
    assert (card["suggestionConfidence"], card["canDraft"]) == ("Not enough yet", False)


def test_the_activity_list_endpoint_shows_who_reversed_it_when_and_why(api, policy):
    at_600 = policy.ids[BASE.index(600)]
    assert _post(api, at_600).status_code == 201
    [(_id, by, at, reason)] = _stored(policy.conn, [at_600])
    r = api.client.get(f"/agent-policies/{policy.key}/firings?limit=50", headers=E.VIEWER)
    assert r.status_code == 200, r.text
    rows = r.json()["firings"]
    assert len(rows) == 33
    assert {f["id"]: f["reversed"] for f in rows if f["reversed"]} == \
        {at_600: {"by": "tst-approver", "at": at.isoformat(), "reason": WHY}}
    # the approval detail's firing rows are a different read: they do not claim to know
    assert "reversed" not in V._firing_view({**_raw(policy.conn, at_600)}, set())


def _raw(conn, firing_id):
    cur = conn.cursor()
    cur.execute(f"SELECT {V._FIRING_COLS} FROM proc.bp_policy_firing WHERE firing_id = %s", (firing_id,))
    return V._rows(cur)[0]
