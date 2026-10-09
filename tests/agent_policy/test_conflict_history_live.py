"""One reader for every screen: complete, masked per reader (design §3.1). Needs PROCWISE_TEST_LIVE_DB=1."""
import json
import os
from types import SimpleNamespace

import pytest

from services.agent_policy import conflict_cases as CC
from services.agent_policy import conflict_history as CH
from services.agent_policy.enforcement import MASK
from tests.agent_policy import test_conflict_live_gate as LG
from tests.agent_policy.test_conflict_endpoints_live import (  # noqa: F401 - fixtures
    ADMIN, NOW, STRANGER, _a, _as, _b, _c, _design_case, _gate, client, conn, world)

pytestmark = pytest.mark.skipif(os.getenv("PROCWISE_TEST_LIVE_DB") != "1", reason="live DB required")
REASON = "Paying the 900 refund is fine."


def _viewer(conn, w, name=None, *, admin=False):
    p = LG.who(w, name) if name else SimpleNamespace(subject="tst-stranger", email="stranger@example.test",
                                                     claims={"cognito:groups": []})
    return CH.viewer(conn, p, is_admin=admin)


def settled_live(conn, w, monkeypatch, reason=REASON):
    """A/B (approve, different sources and deciders, args.amount sensitive) paused on 900 and
    approved by both last-level deciders, B's with `reason`. Returns (a, b, live decision id)."""
    (a, da), (b, db) = _a(conn, w), _b(conn, w)
    _gate(monkeypatch, w, [da, db])
    [lv] = LG.lives(conn, w)
    ma, mb = LG.members(conn, w)
    LG.act(conn, w, ma["decision_id"], w.la2)
    LG.act(conn, w, mb["decision_id"], w.lb, reason=reason)
    return a, b, lv["decision_id"]


def _entry(conn, key, case, v):
    with conn.cursor() as cur:
        [e] = [e for e in CH.read(cur, policy_key=key, viewer=v) if e["caseId"] == f"pc_{case}"]
    return e


def test_owner_decider_and_admin_read_the_full_reason_a_stranger_reads_it_masked(conn, world, monkeypatch):
    a, b, live_id = settled_live(conn, world, monkeypatch)
    for v in (_viewer(conn, world, world.oa), _viewer(conn, world, world.lb), _viewer(conn, world, admin=True)):
        e = _entry(conn, a, live_id, v)
        assert e["decision"]["reason"] == REASON and e["example"]["args.amount"] == 900
    e = _entry(conn, a, live_id, _viewer(conn, world))
    assert e["decision"]["reason"] == f"Paying the {MASK} refund is fine."
    assert e["example"]["args.amount"] == MASK
    assert e["decision"]["decidedBy"] == {"kind": "person", "name": f"sub-{world.people[world.lb]}"}
    assert (e["kind"], e["isOpen"], e["decision"]["option"]) == ("live", False, "approve")
    assert e["policies"] == [{"id": k, "version": 1} for k in sorted([a, b])]
    assert not any(k.startswith("_") for k in e)


def test_the_policy_page_and_the_conflicts_detail_return_identical_entries(client, conn, world):
    a, c, did = _design_case(conn, world)
    CC.decide_policy(conn, did, principal=LG.who(world, world.oa), option=f"change:{a}", reason="Narrow it",
                     limit_text=None, now=NOW)
    for hdr in (_as(world, world.oa), STRANGER):
        page = client.get(f"/agent-policies/{a}", headers=hdr).json()["conflicts"]
        detail = client.get(f"/agent-policies/conflicts/{did}", headers=hdr).json()
        assert detail["pairKey"] == "|".join(sorted([a, c]))
        ids = {h["caseId"] for h in detail["conflictHistory"]}
        mine = [{k: v for k, v in e.items() if k != "otherPolicies"} for e in page if e["caseId"] in ids]
        assert mine and mine == detail["conflictHistory"]
        [one] = [e for e in page if e["caseId"] == f"pc_{did}"]
        assert one["otherPolicies"] == [c]
        assert one["decision"]["decidedBy"] == {"kind": "person", "name": f"sub-{world.email_a}"}


def test_the_reader_reports_the_kind_each_path_stored(conn, world):
    a, c, did = _design_case(conn, world)
    CC.close_moot(conn, c, now=NOW)
    with conn.cursor() as cur:
        [e] = [e for e in CH.raw(cur, policy_key=a) if e["caseId"] == f"pc_{did}"]
    assert e["decision"]["option"] == "moot"
    assert e["decision"]["decidedBy"] == {"kind": "retired", "name": "system:retired"}


def test_a_case_closed_before_the_reader_existed_reads_with_no_kind(conn, world):
    a, c, did = _design_case(conn, world)
    with conn.cursor() as cur:
        cur.execute("UPDATE proc.bp_agent_policy_conflict SET is_open = false, outcome = 'moot', "
                    "decided_by = 'legacy', decided_at = now() WHERE decision_id = %s", (did,))
        [e] = [e for e in CH.raw(cur, policy_key=a) if e["caseId"] == f"pc_{did}"]
    assert e["decision"]["option"] == "moot" and e["decision"]["decidedBy"] == {"kind": None, "name": "legacy"}


def test_history_is_newest_first_and_capped(conn, world, monkeypatch):
    a, c, did = _design_case(conn, world)
    _a2, b, live_id = settled_live(conn, world, monkeypatch)   # a second A/B pair; A is a new key
    with conn.cursor() as cur:
        both = CH.raw(cur, pair_key="|".join(sorted([_a2, b])))
        assert [e["caseId"] for e in both] == [f"pc_{live_id}"]
        one = CH.raw(cur, policy_key=c, limit=1)
    assert [e["caseId"] for e in one] == [f"pc_{did}"]


def test_the_conflicts_screen_masks_a_policy_case_reason_everywhere_for_a_stranger(client, conn, world):
    """Fix round 1: the case's own `decision` and `history` follow the reader's viewer rule (§3.1),
    so the three places a reason appears agree for every viewer."""
    a, c, did = _design_case(conn, world)
    reason = "Narrow it below 10001 please"
    CC.decide_policy(conn, did, principal=LG.who(world, world.oa), option=f"change:{a}", reason=reason,
                     limit_text=None, now=NOW)
    masked = f"Narrow it below {MASK} please"
    detail = client.get(f"/agent-policies/conflicts/{did}", headers=STRANGER).json()
    assert "10001" not in json.dumps(detail)
    assert detail["decision"]["reason"] == masked and [h["reason"] for h in detail["history"]] == [masked]
    [mine] = [h for h in detail["conflictHistory"] if h["caseId"] == f"pc_{did}"]
    assert mine["decision"]["reason"] == masked
    listed = client.get("/agent-policies/conflicts?status=all", headers=STRANGER).json()["conflicts"]
    [row] = [v for v in listed if v["decisionId"] == did]
    assert "10001" not in json.dumps(row) and row["decision"]["reason"] == masked
    for hdr in (_as(world, world.oa), ADMIN):
        got = client.get(f"/agent-policies/conflicts/{did}", headers=hdr).json()
        [mine] = [h for h in got["conflictHistory"] if h["caseId"] == f"pc_{did}"]
        assert got["decision"]["reason"] == reason and [h["reason"] for h in got["history"]] == [reason]
        assert mine["decision"]["reason"] == reason
        [row] = [v for v in client.get("/agent-policies/conflicts?status=all", headers=hdr).json()["conflicts"]
                 if v["decisionId"] == did]
        assert row["decision"]["reason"] == reason
