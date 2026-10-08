"""Approval, notification, decider and firing endpoints through the WHOLE app against bp_testdb.

Needs PROCWISE_TEST_LIVE_DB=1 and DB_NAME=bp_testdb. Runs through `api.main.app`, so the
output-safety middleware is in the path (these routes are NOT on the screen exemption list).
Decider-map rows, decisions and notifications made here are removed afterwards;
proc.bp_policy_firing is append-only by design, so its 'TST-<hex>' rows remain (see
test_approvals_live.py).
"""
import copy
import json
import os
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from api.routers import agent_policies as R
from services.agent_policy import approval_views as V
from services.agent_policy import approvals as A
from services.agent_policy.compiler import compile_policy
from services.agent_policy.enforcement import MASK
from tests.agent_policy.fixtures import FORM_EXAMPLE, SETTINGS

pytestmark = pytest.mark.skipif(os.getenv("PROCWISE_TEST_LIVE_DB") != "1", reason="live DB required")

NOW = datetime(2026, 10, 8, 9, 0, tzinfo=timezone.utc)
IBAN = "GB29NWBK60161331926819"


@pytest.fixture
def conn():
    assert os.getenv("DB_NAME") == "bp_testdb", "these live tests run on bp_testdb only"
    from services.db import get_conn
    with get_conn() as c:
        yield c


@pytest.fixture
def world(conn, monkeypatch):
    tag = str(uuid.uuid4().int)[:9]   # digits: a policy id is <3 capitals>-<4+ digits>
    w = SimpleNamespace(tag=tag, key=f"TST-{tag}", l1=f"TST L1 {tag}", l2=f"TST L2 {tag}",
                        l1_email=f"l1-{tag}@example.test", l2_group=f"TSTGROUP{tag}",  # no underscores: see test_approval_endpoints_scrub.py
                        requester=f"req-{tag}@example.test", made_names=[])
    with conn.cursor() as cur:
        cur.execute("INSERT INTO proc.bp_policy_decider_map (decider_name, groups, emails, last_modified_by) "
                    "VALUES (%s, '{}', %s, 'test'), (%s, %s, '{}', 'test')",
                    (w.l1, [w.l1_email], w.l2, [w.l2_group]))
    form = copy.deepcopy(FORM_EXAMPLE)
    form["checked"] = {"by": "user_8841", "at": "2026-10-08T09:14:00Z"}
    form["deciders"] = [w.l1, w.l2]
    form["hidden"]["inputs"].append({"name": "Bank account", "field": "args.iban", "type": "string",
                                     "from": "action", "showApprover": True, "sensitive": True})
    w.doc = compile_policy(form, policy_key=w.key, version=3, status="live", settings=SETTINGS,
                           never_suggest=False)
    real = V._compiled
    monkeypatch.setattr(V, "_compiled", lambda cur, pairs: {**real(cur, pairs), (w.key, 3): w.doc})
    audits = []
    monkeypatch.setattr(R.agent_actions, "record_action_or_fail", lambda **kw: audits.append(kw))
    w.audits = audits
    replays = []
    monkeypatch.setattr(R, "_replay_later", lambda did: replays.append(did))
    w.replays = replays
    yield w
    with conn.cursor() as cur:
        cur.execute("SELECT decision_id FROM proc.bp_decision WHERE subject_type = %s AND subject_id LIKE %s",
                    (A.SUBJECT_TYPE, f"{w.key}:%"))
        ids = [r[0] for r in cur.fetchall()]
        cur.execute("DELETE FROM proc.bp_policy_notification WHERE link = ANY(%s) OR recipient = ANY(%s)",
                    ([f"decision:{i}" for i in ids], [w.l1, w.l2, w.requester]))
        cur.execute("DELETE FROM proc.bp_decision WHERE decision_id = ANY(%s)", (ids,))
        cur.execute("DELETE FROM proc.bp_decision WHERE subject_type = %s AND subject_id LIKE %s",
                    (V.REPLAY_SUBJECT_TYPE, f"{w.key}:%"))
        cur.execute("DELETE FROM proc.bp_policy_decider_map WHERE decider_name = ANY(%s)",
                    ([w.l1, w.l2, *w.made_names],))


@pytest.fixture
def client(monkeypatch):
    from api.main import app
    monkeypatch.setenv("AGENT_POLICY_GATEWAY_KEY", "k1")
    return TestClient(app)


def _hdr(sub, email="", groups=()):
    return {"X-Gateway-Key": "k1", "X-User-Sub": sub, "X-User-Email": email,
            "X-User-Groups": json.dumps(list(groups))}


def _open(conn, w):
    with conn.cursor() as cur:
        cur.execute("INSERT INTO proc.bp_policy_firing (policy_key, policy_version, checkpoint, action_name, "
                    "agent, requested_by, outcome, result, matched_values) VALUES (%s, 3, 'tool.call.before', "
                    "'refund.issue', 'agent_nick', %s, 'approve', 'paused_for_approval', %s) RETURNING firing_id",
                    (w.key, w.requester, json.dumps({"args.amount": 900, "args.iban": IBAN})))
        fid = cur.fetchone()[0]
    did = A.open_case(conn, policy_doc=w.doc, firing_id=fid,
                      action={"tool": "refund.issue", "args": {"amount": 900, "iban": IBAN},
                              "agent": "agent_nick", "workflowId": "wf-1", "userId": w.requester,
                              "reason": "The customer was charged twice"},
                      requested_by=w.requester, now=NOW)
    with conn.cursor() as cur:
        cur.execute("INSERT INTO proc.bp_policy_notification (firing_id, recipient, message, link) "
                    "VALUES (%s, %s, %s, %s) RETURNING notification_id",
                    (fid, w.l1, "Issuing a refund or credit needs your decision.", f"decision:{did}"))
        nid = cur.fetchone()[0]
    return did, fid, nid


def _withheld(body) -> bool:
    return "[withheld]" in json.dumps(body) or "I can't" in json.dumps(body)


def test_l1_sees_unmasked_case_and_others_see_it_masked(client, conn, world):
    did, fid, _ = _open(conn, world)
    l1 = _hdr("u-l1", world.l1_email)
    r = client.get("/agent-policies/approvals", headers=l1)
    assert r.status_code == 200, r.text
    mine = [c for c in r.json()["approvals"] if c["id"] == did]
    assert len(mine) == 1 and mine[0]["canDecide"] is True
    vals = {i["field"]: i["value"] for i in mine[0]["inputs"]}
    assert vals["args.iban"] == IBAN and vals["args.amount"] == 900
    assert mine[0]["levelName"] == world.l1 and mine[0]["requestedBy"] == world.requester

    # L2 (group) is linked but not at the current level: not listed, masked on direct read
    l2 = _hdr("u-l2", "", [world.l2_group])
    assert did not in [c["id"] for c in client.get("/agent-policies/approvals", headers=l2).json()["approvals"]]
    one = client.get(f"/agent-policies/approvals/{did}", headers=l2).json()
    assert {i["field"]: i["value"] for i in one["inputs"]}["args.iban"] == MASK and one["canDecide"] is False
    assert [f["matchedValues"]["args.iban"] for f in one["history"]["firings"]] == [MASK]

    # An Admin reads every case, masked, and cannot decide
    admin = _hdr("u-admin", "admin@example.test", ["PROCWISE_ADMIN"])
    listed = [c for c in client.get("/agent-policies/approvals", headers=admin).json()["approvals"] if c["id"] == did]
    assert listed and {i["field"]: i["value"] for i in listed[0]["inputs"]}["args.iban"] == MASK
    assert listed[0]["canDecide"] is False
    r = client.post(f"/agent-policies/approvals/{did}/decide", json={"verb": "approve"}, headers=admin)
    assert r.status_code == 403

    # A stranger cannot read it at all
    assert client.get(f"/agent-policies/approvals/{did}", headers=_hdr("u-x", "x@example.test")).status_code == 404


def test_decide_rules(client, conn, world):
    did, fid, _ = _open(conn, world)
    l1 = _hdr("u-l1", world.l1_email)
    r = client.post(f"/agent-policies/approvals/{did}/decide", json={"verb": "reject", "reason": "  "}, headers=l1)
    assert r.status_code == 422
    assert r.json()["problems"][0]["field"] == "reason" and r.json()["problems"][0]["code"] == "reason_required"

    # the requester is linked too, and still barred
    with conn.cursor() as cur:
        cur.execute("UPDATE proc.bp_policy_decider_map SET emails = %s WHERE decider_name = %s",
                    ([world.l1_email, world.requester], world.l1))
    r = client.post(f"/agent-policies/approvals/{did}/decide", json={"verb": "approve"},
                    headers=_hdr("u-req", world.requester))
    assert r.status_code == 403 and "your own request" in r.json()["detail"]

    r = client.post(f"/agent-policies/approvals/{did}/decide",
                    json={"verb": "reject", "reason": "Not a duplicate charge"}, headers=l1)
    assert r.status_code == 200 and r.json()["result"] == "rejected"
    assert [a["status"] for a in world.audits if a["action_type"] == "agent_policy.decide"][-1] == "done"
    assert world.replays == []

    r = client.post(f"/agent-policies/approvals/{did}/decide", json={"verb": "approve"}, headers=l1)
    assert r.status_code == 409

    hist = client.get(f"/agent-policies/approvals/{did}", headers=l1).json()["history"]
    assert [(d["verb"], d["reason"]) for d in hist["decisions"]] == [("reject", "Not a duplicate charge")]
    assert hist["firings"][0]["result"] == "rejected"


def test_approve_queues_the_replay(client, conn, world):
    did, _, _ = _open(conn, world)
    r = client.post(f"/agent-policies/approvals/{did}/decide", json={"verb": "approve"},
                    headers=_hdr("u-l1", world.l1_email))
    assert r.status_code == 200 and r.json()["result"] == "approved"
    assert world.replays == [did]


def _replay_row(conn, did, subject_id, facts):
    """A replay row as replay._insert_replay/_finish_replay write it."""
    with conn.cursor() as cur:
        cur.execute("INSERT INTO proc.bp_decision (subject_type, subject_id, decision, resolution, facts, evidence, "
                    "status, actioned_by, actioned_at, created_by) VALUES (%s, %s, 'replay', 'resolved', %s, '[]', "
                    "'actioned', 'system:replay', now(), 'system:replay')",
                    (V.REPLAY_SUBJECT_TYPE, subject_id, json.dumps({"caseIds": [did], "tool": "refund.issue", **facts})))


def test_case_history_carries_the_latest_replay(client, conn, world):
    did, fid, _ = _open(conn, world)
    other, ofid, _ = _open(conn, world)
    l1 = _hdr("u-l1", world.l1_email)
    assert client.get(f"/agent-policies/approvals/{did}", headers=l1).json()["history"]["replay"] is None
    _replay_row(conn, other, f"{world.key}:{ofid}", {"outcome": "ran", "resultSummary": "other", "error": None})
    assert client.get(f"/agent-policies/approvals/{did}", headers=l1).json()["history"]["replay"] is None
    _replay_row(conn, did, f"{world.key}:{fid}", {"ok": None, "outcome": "running", "resultSummary": None, "error": None})
    _replay_row(conn, did, f"{world.key}:{fid}", {"ok": True, "outcome": "ran",
                                                  "resultSummary": f"Refund of 900 to {MASK} issued", "error": None})
    r = client.get(f"/agent-policies/approvals/{did}", headers=l1).json()["history"]["replay"]
    assert r["outcome"] == "ran" and r["resultSummary"] == f"Refund of 900 to {MASK} issued"
    assert r["error"] is None and r["at"]
    assert set(r) == {"outcome", "resultSummary", "error", "at"}


def test_notifications_mine_and_read(client, conn, world):
    did, _, nid = _open(conn, world)
    l1 = _hdr("u-l1", world.l1_email)
    got = client.get("/agent-policies/notifications?mine=1", headers=l1).json()["notifications"]
    mine = [n for n in got if n["id"] == nid]
    assert mine and mine[0]["read"] is False and mine[0]["decisionId"] == did and mine[0]["policyKey"] == world.key
    other = client.get("/agent-policies/notifications?mine=1", headers=_hdr("u-x", "x@example.test")).json()
    assert nid not in [n["id"] for n in other["notifications"]]
    assert client.post(f"/agent-policies/notifications/{nid}/read", headers=_hdr("u-x", "x@example.test")).status_code == 404
    for _ in range(2):   # idempotent
        assert client.post(f"/agent-policies/notifications/{nid}/read", headers=l1).json() == {"id": nid, "read": True}
    with conn.cursor() as cur:
        cur.execute("SELECT read_by FROM proc.bp_policy_notification WHERE notification_id = %s", (nid,))
        assert cur.fetchone()[0] == ["u-l1"]
    got = client.get("/agent-policies/notifications?mine=1", headers=l1).json()["notifications"]
    assert [n["read"] for n in got if n["id"] == nid] == [True]
    assert any(a["action_type"] == "agent_policy.notification_read" for a in world.audits)


def test_deciders_put_is_admin_only_and_validated(client, conn, world):
    name = f"TST Ops {world.tag}"
    world.made_names.append(name)
    admin = _hdr("u-admin", "admin@example.test", ["PROCWISE_ADMIN"])
    viewer = _hdr("u-v", "v@example.test", ["PROCWISE_VIEWER"])
    assert client.put(f"/agent-policies/deciders/{name}", json={"groups": ["G"]}, headers=viewer).status_code == 403
    r = client.put(f"/agent-policies/deciders/{name}", json={"groups": [], "emails": []}, headers=admin)
    assert r.status_code == 422 and r.json()["problems"][0]["code"] == "someone_required"
    r = client.put(f"/agent-policies/deciders/{name}", json={"emails": ["not-an-email"]}, headers=admin)
    assert r.status_code == 422 and r.json()["problems"][0]["field"] == "emails"
    r = client.put(f"/agent-policies/deciders/{name}", json={"groups": [" "]}, headers=admin)
    assert r.status_code == 422 and r.json()["problems"][0]["field"] == "groups"
    r = client.put(f"/agent-policies/deciders/{name}",
                   json={"groups": ["PROCWISE_FINANCE"], "emails": ["Ops@Example.TEST"], "notes": "Ops on call"},
                   headers=admin)
    assert r.status_code == 200, r.text
    assert r.json()["emails"] == ["ops@example.test"] and r.json()["lastModifiedBy"] == "u-admin"
    listed = {d["name"]: d for d in client.get("/agent-policies/deciders", headers=viewer).json()["deciders"]}
    assert listed[name]["groups"] == ["PROCWISE_FINANCE"]
    assert any(a["action_type"] == "agent_policy.admin" and a["status"] == "allowed" for a in world.audits)


def test_firings_newest_first_masked_and_limited(client, conn, world):
    _open(conn, world)
    _open(conn, world)
    hdr = _hdr("u-v", "v@example.test", ["PROCWISE_VIEWER"])
    r = client.get(f"/agent-policies/{world.key}/firings?limit=50", headers=hdr)
    assert r.status_code == 200, r.text
    rows = r.json()["firings"]
    assert len(rows) == 2 and rows[0]["id"] > rows[1]["id"]
    assert all(f["matchedValues"]["args.iban"] == MASK and f["matchedValues"]["args.amount"] == 900 for f in rows)
    assert client.get(f"/agent-policies/{world.key}/firings?limit=201", headers=hdr).status_code == 422
    assert len(client.get(f"/agent-policies/{world.key}/firings?limit=1", headers=hdr).json()["firings"]) == 1


def test_nothing_in_these_answers_is_withheld_by_output_safety(client, conn, world):
    """Realistic answers through the scrubber. If this fails, the screens would show [withheld]:
    report for a user ruling -- never add an exemption from here."""
    did, _, nid = _open(conn, world)
    l1 = _hdr("u-l1", world.l1_email)
    admin = _hdr("u-admin", "admin@example.test", ["PROCWISE_ADMIN"])
    answers = {
        "list": client.get("/agent-policies/approvals", headers=l1).json(),
        "one": client.get(f"/agent-policies/approvals/{did}", headers=l1).json(),
        "notifications": client.get("/agent-policies/notifications?mine=1", headers=l1).json(),
        "deciders": client.get("/agent-policies/deciders", headers=admin).json(),
        "firings": client.get(f"/agent-policies/{world.key}/firings", headers=admin).json(),
        "read": client.post(f"/agent-policies/notifications/{nid}/read", headers=l1).json(),
        "one_after_replay": (_replay_row(conn, did, f"{world.key}:x", {"outcome": "error", "resultSummary": None,
                                                                     "error": "The supplier service did not answer."})
                             or client.get(f"/agent-policies/approvals/{did}", headers=l1).json()),
        "decide": client.post(f"/agent-policies/approvals/{did}/decide",
                              json={"verb": "reject", "reason": "Not a duplicate charge"}, headers=l1).json(),
    }
    bad = {k: v for k, v in answers.items() if _withheld(v)}
    assert not bad, json.dumps(bad, default=str)[:3000]
