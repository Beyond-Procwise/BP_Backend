"""C1: the generic /decisions reads never serve agent-policy approval or replay rows.

Their facts hold the tool call's raw arguments, context and reason; only /agent-policies/approvals
reads them, masked per caller. Needs PROCWISE_TEST_LIVE_DB=1 and DB_NAME=bp_testdb. Every row made
here is removed afterwards (firing rows are append-only BY DESIGN and stay, key TST-<digits>).
"""
import json
import os
import random
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routers import decisions as decisions_router
from services.agent_policy import approvals as A
from services.agent_policy.compiler import compile_policy
from tests.agent_policy.fixtures import FORM_EXAMPLE, SETTINGS

live = pytest.mark.skipif(os.getenv("PROCWISE_TEST_LIVE_DB") != "1", reason="live DB required")

SECRET = "GB29NWBK60161331926819"


@pytest.fixture
def conn():
    assert os.getenv("DB_NAME") == "bp_testdb", "these live tests run on bp_testdb only"
    from services.db import get_conn
    with get_conn() as c:
        yield c


@pytest.fixture
def client():
    from services.db import get_conn
    app = FastAPI()
    app.include_router(decisions_router.router)
    app.dependency_overrides[decisions_router.require_user] = lambda: SimpleNamespace(subject="someone-signed-in")
    app.state.agent_nick = SimpleNamespace(get_db_connection=get_conn,
                                           policy_engine=SimpleNamespace(get_policy=lambda s: None))
    return TestClient(app)


@pytest.fixture
def seeded(conn):
    key = f"TST-{random.randint(10**7, 10**8 - 1)}"
    form = json.loads(json.dumps(FORM_EXAMPLE))
    form["checked"] = {"by": "user_8841", "at": "2026-10-08T09:14:00Z"}
    form["deciders"] = ["TST nobody"]
    doc = compile_policy(form, policy_key=key, version=1, status="live", settings=SETTINGS, never_suggest=False)
    with conn.cursor() as cur:
        cur.execute("INSERT INTO proc.bp_policy_firing (policy_key, policy_version, checkpoint, action_name, "
                    "outcome, result) VALUES (%s, 1, 'tool.call.before', 'refund.issue', 'approve', "
                    "'paused_for_approval') RETURNING firing_id", (key,))
        fid = cur.fetchone()[0]
    case_id = A.open_case(conn, policy_doc=doc, firing_id=fid,
                          action={"tool": "refund.issue", "args": {"amount": 900, "iban": SECRET},
                                  "agent": "agent_nick", "workflowId": None, "userId": "u-1",
                                  "reason": f"pay {SECRET}"},
                          requested_by="req@example.test", now=datetime.now(timezone.utc))
    with conn.cursor() as cur:
        cur.execute("INSERT INTO proc.bp_decision (subject_type, subject_id, decision, resolution, rationale, "
                    "facts, evidence, status, created_by) VALUES ('agent_policy_replay', %s, 'replay', "
                    "'escalated', 'x', %s, '[]', 'open', 'test') RETURNING decision_id",
                    (f"{key}:{fid}", json.dumps({"args": {"iban": SECRET}})))
        replay_id = cur.fetchone()[0]
        # a control row of an ordinary subject type: still served exactly as before
        cur.execute("INSERT INTO proc.bp_decision (subject_type, subject_id, decision, resolution, rationale, "
                    "facts, evidence, status, created_by) VALUES ('tst_ordinary', %s, 'review', "
                    "'escalated', 'x', '{}', '[]', 'open', 'test') RETURNING decision_id", (key,))
        control_id = cur.fetchone()[0]
    yield SimpleNamespace(case_id=case_id, replay_id=replay_id, control_id=control_id, key=key)
    with conn.cursor() as cur:
        cur.execute("DELETE FROM proc.bp_policy_notification WHERE link = %s", (f"decision:{case_id}",))
        cur.execute("DELETE FROM proc.bp_decision WHERE decision_id = ANY(%s)",
                    ([case_id, replay_id, control_id],))


@live
def test_list_never_serves_agent_policy_rows(client, seeded):
    for q in ("", "?subject_type=agent_policy_approval", "?subject_type=agent_policy_replay", "?limit=500"):
        r = client.get(f"/decisions{q}")
        assert r.status_code == 200, r.text
        assert SECRET not in r.text
        ids = {row["decision_id"] for row in r.json()["data"]}
        assert seeded.case_id not in ids and seeded.replay_id not in ids
    assert client.get("/decisions?subject_type=agent_policy_approval").json() == {"data": [], "total": 0}
    ordinary = client.get("/decisions?subject_type=tst_ordinary").json()
    assert seeded.control_id in {row["decision_id"] for row in ordinary["data"]} and ordinary["total"] >= 1


@live
def test_by_id_answers_404_for_agent_policy_rows(client, seeded):
    for did in (seeded.case_id, seeded.replay_id):
        r = client.get(f"/decisions/{did}")
        assert r.status_code == 404 and SECRET not in r.text
    ok = client.get(f"/decisions/{seeded.control_id}")
    assert ok.status_code == 200 and ok.json()["subject_type"] == "tst_ordinary"


# ------------------------------------------------------------------ stage 4: conflict cases
@pytest.fixture
def conflict_rows(conn):
    """A policy conflict case and a live conflict case (raw rows; their facts quote action values)."""
    key = f"TST-{random.randint(10**7, 10**8 - 1)}|TST-{random.randint(10**7, 10**8 - 1)}"
    ids = {}
    with conn.cursor() as cur:
        for st, decision in (("policy_conflict", "resolve_conflict"), ("live_conflict", "approve_or_reject")):
            cur.execute("INSERT INTO proc.bp_decision (subject_type, subject_id, decision, resolution, rationale, "
                        "facts, evidence, status, created_by) VALUES (%s, %s, %s, 'escalated', 'x', %s, %s, "
                        "'open', 'test') RETURNING decision_id",
                        (st, key, decision, json.dumps({"action": {"args": {"iban": SECRET}}}),
                         json.dumps([{"kind": "overlap", "example": {"args.iban": SECRET}}])))
            ids[st] = cur.fetchone()[0]
    yield ids
    with conn.cursor() as cur:
        cur.execute("DELETE FROM proc.bp_decision WHERE decision_id = ANY(%s)", (list(ids.values()),))


def test_conflict_subject_types_are_hidden():
    assert {"policy_conflict", "live_conflict"} <= set(decisions_router.HIDDEN_SUBJECT_TYPES)
    for t in decisions_router.HIDDEN_SUBJECT_TYPES:
        assert f"'{t}'" in decisions_router._HIDDEN_CLAUSE


@live
def test_list_and_by_id_never_serve_conflict_cases(client, seeded, conflict_rows):
    for q in ("", "?subject_type=policy_conflict", "?subject_type=live_conflict", "?limit=500"):
        r = client.get(f"/decisions{q}")
        assert r.status_code == 200, r.text
        assert SECRET not in r.text
        ids = {row["decision_id"] for row in r.json()["data"]}
        assert not ids & set(conflict_rows.values())
    for st in ("policy_conflict", "live_conflict"):
        assert client.get(f"/decisions?subject_type={st}").json() == {"data": [], "total": 0}
    for did in conflict_rows.values():
        r = client.get(f"/decisions/{did}")
        assert r.status_code == 404 and SECRET not in r.text
    # an ordinary subject type is still served exactly as before
    ordinary = client.get("/decisions?subject_type=tst_ordinary").json()
    assert seeded.control_id in {row["decision_id"] for row in ordinary["data"]}
    assert client.get(f"/decisions/{seeded.control_id}").status_code == 200
