"""The queue endpoint feeds the Todo list. It must return escalations only.

Also covers the sibling action route, POST /decisions/email-reply/{decision_id}/action,
which records a human's send/reject against an already-recorded email decision. It
exists because act_on_finding (and DecisionEngine.execute/decide_finding underneath it)
key off finding_id and read proc.bp_extraction_discrepancy -- a table an email decision
has no row in.
"""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from engines.decision_engine import Decision, ESCALATED
import api.routers.decisions as decisions_router


ROWS = [
    (7, "email_reply", "wf-1-PeopleFirst", "PeopleFirst HR Solutions Ltd", "DEAL-1",
     "escalate", "escalated", "price_change is escalate-only", "EmailReplyAutonomyPolicy",
     {"intent": "price_change"}, "2026-07-28T10:00:00+00:00"),
]


class _Cur:
    description = [("decision_id",), ("subject_type",), ("subject_id",), ("supplier_id",),
                   ("deal_id",), ("decision",), ("resolution",), ("rationale",),
                   ("policy_name",), ("facts",), ("created_at",)]

    def __init__(self):
        self.sql = ""
        self.params = ()
        self.executed = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        self.sql, self.params = sql, params or ()
        self.executed.append((sql, params or ()))

    def fetchall(self):
        return ROWS

    def fetchone(self):
        return (len(ROWS),)


class _Conn:
    def __init__(self, cur):
        self._cur = cur

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def cursor(self):
        return self._cur


@pytest.fixture()
def client():
    cur = _Cur()
    app = FastAPI()
    app.include_router(decisions_router.router)
    app.state.agent_nick = SimpleNamespace(
        get_db_connection=lambda: _Conn(cur),
        policy_engine=SimpleNamespace(get_policy=lambda slug: None),
    )
    app.state._cur = cur
    return TestClient(app)


def test_queue_returns_escalated_email_decisions(client):
    res = client.get("/decisions", params={"subject_type": "email_reply", "status": "open"})
    assert res.status_code == 200
    body = res.json()
    assert body["total"] == 1
    row = body["data"][0]
    assert row["decision_id"] == 7
    assert row["subject_id"] == "wf-1-PeopleFirst"
    assert row["policy_name"] == "EmailReplyAutonomyPolicy"


def test_queue_filters_on_escalations_in_sql_not_in_python(client):
    client.get("/decisions", params={"subject_type": "email_reply", "status": "open"})
    sql = client.app.state._cur.sql.lower()
    # Filtering after the LIMIT would silently drop escalations off the end of a
    # busy queue -- the exact bug that pinned the findings badge at its page size.
    assert "resolution" in sql
    assert "where" in sql


def test_decide_endpoint_returns_the_decision_and_records_it(client, monkeypatch):
    captured = {}

    def fake_decide(self, response_id, *, authority=None, requested=None):
        captured["response_id"] = response_id
        captured["authority"] = authority
        return Decision(subject_type="email_reply", subject_id="wf-1-PeopleFirst",
                        decision="escalate", resolution=ESCALATED,
                        rationale="needs a human")

    monkeypatch.setattr("engines.decision_engine.DecisionEngine.decide_email_reply",
                        fake_decide, raising=True)
    monkeypatch.setattr("engines.decision_engine.DecisionEngine.record",
                        lambda self, d, **k: 42, raising=True)

    res = client.post("/decisions/email-reply/1")
    assert res.status_code == 200
    assert res.json()["decision_id"] == 42
    assert res.json()["decision"]["resolution"] == "escalated"
    assert captured["response_id"] == "1"
    # The endpoint must resolve authority itself: a caller-supplied limit would be a
    # limit chosen by the requester.
    assert captured["authority"] is not None


# ---------------------------------------------------------------------------
# POST /decisions/email-reply/{decision_id}/action
# ---------------------------------------------------------------------------

def test_action_route_wires_to_the_engine_and_never_touches_findings(client, monkeypatch):
    """The router must call DecisionEngine.act_on_email_reply, not execute()/
    decide_finding() -- those read/write proc.bp_extraction_discrepancy, which an
    email decision has no row in.
    """
    captured = {}

    def fake_act(self, decision_id, action, *, user_id="api", override_reason=None):
        captured["decision_id"] = decision_id
        captured["action"] = action
        captured["user_id"] = user_id
        captured["override_reason"] = override_reason
        return {"applied": True, "action": action, "decision_id": 99,
                "overridden": False, "override_reason": None, "actioned_by": user_id,
                "recommendation": {"decision": "escalate", "resolution": "escalated"}}

    def boom(*a, **k):
        raise AssertionError("execute()/decide_finding() must not be called on the email path")

    monkeypatch.setattr("engines.decision_engine.DecisionEngine.act_on_email_reply",
                        fake_act, raising=True)
    monkeypatch.setattr("engines.decision_engine.DecisionEngine.execute", boom, raising=True)
    monkeypatch.setattr("engines.decision_engine.DecisionEngine.decide_finding", boom, raising=True)
    monkeypatch.setattr("engines.decision_engine.DecisionEngine._fetch_finding", boom, raising=True)

    res = client.post("/decisions/email-reply/7/action",
                      json={"action": "reject", "user_id": "alice"})
    assert res.status_code == 200
    body = res.json()
    assert body["applied"] is True
    assert body["decision_id"] == 99
    assert captured == {"decision_id": 7, "action": "reject", "user_id": "alice",
                        "override_reason": None}


def test_action_route_returns_400_on_unknown_decision(client, monkeypatch):
    monkeypatch.setattr(
        "engines.decision_engine.DecisionEngine.act_on_email_reply",
        lambda self, decision_id, action, **k: {
            "applied": False, "error": f"email decision {decision_id} not found",
        },
        raising=True,
    )
    res = client.post("/decisions/email-reply/404/action", json={"action": "send"})
    assert res.status_code == 400


def test_action_route_surfaces_requires_override_as_200_not_error(client, monkeypatch):
    """Needing an override is a normal outcome of the human-in-the-loop convention,
    not a failure -- it must come back 200 so the UI can show the confirmation."""
    monkeypatch.setattr(
        "engines.decision_engine.DecisionEngine.act_on_email_reply",
        lambda self, decision_id, action, **k: {
            "applied": False, "requires_override": True,
            "recommendation": {"decision": "escalate", "resolution": "escalated"},
            "prompt": "The evidence does not support sending here: ...",
        },
        raising=True,
    )
    res = client.post("/decisions/email-reply/7/action", json={"action": "send"})
    assert res.status_code == 200
    assert res.json()["requires_override"] is True


def test_action_route_passes_override_reason_through(client, monkeypatch):
    captured = {}

    def fake_act(self, decision_id, action, *, user_id="api", override_reason=None):
        captured["override_reason"] = override_reason
        return {"applied": True, "action": action, "decision_id": 100, "overridden": True,
                "override_reason": override_reason, "actioned_by": user_id}

    monkeypatch.setattr("engines.decision_engine.DecisionEngine.act_on_email_reply",
                        fake_act, raising=True)
    res = client.post(
        "/decisions/email-reply/7/action",
        json={"action": "send", "override_reason": "supplier confirmed on the phone"},
    )
    assert res.status_code == 200
    assert res.json()["overridden"] is True
    assert captured["override_reason"] == "supplier confirmed on the phone"


# ---------------------------------------------------------------------------
# Fix round 1, concern 1: the queue must actually clear once a human acts.
#
# Everything above monkeypatches act_on_email_reply, which proves the ROUTER is
# wired correctly but says nothing about whether the ENGINE's write actually
# changes what the queue query sees. This test runs the real engine code (no
# monkeypatching of act_on_email_reply/decide_email_reply) against a small
# in-memory stand-in for proc.bp_decision, and asserts on GET /decisions's own
# result set before and after -- the regression that matters is "the row stops
# matching the query", not "an UPDATE statement was issued".
# ---------------------------------------------------------------------------

class _MemoryBpDecisionTable:
    """A tiny in-memory stand-in for proc.bp_decision, just enough to answer the
    exact queries `_fetch_email_decision`, `_record_human_action`, `_close_original_
    email_decision`, and `list_decisions` issue.
    """

    def __init__(self, rows):
        self.rows = [dict(r) for r in rows]
        self.next_id = max((r["decision_id"] for r in self.rows), default=0) + 1


def _queue_matches(row, sql, params):
    """Mirror list_decisions's WHERE clause: resolution='escalated' is always
    required (it's baked into the SQL, not a parameter); subject_type/status are
    consumed from params in the same order the router appends them."""
    if row.get("resolution") != "escalated":
        return False
    idx = 0
    if "d.subject_type = %s" in sql:
        if row.get("subject_type") != params[idx]:
            return False
        idx += 1
    if "d.status = %s" in sql:
        if row.get("status") != params[idx]:
            return False
        idx += 1
    return True


class _MemoryCursor:
    def __init__(self, table):
        self.table = table
        self.description = None
        self._result = None
        self._rows_result = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        params = params or ()
        norm = " ".join(sql.split())

        if norm.startswith("UPDATE proc.bp_decision"):
            status, decision_id, subject_type = params
            for row in self.table.rows:
                if row["decision_id"] == decision_id and row["subject_type"] == subject_type:
                    row["status"] = status
            self._result = None
            return

        if norm.startswith("INSERT INTO proc.bp_decision"):
            (subject_type, subject_id, deal_id, supplier_id, decision, resolution,
             rationale, policy_id, policy_name, facts, evidence, status, actioned_by,
             override_reason, agent, created_by) = params
            new_id = self.table.next_id
            self.table.next_id += 1
            self.table.rows.append({
                "decision_id": new_id, "subject_type": subject_type,
                "subject_id": subject_id, "deal_id": deal_id, "supplier_id": supplier_id,
                "decision": decision, "resolution": resolution, "rationale": rationale,
                "policy_id": policy_id, "policy_name": policy_name, "facts": facts,
                "evidence": evidence, "status": status, "actioned_by": actioned_by,
                "override_reason": override_reason, "agent": agent,
                "created_by": created_by, "created_at": "2026-07-28T11:00:00+00:00",
            })
            self.description = [("decision_id",)]
            self._result = (new_id,)
            return

        if norm.startswith("SELECT") and "count(*)" in norm.lower():
            matched = [r for r in self.table.rows if _queue_matches(r, norm, params)]
            self._result = (len(matched),)
            return

        if norm.startswith("SELECT") and "FROM proc.bp_decision d" in norm:
            matched = [r for r in self.table.rows if _queue_matches(r, norm, params)]
            self.description = [
                ("decision_id",), ("subject_type",), ("subject_id",), ("supplier_id",),
                ("deal_id",), ("decision",), ("resolution",), ("rationale",),
                ("policy_name",), ("facts",), ("created_at",),
            ]
            self._rows_result = [
                (r["decision_id"], r["subject_type"], r["subject_id"], r["supplier_id"],
                 r["deal_id"], r["decision"], r["resolution"], r["rationale"],
                 r["policy_name"], r["facts"], r["created_at"])
                for r in matched
            ]
            return

        if norm.startswith("SELECT") and "WHERE decision_id = %s AND subject_type = %s" in norm:
            decision_id, subject_type = params
            match = next(
                (r for r in self.table.rows
                 if r["decision_id"] == decision_id and r["subject_type"] == subject_type),
                None,
            )
            self.description = [
                ("decision_id",), ("subject_type",), ("subject_id",), ("deal_id",),
                ("supplier_id",), ("decision",), ("resolution",), ("rationale",),
                ("policy_id",), ("policy_name",), ("facts",), ("evidence",),
            ]
            self._result = None if match is None else (
                match["decision_id"], match["subject_type"], match["subject_id"],
                match["deal_id"], match["supplier_id"], match["decision"],
                match["resolution"], match["rationale"], match.get("policy_id"),
                match["policy_name"], match["facts"], match["evidence"],
            )
            return

        raise AssertionError(f"unexpected SQL in the in-memory fake: {norm}")

    def fetchone(self):
        return self._result

    def fetchall(self):
        return self._rows_result


class _MemoryConn:
    def __init__(self, table):
        self.table = table

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def cursor(self):
        return _MemoryCursor(self.table)

    def commit(self):
        pass


def test_after_an_action_the_original_decision_no_longer_matches_the_queue():
    """The regression that matters: assert on GET /decisions's own result set,
    not on the UPDATE having merely been issued."""
    table = _MemoryBpDecisionTable(rows=[{
        "decision_id": 7, "subject_type": "email_reply", "subject_id": "wf-1-PeopleFirst",
        "deal_id": "DEAL-1", "supplier_id": "PeopleFirst HR Solutions Ltd",
        "decision": "escalate", "resolution": "escalated",
        "rationale": "price_change is escalate-only", "policy_id": 11,
        "policy_name": "EmailReplyAutonomyPolicy", "facts": {"intent": "price_change"},
        "evidence": [], "status": "open", "created_at": "2026-07-28T10:00:00+00:00",
    }])
    app = FastAPI()
    app.include_router(decisions_router.router)
    app.state.agent_nick = SimpleNamespace(
        get_db_connection=lambda: _MemoryConn(table),
        policy_engine=SimpleNamespace(get_policy=lambda slug: None),
    )
    memory_client = TestClient(app)

    before = memory_client.get("/decisions", params={"subject_type": "email_reply", "status": "open"})
    assert before.status_code == 200
    assert before.json()["total"] == 1
    assert before.json()["data"][0]["decision_id"] == 7

    res = memory_client.post(
        "/decisions/email-reply/7/action",
        json={"action": "send", "user_id": "alice",
              "override_reason": "supplier confirmed on the phone"},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["applied"] is True
    assert body["queue_closed"] is True
    assert "warning" not in body

    after = memory_client.get("/decisions", params={"subject_type": "email_reply", "status": "open"})
    assert after.json()["total"] == 0
    assert after.json()["data"] == []

    # The audit trail exists as a SEPARATE row -- closing the original did not
    # erase or rewrite what the agent originally decided.
    original = next(r for r in table.rows if r["decision_id"] == 7)
    assert original["status"] == "overridden"
    assert original["rationale"] == "price_change is escalate-only"  # untouched
    audit_rows = [r for r in table.rows if r["decision_id"] != 7]
    assert len(audit_rows) == 1
    assert audit_rows[0]["decision"] == "send"
    assert audit_rows[0]["actioned_by"] == "alice"
    assert audit_rows[0]["override_reason"] == "supplier confirmed on the phone"

    # And a decision NOT actioned still shows up -- proving the queue query
    # itself, not just this one row, still works.
    table.rows.append({
        "decision_id": 42, "subject_type": "email_reply", "subject_id": "wf-2-Acme",
        "deal_id": None, "supplier_id": "Acme Ltd", "decision": "escalate",
        "resolution": "escalated", "rationale": "no prior offer", "policy_id": 11,
        "policy_name": "EmailReplyAutonomyPolicy", "facts": {}, "evidence": [],
        "status": "open", "created_at": "2026-07-28T12:00:00+00:00",
    })
    still_open = memory_client.get("/decisions", params={"subject_type": "email_reply", "status": "open"})
    assert still_open.json()["total"] == 1
    assert still_open.json()["data"][0]["decision_id"] == 42
