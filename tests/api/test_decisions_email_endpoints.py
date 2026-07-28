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
