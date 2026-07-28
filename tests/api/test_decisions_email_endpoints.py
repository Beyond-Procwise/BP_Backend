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
     {"intent": "price_change"}, "2026-07-28T10:00:00+00:00",
     "We must raise our rate to GBP 94,000 for the coming term.", "verbatim"),
]


class _Cur:
    description = [("decision_id",), ("subject_type",), ("subject_id",), ("supplier_id",),
                   ("deal_id",), ("decision",), ("resolution",), ("rationale",),
                   ("policy_name",), ("facts",), ("created_at",),
                   ("supporting_sentence",), ("supporting_sentence_grounding",)]

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


# ---------------------------------------------------------------------------
# Task 10: the supplier's own cited sentence has to reach the client. The
# decision records it in `evidence` as `supporting_sentence`; this query used to
# select `facts` and not `evidence`, so the review panel had nothing to quote.
# ---------------------------------------------------------------------------

def test_queue_returns_the_supplier_s_cited_sentence_and_its_grounding(client):
    res = client.get("/decisions", params={"subject_type": "email_reply", "status": "open"})
    row = res.json()["data"][0]
    assert row["supporting_sentence"] == (
        "We must raise our rate to GBP 94,000 for the coming term."
    )
    # Returned WITH the sentence: a caller must not attribute an ungrounded
    # sentence to the supplier, so it never has to assume this.
    assert row["supporting_sentence_grounding"] == "verbatim"


def test_the_sentence_is_read_out_of_evidence_without_shipping_all_of_it(client):
    client.get("/decisions", params={"subject_type": "email_reply", "status": "open"})
    # `.sql` holds the LAST statement, which is the count query -- the row query is
    # the first one issued.
    sql = " ".join(client.app.state._cur.executed[0][0].split())
    assert "supporting_sentence" in sql
    assert "jsonb_array_elements" in sql
    # A polled list endpoint that can be asked for 500 rows must not return the whole
    # provenance array to render one sentence -- `GET /decisions/{id}` is for that.
    assert "d.evidence," not in sql
    assert "d.evidence FROM" not in sql
    # And the array is only walked when it really is an array: `evidence` is nullable.
    assert "jsonb_typeof(d.evidence) = 'array'" in sql


def test_an_ungrounded_sentence_is_flagged_and_a_dataless_row_stays_absent():
    """Two honest edges, through the real query path's own column mapping: a quote the
    classifier could not find in the supplier's message is returned WITH that verdict,
    and a decision carrying no evidence at all returns None rather than a guess."""
    table = _MemoryBpDecisionTable(rows=[
        {
            "decision_id": 11, "subject_type": "email_reply", "subject_id": "wf-11",
            "deal_id": None, "supplier_id": "Acme Ltd", "decision": "escalate",
            "resolution": "escalated", "rationale": "could not be grounded",
            "policy_id": 11, "policy_name": "EmailReplyAutonomyPolicy", "facts": {},
            "evidence": [{"fact": "supporting_sentence",
                          "value": "we accept your price",
                          "reference": "NOT FOUND in source"}],
            "status": "open", "created_at": "2026-07-28T13:00:00+00:00",
        },
        {
            "decision_id": 12, "subject_type": "email_reply", "subject_id": "wf-12",
            "deal_id": None, "supplier_id": "Beta Ltd", "decision": "escalate",
            "resolution": "escalated", "rationale": "no send authority",
            "policy_id": None, "policy_name": None, "facts": {}, "evidence": [],
            "status": "open", "created_at": "2026-07-28T12:00:00+00:00",
        },
    ])
    app = FastAPI()
    app.include_router(decisions_router.router)
    app.state.agent_nick = SimpleNamespace(
        get_db_connection=lambda: _MemoryConn(table),
        policy_engine=SimpleNamespace(get_policy=lambda slug: None),
    )
    body = TestClient(app).get(
        "/decisions", params={"subject_type": "email_reply", "status": "open"}
    ).json()
    ungrounded, dataless = body["data"][0], body["data"][1]
    assert ungrounded["supporting_sentence"] == "we accept your price"
    assert ungrounded["supporting_sentence_grounding"] == "NOT FOUND in source"
    assert dataless["supporting_sentence"] is None
    assert dataless["supporting_sentence_grounding"] is None


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

    def __init__(self, rows, fail_insert=False):
        self.rows = [dict(r) for r in rows]
        self.next_id = max((r["decision_id"] for r in self.rows), default=0) + 1
        # Fix round 2: lets a test simulate `_record_human_action`'s INSERT
        # failing, the same way that method's own pre-existing try/except would
        # see a real DB error -- it catches it and returns None, it never raises
        # out to the caller.
        self.fail_insert = fail_insert


def _cited(evidence, key):
    """The `supporting_sentence` evidence record's `value`/`reference`, or None.

    Stands in for the CASE + jsonb_array_elements subqueries in list_decisions's SQL,
    including their tolerance of a non-array payload.
    """
    if not isinstance(evidence, list):
        return None
    for item in evidence:
        if isinstance(item, dict) and item.get("fact") == "supporting_sentence":
            return item.get(key)
    return None


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
            if self.table.fail_insert:
                # Exercises `_record_human_action`'s own pre-existing try/except:
                # it catches this and returns None -- it never propagates.
                raise RuntimeError("simulated: the audit write failed")
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
            # Mirror `ORDER BY d.created_at DESC LIMIT %s` for real: the LIMIT is
            # the trailing param the router appends AFTER subject_type/status, and
            # without actually slicing here, a test could never tell a true
            # server-side count apart from `len(data)` -- exactly the regression
            # this fake exists to catch.
            matched.sort(key=lambda r: r.get("created_at") or "", reverse=True)
            limit = params[-1] if params else None
            if isinstance(limit, int):
                matched = matched[:limit]
            self.description = [
                ("decision_id",), ("subject_type",), ("subject_id",), ("supplier_id",),
                ("deal_id",), ("decision",), ("resolution",), ("rationale",),
                ("policy_name",), ("facts",), ("created_at",),
                ("supporting_sentence",), ("supporting_sentence_grounding",),
            ]
            self._rows_result = [
                (r["decision_id"], r["subject_type"], r["subject_id"], r["supplier_id"],
                 r["deal_id"], r["decision"], r["resolution"], r["rationale"],
                 r["policy_name"], r["facts"], r["created_at"],
                 # Mirrors the two derived columns in list_decisions's SQL: the first
                 # `supporting_sentence` record in `evidence`, and its own reference
                 # (the grounding verdict), or None when there is no such record.
                 _cited(r.get("evidence"), "value"), _cited(r.get("evidence"), "reference"))
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


# ---------------------------------------------------------------------------
# Fix round 2, CRITICAL: the close must be gated on the audit write succeeding.
# `_record_human_action` catches its own exceptions and returns None -- it
# never raises. Before this fix, `act_on_email_reply` closed the original row
# and reported `applied: True` regardless, which on a failed write is the worst
# available outcome: the record of who acted is lost, the task disappears from
# the human's queue, and the caller is told it worked.
# ---------------------------------------------------------------------------

def test_a_failed_audit_write_leaves_the_original_decision_in_the_queue():
    """Runs the real failure path (no monkeypatching of act_on_email_reply)
    against the in-memory table. Asserts BOTH that the engine reports
    applied=False directly, and that the real GET /decisions queue still
    contains the original, un-actioned decision afterward.
    """
    from engines.decision_engine import DecisionEngine

    table = _MemoryBpDecisionTable(rows=[{
        "decision_id": 7, "subject_type": "email_reply", "subject_id": "wf-1-PeopleFirst",
        "deal_id": "DEAL-1", "supplier_id": "PeopleFirst HR Solutions Ltd",
        "decision": "escalate", "resolution": "escalated",
        "rationale": "price_change is escalate-only", "policy_id": 11,
        "policy_name": "EmailReplyAutonomyPolicy", "facts": {"intent": "price_change"},
        "evidence": [], "status": "open", "created_at": "2026-07-28T10:00:00+00:00",
    }], fail_insert=True)

    app = FastAPI()
    app.include_router(decisions_router.router)
    nick = SimpleNamespace(
        get_db_connection=lambda: _MemoryConn(table),
        policy_engine=SimpleNamespace(get_policy=lambda slug: None),
    )
    app.state.agent_nick = nick
    memory_client = TestClient(app)

    result = DecisionEngine(nick).act_on_email_reply(7, "reject", user_id="bob")
    assert result["applied"] is False
    assert "error" in result
    assert "audit_decision_id" not in result

    # The row was never closed -- it genuinely has not been dealt with.
    assert table.rows[0]["status"] == "open"
    # No second row was inserted either: the audit write itself failed.
    assert len(table.rows) == 1

    still_open = memory_client.get("/decisions", params={"subject_type": "email_reply", "status": "open"})
    assert still_open.status_code == 200
    assert still_open.json()["total"] == 1
    assert still_open.json()["data"][0]["decision_id"] == 7

    # Also confirmed at the HTTP layer: the router maps a plain error (no
    # requires_override) to 400, so the human sees the action did NOT succeed.
    res = memory_client.post("/decisions/email-reply/7/action",
                             json={"action": "reject", "user_id": "bob"})
    assert res.status_code == 400


# ---------------------------------------------------------------------------
# Fix round 2: the true-count claim was untested. Neither the brief's fixture
# nor the Fix round 1 in-memory table ever created limit < total_escalations,
# so a regression to `total = len(rows)` would have passed every test here.
# ---------------------------------------------------------------------------

def test_total_reflects_the_true_server_side_count_not_the_page_size():
    rows = [
        {
            "decision_id": i, "subject_type": "email_reply", "subject_id": f"wf-{i}",
            "deal_id": None, "supplier_id": f"Supplier {i}", "decision": "escalate",
            "resolution": "escalated", "rationale": "needs a human", "policy_id": 11,
            "policy_name": "EmailReplyAutonomyPolicy", "facts": {}, "evidence": [],
            "status": "open", "created_at": f"2026-07-28T{10 + i}:00:00+00:00",
        }
        for i in range(1, 4)  # three escalated decisions
    ]
    table = _MemoryBpDecisionTable(rows=rows)
    app = FastAPI()
    app.include_router(decisions_router.router)
    app.state.agent_nick = SimpleNamespace(
        get_db_connection=lambda: _MemoryConn(table),
        policy_engine=SimpleNamespace(get_policy=lambda slug: None),
    )
    memory_client = TestClient(app)

    res = memory_client.get(
        "/decisions", params={"subject_type": "email_reply", "status": "open", "limit": 2}
    )
    assert res.status_code == 200
    body = res.json()
    # The page is capped at 2 -- but the true backlog is 3. A regression to
    # `total = len(data)` would report 2, not 3, and this is the assertion that
    # catches it.
    assert len(body["data"]) == 2
    assert body["total"] == 3
    assert body["total"] > len(body["data"])


# ---------------------------------------------------------------------------
# Fix round 1 (A): GET /decisions/email-reply/{decision_id}/message
#
# The supplier's own message, so a person can read it beside the reply they are
# approving. This is the feature the restored panel exists for; the text was stored
# and nothing served it.
#
# The fake below answers the route's two statements and NOTHING else -- an
# unexpected statement is an assertion failure, which is how these tests can claim
# the lookup really is scoped by decision_id AND subject_type rather than merely
# looking like it in the source.
# ---------------------------------------------------------------------------

DECISION_ROW = {
    "decision_id": 7, "subject_type": "email_reply",
    "subject_id": "089580f2-PeopleFirst", "supplier_id": "PeopleFirst HR Solutions Ltd",
    "deal_id": None,
}
FINDING_ROW = {
    "decision_id": 7, "subject_type": "finding", "subject_id": "421",
    "supplier_id": None, "deal_id": None,
}
REPLY_ROW = {
    "unique_id": "089580f2-PeopleFirst", "id": 1,
    "response_from": "billing@peoplefirst.invalid",
    "response_subject": "RE: Negotiation",
    "response_text": "Thank you for the proposal. We can offer 94,000.00 GBP with 45 day payment terms",
    "response_body": None,
    "received_time": None,
    "response_date": None,
    "supplier_id": "PeopleFirst HR Solutions Ltd",
}


class _MessageCursor:
    def __init__(self, decisions, replies, raise_on_message=False):
        self.decisions, self.replies = decisions, replies
        self.raise_on_message = raise_on_message
        self.executed = []
        self.description = None
        self._result = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        norm = " ".join((sql or "").split())
        self.executed.append((norm, params or ()))
        if "FROM proc.bp_decision" in norm:
            # The scope is the point: BOTH the id and the subject type, in SQL.
            assert "decision_id = %s AND subject_type = %s" in norm
            decision_id, subject_type = params
            match = next((d for d in self.decisions
                          if d["decision_id"] == decision_id
                          and d["subject_type"] == subject_type), None)
            self.description = [("decision_id",), ("subject_id",), ("supplier_id",), ("deal_id",)]
            self._result = None if match is None else (
                match["decision_id"], match["subject_id"], match["supplier_id"], match["deal_id"])
            return
        if "FROM proc.supplier_response" in norm:
            if self.raise_on_message:
                raise RuntimeError('relation "proc.supplier_response" does not exist')
            subject_id = params[0]
            match = next((r for r in self.replies
                          if r["unique_id"] == subject_id or str(r["id"]) == str(subject_id)), None)
            self.description = [("response_from",), ("response_subject",), ("response_text",),
                                ("response_body",), ("received_time",), ("response_date",),
                                ("supplier_id",)]
            self._result = None if match is None else (
                match["response_from"], match["response_subject"], match["response_text"],
                match["response_body"], match["received_time"], match["response_date"],
                match["supplier_id"])
            return
        raise AssertionError(f"unexpected SQL on the message route: {norm}")

    def fetchone(self):
        return self._result


class _MessageConn:
    def __init__(self, cur):
        self._cur = cur

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def cursor(self):
        return self._cur


def _message_client(decisions, replies, raise_on_message=False):
    cur = _MessageCursor(decisions, replies, raise_on_message)
    app = FastAPI()
    app.include_router(decisions_router.router)
    app.state.agent_nick = SimpleNamespace(
        get_db_connection=lambda: _MessageConn(cur),
        policy_engine=SimpleNamespace(get_policy=lambda slug: None),
    )
    app.state._cur = cur
    return TestClient(app), cur


def test_the_message_route_returns_the_supplier_s_own_words():
    client, _cur = _message_client([DECISION_ROW], [REPLY_ROW])
    res = client.get("/decisions/email-reply/7/message")
    assert res.status_code == 200
    body = res.json()
    assert body["available"] is True
    assert body["note"] is None
    msg = body["message"]
    assert msg["from"] == "billing@peoplefirst.invalid"
    assert msg["subject"] == "RE: Negotiation"
    assert msg["body"].startswith("Thank you for the proposal.")
    assert body["supplier_id"] == "PeopleFirst HR Solutions Ltd"
    assert body["subject_id"] == "089580f2-PeopleFirst"


def test_the_message_lookup_is_scoped_by_subject_type_not_by_id_alone():
    """A finding's decision_id must not reach an email message. The two id sequences
    are independent and both start at 1 -- an unscoped lookup by id is the same class
    of bug that already had to be fixed on the card path."""
    client, cur = _message_client([FINDING_ROW], [REPLY_ROW])
    res = client.get("/decisions/email-reply/7/message")
    assert res.status_code == 404
    assert "7" in res.json()["detail"]
    # And it never went looking for a message at all.
    assert not any("supplier_response" in sql for sql, _p in cur.executed)
    # The scope really was in the parameters, not filtered afterwards in Python.
    assert cur.executed[0][1] == (7, "email_reply")


def test_an_unknown_decision_is_a_404_and_names_nothing_internal():
    client, _cur = _message_client([], [])
    res = client.get("/decisions/email-reply/404/message")
    assert res.status_code == 404
    detail = res.json()["detail"]
    assert "404" in detail
    for token in ("proc.", "supplier_response", "bp_decision", "subject_type"):
        assert token not in detail


def test_a_missing_message_is_an_honest_absence_not_an_error_or_an_invention():
    """The decision exists; its message does not. That is 200 + available:false, so
    the panel renders the absence -- never a synthesised body, never a 500."""
    client, _cur = _message_client([DECISION_ROW], [])
    res = client.get("/decisions/email-reply/7/message")
    assert res.status_code == 200
    body = res.json()
    assert body["available"] is False
    assert body["message"] is None
    assert body["note"] and "could not be found" in body["note"]
    for token in ("proc.", "supplier_response", "response_text"):
        assert token not in body["note"]


def test_a_stored_row_with_no_body_reads_as_unavailable():
    """Headers without a body is not a message to review. It must not render as an
    empty quotation with a From line implying there is something there."""
    client, _cur = _message_client([DECISION_ROW], [{**REPLY_ROW, "response_text": None,
                                                    "response_body": None}])
    body = client.get("/decisions/email-reply/7/message").json()
    assert body["available"] is False
    assert body["message"] is None


def test_a_read_failure_says_so_without_quoting_the_driver():
    client, _cur = _message_client([DECISION_ROW], [REPLY_ROW], raise_on_message=True)
    res = client.get("/decisions/email-reply/7/message")
    assert res.status_code == 500
    detail = res.json()["detail"]
    assert "could not be read" in detail
    # The driver's message named a table; the reader must not see it.
    for token in ("proc.", "supplier_response", "relation", "RuntimeError"):
        assert token not in detail


def test_the_received_time_is_serialised_and_falls_back_to_the_sent_date():
    from datetime import datetime, timezone
    when = datetime(2026, 7, 28, 9, 30, tzinfo=timezone.utc)
    client, _cur = _message_client([DECISION_ROW], [{**REPLY_ROW, "received_time": when}])
    assert client.get("/decisions/email-reply/7/message").json()["message"]["received_at"] \
        == when.isoformat()
    # No received time recorded -> the date the supplier's mail carries, if any.
    client, _cur = _message_client([DECISION_ROW], [{**REPLY_ROW, "received_time": None,
                                                     "response_date": when}])
    assert client.get("/decisions/email-reply/7/message").json()["message"]["received_at"] \
        == when.isoformat()
    # Neither recorded -> absent, not "now".
    client, _cur = _message_client([DECISION_ROW], [REPLY_ROW])
    assert client.get("/decisions/email-reply/7/message").json()["message"]["received_at"] is None
