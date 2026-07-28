"""DecisionEngine.act_on_email_reply -- record a human's send/reject on an
already-decided, escalated email-reply decision.

This is the sibling of `execute()` for the email path. `execute()` and
`decide_finding()` key off `finding_id` and read/write proc.bp_extraction_discrepancy
-- an email decision_id has no row there at all, so this path must never reach that
table. It reads and writes proc.bp_decision only: `_fetch_email_decision` (SELECT,
scoped to subject_type='email_reply') and the shared `_record_human_action` (INSERT),
the exact persistence `execute()` already uses for findings.

No real database is touched: a fake cursor/connection stand in, and every SQL
string executed is asserted never to mention bp_extraction_discrepancy.
"""
import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from engines.decision_engine import DecisionEngine, ESCALATED, RESOLVED

# Column order of the SELECT in DecisionEngine._fetch_email_decision:
#   decision_id, subject_type, subject_id, deal_id, supplier_id,
#   decision, resolution, rationale, policy_id, policy_name, facts, evidence
_SELECT_COLUMNS = [
    "decision_id", "subject_type", "subject_id", "deal_id", "supplier_id",
    "decision", "resolution", "rationale", "policy_id", "policy_name",
    "facts", "evidence",
]

ESCALATED_ROW = (
    7, "email_reply", "wf-1-PeopleFirst", None, "PeopleFirst HR Solutions Ltd",
    "escalate", "escalated", "price_change is escalate-only", 11,
    "EmailReplyAutonomyPolicy", {"intent": "price_change"},
    [{"fact": "intent", "value": "price_change",
      "source": "AgentNick classification of supplier_response.response_text",
      "reference": "1"}],
)

RESOLVED_SEND_ROW = (
    9, "email_reply", "wf-2-Acme", None, "Acme Ltd",
    "send", "resolved", "on the governed auto-reply list", 11,
    "EmailReplyAutonomyPolicy", {"intent": "acknowledge"}, [],
)


class FakeCursor:
    """Stands in for a psycopg2 cursor across BOTH the SELECT in
    `_fetch_email_decision` and the INSERT in `_record_human_action`.
    """

    def __init__(self, select_row):
        self.select_row = select_row
        self.description = None
        self.executed = []
        self.insert_params = None
        self._last_result = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        self.executed.append((sql, params))
        norm = " ".join(sql.split())
        if norm.startswith("SELECT") and "FROM proc.bp_decision" in norm:
            self.description = [(c,) for c in _SELECT_COLUMNS]
            self._last_result = self.select_row
        elif norm.startswith("INSERT INTO proc.bp_decision"):
            self.insert_params = params
            self._last_result = (555,)
        else:  # pragma: no cover - would indicate an unexpected query
            self._last_result = None

    def fetchone(self):
        return self._last_result


class FakeConn:
    def __init__(self, cur):
        self._cur = cur
        self.committed = False

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def cursor(self):
        return self._cur

    def commit(self):
        self.committed = True


def _engine(select_row):
    cur = FakeCursor(select_row)
    conn = FakeConn(cur)
    nick = SimpleNamespace(
        get_db_connection=lambda: conn,
        policy_engine=SimpleNamespace(get_policy=lambda slug: None),
    )
    eng = DecisionEngine(nick)
    return eng, cur


def test_unknown_action_is_rejected_without_touching_the_db():
    nick = SimpleNamespace(
        get_db_connection=lambda: (_ for _ in ()).throw(
            RuntimeError("must not touch the db for an unknown action")
        ),
        policy_engine=None,
    )
    result = DecisionEngine(nick).act_on_email_reply(7, "approve", user_id="alice")
    assert result["applied"] is False
    assert "unknown action" in result["error"]


def test_decision_not_found_returns_an_error_not_a_500():
    eng, cur = _engine(select_row=None)
    result = eng.act_on_email_reply(404, "send", user_id="alice")
    assert result == {"applied": False, "error": "email decision 404 not found"}
    # No INSERT was attempted for a decision that could not be read.
    assert cur.insert_params is None


def test_sending_against_an_escalated_recommendation_requires_an_override():
    eng, cur = _engine(select_row=ESCALATED_ROW)
    result = eng.act_on_email_reply(7, "send", user_id="alice")
    assert result["applied"] is False
    assert result["requires_override"] is True
    assert "price_change is escalate-only" in result["prompt"]
    # Nothing was written until the human means it.
    assert cur.insert_params is None


def test_overriding_a_send_records_who_and_why():
    eng, cur = _engine(select_row=ESCALATED_ROW)
    result = eng.act_on_email_reply(
        7, "send", user_id="alice", override_reason="supplier confirmed on the phone"
    )
    assert result["applied"] is True
    assert result["overridden"] is True
    assert result["override_reason"] == "supplier confirmed on the phone"
    assert result["decision_id"] == 555

    params = cur.insert_params
    assert params is not None
    # (subject_type, subject_id, deal_id, supplier_id, decision, resolution,
    #  rationale, policy_id, policy_name, facts, evidence, status, actioned_by,
    #  override_reason, agent, created_by)
    assert params[0] == "email_reply"
    assert params[1] == "wf-1-PeopleFirst"
    assert params[4] == "send"          # decision column carries the HUMAN's action
    assert params[5] == "escalated"     # resolution is the recommendation's, unchanged
    assert params[11] == "overridden"   # status
    assert params[12] == "alice"        # actioned_by
    assert params[13] == "supplier confirmed on the phone"  # override_reason
    assert params[14] == "decision_engine"
    assert params[15] == "alice"        # created_by
    # facts/evidence round-trip the STORED decision's, not an empty shell.
    assert json.loads(params[9]) == {"intent": "price_change"}
    assert json.loads(params[10])[0]["fact"] == "intent"


def test_rejecting_an_escalated_recommendation_needs_no_override():
    """'Do not send' is exactly what an escalation asks a human to weigh -- it does
    not contradict the recommendation, so no override_reason is required."""
    eng, cur = _engine(select_row=ESCALATED_ROW)
    result = eng.act_on_email_reply(7, "reject", user_id="bob")
    assert result["applied"] is True
    assert result["overridden"] is False
    assert result["override_reason"] is None

    params = cur.insert_params
    assert params[4] == "reject"
    assert params[11] == "actioned"     # not "overridden" -- nothing was overridden
    assert params[13] is None           # override_reason


def test_sending_a_resolved_send_recommendation_does_not_conflict():
    """Acting on a decision the engine already resolved as 'send' is not a
    contradiction and needs no override."""
    eng, cur = _engine(select_row=RESOLVED_SEND_ROW)
    result = eng.act_on_email_reply(9, "send", user_id="carol")
    assert result["applied"] is True
    assert result["overridden"] is False
    params = cur.insert_params
    assert params[4] == "send"
    assert params[5] == "resolved"
    assert params[11] == "actioned"


def test_the_fetch_is_scoped_to_email_reply_in_sql():
    eng, cur = _engine(select_row=ESCALATED_ROW)
    eng.act_on_email_reply(7, "reject", user_id="bob")
    select_sql, select_params = cur.executed[0]
    assert "subject_type" in select_sql
    assert select_params == (7, "email_reply")


def test_this_path_never_touches_bp_extraction_discrepancy():
    eng, cur = _engine(select_row=ESCALATED_ROW)
    eng.act_on_email_reply(7, "send", user_id="alice", override_reason="confirmed")
    for sql, _params in cur.executed:
        assert "bp_extraction_discrepancy" not in sql
    # And the finding-path methods were never called.
    assert cur.executed  # sanity: something was actually executed


def test_stringified_jsonb_facts_and_evidence_are_still_parsed():
    """psycopg2 normally hands back jsonb as already-parsed python objects, but this
    must not assume it -- a plain string must still round-trip.
    """
    row = list(ESCALATED_ROW)
    row[10] = json.dumps({"intent": "price_change"})
    row[11] = json.dumps([{"fact": "intent", "value": "price_change",
                           "source": "x", "reference": "1"}])
    eng, cur = _engine(select_row=tuple(row))
    result = eng.act_on_email_reply(7, "reject", user_id="bob")
    assert result["applied"] is True
    assert result["recommendation"]["facts"] == {"intent": "price_change"}
    assert result["recommendation"]["evidence"][0]["fact"] == "intent"
