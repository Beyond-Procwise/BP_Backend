"""A suspected payment-detail-change reply stops the agent drafting against that thread, marks a human's draft, and stops any send.

Three layers, because each is the last line when another is bypassed: the agent-initiated drafting paths REFUSE; a person's own draft
is allowed but carries a failing violation and is not ready; and the send guard denies regardless.
"""

import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from agents.base_agent import AgentContext, AgentStatus
from src.services.draft_assurance import connections, inbound, run as R
from src.services import email_dispatch_guard as guard
from tests.guardrails.test_send_path_gate import base_kwargs
from tests.services.test_draft_agent_stages import DECISION, _agent, _decision_agent, _prompt, good_model
from tests.services.test_draft_run import Conn, _policies, env
from tests.services.test_draft_assurance import TABLES


class FlagStore:
    """The email_agent door, answering exactly the two statements blocking_flags makes."""

    def __init__(self, flags=(), table=True, boom=False):
        self.flags, self.table, self.boom, self.queries = list(flags), table, boom, []

    def cursor(self):
        return FlagCur(self)


class FlagCur:
    def __init__(self, store):
        self.store, self.rows = store, []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        if self.store.boom:
            raise RuntimeError("email_agent unreachable")
        self.store.queries.append((" ".join(sql.split()), params))
        if "to_regclass" in sql:
            self.rows = [("email_agent.bp_inbound_flag" if self.store.table else None,)]
        elif "bp_inbound_flag" in sql:
            self.rows = [(f["id"], f["status"], f.get("kinds", ["payment_detail_change"]), f.get("message_id", "<m>"), None) for f in self.store.flags]
        else:
            self.rows = []

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def fetchall(self):
        return self.rows


OPEN = {"id": 7, "status": "open"}


def wire(monkeypatch, agent, store):
    @contextmanager
    def writer(agent_nick=None):
        yield store
    monkeypatch.setattr(connections, "writer", writer)
    base = _policies()
    human = json.loads((Path(__file__).resolve().parents[2] / "deploy/sql/2026-10-08_email_family_human_written.sql").read_text().split("$json$")[1])["rules"]
    rows = {"email_family_human_written": {"details": {"rules": human}, "version": 1, "policy_desc": "human"}}
    agent.agent_nick.policy_engine = SimpleNamespace(get_policy=lambda slug: rows.get(slug) or base.get_policy(slug), list_policies=base.list_policies)
    agent.agent_nick.get_db_connection = lambda: Conn(TABLES)
    return agent


# --- the drafting paths an agent starts: REFUSE ----------------------------------------------------------------------------------

def counter(agent):
    payload = {**DECISION, "recipients": ["a@x.test"]}
    stored = []
    agent._store_draft = lambda d: stored.append(d)
    agent._record_learning_events = lambda *a, **k: None
    out = agent._handle_negotiation_counter(AgentContext(workflow_id="wf-1", agent_id="email_drafting", user_id="u", input_data=payload), payload)
    return out, stored


def test_the_counter_path_refuses_to_draft_against_a_flagged_thread_and_calls_no_model_and_stores_nothing(monkeypatch):
    agent = wire(monkeypatch, _decision_agent(monkeypatch), FlagStore([OPEN]))
    calls = []
    monkeypatch.setattr(agent, "call_ollama", lambda **kw: calls.append(kw) or {"message": {"content": "x"}})
    out, stored = counter(agent)
    assert out.status == AgentStatus.FAILED and out.data["blocked"] is True and out.data["flag_ids"] == [7]
    assert "payment details" in out.data["blocked_reason"] and out.error == out.data["blocked_reason"]
    assert stored == [] and not [c for c in calls if "negotiation" in json.dumps(c, default=str).lower() and "Context" in json.dumps(c, default=str)]


def test_from_decision_refuses_and_the_run_returns_a_failure_without_storing(monkeypatch):
    agent = wire(monkeypatch, _decision_agent(monkeypatch), FlagStore([OPEN]))
    d = agent.from_decision(dict(DECISION))
    assert d["blocked"] is True and "body" not in d and d["flag_ids"] == [7]
    stored = []
    monkeypatch.setattr(agent, "_store_draft", lambda x: stored.append(x))
    out = agent.run(AgentContext(workflow_id="wf-1", agent_id="email_drafting", user_id="u", input_data={"decision": dict(DECISION)}))
    assert out.status == AgentStatus.FAILED and out.data["blocked"] is True and stored == []


def test_a_lookup_that_fails_blocks_automatic_drafting_rather_than_assuming_all_is_well(monkeypatch):
    agent = wire(monkeypatch, _decision_agent(monkeypatch), FlagStore(boom=True))
    out, stored = counter(agent)
    assert out.status == AgentStatus.FAILED and "could not be checked" in out.data["blocked_reason"] and stored == []


def test_no_flag_and_an_absent_flag_table_leave_drafting_exactly_as_before(monkeypatch):
    for store in (FlagStore([]), FlagStore(table=False)):
        agent = wire(monkeypatch, _decision_agent(monkeypatch), store)
        out, stored = counter(agent)
        assert out.status == AgentStatus.SUCCESS and len(stored) == 1
        assert not any(v["kind"] == "inbound_flag_unreviewed" for v in stored[0]["assurance"]["violations"])


def test_the_lookup_is_scoped_to_this_workflow_and_supplier(monkeypatch):
    store = FlagStore([])
    agent = wire(monkeypatch, _decision_agent(monkeypatch), store)
    counter(agent)
    scoped = [p for sql, p in store.queries if "FROM email_agent.bp_inbound_flag" in sql]
    assert scoped and scoped[0][0] == "wf-1" and scoped[0][1] == "S-1"


# --- a person's own draft: ALLOWED, but marked and not ready ----------------------------------------------------------------------

def test_a_draft_a_person_asked_for_goes_ahead_but_carries_a_failing_violation_and_is_not_ready(monkeypatch):
    agent = wire(monkeypatch, _agent(monkeypatch, good_model), FlagStore([OPEN]))
    a = _prompt(agent, requested_by="nick")["assurance"]
    bad = [v for v in a["violations"] if v["kind"] == "inbound_flag_unreviewed"]
    assert len(bad) == 1 and bad[0]["severity"] == "fail" and "payment details" in bad[0]["detail"]
    assert a["status"] == "needs_review" and a["ready"] is False and a["inbound_block"]["state"] == "blocked"
    assert a["inbound_block"]["flag_ids"] == [7]


def test_a_person_writing_on_a_flagged_thread_by_hand_is_marked_the_same_way(monkeypatch):
    agent = wire(monkeypatch, _agent(monkeypatch, good_model), FlagStore([OPEN]))
    a = agent.assure_human_written(text="Thanks, agreed.", recipients=["a@x.test"], supplier_id="S-1", workflow_id="wf-1", requested_by="nick")
    assert any(v["kind"] == "inbound_flag_unreviewed" and v["severity"] == "fail" for v in a["violations"]) and a["ready"] is False


def test_the_record_says_the_check_ran_and_found_nothing_when_it_found_nothing(monkeypatch):
    agent = wire(monkeypatch, _agent(monkeypatch, good_model), FlagStore([]))
    a = _prompt(agent, requested_by="nick")["assurance"]
    assert a.get("inbound_block") in (None, {"state": "clear", "flag_ids": []}) and not [v for v in a["violations"] if v["kind"] == "inbound_flag_unreviewed"]


def test_the_run_exposes_blocked_reason_only_when_blocked():
    clear = R.AssuranceRun(env=env(), data={}, family_source="declared", inbound_block={"state": "clear", "flag_ids": []})
    blocked = R.AssuranceRun(env=env(), data={}, family_source="declared", inbound_block={"state": "blocked", "flag_ids": [3], "phrase": "asks to change payment details"})
    injected = R.AssuranceRun(env=env(), data={}, family_source="declared", inbound_block={"state": "blocked", "flag_ids": [4], "phrase": "contains text that tries to instruct the assistant"})
    unknown = R.AssuranceRun(env=env(), data={}, family_source="declared", inbound_block={"state": "unknown", "flag_ids": []})
    assert clear.blocked_reason() is None and R.AssuranceRun(env=env(), data={}, family_source="declared").blocked_reason() is None
    assert "payment details" in blocked.blocked_reason() and "could not be checked" in unknown.blocked_reason()
    assert "tries to instruct the assistant" in injected.blocked_reason() and "payment" not in injected.blocked_reason()


# --- the send guard: the last line, whatever the drafting layer did ----------------------------------------------------------------

class GuardConn:
    """The guard's own lookups (allow-list etc.) plus the flag query."""

    def __init__(self, flags=(), table=True, boom=False):
        from tests.guardrails.test_send_path_gate import FakeConn
        self._base = FakeConn()
        self.store = FlagStore(flags, table, boom)

    def __getattr__(self, name):
        return getattr(self._base, name)

    def cursor(self):
        return self.store.cursor()


def test_the_guard_still_allows_a_clean_send_and_a_connection_that_cannot_run_sql():
    assert guard.check_dispatch(**base_kwargs(conn=GuardConn())).allowed is True
    assert guard.check_dispatch(**base_kwargs()).allowed is True                    # the plain test double has no cursor at all


def test_the_guard_denies_a_send_while_a_flag_is_open_or_confirmed_and_names_it_without_text():
    for status in ("open", "confirmed_fraud"):
        d = guard.check_dispatch(**base_kwargs(conn=GuardConn([{"id": 9, "status": status}])))
        assert d.allowed is False and "payment details" in d.reason
        assert d.evidence["flag_ids"] == [9] and "GB29" not in json.dumps(d.evidence, default=str)


def test_the_guard_denies_when_the_flags_cannot_be_read():
    d = guard.check_dispatch(**base_kwargs(conn=GuardConn(boom=True)))
    assert d.allowed is False and "could not be checked" in d.reason


def test_an_absent_flag_table_does_not_stop_a_send():
    assert guard.check_dispatch(**base_kwargs(conn=GuardConn(table=False))).allowed is True


def test_the_guard_asks_about_the_drafts_own_workflow_and_supplier():
    conn = GuardConn([])
    guard.check_dispatch(**base_kwargs(conn=conn))
    asked = [p for sql, p in conn.store.queries if "FROM email_agent.bp_inbound_flag" in sql]
    assert asked == [("WF-1", "SUP-1")]


def test_a_valid_approval_does_not_override_the_block():
    d = guard.check_dispatch(**base_kwargs(conn=GuardConn([OPEN]), approval_lookup=lambda **_: {
        "approval_id": 1, "status": "approved", "actioned_by": "buyer@ourcompany.com",
        "grounding": {"content_hash": __import__("tests.guardrails.test_send_path_gate", fromlist=["x"]).BASE_APPROVED_HASH}}))
    assert d.allowed is False


def test_a_draft_on_a_flagged_thread_is_marked_even_when_no_family_could_be_loaded(monkeypatch):
    agent = wire(monkeypatch, _agent(monkeypatch, good_model), FlagStore([OPEN]))
    agent.agent_nick.policy_engine = SimpleNamespace(get_policy=lambda slug: None, list_policies=lambda: [])      # no family rows at all
    a = agent.assure_human_written(text="Thanks.", recipients=["a@x.test"], supplier_id="S-1", workflow_id="wf-1", requested_by="nick")
    assert a["status"] == "unassured" and a["ready"] is False
    assert [v["kind"] for v in a["violations"]] == ["inbound_flag_unreviewed"] and a["inbound_block"]["state"] == "blocked"


def test_the_guard_says_an_injection_flag_is_about_the_assistant_not_about_payment():
    d = guard.check_dispatch(**base_kwargs(conn=GuardConn([{"id": 9, "status": "open", "kinds": ["instruction_override"]}])))
    assert d.allowed is False and "instruct the assistant" in d.reason and "payment" not in d.reason
