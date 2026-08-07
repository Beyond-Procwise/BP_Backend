"""ApprovalsAgent must ground its verdict, and must never invent a threshold.

The old agent defaulted the spend-authority limit to a hardcoded 1000 whenever
its (non-existent) policy table lookup failed. A fabricated approval threshold
can auto-approve real money, so the replacement escalates instead.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agents.approvals_agent import (
    ApprovalsAgent,
    DECISION_APPROVE,
    DECISION_ESCALATE,
)
from agents.base_agent import AgentContext, AgentStatus


class _FakePolicyEngine:
    def __init__(self, policy):
        self._policy = policy

    def get_policy(self, slug):
        return self._policy if slug == "approval_threshold" else None


def _agent(policy=None) -> ApprovalsAgent:
    """Build an ApprovalsAgent with the DB stubbed out."""
    nick = SimpleNamespace(
        policy_engine=_FakePolicyEngine(policy),
        settings=SimpleNamespace(),
        get_db_connection=MagicMock(side_effect=RuntimeError("db disabled in test")),
    )
    agent = ApprovalsAgent.__new__(ApprovalsAgent)  # skip GPU / heavy __init__
    agent.agent_nick = nick
    agent.settings = nick.settings
    agent.policy_engine = nick.policy_engine
    return agent


def _ctx(**payload) -> AgentContext:
    return AgentContext(
        workflow_id="wf-test",
        agent_id="approvals",
        user_id="tester",
        input_data=payload,
    )


# The shape PolicyEngine._normalise_policy_row ACTUALLY returns. The first cut of
# these tests invented a friendlier shape ({"rules": ..., "policy_name": ...}) and
# passed green while the live agent read None from every field and escalated
# everything. A fixture that does not match production is worse than no fixture.
_GOVERNED = {
    "policyId": "approval_threshold",          # the identifier string, NOT the DB key
    "policyName": "ApprovalThresholdPolicy",
    "policy_type": "approval",
    "details": {"rules": {"default_threshold_gbp": 10000, "currency": "GBP"}},
    "aliases": {"approval_threshold"},
    "slug": "approval_threshold_policy",
    "raw_row": {"policy_id": 10, "policy_name": "ApprovalThresholdPolicy"},
}


def test_amount_within_threshold_is_approved():
    out = _agent(_GOVERNED).run(_ctx(amount=5000, currency="GBP"))
    assert out.status is AgentStatus.SUCCESS
    assert out.data["decision"] == DECISION_APPROVE
    assert out.data["approved"] is True
    assert out.data["threshold"] == 10000.0


def test_amount_above_threshold_escalates():
    out = _agent(_GOVERNED).run(_ctx(amount=25000, currency="GBP"))
    assert out.data["decision"] == DECISION_ESCALATE
    assert out.data["approved"] is False


def test_the_trace_never_records_a_false_comparison():
    """An escalation once wrote "25000 <= 10000" into the audit trail.

    The grounding block is the thing a human uses to check the verdict, so every
    statement in it has to actually be true.
    """
    approved = _agent(_GOVERNED).run(_ctx(amount=5000))
    assert approved.data["grounding"]["comparison"] == "5000 <= 10000"

    escalated = _agent(_GOVERNED).run(_ctx(amount=25000))
    assert escalated.data["grounding"]["comparison"] == "25000 > 10000"


def test_boundary_amount_equal_to_threshold_is_approved():
    out = _agent(_GOVERNED).run(_ctx(amount=10000))
    assert out.data["decision"] == DECISION_APPROVE


def test_threshold_comes_from_the_governed_policy_not_a_constant():
    out = _agent(_GOVERNED).run(_ctx(amount=1))
    g = out.data["grounding"]
    assert g["threshold_source"] == "governed_policy"
    assert g["policy_name"] == "ApprovalThresholdPolicy"
    assert g["policy_id"] == 10


def test_no_governed_policy_escalates_and_never_fabricates_a_threshold():
    """The regression that matters: no policy must NOT mean 'default to 1000'."""
    out = _agent(policy=None).run(_ctx(amount=5000))
    assert out.data["decision"] == DECISION_ESCALATE
    assert out.data["threshold"] is None
    assert out.data["grounding"]["threshold_source"] == "unavailable"
    # 5000 would have been APPROVED under a fabricated 10000 default, and
    # ESCALATED under the old hardcoded 1000. Neither is acceptable: with no
    # governed rule the only honest answer is "a human decides".
    assert "human" in out.data["decision_reason"].lower()


def test_explicit_threshold_is_recorded_as_a_caller_override():
    out = _agent(_GOVERNED).run(_ctx(amount=5000, threshold=100))
    assert out.data["decision"] == DECISION_ESCALATE  # 5000 > 100
    assert out.data["grounding"]["threshold_source"] == "request_override"


def test_missing_amount_fails_rather_than_guessing():
    out = _agent(_GOVERNED).run(_ctx(supplier_id="SUP001"))
    assert out.status is AgentStatus.FAILED
    assert "amount" in (out.error or "")


def test_currency_is_not_invented_when_absent():
    out = _agent(_GOVERNED).run(_ctx(amount=500))
    assert out.data["currency"] is None


def test_best_quote_price_is_used_when_amount_is_absent():
    out = _agent(_GOVERNED).run(
        _ctx(best_quote={"price": 250, "supplier_id": "S1"}, rfq_id="RFQ-1")
    )
    assert out.data["amount"] == 250.0
    assert out.data["decision"] == DECISION_APPROVE


def test_persistence_failure_surfaces_as_null_id_not_a_false_success():
    """The DB is stubbed to raise; the agent must not pretend it stored anything."""
    out = _agent(_GOVERNED).run(_ctx(amount=5000))
    assert out.status is AgentStatus.SUCCESS  # the decision itself is still valid
    assert out.data["approval_id"] is None    # but we do NOT claim it was persisted


def test_no_qdrant_references_are_fabricated():
    """The old agent embedded str(amount) and called the hits 'references'."""
    out = _agent(_GOVERNED).run(_ctx(amount=5000))
    assert "references" not in out.data


def test_fixture_matches_what_policy_engine_really_emits():
    """Guard the fixture above against drifting from the real engine.

    This is the check that would have caught the bug the mocked tests missed: the
    unit tests were green while the live agent read None from every policy field,
    because the fixture's shape was imagined rather than observed. Feed a real
    bp_policy row through the real normaliser and assert the keys the agent relies
    on are the ones that come out.
    """
    from engines.policy_engine import PolicyEngine

    row = {
        "policy_id": 10,
        "policy_name": "ApprovalThresholdPolicy",
        "policy_type": "approval",
        "policy_desc": "Spend authority gate.",
        "policy_details": {
            "policy_identifier": "approval_threshold",
            "rules": {"default_threshold_gbp": 10000, "currency": "GBP"},
        },
        "policy_linked_agents": "approvals_agent",
    }
    engine = PolicyEngine(policy_rows=[row])
    policy = engine.get_policy("approval_threshold")

    assert policy is not None, "alias 'approval_threshold' must resolve"
    # These are the exact paths ApprovalsAgent._governed_threshold reads.
    assert policy["details"]["rules"]["default_threshold_gbp"] == 10000
    assert policy["raw_row"]["policy_id"] == 10
    assert policy["raw_row"]["policy_name"] == "ApprovalThresholdPolicy"


# --- C3 (fix round 2): an automated verdict must never become a findable
# dispatch approval, and a caller-supplied "actioned_by" must not be able to
# forge one either. -----------------------------------------------------
#
# Fix round 1 accepted a non-blank payload["actioned_by"] as proof a human
# had approved, and wrote a genuinely findable approval (status='approved',
# actioned_by set) through approval_store.record_approval when it was
# present. That was a forgery hole: payload is context.input_data, the
# caller-supplied body of POST /agent-workflows/{workflow_id}/run, which has
# NO auth dependency at all. An unauthenticated caller could name themselves
# actioned_by and plant a real approval row for a workflow_id an
# email_dispatch node elsewhere in the same graph would then find -- and it
# ignored this agent's OWN verdict, writing an "approved" row even when the
# amount was above threshold and the real decision was escalate.
#
# AgentContext carries no authenticated principal anywhere in this codebase
# (no orchestrator path attaches one, and context.user_id is exactly as
# caller-suppliable as the removed payload["actioned_by"] was). So rather
# than invent a middle ground, this agent now NEVER produces a findable
# approval, full stop -- the same as before fix round 1 existed. A human
# approving a dispatch must do so through a surface that authenticates them
# and records its own approval.

class _FakeApprovalCursor:
    def __init__(self, log):
        self._log = log
        self._next_id = len(log) + 1

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self._log.append((sql, params))

    def fetchone(self):
        return (self._next_id,)


class _FakeApprovalConn:
    """Stands in for agent_nick.get_db_connection() so the automated
    raw-INSERT path can be exercised without a database."""

    def __init__(self):
        self.log = []
        self.committed = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def cursor(self):
        return _FakeApprovalCursor(self.log)

    def commit(self):
        self.committed += 1


def test_a_forged_actioned_by_in_the_payload_writes_nothing_findable():
    """The exploit this round closes: an unauthenticated caller cannot name
    themselves actioned_by in the workflow-run payload and get a genuinely
    findable (status='approved', actioned_by set) row. There is exactly one
    INSERT this agent ever issues, and it never carries those columns --
    regardless of what the payload says.
    """
    fake_conn = _FakeApprovalConn()
    agent = _agent(_GOVERNED)
    agent.agent_nick.get_db_connection = lambda: fake_conn

    out = agent.run(_ctx(
        amount=5000, currency="GBP", rfq_id="RFQ-1", workflow_id="WF-1",
        unique_id="PROC-WF-1", supplier_id="SUP-1", deal_id="DEAL-1",
        actioned_by="buyer@ourcompany.com",  # forged: no authenticated caller
    ))

    assert out.data["decision"] == DECISION_APPROVE
    assert out.data["approval_id"] == 1
    assert len(fake_conn.log) == 1, "must issue exactly one INSERT, never a second one"
    sql, params = fake_conn.log[0]
    assert "INSERT INTO proc.bp_approval" in sql
    assert "status" not in sql.lower()
    assert "actioned_by" not in sql.lower()
    # The forged name must not even reach the row as free text anywhere.
    assert "buyer@ourcompany.com" not in params


def test_an_escalate_verdict_writes_nothing_findable_even_with_a_forged_actioned_by():
    """An amount above threshold yields decision=escalate from run(). A
    forged actioned_by must not override that into an approved, findable
    row -- the exact "ignored the agent's own verdict" half of the exploit.
    """
    fake_conn = _FakeApprovalConn()
    agent = _agent(_GOVERNED)
    agent.agent_nick.get_db_connection = lambda: fake_conn

    out = agent.run(_ctx(
        amount=25000, currency="GBP", rfq_id="RFQ-2", workflow_id="WF-2",
        actioned_by="buyer@ourcompany.com",
    ))

    assert out.data["decision"] == DECISION_ESCALATE
    sql, params = fake_conn.log[0]
    assert "status" not in sql.lower()
    assert "actioned_by" not in sql.lower()
    assert "buyer@ourcompany.com" not in params
    # The row that IS written must honestly say escalate, not approve.
    assert DECISION_ESCALATE in params


def test_automated_verdict_never_sets_status_or_actioned_by():
    """No actioned_by in the payload -> the automated INSERT runs, and it
    must not name status/actioned_by columns at all: an unattended
    threshold comparison must never satisfy
    approval_store.find_dispatch_approval's 'a human signed this' check."""
    fake_conn = _FakeApprovalConn()
    agent = _agent(_GOVERNED)
    agent.agent_nick.get_db_connection = lambda: fake_conn

    out = agent.run(_ctx(amount=5000, currency="GBP"))

    assert out.data["approval_id"] == 1
    sql, _params = fake_conn.log[0]
    assert "status" not in sql.lower()
    assert "actioned_by" not in sql.lower()


def test_escalation_with_no_human_actor_still_uses_the_automated_path():
    """An ESCALATE verdict with nobody named must not accidentally become
    findable either."""
    fake_conn = _FakeApprovalConn()
    agent = _agent(_GOVERNED)
    agent.agent_nick.get_db_connection = lambda: fake_conn

    out = agent.run(_ctx(amount=25000, currency="GBP"))

    assert out.data["decision"] == DECISION_ESCALATE
    sql, _params = fake_conn.log[0]
    assert "status" not in sql.lower()
    assert "actioned_by" not in sql.lower()


def test_agent_resolves_threshold_from_a_real_policy_engine():
    """End-to-end through the real PolicyEngine, not a hand-written stub."""
    from engines.policy_engine import PolicyEngine

    engine = PolicyEngine(
        policy_rows=[
            {
                "policy_id": 10,
                "policy_name": "ApprovalThresholdPolicy",
                "policy_type": "approval",
                "policy_desc": "Spend authority gate.",
                "policy_details": {
                    "policy_identifier": "approval_threshold",
                    "rules": {"default_threshold_gbp": 10000, "currency": "GBP"},
                },
                "policy_linked_agents": "approvals_agent",
            }
        ]
    )
    agent = _agent()
    agent.policy_engine = engine

    out = agent.run(_ctx(amount=5000, currency="GBP"))
    assert out.data["decision"] == DECISION_APPROVE
    assert out.data["threshold"] == 10000.0
    g = out.data["grounding"]
    assert g["threshold_source"] == "governed_policy"
    assert g["policy_id"] == 10
    assert g["policy_name"] == "ApprovalThresholdPolicy"
