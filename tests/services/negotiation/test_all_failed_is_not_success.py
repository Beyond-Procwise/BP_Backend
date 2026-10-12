"""A negotiation in which every supplier failed has not succeeded.

The mailbox watcher reads the negotiation's status to decide whether a reply
has been dealt with: on SUCCESS it clears the replies down and stops watching.
The multi-round path returned SUCCESS unconditionally, so on 2026-10-12 a round
whose only draft failed (`negotiation_session_locked`) was reported as a
success "awaiting approval" -- with nothing to approve -- and the conversation
was closed with the supplier's reply unanswered.
"""
import pytest

from agents.base_agent import AgentContext, AgentOutput, AgentStatus
from agents.negotiation_agent import NegotiationAgent
from tests.test_negotiation_agent import DummyNick


def _run(monkeypatch, outcomes):
    """Run one multi-round negotiation; ``outcomes`` maps supplier -> AgentStatus."""

    def fake_single(self, context):
        supplier_id = context.input_data.get("supplier_id")
        status = outcomes[supplier_id]
        data = {"supplier": supplier_id, "supplier_id": supplier_id, "drafts": []}
        error = None if status == AgentStatus.SUCCESS else "negotiation_session_locked"
        return self._with_plan(
            context, AgentOutput(status=status, data=data, pass_fields=dict(data), error=error)
        )

    monkeypatch.setattr(NegotiationAgent, "_run_single_negotiation", fake_single)
    agent = NegotiationAgent(DummyNick())
    context = AgentContext(
        workflow_id="wf-all-failed",
        agent_id="NegotiationAgent",
        user_id=None,
        input_data={
            "negotiation_batch": [{"supplier_id": s, "rfq_id": f"RFQ-{s}"} for s in outcomes],
            "shared_context": {"target_price": 455.0},
            "multi_round_enabled": True,
        },
    )
    return agent.run(context)


def test_every_supplier_failing_is_reported_as_failed(monkeypatch):
    output = _run(monkeypatch, {"S1": AgentStatus.FAILED})
    assert output.status == AgentStatus.FAILED
    assert output.data["final_status"] == "ALL_FAILED"
    assert output.error


def test_a_failed_run_is_not_described_as_awaiting_approval(monkeypatch):
    """Nothing was drafted, so there is nothing for a person to approve."""
    output = _run(monkeypatch, {"S1": AgentStatus.FAILED, "S2": AgentStatus.FAILED})
    assert output.data["final_status"] == "ALL_FAILED"


def test_one_supplier_succeeding_is_still_a_success(monkeypatch):
    output = _run(monkeypatch, {"S1": AgentStatus.SUCCESS, "S2": AgentStatus.FAILED})
    assert output.status == AgentStatus.SUCCESS


def test_a_clean_run_is_unchanged(monkeypatch):
    output = _run(monkeypatch, {"S1": AgentStatus.SUCCESS})
    assert output.status == AgentStatus.SUCCESS
    assert output.data["final_status"] != "ALL_FAILED"
