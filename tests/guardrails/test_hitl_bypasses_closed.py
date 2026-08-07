"""Neither bypass may return an approval.

A caller waiving its own checkpoint and a global off-switch are the same
failure wearing different clothes: automation deciding it does not need a
human. Both must produce 'pending', and the attempt must be visible.
"""

from src.agents.negotiation_agent import NegotiationAgent


class Ctx:
    def __init__(self, input_data):
        self.input_data = input_data
        self.workflow_id = "WF-1"


def _agent():
    return NegotiationAgent.__new__(NegotiationAgent)


def test_payload_auto_approve_does_not_approve():
    agent = _agent()
    state = {}
    result = agent._resolve_hitl_decision(
        context=Ctx({"hitl_auto_approve": True}),
        shared_context={},
        negotiation_state=state,
        round_num=1,
    )
    assert result["status"] == "pending"
    assert result["source"] != "auto_approved"


def test_shared_context_auto_approve_does_not_approve():
    agent = _agent()
    result = agent._resolve_hitl_decision(
        context=Ctx({}),
        shared_context={"hitl_auto_approve": True},
        negotiation_state={},
        round_num=1,
    )
    assert result["status"] == "pending"


def test_the_attempt_is_recorded():
    """An attempted bypass is an event an auditor needs to see."""
    agent = _agent()
    result = agent._resolve_hitl_decision(
        context=Ctx({"hitl_auto_approve": True}),
        shared_context={},
        negotiation_state={},
        round_num=1,
    )
    assert result.get("bypass_attempted") is True


def test_an_explicit_human_decision_still_works():
    agent = _agent()
    result = agent._resolve_hitl_decision(
        context=Ctx({}),
        shared_context={},
        negotiation_state={"hitl_decisions": {"1": "approved"}},
        round_num=1,
    )
    assert result["status"] == "approved"
    assert result["source"] == "provided"


def test_hitl_enabled_false_does_not_auto_approve(monkeypatch):
    """The global off-switch must not manufacture an approval."""
    agent = _agent()
    monkeypatch.setattr(
        type(agent), "_hitl_enforced", lambda self: False, raising=True
    )
    result = agent._resolve_hitl_decision(
        context=Ctx({}),
        shared_context={},
        negotiation_state={},
        round_num=1,
    )
    assert result["status"] == "pending"
    assert result.get("hitl_disabled_ignored") is True
