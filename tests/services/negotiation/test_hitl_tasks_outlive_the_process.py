"""A round held for approval must still exist after the process that drafted it.

When `_resolve_hitl_decision` withholds approval, `_finalize_round_email_bundle`
stashes the drafted counters in `negotiation_state["hitl_email_tasks"]` and
returns without sending. That dict is process memory. It was copied into
`final_output.data` (negotiation_agent.py:3499) and nowhere else, and the one
live caller -- `email_watcher.py:1401` -- discards the AgentOutput. So the
counters a human was asked to approve did not survive the run in which they were
written, and approving the round afterwards had nothing left to release.

The session is already the thing that persists between rounds. These tests say
the held tasks belong on it.
"""
import json

import pytest

from agents.base_agent import AgentContext, AgentOutput, AgentStatus
from agents.negotiation_agent import NegotiationAgent, NegotiationIdentifier
from tests.test_negotiation_agent import DummyNick

WORKFLOW_ID = "wf-hitl-1"


class FakeRedis:
    """Enough of the client for the session round-trip, and no more."""

    def __init__(self):
        self.store = {}

    def get(self, key):
        return self.store.get(key)

    def set(self, key, value):
        self.store[key] = value

    def delete(self, key):
        self.store.pop(key, None)


@pytest.fixture
def agent():
    agent = NegotiationAgent(DummyNick())
    agent._redis_client = FakeRedis()
    return agent


@pytest.fixture
def context():
    ctx = AgentContext(
        workflow_id=WORKFLOW_ID,
        agent_id="NegotiationAgent",
        user_id="system",
        input_data={},
    )
    identifier = NegotiationIdentifier(
        workflow_id=WORKFLOW_ID,
        supplier_id="SUP-1",
        session_reference="ref-1",
        round_number=1,
    )
    # The shape _register_pending_email_task builds: live objects, not JSON.
    ctx._pending_email_round_tasks = {
        1: [
            {
                "identifier": identifier,
                "task": {"round_no": 1, "draft_stub": {"counter_price": 88.0}},
                "result": AgentOutput(status=AgentStatus.SUCCESS, data={}),
            }
        ]
    }
    return ctx


@pytest.fixture
def state(agent):
    session = agent._load_session_state_obj(WORKFLOW_ID, max_rounds=3)
    session.register_supplier("SUP-1")
    return {
        "workflow_id": WORKFLOW_ID,
        "session": session,
        "max_rounds": 3,
        "active_suppliers": {"SUP-1": {"status": "PENDING"}},
        "completed_suppliers": set(),
    }


def _hold_round(agent, context, state):
    agent._finalize_round_email_bundle(
        context=context,
        round_number=1,
        drafts=[],
        draft_bundles=[],
        negotiation_state=state,
        require_hitl=True,
    )


def test_the_held_round_is_still_stashed_in_memory(agent, context, state):
    """Guards the premise. If this breaks the rest of the file is describing
    behaviour that no longer exists."""
    _hold_round(agent, context, state)
    assert state["hitl_email_tasks"][1], "the round should be held, not sent"


def test_the_held_round_survives_a_reload_of_the_session(agent, context, state):
    """The harm, stated as a restart states it: load the session back from the
    store and the approved-pending counters should still be there."""
    _hold_round(agent, context, state)

    reloaded = agent._load_session_state_obj(WORKFLOW_ID, max_rounds=3)

    held = reloaded.negotiation_parameters.get("hitl_email_tasks") or {}
    assert held, (
        "no held email tasks on the reloaded session; the counters a human was "
        "asked to approve did not outlive the process that drafted them"
    )
    assert "1" in held or 1 in held, f"round 1 not in held rounds {sorted(held)}"


def test_the_counter_a_human_must_approve_is_in_what_survived(agent, context, state):
    """Not just that a row exists -- that the drafted counter is still in it.

    The stashed entry carries a NegotiationIdentifier and an AgentOutput, neither
    of which is JSON. Persisting it raw makes `_save_session_state_obj` raise into
    its own except-block, which stores nothing at all; reducing it too far stores
    a husk. This asserts the round survived with its content."""
    _hold_round(agent, context, state)

    reloaded = agent._load_session_state_obj(WORKFLOW_ID, max_rounds=3)
    held = reloaded.negotiation_parameters.get("hitl_email_tasks") or {}
    entries = held.get("1") or held.get(1) or []

    assert entries, f"round 1 held nothing; held rounds were {sorted(held)}"
    assert entries[0]["task"]["draft_stub"]["counter_price"] == 88.0
    assert entries[0]["identifier"], "the supplier this counter was written to is gone"
    json.dumps(reloaded.to_dict())  # and the session as a whole still stores


def test_holding_a_round_does_not_lose_existing_session_state(agent, context, state):
    """Persisting the held tasks must not clobber what the session already knows."""
    state["session"].negotiation_parameters["negotiation_style"] = "Competitive"

    _hold_round(agent, context, state)

    reloaded = agent._load_session_state_obj(WORKFLOW_ID, max_rounds=3)
    assert reloaded.negotiation_parameters.get("negotiation_style") == "Competitive"
    assert "SUP-1" in reloaded.supplier_negotiations


def test_a_sent_round_persists_no_held_tasks(agent, context, state):
    """require_hitl=False means the bundle went out. Nothing is being held, so
    nothing should be recorded as awaiting approval."""
    agent._finalize_email_round = lambda *a, **k: AgentOutput(
        status=AgentStatus.SUCCESS, data={"drafts": []}
    )

    agent._finalize_round_email_bundle(
        context=context,
        round_number=1,
        drafts=[],
        draft_bundles=[],
        negotiation_state=state,
        require_hitl=False,
    )

    reloaded = agent._load_session_state_obj(WORKFLOW_ID, max_rounds=3)
    assert not (reloaded.negotiation_parameters.get("hitl_email_tasks") or {})
