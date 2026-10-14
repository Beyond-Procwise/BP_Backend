"""A failed negotiation must not take the supplier's reply down with it.

`_process_agents` called `self.negotiation_agent.execute(neg_context)` and threw
the AgentOutput away (email_watcher.py:1401), then deleted the pending supplier
responses unconditionally. So when the negotiation failed -- or raised, which is
caught and logged -- the only record of what the supplier said was deleted
anyway, and there was nothing to retry from and nothing to read afterwards.

This is the live route: the inbound-reply path has no orchestrator, so this is
the only place the negotiation result can be acted on at all.
"""
from datetime import datetime, timezone

import pytest

from agents.base_agent import AgentOutput, AgentStatus
from services import email_watcher as email_watcher_module
from services.email_watcher import EmailWatcherV2

WORKFLOW_ID = "wf-neg-1"
UNIQUE_ID = "PROC-WF-DEADBEEF"

PENDING_ROW = {
    "unique_id": UNIQUE_ID,
    "supplier_id": "SUP-1",
    "response_text": "We can do 95 per unit.",
    "response_subject": "Re: Quote request",
    "response_message_id": "<reply-1>",
    "response_from": "supplier@example.com",
    "supplier_email": "supplier@example.com",
}


class StubSupplierAgent:
    """Reports a complete batch, which is what lets the negotiation run."""

    def execute(self, context):
        return AgentOutput(
            status=AgentStatus.SUCCESS,
            data={
                "workflow_id": WORKFLOW_ID,
                "batch_ready": True,
                "expected_responses": 1,
                "collected_responses": 1,
                "supplier_responses": [
                    {"unique_id": UNIQUE_ID, "supplier_id": "SUP-1"}
                ],
                "unique_ids": [UNIQUE_ID],
            },
        )


class RecordingNegotiationAgent:
    def __init__(self, outcome):
        self.outcome = outcome
        self.calls = 0

    def execute(self, context):
        self.calls += 1
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


@pytest.fixture
def deletions(monkeypatch):
    """Every delete_responses call the run makes."""
    recorded = []

    def _fetch_pending(*, workflow_id):
        return [dict(PENDING_ROW)]

    def _delete(*, workflow_id, unique_ids):
        recorded.append(list(unique_ids))

    monkeypatch.setattr(
        email_watcher_module.supplier_response_repo, "fetch_pending", _fetch_pending
    )
    monkeypatch.setattr(
        email_watcher_module.supplier_response_repo, "delete_responses", _delete
    )
    return recorded


def _run(negotiation_agent):
    watcher = EmailWatcherV2(
        supplier_agent=StubSupplierAgent(),
        negotiation_agent=negotiation_agent,
        dispatch_wait_seconds=0,
        poll_interval_seconds=0,
        max_poll_attempts=1,
        email_fetcher=lambda **_: [],
        sleep=lambda _: None,
        now=lambda: datetime.now(timezone.utc),
    )
    tracker = watcher._ensure_tracker(WORKFLOW_ID)
    tracker.all_dispatched = True
    tracker.all_responded = True
    watcher._process_agents(tracker)
    return watcher


def test_a_successful_negotiation_still_clears_the_replies(deletions):
    """Guards the existing behaviour: on success the round is done with them."""
    agent = RecordingNegotiationAgent(
        AgentOutput(status=AgentStatus.SUCCESS, data={"round": 1})
    )

    _run(agent)

    assert agent.calls == 1
    assert deletions, "a completed round should clear its pending replies"
    assert UNIQUE_ID in deletions[0]


def test_a_failed_negotiation_keeps_the_supplier_reply(deletions):
    """The harm: the reply is the only record of what the supplier offered."""
    agent = RecordingNegotiationAgent(
        AgentOutput(
            status=AgentStatus.FAILED, data={}, error="negotiation_session_locked"
        )
    )

    _run(agent)

    assert agent.calls == 1
    assert not deletions, (
        f"the negotiation failed but the supplier's reply was deleted anyway "
        f"({deletions}); there is nothing left to retry from"
    )


def test_a_raising_negotiation_keeps_the_supplier_reply(deletions):
    """The exception is caught and logged, so the run continues to the delete."""
    agent = RecordingNegotiationAgent(RuntimeError("ollama unreachable"))

    _run(agent)

    assert agent.calls == 1
    assert not deletions, (
        f"the negotiation raised but the supplier's reply was deleted anyway "
        f"({deletions})"
    )


def test_no_negotiation_agent_configured_still_clears_the_replies(deletions):
    """Absent an agent there is no negotiation to fail, and the supplier
    interaction has already been processed -- unchanged behaviour."""
    _run(None)

    assert deletions, "with no negotiation agent the replies should still clear"
