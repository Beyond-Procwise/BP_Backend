"""When a supplier's offer is final, the agent saves a reply for a person to approve.

Drives `_raise_final_offer`, the step the halt-on-final-offer path now takes,
with the draft store and the dispatch record stubbed. The reply must go to the
address our own email was sent to, under a reference of its own -- reusing the
sent email's reference would overwrite it, because the draft store upserts on
(workflow, reference).
"""
from types import SimpleNamespace

import pytest

from agents.negotiation_agent import NegotiationAgent
from src.services.negotiation.final_offer import ACCEPT, ESCALATE
from tests.test_negotiation_agent import DummyNick


class _EmailAgent:
    """Stands in for the draft store: gives the draft its own reference, as the real one does."""

    def __init__(self, fail=False):
        self.stored = []
        self.fail = fail

    def _store_draft(self, draft):
        if self.fail:
            raise RuntimeError("draft store unavailable")
        assert not draft.get("unique_id"), "the reply must not reuse another email's reference"
        draft["unique_id"] = "PROC-WF-NEWREPLY"
        draft["id"] = 9001
        self.stored.append(dict(draft))


@pytest.fixture
def agent(monkeypatch):
    agent = NegotiationAgent(DummyNick())
    agent.stub_email = _EmailAgent()
    monkeypatch.setattr(agent, "_ensure_email_agent", lambda: agent.stub_email)
    monkeypatch.setattr(agent, "_final_offer_recipients", lambda workflow_id, sent_reference, context: ["nick@supplier.example"])
    return agent


def _raise(agent, **overrides):
    args = dict(
        context=SimpleNamespace(input_data={"contact_name": "Nicholas Geelen"}),
        workflow_id="wf-1", supplier="SUP-1", supplier_name="Test Supplies Ltd",
        sent_reference="PROC-WF-SENT", round_no=1, state={"base_subject": "Pricing Discussion"},
        offer=470.0, target=455.0, walkaway=None, currency="GBP",
    )
    args.update(overrides)
    return agent._raise_final_offer(**args)


def test_an_over_target_final_offer_escalates_with_a_holding_reply(agent):
    advice, draft = _raise(agent)
    assert advice["recommendation"] == ESCALATE and advice["over_by"] == 15.0
    (stored,) = agent.stub_email.stored
    assert "reviewing it internally" in stored["body"] and "accept" not in stored["body"].lower()
    assert draft["unique_id"] == "PROC-WF-NEWREPLY"


def test_a_within_target_final_offer_drafts_an_acceptance(agent):
    advice, _ = _raise(agent, offer=450.0)
    assert advice["recommendation"] == ACCEPT
    assert "We accept the price of 450.00 GBP" in agent.stub_email.stored[0]["body"]


def test_the_reply_is_an_unsent_draft_to_the_address_we_wrote_to(agent):
    _raise(agent)
    stored = agent.stub_email.stored[0]
    assert stored["recipients"] == ["nick@supplier.example"]
    assert stored["sent_status"] is False
    assert (stored["workflow_id"], stored["supplier_id"]) == ("wf-1", "SUP-1")
    assert stored["subject"] == "Re: Pricing Discussion"
    assert stored["body"].startswith("<p>Dear Nicholas Geelen,</p>")


def test_the_recommendation_travels_with_the_draft(agent):
    """The approval screen reads it from the saved draft."""
    advice, _ = _raise(agent)
    stored = agent.stub_email.stored[0]
    assert stored["final_offer"] == advice
    assert stored["in_reply_to_unique_id"] == "PROC-WF-SENT"


def test_no_known_recipient_means_advice_but_no_draft(agent, monkeypatch):
    """An email is never addressed to a guess."""
    monkeypatch.setattr(agent, "_final_offer_recipients", lambda *a: [])
    advice, draft = _raise(agent)
    assert advice["recommendation"] == ESCALATE
    assert draft is None and agent.stub_email.stored == []


def test_a_failing_draft_store_still_returns_the_advice(agent):
    agent.stub_email.fail = True
    advice, draft = _raise(agent)
    assert advice["recommendation"] == ESCALATE and draft is None


def test_the_supplier_name_is_not_used_as_a_persons_name(agent):
    _raise(agent, context=SimpleNamespace(input_data={}))
    assert agent.stub_email.stored[0]["body"].startswith("<p>Hello,</p>")
