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

    def __init__(self, fail=False, blocked=None):
        self.stored = []
        self.fail = fail
        self.blocked = blocked
        self.assured = []
        self.master = SimpleNamespace(emails=["nick@supplier.example"], name="Nicholas Geelen")
        self.looked_up = []

    def _master_contact(self, supplier_id):
        self.looked_up.append(supplier_id)
        return self.master

    def assure_final_offer_reply(self, data, *, text, recipients, supplier_id, workflow_id):
        self.assured.append({"data": dict(data), "text": text, "recipients": list(recipients),
                             "supplier_id": supplier_id, "workflow_id": workflow_id})
        if self.blocked:
            return self.blocked, {}
        return None, {"family_id": "final_offer_reply", "status": "ok", "violations": [], "ready": True}

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
    return agent


def _raise(agent, **overrides):
    args = dict(
        context=SimpleNamespace(input_data={"contact_name": "Somebody Else", "supplier_email": "elsewhere@example.com"}),
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


def test_no_address_on_the_supplier_master_means_advice_but_no_draft(agent):
    """An email is never addressed from what travelled with the workflow."""
    agent.stub_email.master = SimpleNamespace(emails=[], name="Nicholas Geelen")
    advice, draft = _raise(agent)
    assert advice["recommendation"] == ESCALATE
    assert draft is None and agent.stub_email.stored == []
    assert "supplier master" in advice["reply_not_drafted"]


def test_the_address_and_name_come_from_the_supplier_master_not_the_workflow(agent):
    """The workflow carried a different name and address; neither is used."""
    _raise(agent)
    stored = agent.stub_email.stored[0]
    assert agent.stub_email.looked_up == ["SUP-1"]
    assert stored["recipients"] == ["nick@supplier.example"]
    assert "Somebody Else" not in stored["body"] and "elsewhere@example.com" not in str(stored["recipients"])


def test_a_failing_draft_store_still_returns_the_advice(agent):
    agent.stub_email.fail = True
    advice, draft = _raise(agent)
    assert advice["recommendation"] == ESCALATE and draft is None


def test_with_no_contact_name_on_the_master_nobody_is_named(agent):
    """Not the supplier's company name, and not a name the workflow carried."""
    agent.stub_email.master = SimpleNamespace(emails=["nick@supplier.example"], name=None)
    _raise(agent)
    assert agent.stub_email.stored[0]["body"].startswith("<p>Hello,</p>")


class TestItIsOneOfTheAgentsGovernedEmails:
    """The reply takes the same assurance and the same mandate as everything else the agent writes."""

    def test_the_draft_carries_its_assurance_record(self, agent):
        _raise(agent)
        assert agent.stub_email.stored[0]["assurance"]["family_id"] == "final_offer_reply"

    def test_assurance_is_told_our_limits_so_it_can_check_they_are_not_stated(self, agent):
        _raise(agent, walkaway=480.0)
        (call,) = agent.stub_email.assured
        assert call["data"]["target_price"] == 455.0 and call["data"]["walkaway_price"] == 480.0
        assert call["data"]["current_offer"] == 470.0 and call["data"]["currency"] == "GBP"
        assert (call["supplier_id"], call["workflow_id"]) == ("SUP-1", "wf-1")
        assert call["text"] == agent.stub_email.stored[0]["body"]

    def test_a_thread_with_an_unreviewed_flagged_reply_gets_no_draft(self, agent):
        """A suspected payment-detail or injection reply stops an agent writing on the thread."""
        agent.stub_email.blocked = "a reply from this supplier has been flagged and has not been reviewed by a person"
        advice, draft = _raise(agent)
        assert draft is None and agent.stub_email.stored == []
        assert advice["recommendation"] == ESCALATE
        assert "flagged" in advice["reply_not_drafted"]

    def test_a_good_price_the_agent_has_no_mandate_for_is_escalated(self, agent):
        advice, _ = _raise(agent, offer=450.0, hold_reason="this counter commits GBP 450.00, above the governed GBP 100 limit")
        assert advice["recommendation"] == ESCALATE
        assert "above the governed GBP 100 limit" in advice["reason"]
        assert "accept" not in agent.stub_email.stored[0]["body"].lower()

    def test_no_hold_leaves_a_good_price_accepted(self, agent):
        advice, _ = _raise(agent, offer=450.0, hold_reason=None)
        assert advice["recommendation"] == ACCEPT
