"""The reply to a final offer takes the email assurance every other draft takes.

`EmailDraftingAgent.assure_final_offer_reply` is the family's entry point. It must
stop an agent writing on a thread that holds an unreviewed flagged reply, hand
back the family's record otherwise, and never raise: a failure is carried on the
draft as `unassured`, because silence would read as "checked and clean".
"""
from types import SimpleNamespace

import pytest

from agents.email_drafting_agent import EmailDraftingAgent


class _Run:
    def __init__(self, blocked=None):
        self._blocked = blocked

    def blocked_reason(self):
        return self._blocked


@pytest.fixture
def agent(monkeypatch):
    agent = EmailDraftingAgent.__new__(EmailDraftingAgent)
    agent.calls = SimpleNamespace(prepared=[], finalized=[])

    def prepare(data, *, workflow_id, slug=None, **kw):
        agent.calls.prepared.append({"data": data, "workflow_id": workflow_id, "slug": slug})
        return agent.run

    def finalize(run, composed, recipients, supplier_id, *, repaired):
        agent.calls.finalized.append({"composed": composed, "recipients": recipients,
                                      "supplier_id": supplier_id, "repaired": repaired})
        return {"family_id": "final_offer_reply", "status": "ok", "ready": True, "violations": []}

    agent.run = _Run()
    monkeypatch.setattr(agent, "_assurance_prepare", prepare, raising=False)
    monkeypatch.setattr(agent, "_assurance_finalize", finalize, raising=False)
    return agent


def _assure(agent, **over):
    args = dict(text="<p>Hello,</p><p>Thank you for your final offer of 470.00 GBP.</p>",
                recipients=["nick@supplier.example"], supplier_id="SUP-1", workflow_id="wf-1")
    args.update(over)
    return agent.assure_final_offer_reply({"current_offer": 470.0, "target_price": 455.0}, **args)


def test_it_runs_under_its_own_governed_family(agent):
    _assure(agent)
    (prepared,) = agent.calls.prepared
    assert prepared["slug"] == "email_family_final_offer_reply"
    assert prepared["workflow_id"] == "wf-1"
    assert prepared["data"]["supplier_id"] == "SUP-1" and prepared["data"]["target_price"] == 455.0


def test_the_words_checked_are_the_words_of_the_email_and_are_not_repaired(agent):
    blocked, record = _assure(agent)
    assert blocked is None and record["family_id"] == "final_offer_reply"
    (finalized,) = agent.calls.finalized
    assert "470.00 GBP" in finalized["composed"] and "<p>" not in finalized["composed"]
    assert finalized["repaired"] is False and finalized["recipients"] == ["nick@supplier.example"]


def test_a_flagged_thread_blocks_the_draft_and_nothing_is_finalised(agent):
    agent.run = _Run(blocked="a reply from this supplier has been flagged")
    blocked, record = _assure(agent)
    assert blocked == "a reply from this supplier has been flagged"
    assert record == {} and agent.calls.finalized == []


def test_a_failure_is_carried_as_unassured_and_never_raised(agent, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("policy store unreachable")

    monkeypatch.setattr(agent, "_assurance_prepare", boom, raising=False)
    blocked, record = _assure(agent)
    assert blocked is None
    assert record["status"] == "unassured" and record["family_id"] == "final_offer_reply"
    assert "policy store unreachable" in record["reason"] and record["ready"] is False
