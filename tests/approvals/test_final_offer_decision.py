"""A person decides a supplier's final offer, and the waiting reply becomes their answer.

When the agent escalates a final offer it leaves a holding reply. Before this, a person
who decided to take the price (or walk away) had to write the email by hand. The decision
rewrites that same draft in fixed wording, records who decided, and sends nothing.

The email store and the drafting agent are stubbed; the wording is the real module's.
"""
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from src.api.routers import approvals as approvals_router
from src.services.approval_content import content_hash
from src.services.draft_hydration import resolve_effective_content

UID = "PROC-WF-FINAL0000001"
ADVICE = {
    "recommendation": "escalate", "reason": "The supplier's final offer of 470.00 GBP is 15.00 GBP (3.3%) above the target of 455.00 GBP.",
    "offer": 470.0, "currency": "GBP", "limit": 455.0, "limit_kind": "target", "over_by": 15.0, "over_pct": 3.3,
}
HOLDING = ("<p>Hello,</p><p>Thank you for your final offer of 470.00 GBP.</p>"
           "<p>We are reviewing it internally and will come back to you with our decision.</p>")


class _Principal:
    subject = "sub-buyer-007"
    email = "buyer@ourcompany.com"
    claims = {"cognito:groups": ["bp-buyers"]}


class _Agent:
    def __init__(self, store, blocked=None, save=True):
        self.store, self.blocked, self.save = store, blocked, save
        self.assured = []

    def _master_contact(self, supplier_id):
        return SimpleNamespace(emails=["nadia@ashcroft.example"], name="Nadia Rahman")

    def assure_final_offer_reply(self, data, *, text, recipients, supplier_id, workflow_id, requested_by=None):
        self.assured.append({"data": dict(data), "text": text, "requested_by": requested_by})
        if self.blocked:
            return self.blocked, {}
        return None, {"family_id": "final_offer_reply", "status": "needs_review", "violations": []}

    def _store_draft(self, draft):
        if not self.save:
            return                      # the real one logs and swallows a database error
        draft["draft_record_id"] = 8200
        self.store["draft"] = {k: v for k, v in draft.items() if k != "draft_record_id"}


def _draft(**over):
    draft = {
        "unique_id": UID, "workflow_id": "wf-1", "rfq_id": None, "supplier_id": "SUP-A",
        "supplier_name": "Ashcroft Associates", "recipients": ["nadia@ashcroft.example"],
        "subject": "Re: Pricing Discussion", "body": HOLDING, "requested_by": None,
        "attachments": None, "final_offer": dict(ADVICE),
    }
    draft.update(over)
    return draft


@pytest.fixture
def world(monkeypatch):
    store = {"draft": _draft(), "gate": []}
    agent = _Agent(store)
    monkeypatch.setattr(approvals_router, "record_action_or_fail", lambda **kw: None)
    monkeypatch.setattr(approvals_router, "_load_draft", lambda uid, conn=None: store["draft"])
    monkeypatch.setattr(approvals_router, "_drafting_agent", lambda request: agent)
    monkeypatch.setattr(approvals_router.approval_store, "find_dispatch_approval", lambda **k: None)
    monkeypatch.setattr(approvals_router, "gate",
                        lambda action, principal, **kw: store["gate"].append((action, kw.get("context"))))
    store["agent"] = agent
    return store


def _post(decision, principal=_Principal(), **body):
    app = FastAPI()
    app.include_router(approvals_router.router)
    app.dependency_overrides[approvals_router.require_user] = lambda: principal
    return TestClient(app).post(f"/approvals/drafts/{UID}/final-offer", json={"decision": decision, **body})


def test_accepting_turns_the_holding_reply_into_an_acceptance(world):
    res = _post("accept")
    assert res.status_code == 200
    body = world["draft"]["body"]
    assert "We accept the price of 470.00 GBP" in body and "reviewing it internally" not in body
    assert body.startswith("<p>Dear Nadia Rahman,</p>")
    assert "455" not in body, "our target is never told to the supplier"
    assert "We accept the price of 470.00 GBP" in res.json()["body_text"]


def test_declining_turns_it_into_a_decline_that_names_no_figure_of_ours(world):
    assert _post("decline").status_code == 200
    body = world["draft"]["body"]
    assert "not able to proceed at this price" in body and "accept" not in body.lower()
    assert "455" not in body and "15.00" not in body


def test_who_decided_is_recorded_from_their_sign_in(world):
    _post("accept")
    stored = world["draft"]
    assert stored["requested_by"] == "sub-buyer-007"
    decision = stored["final_offer"]["decision"]
    assert (decision["decision"], decision["decided_by"]) == ("accept", "sub-buyer-007")
    assert decision["decided_at"]
    assert stored["final_offer"]["recommendation"] == "escalate", "the agent's advice is kept beside the decision"


def test_the_decided_email_is_assured_like_any_other(world):
    _post("accept")
    (call,) = world["agent"].assured
    assert call["data"]["target_price"] == 455.0 and call["data"]["current_offer"] == 470.0
    assert "We accept the price of 470.00 GBP" in call["text"]
    assert world["draft"]["assurance"]["family_id"] == "final_offer_reply"
    assert call["requested_by"] == "sub-buyer-007", "the checks are accounted to the person who decided"


def test_it_is_gated_as_writing_a_draft_and_sends_nothing(world):
    _post("accept")
    ((action, context),) = world["gate"]
    assert action == "email.draft"
    assert context == {"unique_id": UID, "final_offer_decision": "accept"}
    assert world["draft"].get("sent_status") in (None, False)


def test_a_caller_the_gate_refuses_changes_nothing(world, monkeypatch):
    def refuse(action, principal, **kw):
        raise HTTPException(status_code=403, detail="role Viewer may not perform write")

    monkeypatch.setattr(approvals_router, "gate", refuse)
    assert _post("accept").status_code == 403
    assert world["draft"]["body"] == HOLDING


def test_an_ordinary_email_has_nothing_to_decide(world):
    world["draft"] = _draft(final_offer=None)
    res = _post("accept")
    assert res.status_code == 409 and "not a reply to a supplier's final offer" in res.json()["detail"]


def test_an_email_that_changed_since_it_was_opened_is_refused(world):
    res = _post("accept", content_hash="0" * 64)
    assert res.status_code == 409 and "changed after you opened it" in res.json()["detail"]
    assert world["draft"]["body"] == HOLDING


def test_the_version_on_screen_is_accepted(world):
    seen = content_hash(resolve_effective_content(world["draft"]))
    assert _post("accept", content_hash=seen).status_code == 200


def test_a_price_nobody_could_read_cannot_be_accepted_but_can_be_declined(world):
    world["draft"] = _draft(final_offer={**ADVICE, "offer": None})
    res = _post("accept")
    assert res.status_code == 409 and "could not be read" in res.json()["detail"]
    assert world["draft"]["body"] == HOLDING
    assert _post("decline").status_code == 200


def test_a_flagged_thread_refuses_the_decision_email(world):
    world["agent"].blocked = "a reply from this supplier has been flagged and has not been reviewed by a person"
    res = _post("accept")
    assert res.status_code == 409 and "flagged" in res.json()["detail"]
    assert world["draft"]["body"] == HOLDING


@pytest.mark.parametrize("word", ["approve", "", "ACCEPT ALL"])
def test_only_accept_or_decline_is_a_decision(world, word):
    assert _post(word).status_code == 422
    assert world["draft"]["body"] == HOLDING


def test_a_decision_that_could_not_be_saved_says_so(world):
    world["agent"].save = False
    res = _post("accept")
    assert res.status_code == 500 and "unchanged" in res.json()["detail"]


def test_a_missing_draft_is_not_found(world):
    world["draft"] = None
    assert _post("accept").status_code == 404
