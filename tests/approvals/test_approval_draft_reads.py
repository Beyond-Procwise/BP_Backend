"""What an approver sees before approving: the waiting drafts, and one email in full.

``GET /approvals/pending`` lists only drafts nobody has approved, and only by id and
subject. A screen built on it could not show the email being approved, could not show a
draft that is approved and waiting to be sent, and could not say that an approval no
longer covers the draft because the draft changed afterwards. These two reads close that.
"""

from datetime import datetime, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routers import approvals as approvals_router
from src.services.approval_content import content_hash
from src.services.draft_hydration import resolve_effective_content

MARKER = "<!-- PROCWISE_MARKER:TRACKING:PROC-WF-AAAA11112222|SUPPLIER:SUP-A|TOKEN:C2E75A83D69C -->"
BODY = f"<p>Dear Nadia,</p><p>We can offer 455.00 GBP per unit.<br>Please reply by 17 October 2026.</p>\n\n{MARKER}"


class _Principal:
    subject = "sub-approver-001"
    email = "approver@ourcompany.com"
    claims = {"cognito:groups": ["bp-buyers"]}


@pytest.fixture(autouse=True)
def _stub_audit_write(monkeypatch):
    monkeypatch.setattr(approvals_router, "record_action_or_fail", lambda **kwargs: None)


def _client(principal=_Principal()):
    app = FastAPI()
    app.include_router(approvals_router.router)
    app.dependency_overrides[approvals_router.require_user] = lambda: principal
    return TestClient(app)


def _allow(*a, **k):
    return approvals_router.guardrail.Decision(allowed=True, reason="ok")


def _deny(*a, **k):
    return approvals_router.guardrail.Decision(
        allowed=False, reason="role Viewer may not perform approve_email"
    )


def _draft(**over):
    draft = {
        "unique_id": "PROC-WF-AAAA11112222",
        "workflow_id": "wf-1",
        "rfq_id": None,
        "supplier_id": "SUP-A",
        "supplier_name": "Ashcroft Associates",
        "recipient_email": "nadia@ashcroft.example",
        "recipients": ["nadia@ashcroft.example"],
        "subject": "Re: Pricing Discussion",
        "body": BODY,
        "requested_by": None,
        "attachments": None,
    }
    draft.update(over)
    return draft


def _approval_of(draft, **over):
    row = {
        "approval_id": 41,
        "status": "approved",
        "actioned_by": "sub-approver-002",
        "actioned_at": datetime(2026, 10, 11, 9, 0, tzinfo=timezone.utc),
        "grounding": {"content_hash": content_hash(resolve_effective_content(draft))},
    }
    row.update(over)
    return row


def _listing(monkeypatch, entries):
    monkeypatch.setattr(approvals_router.guardrail, "authorize", _allow)
    monkeypatch.setattr(
        approvals_router.approval_store, "list_unsent_drafts", lambda **k: entries
    )


# --- GET /approvals/drafts -------------------------------------------------------------


def test_drafts_an_unauthenticated_caller_sees_nothing(monkeypatch):
    called = {"listed": False}
    monkeypatch.setattr(
        approvals_router.approval_store, "list_unsent_drafts",
        lambda **k: called.__setitem__("listed", True) or [],
    )

    response = _client(principal=None).get("/approvals/drafts")

    assert response.status_code in (401, 403)
    assert called["listed"] is False


def test_drafts_a_denied_capability_lists_nothing(monkeypatch):
    called = {"listed": False}
    monkeypatch.setattr(
        approvals_router.approval_store, "list_unsent_drafts",
        lambda **k: called.__setitem__("listed", True) or [],
    )
    monkeypatch.setattr(approvals_router.guardrail, "authorize", _deny)

    response = _client().get("/approvals/drafts")

    assert response.status_code == 403
    assert called["listed"] is False


def test_a_draft_nobody_has_approved_is_listed_as_waiting(monkeypatch):
    draft = _draft()
    created = datetime(2026, 10, 10, 18, 41, tzinfo=timezone.utc)
    _listing(monkeypatch, [{"draft": draft, "approval": None, "created_on": created}])

    body = _client().get("/approvals/drafts").json()

    assert body["count"] == 1
    row = body["drafts"][0]
    assert row["unique_id"] == "PROC-WF-AAAA11112222"
    assert row["supplier_name"] == "Ashcroft Associates"
    assert row["recipients"] == ["nadia@ashcroft.example"]
    assert row["subject"] == "Re: Pricing Discussion"
    assert row["created_on"].startswith("2026-10-10T18:41")
    assert row["content_hash"] == content_hash(resolve_effective_content(draft))
    assert row["approval"] is None


def test_an_approved_draft_names_its_approver_and_is_ready_to_send(monkeypatch):
    draft = _draft()
    _listing(monkeypatch, [{"draft": draft, "approval": _approval_of(draft), "created_on": None}])

    row = _client().get("/approvals/drafts").json()["drafts"][0]

    assert row["approval"]["approval_id"] == 41
    assert row["approval"]["actioned_by"] == "sub-approver-002"
    assert row["approval"]["actioned_at"].startswith("2026-10-11T09:00")
    assert row["approval"]["covers_this_version"] is True


def test_an_approval_of_an_earlier_version_does_not_cover_the_draft(monkeypatch):
    approved = _draft()
    edited = _draft(body=BODY.replace("455.00", "405.00"))
    assert content_hash(resolve_effective_content(approved)) != content_hash(
        resolve_effective_content(edited)
    )  # the precondition: the edit really moves the hash
    _listing(monkeypatch, [{"draft": edited, "approval": _approval_of(approved), "created_on": None}])

    row = _client().get("/approvals/drafts").json()["drafts"][0]

    assert row["approval"]["covers_this_version"] is False


def test_an_approval_that_recorded_no_content_does_not_cover_the_draft(monkeypatch):
    draft = _draft()
    _listing(monkeypatch, [{"draft": draft, "approval": _approval_of(draft, grounding={}), "created_on": None}])

    row = _client().get("/approvals/drafts").json()["drafts"][0]

    assert row["approval"]["covers_this_version"] is False


def test_the_list_carries_no_email_text(monkeypatch):
    draft = _draft()
    _listing(monkeypatch, [{"draft": draft, "approval": None, "created_on": None}])

    raw = _client().get("/approvals/drafts").text

    assert "455.00" not in raw and "body" not in raw


# --- GET /approvals/drafts/{unique_id} -------------------------------------------------


def _one(monkeypatch, draft, approval=None):
    monkeypatch.setattr(approvals_router.guardrail, "authorize", _allow)
    monkeypatch.setattr(approvals_router, "_load_draft", lambda uid, conn=None: draft)
    monkeypatch.setattr(
        approvals_router.approval_store, "find_dispatch_approval", lambda **k: approval
    )


def test_one_draft_is_read_as_the_words_the_supplier_would_see(monkeypatch):
    draft = _draft()
    _one(monkeypatch, draft)

    body = _client().get("/approvals/drafts/PROC-WF-AAAA11112222").json()

    assert body["body_text"] == (
        "Dear Nadia,\nWe can offer 455.00 GBP per unit.\nPlease reply by 17 October 2026."
    )
    assert "PROCWISE_MARKER" not in body["body_text"] and "<" not in body["body_text"]
    assert body["content_hash"] == content_hash(resolve_effective_content(draft))
    assert body["approval"] is None


def test_one_draft_carries_its_approval(monkeypatch):
    draft = _draft()
    _one(monkeypatch, draft, _approval_of(draft))

    body = _client().get("/approvals/drafts/PROC-WF-AAAA11112222").json()

    assert body["approval"]["actioned_by"] == "sub-approver-002"
    assert body["approval"]["covers_this_version"] is True


def test_the_requester_is_named_so_a_screen_can_explain_a_self_approval_refusal(monkeypatch):
    _one(monkeypatch, _draft(requested_by="sub-approver-001"))

    body = _client().get("/approvals/drafts/PROC-WF-AAAA11112222").json()

    assert body["requested_by"] == "sub-approver-001"
    assert body["requested_by_you"] is True


def test_a_missing_or_already_sent_draft_is_not_found(monkeypatch):
    _one(monkeypatch, None)

    assert _client().get("/approvals/drafts/PROC-WF-NOPE").status_code == 404


def test_one_draft_a_denied_capability_reads_nothing(monkeypatch):
    called = {"loaded": False}
    monkeypatch.setattr(
        approvals_router, "_load_draft",
        lambda uid, conn=None: called.__setitem__("loaded", True) or _draft(),
    )
    monkeypatch.setattr(approvals_router.guardrail, "authorize", _deny)

    response = _client().get("/approvals/drafts/PROC-WF-AAAA11112222")

    assert response.status_code == 403
    assert called["loaded"] is False


def test_one_draft_an_unauthenticated_caller_reads_nothing(monkeypatch):
    called = {"loaded": False}
    monkeypatch.setattr(
        approvals_router, "_load_draft",
        lambda uid, conn=None: called.__setitem__("loaded", True) or _draft(),
    )

    response = _client(principal=None).get("/approvals/drafts/PROC-WF-AAAA11112222")

    assert response.status_code in (401, 403)
    assert called["loaded"] is False


# --- POST /approvals/dispatch/{unique_id}: approving the version that was read ---------


def _approving(monkeypatch, draft):
    recorded = []
    monkeypatch.setattr(approvals_router.guardrail, "authorize", _allow)
    monkeypatch.setattr(approvals_router, "_load_draft", lambda uid, conn=None: draft)
    monkeypatch.setattr(
        approvals_router.approval_store, "record_approval",
        lambda **kwargs: recorded.append(kwargs) or 99,
    )
    return recorded


def test_approving_the_version_that_was_read_is_recorded(monkeypatch):
    draft = _draft()
    recorded = _approving(monkeypatch, draft)
    seen = content_hash(resolve_effective_content(draft))

    response = _client().post("/approvals/dispatch/PROC-WF-AAAA11112222", json={"content_hash": seen})

    assert response.status_code == 200
    assert recorded and recorded[0]["grounding_extra"]["content_hash"] == seen


def test_approving_a_draft_that_changed_since_it_was_read_is_refused(monkeypatch):
    read = _draft()
    now = _draft(body=BODY.replace("455.00", "405.00"))
    recorded = _approving(monkeypatch, now)
    seen = content_hash(resolve_effective_content(read))

    response = _client().post("/approvals/dispatch/PROC-WF-AAAA11112222", json={"content_hash": seen})

    assert response.status_code == 409
    assert "changed after you opened it" in response.json()["detail"]
    assert recorded == [], "an approval was written for words the approver did not read"


def test_an_approval_that_names_no_version_still_approves_the_draft_as_it_stands(monkeypatch):
    recorded = _approving(monkeypatch, _draft())

    assert _client().post("/approvals/dispatch/PROC-WF-AAAA11112222", json={}).status_code == 200
    assert len(recorded) == 1


# ---- a reply to a supplier's final offer carries its recommendation -----------

_ADVICE = {
    "recommendation": "escalate",
    "reason": "The supplier's final offer of 470.00 GBP is 15.00 GBP (3.3%) above the target of 455.00 GBP. "
              "A person needs to decide whether to accept it or walk away.",
    "offer": 470.0, "currency": "GBP", "limit": 455.0, "limit_kind": "target", "over_by": 15.0, "over_pct": 3.3,
}


def _read(monkeypatch, draft):
    _one(monkeypatch, draft)
    return _client().get("/approvals/drafts/PROC-WF-AAAA11112222").json()


def test_a_final_offer_reply_shows_the_recommendation_to_its_approver(monkeypatch):
    """Without it the approver sees only a holding email, and not why it is one."""
    assert _read(monkeypatch, _draft(final_offer=_ADVICE))["final_offer"] == _ADVICE


def test_the_list_shows_which_drafts_answer_a_final_offer(monkeypatch):
    _listing(monkeypatch, [{"draft": _draft(final_offer=_ADVICE), "approval": None, "created_on": None}])
    (row,) = _client().get("/approvals/drafts").json()["drafts"]
    assert row["final_offer"]["recommendation"] == "escalate"


def test_an_ordinary_draft_carries_no_recommendation(monkeypatch):
    assert _read(monkeypatch, _draft())["final_offer"] is None


def test_a_malformed_recommendation_is_not_passed_on(monkeypatch):
    assert _read(monkeypatch, _draft(final_offer="accept"))["final_offer"] is None
