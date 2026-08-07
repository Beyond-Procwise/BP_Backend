"""An approval covers the email that was approved, not the draft as a moving target."""

from src.services import email_dispatch_guard as guard
from src.services.approval_content import content_hash
from tests.guardrails.test_send_path_gate import base_kwargs


DRAFT = {
    "unique_id": "PROC-WF-1",
    "rfq_id": "RFQ-1",
    "workflow_id": "WF-1",
    "supplier_id": "SUP-1",
    "recipients": ["buyer@supplier-b.com"],
    "subject": "Request for quotation",
    "body": "Please quote for 100 units.",
}


def _approval(hash_value):
    return {
        "approval_id": 1,
        "status": "approved",
        "actioned_by": "buyer@ourcompany.com",
        "deal_id": None,
        "grounding": {"content_hash": hash_value},
    }


def test_an_unchanged_draft_is_allowed():
    """So the check cannot be satisfied by simply always denying."""
    decision = guard.check_dispatch(
        **base_kwargs(
            draft=dict(DRAFT),
            body=DRAFT["body"],
            subject=DRAFT["subject"],
            approval_lookup=lambda **_: _approval(content_hash(DRAFT)),
        )
    )
    assert decision.allowed is True, decision.reason


def test_an_edited_body_is_refused():
    edited = dict(DRAFT, body="Supplier B quoted 12,450.00.")
    decision = guard.check_dispatch(
        **base_kwargs(
            draft=edited,
            body=edited["body"],
            subject=edited["subject"],
            approval_lookup=lambda **_: _approval(content_hash(DRAFT)),
        )
    )
    assert decision.allowed is False
    assert "changed" in decision.reason.lower()


def test_a_changed_recipient_is_refused():
    edited = dict(DRAFT, recipients=["someone@elsewhere.com"])
    decision = guard.check_dispatch(
        **base_kwargs(
            draft=edited,
            body=edited["body"],
            subject=edited["subject"],
            approval_lookup=lambda **_: _approval(content_hash(DRAFT)),
        )
    )
    assert decision.allowed is False


def test_an_approval_with_no_hash_is_refused():
    """An approval predating content binding cannot vouch for content."""
    decision = guard.check_dispatch(
        **base_kwargs(
            draft=dict(DRAFT),
            body=DRAFT["body"],
            subject=DRAFT["subject"],
            approval_lookup=lambda **_: _approval(None),
        )
    )
    assert decision.allowed is False
