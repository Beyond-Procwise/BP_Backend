"""C1: an approval binds to what is transmitted, not to the stored row.

check_dispatch used to hash `content_hash(draft)` -- the *stored* row -- even
though send_draft can transmit a caller-supplied subject_override/
body_override straight from an HTTP request body (all three dispatch
endpoints in workflows.py feed those directly). The stored row never
changed, so its approved hash still matched, and an override sailed through
unchecked. That was verbatim the attack spec Sec.5.3 says this feature exists
to prevent: approve a routine RFQ, then send it with a competitor's price in
the body.

Every test here keeps `draft` (the stored row) untouched and instead varies
the `recipients`/`subject`/`body` kwargs check_dispatch receives -- those are
send_draft's *resolved* values (post-override), which is what the fix must
hash. A test that edits `draft` instead of these kwargs conflates the two and
proves nothing about this bug (that was the whole reason C1 shipped
unnoticed: every prior test here passed body=DRAFT["body"]).
"""

from src.services import email_dispatch_guard as guard
from src.services.approval_content import content_hash
from src.services.draft_hydration import resolve_effective_content
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

# What an approver actually approved: the resolved material for DRAFT with no
# overrides, computed the identical way the approval endpoint computes it
# (see approvals.py:approve_dispatch and C2).
APPROVED_HASH = content_hash(resolve_effective_content(DRAFT))


def _approval(hash_value=APPROVED_HASH):
    return {
        "approval_id": 1,
        "status": "approved",
        "actioned_by": "buyer@ourcompany.com",
        "deal_id": None,
        "grounding": {"content_hash": hash_value},
    }


def _kwargs(**over):
    kwargs = dict(
        draft=dict(DRAFT),  # the stored row -- never mutated by these tests
        recipients=DRAFT["recipients"],
        subject=DRAFT["subject"],
        body=DRAFT["body"],
        approval_lookup=lambda **_: _approval(),
    )
    kwargs.update(over)
    return base_kwargs(**kwargs)


def test_no_override_still_sends():
    """The baseline: nothing was overridden, so the approved hash matches."""
    decision = guard.check_dispatch(**_kwargs())
    assert decision.allowed is True, decision.reason


def test_a_byte_identical_override_still_sends():
    """An override that merely repeats the approved content is not itself
    suspicious -- proves the fix is not simply 'deny whenever an override is
    present'."""
    decision = guard.check_dispatch(
        **_kwargs(body="Please quote for 100 units.", subject="Request for quotation")
    )
    assert decision.allowed is True, decision.reason


def test_an_override_that_differs_from_the_approved_body_is_refused():
    """The exact attack this closes: the stored draft is untouched, but what
    would actually be transmitted has changed."""
    decision = guard.check_dispatch(
        **_kwargs(body="Our competitor quoted 4.20 per unit.")
    )
    assert decision.allowed is False
    assert "changed" in decision.reason.lower()


def test_an_override_that_differs_from_the_approved_subject_is_refused():
    decision = guard.check_dispatch(**_kwargs(subject="Revised quotation"))
    assert decision.allowed is False
    assert "changed" in decision.reason.lower()


def test_an_override_that_differs_from_the_approved_recipients_is_refused():
    decision = guard.check_dispatch(**_kwargs(recipients=["someone@elsewhere.com"]))
    assert decision.allowed is False
    assert "changed" in decision.reason.lower()


def test_an_approval_with_no_hash_is_refused():
    """An approval predating content binding cannot vouch for content."""
    decision = guard.check_dispatch(**_kwargs(approval_lookup=lambda **_: _approval(None)))
    assert decision.allowed is False
