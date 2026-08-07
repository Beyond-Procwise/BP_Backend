"""What was approved must be what gets sent.

Without this, an approval is standing permission on a mutable object: approve
a routine RFQ, someone edits the body to carry a competitor's price, and the
original approval still releases it.
"""

from src.services.approval_content import content_hash


def _draft(**over):
    base = {
        "unique_id": "PROC-WF-1",
        "recipients": ["buyer@supplier-b.com"],
        "subject": "Request for quotation",
        "body": "Please quote for 100 units.",
        "attachments": [{"filename": "spec.pdf"}],
    }
    base.update(over)
    return base


def test_the_same_draft_hashes_the_same_way():
    assert content_hash(_draft()) == content_hash(_draft())


def test_editing_the_body_changes_the_hash():
    assert content_hash(_draft()) != content_hash(
        _draft(body="Supplier B quoted 12,450.00.")
    )


def test_changing_a_recipient_changes_the_hash():
    assert content_hash(_draft()) != content_hash(
        _draft(recipients=["someone.else@elsewhere.com"])
    )


def test_changing_the_subject_changes_the_hash():
    assert content_hash(_draft()) != content_hash(_draft(subject="Revised"))


def test_adding_an_attachment_changes_the_hash():
    assert content_hash(_draft()) != content_hash(
        _draft(attachments=[{"filename": "spec.pdf"}, {"filename": "po.pdf"}])
    )


def test_recipient_order_does_not_change_the_hash():
    """Two recipients in a different order are the same set of recipients."""
    a = _draft(recipients=["a@supplier-b.com", "b@supplier-b.com"])
    b = _draft(recipients=["b@supplier-b.com", "a@supplier-b.com"])
    assert content_hash(a) == content_hash(b)


def test_it_hashes_the_recipients_the_send_path_will_use():
    """draft_rfq_emails has recipient_email, not recipients.

    Hashing a raw column instead of the resolved list would let the approved
    set and the sent set diverge.
    """
    from_singular = _draft(recipients=None, receiver="buyer@supplier-b.com")
    assert content_hash(from_singular) == content_hash(_draft())


import pytest


@pytest.mark.parametrize("bad", [None, "a string", ["a", "b"], 42])
def test_a_draft_that_is_not_a_mapping_still_hashes(bad):
    """The send path calls this. Crashing here turns a refusal into an outage."""
    assert isinstance(content_hash(bad), str)


def test_a_failure_inside_recipient_resolution_still_hashes(monkeypatch):
    """Force the guarded branch -- nothing else in the suite reaches it."""
    import src.services.approval_content as mod

    def explode(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        "src.services.email_dispatch_guard.resolve_recipients", explode
    )
    assert isinstance(content_hash({"subject": "s", "body": "b"}), str)
