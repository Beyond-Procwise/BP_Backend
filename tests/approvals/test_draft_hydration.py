"""C2: the approval side and the send side must hydrate the same draft.

The approval endpoint (approvals.py:_load_draft) and the pending list
(approval_store.list_pending_dispatch_approvals) used to read raw columns.
The send path (EmailDispatchService._hydrate_draft) started from the JSON
``payload`` blob and filled in columns only via ``setdefault`` -- so payload
won on the send side for subject/body/recipients/attachments, and the
approval side never applied that same precedence. A real draft in bp_testdb
reproduced it: the approval-side body and the send-side body were different
strings, so a genuinely approved draft could never pass its own
content-binding check.

Both sides now delegate to the one hydration function in
draft_hydration.py. These tests load a single, real-shaped row (payload
that disagrees with its own columns, exactly like the reproduction) through
both entry points and assert they agree -- not just that the shared
function exists, but that each caller actually uses it.
"""

from src.api.routers import approvals as approvals_router
from src.services.approval_content import content_hash
from src.services.draft_hydration import DRAFT_COLUMNS, resolve_effective_content
from src.services.email_dispatch_service import EmailDispatchService


class _FakeSettings:
    ses_default_sender = "sender@ourcompany.com"


class _FakeAgentNickForHydration:
    """EmailDispatchService._hydrate_draft only ever touches self.settings."""

    settings = _FakeSettings()


class _FakeCursor:
    def __init__(self, row):
        self._row = dict(row)

    def execute(self, sql, params):
        pass

    def fetchone(self):
        return dict(self._row)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeConn:
    def __init__(self, row):
        self._row = row

    def cursor(self, cursor_factory=None):
        return _FakeCursor(self._row)


def _row(**over):
    """A row shaped like the real proc.draft_rfq_emails SELECT, payload
    disagreeing with its own columns -- exactly the shape that reproduced C2
    against the live draft in bp_testdb."""

    base = dict.fromkeys(DRAFT_COLUMNS, None)
    base.update(
        {
            "id": 1,
            "rfq_id": "RFQ-1",
            "supplier_id": "SUP-1",
            "subject": "stale column subject",
            "body": "stale column body",
            "sent": False,
            "recipient_email": "buyer@supplier-b.com",
            "payload": {
                "subject": "the real, current subject",
                "body": "the real, current body",
                "recipients": ["buyer@supplier-b.com"],
            },
            "workflow_id": "WF-1",
            "unique_id": "PROC-WF-1",
            "attachments": None,
        }
    )
    base.update(over)
    return base


def _send_side_hydrate(row):
    """The exact tuple EmailDispatchService._fetch_latest_draft would
    produce, run through the real (unbound) _hydrate_draft method."""

    row_tuple = tuple(row[col] for col in DRAFT_COLUMNS)
    return EmailDispatchService._hydrate_draft(_FakeAgentNickForHydration(), row_tuple)


def _approval_side_hydrate(row):
    return approvals_router._load_draft("PROC-WF-1", conn=_FakeConn(row))


def test_payload_wins_over_stale_columns_on_both_sides():
    row = _row()
    send_side = _send_side_hydrate(row)
    approval_side = _approval_side_hydrate(row)

    assert send_side["subject"] == "the real, current subject"
    assert approval_side["subject"] == "the real, current subject"
    assert send_side["body"] == "the real, current body"
    assert approval_side["body"] == "the real, current body"


def test_both_sides_hash_the_same_resolved_material():
    row = _row()
    send_side = _send_side_hydrate(row)
    approval_side = _approval_side_hydrate(row)

    send_hash = content_hash(resolve_effective_content(send_side))
    approval_hash = content_hash(resolve_effective_content(approval_side))
    assert send_hash == approval_hash


def test_multi_recipient_case_recipient_email_holds_only_the_first():
    """proc.draft_rfq_emails.recipient_email is singular; payload.recipients
    can carry the full list. Both sides must resolve to the full list, or
    approving what an approver saw (all recipients) would not match what
    the send path would transmit (fewer recipients, from the stale column)."""

    row = _row(
        recipient_email="first@supplier-b.com",
        payload={
            "subject": "Request for quotation",
            "body": "Please quote for 100 units.",
            "recipients": ["first@supplier-b.com", "second@supplier-b.com"],
        },
    )
    send_side = _send_side_hydrate(row)
    approval_side = _approval_side_hydrate(row)

    assert sorted(send_side["recipients"]) == [
        "first@supplier-b.com",
        "second@supplier-b.com",
    ]
    assert sorted(approval_side["recipients"]) == [
        "first@supplier-b.com",
        "second@supplier-b.com",
    ]

    send_hash = content_hash(resolve_effective_content(send_side))
    approval_hash = content_hash(resolve_effective_content(approval_side))
    assert send_hash == approval_hash
