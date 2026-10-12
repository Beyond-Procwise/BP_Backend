"""A draft that was replaced before it was sent is not an email anyone is waiting to send.

The mailbox watcher will not read replies for a workflow until every expected
email has gone out, and "expected" was every draft row the workflow had. When a
counter-offer was redrafted, the earlier version stayed behind unsent, so the
workflow never counted as fully sent and the supplier's reply to the version
that DID go out was left unread in the mailbox.

An unsent draft with a later draft to the same supplier has been superseded and
is dropped from the expected set. Nothing else changes: an unsent draft that is
the supplier's latest is still waited for, and so is every supplier's own draft.
"""
from contextlib import contextmanager

import pytest

from repositories import draft_rfq_emails_repo as repo


class _Cursor:
    def __init__(self, rows):
        self.rows = rows

    def execute(self, sql, params=None):
        self.sql = sql

    def fetchall(self):
        return list(self.rows)

    def close(self):
        pass


class _Conn:
    def __init__(self, rows):
        self.rows = rows

    def cursor(self):
        return _Cursor(self.rows)


@pytest.fixture
def drafts(monkeypatch):
    """Rows as the query returns them: (unique_id, supplier_id, dispatched_at, sent, id)."""

    def _install(rows):
        @contextmanager
        def _get_conn():
            yield _Conn(rows)

        monkeypatch.setattr(repo, "get_conn", _get_conn)

    return _install


def _expected(workflow_id="wf"):
    ids, suppliers, _ = repo.expected_unique_ids_and_last_dispatch(workflow_id=workflow_id)
    return ids, suppliers


def test_a_redrafted_email_leaves_only_the_version_that_was_sent(drafts):
    """The live case: draft 7921 was replaced by 7954, and 7954 was sent."""
    drafts([
        ("UID-OLD", "SUP-1", None, False, 7921),
        ("UID-NEW", "SUP-1", "2026-10-12T07:28:06+00:00", True, 7954),
    ])
    ids, suppliers = _expected()
    assert ids == {"UID-NEW"}
    assert suppliers == {"UID-NEW": "SUP-1"}


def test_an_unsent_draft_that_is_the_latest_is_still_waited_for(drafts):
    """Round 1 went out; round 2 is drafted and not yet sent. Both are expected."""
    drafts([
        ("UID-R1", "SUP-1", "2026-10-12T07:28:06+00:00", True, 10),
        ("UID-R2", "SUP-1", None, False, 11),
    ])
    assert _expected()[0] == {"UID-R1", "UID-R2"}


def test_another_suppliers_unsent_draft_is_still_waited_for(drafts):
    """A batch to several suppliers still waits until each has been sent."""
    drafts([
        ("UID-A", "SUP-A", "2026-10-12T07:28:06+00:00", True, 10),
        ("UID-B", "SUP-B", None, False, 9),
    ])
    assert _expected()[0] == {"UID-A", "UID-B"}


def test_a_sent_email_is_never_dropped_however_old(drafts):
    drafts([
        ("UID-R1", "SUP-1", "2026-10-10T07:00:00+00:00", True, 10),
        ("UID-R2", "SUP-1", "2026-10-12T07:28:06+00:00", True, 11),
    ])
    assert _expected()[0] == {"UID-R1", "UID-R2"}


def test_a_draft_with_no_supplier_is_never_judged_superseded(drafts):
    """With no supplier on the row there is nothing to say what replaced it."""
    drafts([
        ("UID-X", None, None, False, 5),
        ("UID-Y", None, "2026-10-12T07:28:06+00:00", True, 6),
    ])
    assert _expected()[0] == {"UID-X", "UID-Y"}


def test_several_abandoned_versions_all_drop(drafts):
    drafts([
        ("UID-1", "SUP-1", None, False, 1),
        ("UID-2", "SUP-1", None, False, 2),
        ("UID-3", "SUP-1", None, False, 3),
    ])
    assert _expected()[0] == {"UID-3"}
