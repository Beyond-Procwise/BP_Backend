"""An email that has gone out is never turned back into a draft.

Live, 2026-10-14: the negotiation agent handed the email agent the reference of
the email already sent to the supplier as the id for its next counter. The
drafting agent reused it, and ``_store_draft``'s upsert on (workflow_id,
unique_id) set ``sent`` back to false and replaced the body -- the record of
what the supplier was sent was gone. The reply to that email then counted as an
answer to the new draft, so the watcher ran the negotiation a second time.

Two rules, at the two places that decide:
- a new draft never takes the id of an email already sent;
- the save never overwrites a row that has been sent.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from agents.email_drafting_agent import EmailDraftingAgent


class _Cursor:
    def __init__(self, sent_ids, returning=(1,), fail=False):
        self.sent_ids, self.returning, self.fail = set(sent_ids), returning, fail
        self.sql = []
        self._last = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        if self.fail:
            raise RuntimeError("database unreachable")
        self.sql.append(sql)
        if "sent IS TRUE" in sql:
            self._last = (1,) if params and params[0] in self.sent_ids else None
        else:
            self._last = self.returning

    def fetchone(self):
        return self._last


class _Conn:
    def __init__(self, cursor):
        self._cursor = cursor

    def cursor(self):
        return self._cursor

    def commit(self):
        pass

    def rollback(self):
        pass


def _agent(cursor):
    agent = EmailDraftingAgent()

    @contextmanager
    def _connect():
        yield _Conn(cursor)

    agent.agent_nick = SimpleNamespace(get_db_connection=_connect, settings=SimpleNamespace())
    return agent


def test_a_new_draft_does_not_take_the_id_of_a_sent_email():
    agent = _agent(_Cursor(sent_ids={"PROC-WF-SENT00000001"}))

    uid = agent._unique_id_for_new_draft("PROC-WF-SENT00000001", "wf-1", "SUP-1")

    assert uid and uid != "PROC-WF-SENT00000001"


def test_an_unsent_draft_keeps_its_id():
    """Redrafting an email nobody has sent yet updates that draft, as before."""

    agent = _agent(_Cursor(sent_ids=set()))

    assert agent._unique_id_for_new_draft("PROC-WF-UNSENT0000001", "wf-1", "SUP-1") == "PROC-WF-UNSENT0000001"


def test_when_it_cannot_be_checked_the_draft_gets_its_own_id():
    """An unreadable table is no licence to reuse an id that may belong to a sent email."""

    agent = _agent(_Cursor(sent_ids=set(), fail=True))

    uid = agent._unique_id_for_new_draft("PROC-WF-MAYBESENT0001", "wf-1", "SUP-1")

    assert uid and uid != "PROC-WF-MAYBESENT0001"


def test_no_id_given_gets_a_generated_one():
    agent = _agent(_Cursor(sent_ids=set()))

    assert agent._unique_id_for_new_draft(None, "wf-1", "SUP-1")


def test_the_save_never_overwrites_a_sent_row(monkeypatch):
    """The upsert matched a sent row and returned nothing: nothing is captured under the sent email's id."""

    cursor = _Cursor(sent_ids={"PROC-WF-SENT00000001"}, returning=None)
    agent = _agent(cursor)
    monkeypatch.setattr(agent, "_ensure_table_exists", lambda conn: None)
    captured = []
    monkeypatch.setattr(agent, "_capture_draft", lambda d: captured.append(d))

    draft = {"supplier_id": "SUP-1", "unique_id": "PROC-WF-SENT00000001", "workflow_id": "wf-1",
             "subject": "Re: Pricing", "body": "<p>new words</p>", "recipients": ["a@b.test"],
             "assurance": {"status": "verified"}}
    agent._store_draft(draft)

    assert not [s for s in cursor.sql if "INSERT INTO proc.draft_rfq_emails" in s]
    assert captured == []
    assert "id" not in draft


@pytest.mark.parametrize("returning", [(42,)])
def test_an_unsent_row_is_still_saved_and_captured(monkeypatch, returning):
    cursor = _Cursor(sent_ids=set(), returning=returning)
    agent = _agent(cursor)
    monkeypatch.setattr(agent, "_ensure_table_exists", lambda conn: None)
    captured = []
    monkeypatch.setattr(agent, "_capture_draft", lambda d: captured.append(d))

    draft = {"supplier_id": "SUP-1", "unique_id": "PROC-WF-UNSENT0000001", "workflow_id": "wf-1",
             "subject": "Re: Pricing", "body": "<p>words</p>", "recipients": ["a@b.test"]}
    agent._store_draft(draft)

    assert draft["id"] == 42
    assert len(captured) == 1


def test_the_upsert_itself_cannot_overwrite_a_sent_row(monkeypatch):
    """A backstop for a send that lands between the check and the save."""

    cursor = _Cursor(sent_ids=set(), returning=(7,))
    agent = _agent(cursor)
    monkeypatch.setattr(agent, "_ensure_table_exists", lambda conn: None)
    monkeypatch.setattr(agent, "_capture_draft", lambda d: None)
    agent._store_draft({"supplier_id": "SUP-1", "unique_id": "PROC-WF-X", "workflow_id": "wf-1",
                        "subject": "s", "body": "<p>b</p>", "recipients": ["a@b.test"]})
    upsert = [s for s in cursor.sql if "INSERT INTO proc.draft_rfq_emails" in s][0]
    assert "WHERE proc.draft_rfq_emails.sent IS NOT TRUE" in upsert
