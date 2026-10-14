"""A reply answers an email only if it came after it, and only once.

Live, 2026-10-14: the supplier's 06:44 reply to the first test email was matched
again -- "match_confidence 1.00" on the sender's address alone -- to a second
email dispatched at 07:26. The supplier's real reply to the second email then
had nothing left to answer and was ignored. SES replaces our Message-ID, so a
Gmail reply's In-Reply-To never names ours and matching falls back to scoring.
Two facts the scoring never looked at:
- a reply cannot answer an email sent after it arrived;
- a message already stored as the answer to one email is not a new reply.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

import services.email_watcher as ew

T0 = datetime(2026, 10, 14, 7, 26, 32, tzinfo=timezone.utc)


def _watcher(monkeypatch, already=()):
    stored = []
    monkeypatch.setattr(ew, "_store_matched_response", lambda row: stored.append(row))
    monkeypatch.setattr(ew.tracking_repo, "mark_response", lambda **k: None)
    monkeypatch.setattr(ew.workflow_round_response_repo, "mark_response_received", lambda **k: None)
    monkeypatch.setattr(ew.supplier_response_repo, "message_already_matched",
                        lambda message_id, workflow_id, unique_id: message_id in already)
    watcher = ew.EmailWatcher.__new__(ew.EmailWatcher)
    watcher.config = SimpleNamespace(match_threshold=0.45)
    return watcher, stored


def _tracker():
    tracker = ew.WorkflowTracker(workflow_id="wf-b")
    tracker.register_dispatches([ew.EmailDispatchRecord(
        unique_id="PROC-WF-NEW", supplier_id="SUP-1", supplier_email="nick@gmail.test",
        message_id="<ours@procwise.test>", subject="Re: Pricing Discussion", workflow_id="wf-b", dispatched_at=T0)])
    return tracker


def _reply(message_id, received_at):
    return ew.EmailResponse(unique_id=None, supplier_id="SUP-1", supplier_email="nick@gmail.test",
                            from_address="nick@gmail.test", message_id=message_id,
                            subject="Re: Pricing Discussion", body="I find your tone rude.", received_at=received_at,
                            in_reply_to=("<ses-rewritten@amazonses.test>",))


def test_a_reply_that_arrived_before_the_email_was_sent_does_not_answer_it(monkeypatch):
    watcher, stored = _watcher(monkeypatch)
    rows = watcher._match_responses(_tracker(), [_reply("old@mail.gmail.test", T0 - timedelta(minutes=42))])
    assert rows == [] and stored == []


def test_a_message_already_matched_elsewhere_is_not_a_new_reply(monkeypatch):
    watcher, stored = _watcher(monkeypatch, already={"dup@mail.gmail.test"})
    rows = watcher._match_responses(_tracker(), [_reply("dup@mail.gmail.test", T0 + timedelta(minutes=2))])
    assert rows == [] and stored == []


def test_the_real_reply_still_matches(monkeypatch):
    watcher, stored = _watcher(monkeypatch)
    rows = watcher._match_responses(_tracker(), [_reply("new@mail.gmail.test", T0 + timedelta(minutes=2))])
    assert [r.unique_id for r in rows] == ["PROC-WF-NEW"]


def test_the_old_reply_does_not_take_the_place_of_the_new_one(monkeypatch):
    """Both are in the mailbox (it is searched by date); only the new one answers the new email."""

    watcher, stored = _watcher(monkeypatch)
    rows = watcher._match_responses(_tracker(), [_reply("old@mail.gmail.test", T0 - timedelta(minutes=42)),
                                                 _reply("new@mail.gmail.test", T0 + timedelta(minutes=2))])
    assert [r.response_message_id for r in rows] == ["new@mail.gmail.test"]
