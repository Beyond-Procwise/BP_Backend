"""A re-send that sends nothing must not be reported as a send.

`EmailDispatchService.send_draft` short-circuits a draft that has already gone out
(`_maybe_return_existing_dispatch`): it logs "Skipping duplicate supplier dispatch" and
returns ``sent: True`` with the message id of the ORIGINAL send, having put nothing on
the wire. That is correct behaviour -- it is what stops a retry emailing a supplier
twice -- but the flag is indistinguishable from a real send, and the caller here is a
human approval screen whose entire promise is that it never fabricates a success. A
second click on Send would have reported "Sent to …" and recorded a second audit entry
for an email that was never sent.

So the outcome is now stated outright rather than inferred: `dispatched_now` says what
THIS call did, `duplicate` says why, and `duplicate_note` says it in English. Nothing
about the shape of a message id has to be guessed at by a client.
"""
import logging

from services.email_dispatch_service import EmailDispatchService


def _service():
    """The method under test needs no database, no SES and no settings -- only the
    class's own small coercion helpers and a logger."""
    svc = object.__new__(EmailDispatchService)
    svc.logger = logging.getLogger("test.email_dispatch")
    return svc


def _call(svc, draft, existing_metadata):
    return svc._maybe_return_existing_dispatch(
        draft=draft,
        existing_dispatch_metadata=existing_metadata,
        backend_metadata={},
        dispatch_payload={"supplier_id": "PeopleFirst HR Solutions Ltd"},
        dispatch_payload_context=None,
        thread_headers=None,
        workflow_identifier="wf-1",
        unique_id="089580f2-PeopleFirst",
        recipient_list=["billing@peoplefirst.invalid"],
        sender_email="procurement@example.com",
        subject="RE: Negotiation",
        body="…",
        workflow_email_flag=True,
    )


def test_an_already_sent_draft_is_reported_as_a_duplicate_not_a_send(caplog):
    svc = _service()
    with caplog.at_level(logging.INFO, logger="test.email_dispatch"):
        result = _call(
            svc,
            {"sent_status": True, "supplier_id": "PeopleFirst HR Solutions Ltd"},
            {"message_id": "ses-original-0001"},
        )
    assert result is not None, "the short-circuit must fire for an already-sent draft"
    # Unchanged for every existing caller: the draft HAS been sent.
    assert result["sent"] is True
    # New, and the point: nothing was sent by THIS call.
    assert result["dispatched_now"] is False
    assert result["duplicate"] is True
    assert result["duplicate_note"]
    # The id belongs to the earlier send, which is exactly why a caller must not
    # treat its presence as proof of a send it just caused.
    assert result["message_id"] == "ses-original-0001"
    assert "Skipping duplicate supplier dispatch" in caplog.text


def test_the_duplicate_note_reads_as_english_and_names_nothing_internal():
    result = _call(_service(), {"sent": True}, {"message_id": "ses-original-0002"})
    note = result["duplicate_note"]
    assert "already been sent" in note
    assert "nothing was sent again" in note
    for token in ("proc.", "draft_rfq_emails", "sent_status", "message_id", "SES"):
        assert token not in note


def test_a_draft_that_has_not_been_sent_does_not_short_circuit():
    """The marker must not become a way to skip a real send: with nothing recorded as
    sent, this returns None and the caller goes on to dispatch for real."""
    assert _call(_service(), {"sent_status": False, "sent": False}, None) is None
    assert _call(_service(), {}, {}) is None


def test_a_sent_flag_with_no_message_id_still_dispatches():
    """Fail towards sending, not towards a silent skip: `sent` with no id anywhere means
    we cannot prove what went out, so the caller must not be told 'already sent'."""
    assert _call(_service(), {"sent_status": True}, None) is None


def test_the_markers_survive_the_response_model_to_the_client():
    """The router serialises through EmailDispatchResponse; a field it drops never
    reaches the UI, so the contract is pinned here rather than assumed."""
    from api.routers.workflows import EmailDispatchResponse

    payload = EmailDispatchResponse(
        success=True, unique_id="089580f2-PeopleFirst", sent=True,
        dispatched_now=False, duplicate=True,
        duplicate_note="This draft had already been sent, so nothing was sent again.",
        message_id="ses-original-0001", recipients=["billing@peoplefirst.invalid"],
        sender="procurement@example.com", subject="RE: Negotiation",
    ).model_dump()
    assert payload["dispatched_now"] is False
    assert payload["duplicate"] is True
    assert payload["duplicate_note"]

    # And a real send serialises the other way round, so a client can rely on the
    # field being present rather than on its absence meaning anything.
    real = EmailDispatchResponse(
        success=True, unique_id="u", sent=True, dispatched_now=True, duplicate=False,
        message_id="ses-new-0003", recipients=["a@b.co"], sender="s@b.co", subject="s",
    ).model_dump()
    assert real["dispatched_now"] is True
    assert real["duplicate"] is False
