"""Tests for POST /workflows/email/prepare.

Background: the report's email panel edits {recipients, subject, body} for a
deal but has no unique_id/rfq_id of its own, so POST /workflows/email 400s
(EmailDispatchService.resolve_workflow_id needs an identifier that already
resolves to a *persisted* draft). ``prepare_email_draft`` closes that gap by
persisting the panel's edit into proc.draft_rfq_emails (via
EmailDraftingAgent._store_draft, the same helper the drafting agent itself
uses) and handing back the unique_id the panel can then pass to
POST /workflows/email.

These tests exercise the persist path ONLY, against a fake DB connection.
They never call send_email / EmailDispatchService.send_draft / SES, and one
test explicitly asserts the send path is never invoked.
"""

import importlib
import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

mod = importlib.import_module("src.api.routers.workflows")
base_agent_mod = importlib.import_module("agents.base_agent")


class _FakeCursor:
    """Records every statement; answers just enough to satisfy _store_draft."""

    def __init__(self, calls):
        self._calls = calls
        self._last_result = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, sql, params=None):
        self._calls.append((" ".join(sql.split()), params))
        normalised = " ".join(sql.split())
        if normalised.startswith("INSERT INTO proc.draft_rfq_emails"):
            self._last_result = (101,)
        elif "COALESCE(MAX(thread_index)" in normalised:
            self._last_result = (0,)
        else:
            self._last_result = None

    def fetchone(self):
        return self._last_result

    def fetchall(self):
        return []


class _FakeConn:
    def __init__(self):
        self.calls = []
        self.committed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def cursor(self):
        return _FakeCursor(self.calls)

    def commit(self):
        self.committed = True


def _make_agent_nick(conn):
    return SimpleNamespace(
        settings=SimpleNamespace(ses_default_sender="buyer@example.com"),
        prompt_engine=SimpleNamespace(get_prompt=lambda *a, **k: None),
        learning_repository=None,
        process_routing_service=SimpleNamespace(
            log_process=lambda **k: None,
            log_run_detail=lambda **k: None,
            log_action=lambda **k: None,
        ),
        _context_dataset_writer=SimpleNamespace(),
        workflow_memory=None,
        get_db_connection=lambda: conn,
    )


@pytest.fixture(autouse=True)
def _fast_gpu_stub(monkeypatch):
    # Avoid any real GPU probing while constructing EmailDraftingAgent.
    monkeypatch.setattr(base_agent_mod, "configure_gpu", lambda *_, **__: "cpu")


def _insert_call(conn):
    inserts = [
        c for c in conn.calls if c[0].startswith("INSERT INTO proc.draft_rfq_emails")
    ]
    assert len(inserts) == 1, f"expected exactly one INSERT, got {inserts}"
    return inserts[0]


def test_prepare_email_draft_persists_recipient_subject_body(monkeypatch):
    conn = _FakeConn()
    agent_nick = _make_agent_nick(conn)

    payload = mod.EmailPrepareRequest(
        deal_id="DEAL-123",
        to=["buyer@example.com", "supplier@example.com"],
        subject="Quarterly Spend Report",
        body="Please see attached quote request for review.",
    )

    response = mod.prepare_email_draft(payload, agent_nick=agent_nick)

    assert isinstance(response, mod.EmailPrepareResponse)
    assert response.status == "prepared"
    assert response.unique_id  # (b) non-empty identifier
    assert response.workflow_id

    assert conn.committed is True

    _, params = _insert_call(conn)
    # column order in the INSERT:
    # rfq_id, unique_id, supplier_id, supplier_name, subject, body, sent,
    # recipient_email, contact_level, thread_index, sender, payload,
    # workflow_id, run_id, mailbox
    assert params[4] == "Quarterly Spend Report"
    assert "Please see attached quote request for review." in params[5]
    assert params[7] == "buyer@example.com"

    persisted_payload = json.loads(params[11])
    assert persisted_payload["recipients"] == [
        "buyer@example.com",
        "supplier@example.com",
    ]
    assert persisted_payload["unique_id"] == response.unique_id
    assert persisted_payload["workflow_id"] == response.workflow_id


def test_prepare_email_draft_returns_resolvable_identifier(monkeypatch):
    """The identifier this endpoint returns must be exactly what
    resolve_workflow_id(identifier) would later look up: unique_id, matching
    the rfq_id column (both are set to the same generated value)."""

    conn = _FakeConn()
    agent_nick = _make_agent_nick(conn)

    payload = mod.EmailPrepareRequest(
        deal_id="DEAL-999",
        to=["ops@example.com"],
        subject="Report attached",
        body="See the attached spend report.",
    )

    response = mod.prepare_email_draft(payload, agent_nick=agent_nick)

    _, params = _insert_call(conn)
    rfq_id_param, unique_id_param = params[0], params[1]
    assert rfq_id_param == unique_id_param == response.unique_id


def test_prepare_email_draft_never_touches_the_send_path(monkeypatch):
    """Fail-closed guarantee: preparing a draft must never dispatch a real
    email. Assert the dispatch service / SES-backed EmailService are never
    constructed or invoked by the prepare handler."""

    conn = _FakeConn()
    agent_nick = _make_agent_nick(conn)

    send_calls = []

    class _GuardEmailDispatchService:
        def __init__(self, *a, **k):
            send_calls.append(("EmailDispatchService.__init__", a, k))

        def send_draft(self, *a, **k):
            send_calls.append(("EmailDispatchService.send_draft", a, k))
            raise AssertionError("send_draft must never be called by prepare")

        def resolve_workflow_id(self, *a, **k):
            send_calls.append(("EmailDispatchService.resolve_workflow_id", a, k))
            raise AssertionError("resolve_workflow_id must never be called by prepare")

    def _guard_send_email(self, *a, **k):
        send_calls.append(("EmailService.send_email", a, k))
        raise AssertionError("EmailService.send_email must never be called by prepare")

    monkeypatch.setattr(mod, "EmailDispatchService", _GuardEmailDispatchService)

    email_service_mod = importlib.import_module("services.email_service")
    monkeypatch.setattr(email_service_mod.EmailService, "send_email", _guard_send_email)

    payload = mod.EmailPrepareRequest(
        deal_id="DEAL-1",
        to=["buyer@example.com"],
        subject="Draft only",
        body="This must only be persisted, never sent.",
    )

    response = mod.prepare_email_draft(payload, agent_nick=agent_nick)

    assert response.unique_id
    assert send_calls == []
    assert conn.committed is True


def test_prepare_email_draft_rejects_empty_recipients():
    conn = _FakeConn()
    agent_nick = _make_agent_nick(conn)

    with pytest.raises(Exception):
        mod.EmailPrepareRequest(deal_id="DEAL-1", to=[], subject="x", body="y")

    # Belt-and-braces: even if an empty list slipped past validation, the
    # handler itself must fail closed rather than fabricate a draft.
    payload = mod.EmailPrepareRequest.model_construct(
        deal_id="DEAL-1", to=[], subject="x", body="y"
    )
    with pytest.raises(HTTPException) as exc_info:
        mod.prepare_email_draft(payload, agent_nick=agent_nick)
    assert exc_info.value.status_code == 400
    assert conn.calls == []


# ---------------------------------------------------------------------------
# The REPLY path (reply_to_unique_id) -- Action Centre supplier-reply panel.
#
# The panel used to dispatch the escalation's own subject_id, which IS the outbound RFQ
# the supplier already replied to. send_draft resolves that to the same already-sent draft
# row and _maybe_return_existing_dispatch returns duplicate:true / dispatched_now:false
# without putting anything on the wire -- and every escalation refers to an already-sent
# thread by construction, so that was the only case in production. A reply is a NEW
# message and needs its own draft; these tests pin the two things a new draft would
# otherwise LOSE.
# ---------------------------------------------------------------------------
_SOURCE_UID = "PROC-WF-AAAAAAAAAAAA"

_SOURCE_DRAFT = {
    "id": 68,
    "rfq_id": _SOURCE_UID,
    "unique_id": _SOURCE_UID,
    "workflow_id": "WF-THREAD-1",
    "supplier_id": "PeopleFirst HR Solutions Ltd",
    "supplier_name": "PeopleFirst HR Solutions Ltd",
    "subject": "Negotiation",
    "body": "our offer",
    "sent": True,           # the whole point: the source thread is already sent
    "sender": "buyer@example.com",
    "payload": json.dumps({"unique_id": _SOURCE_UID, "metadata": {"round": 1}}),
    "attachments": [
        {"filename": "terms.pdf", "s3_key": f"email-attachments/{_SOURCE_UID}/terms.pdf",
         "bytes": 12, "content_type": "application/pdf"},
    ],
}

_TRACKED = SimpleNamespace(
    workflow_id="WF-THREAD-1",
    unique_id=_SOURCE_UID,
    message_id="<our-rfq@ses>",
    response_message_id="<their-reply@supplier>",
    thread_headers={"References": ["<older@ses>"]},
)


def _wire_reply_path(monkeypatch, *, tracked=_TRACKED, source=None):
    """Source draft + tracking row, and a recorder for the attachment copy."""
    persisted = {}
    monkeypatch.setattr(
        mod.draft_rfq_emails_repo, "load_by_unique_id",
        lambda uid: (source if source is not None else dict(_SOURCE_DRAFT))
        if uid == _SOURCE_UID else None,
    )
    monkeypatch.setattr(
        mod.workflow_email_tracking_repo, "lookup_dispatch_row",
        lambda **kw: tracked,
    )
    monkeypatch.setattr(
        mod, "_persist_draft_attachments",
        lambda agent_nick, uid, records: persisted.update({"uid": uid, "records": records}),
    )
    return persisted


def _reply_payload():
    return mod.EmailPrepareRequest(
        to=["billing@peoplefirst.example.com"],
        subject="RE: Negotiation",
        body="Thanks -- we accept.",
        reply_to_unique_id=_SOURCE_UID,
    )


def test_a_reply_gets_its_own_draft_and_never_reuses_the_sent_one(monkeypatch):
    conn = _FakeConn()
    _wire_reply_path(monkeypatch)

    response = mod.prepare_email_draft(_reply_payload(), agent_nick=_make_agent_nick(conn))

    assert response.unique_id != _SOURCE_UID, (
        "dispatching the source id is what _maybe_return_existing_dispatch short-circuits"
    )
    _, params = _insert_call(conn)
    # sent = False on the new row, so the duplicate guard cannot fire on it.
    assert params[6] is False
    # Same workflow as the thread it answers -- a new row in it, not an orphan run.
    assert params[12] == "WF-THREAD-1"
    assert response.workflow_id == "WF-THREAD-1"
    # Attributed to the real supplier off the thread, not a report-panel placeholder.
    assert params[2] == "PeopleFirst HR Solutions Ltd"


def test_a_reply_carries_the_thread_headers_so_it_lands_in_the_conversation(monkeypatch):
    conn = _FakeConn()
    _wire_reply_path(monkeypatch)

    response = mod.prepare_email_draft(_reply_payload(), agent_nick=_make_agent_nick(conn))

    assert response.thread_headers_carried is True
    # In-Reply-To is the SUPPLIER'S message -- the one being answered.
    assert response.in_reply_to == "<their-reply@supplier>"

    _, params = _insert_call(conn)
    persisted = json.loads(params[11])
    # TOP-LEVEL: _resolve_initial_thread_headers reads draft["thread_headers"] first and
    # draft["headers"] second, and _store_draft always fills `headers` with the
    # X-ProcWise tracking headers -- so a metadata-only copy would never reach the wire.
    assert persisted["thread_headers"]["In-Reply-To"] == ["<their-reply@supplier>"]
    refs = persisted["thread_headers"]["References"]
    assert refs == ["<older@ses>", "<our-rfq@ses>", "<their-reply@supplier>"], refs
    assert persisted["metadata"]["reply_to_unique_id"] == _SOURCE_UID


def test_a_thread_with_no_recorded_message_id_reports_no_threading(monkeypatch):
    """Honest absence, not a fabricated header. The panel refuses to send on this."""
    conn = _FakeConn()
    _wire_reply_path(monkeypatch, tracked=None)

    response = mod.prepare_email_draft(_reply_payload(), agent_nick=_make_agent_nick(conn))

    assert response.thread_headers_carried is False
    assert response.in_reply_to is None
    persisted = json.loads(_insert_call(conn)[1][11])
    assert "thread_headers" not in persisted


def test_a_replys_attachments_are_carried_onto_the_new_draft(monkeypatch):
    """The human uploaded them against the ORIGINAL draft -- the only id the panel had."""
    conn = _FakeConn()
    persisted = _wire_reply_path(monkeypatch)

    response = mod.prepare_email_draft(_reply_payload(), agent_nick=_make_agent_nick(conn))

    assert response.attachments_carried == 1
    assert persisted["uid"] == response.unique_id, "onto the NEW draft, not the source"
    assert persisted["records"][0]["s3_key"] == (
        f"email-attachments/{_SOURCE_UID}/terms.pdf"
    ), "the records are copied; the S3 objects are not moved, so the source stays intact"


def test_a_failed_attachment_copy_fails_the_prepare_rather_than_sending_without_them(
    monkeypatch,
):
    conn = _FakeConn()
    _wire_reply_path(monkeypatch)

    def _boom(agent_nick, uid, records):
        raise RuntimeError("attachments column update failed")

    monkeypatch.setattr(mod, "_persist_draft_attachments", _boom)

    with pytest.raises(HTTPException) as exc_info:
        mod.prepare_email_draft(_reply_payload(), agent_nick=_make_agent_nick(conn))
    assert exc_info.value.status_code == 500
    assert "nothing was sent" in str(exc_info.value.detail)


def test_references_are_bracketed_and_deduped_across_both_stored_forms(monkeypatch):
    """The two sources of message ids disagree on form, and both reach ``References``.

    ``workflow_email_tracking._parse_thread_headers`` strips ``<>`` off every id it reads
    back (and returns tuples), while the ``message_id``/``response_message_id`` COLUMNS
    are returned verbatim, brackets intact. Folding them together naively produces a
    ``References`` header that is malformed under RFC 5322 -- every msg-id there must be
    bracketed -- and that dedupes by raw string, so one message listed in both forms
    survives twice. This pins the canonical form on the way out.
    """
    conn = _FakeConn()
    _wire_reply_path(
        monkeypatch,
        tracked=SimpleNamespace(
            workflow_id="WF-THREAD-1",
            unique_id=_SOURCE_UID,
            message_id="<our-rfq@ses>",
            # exactly what the repo hands back: bare ids, in a tuple, and here naming
            # the very message the ``message_id`` column already gave us bracketed.
            thread_headers={"References": ("older@ses", "our-rfq@ses")},
            response_message_id="<their-reply@supplier>",
        ),
    )

    response = mod.prepare_email_draft(_reply_payload(), agent_nick=_make_agent_nick(conn))

    assert response.in_reply_to == "<their-reply@supplier>"
    refs = json.loads(_insert_call(conn)[1][11])["thread_headers"]["References"]
    assert refs == ["<older@ses>", "<our-rfq@ses>", "<their-reply@supplier>"], refs


def test_an_unknown_thread_404s_rather_than_preparing_an_unthreaded_reply(monkeypatch):
    conn = _FakeConn()
    _wire_reply_path(monkeypatch)

    payload = mod.EmailPrepareRequest(
        to=["someone@example.com"], subject="RE: x", body="y",
        reply_to_unique_id="PROC-WF-DOESNOTEXIST",
    )
    with pytest.raises(HTTPException) as exc_info:
        mod.prepare_email_draft(payload, agent_nick=_make_agent_nick(conn))
    assert exc_info.value.status_code == 404
    assert conn.calls == []


# ---------------------------------------------------------------------------
# who asked for the draft (P3)
# ---------------------------------------------------------------------------
class _Principal:
    def __init__(self, subject):
        self.subject = subject
        self.email = f"{subject}@ourcompany.com"


def test_the_draft_records_the_person_who_prepared_it():
    """This endpoint is the human path into proc.draft_rfq_emails, and the
    approvals surface has to be able to tell whether the person approving is
    the person who asked. It could not: the row recorded no requester at all.

    See tests/approvals/test_self_approval_barred.py for the bar this feeds.
    """
    conn = _FakeConn()
    agent_nick = _make_agent_nick(conn)

    payload = mod.EmailPrepareRequest(
        deal_id="DEAL-770",
        to=["supplier@example.com"],
        subject="Request for quote",
        body="Please quote.",
    )

    mod.prepare_email_draft(payload, agent_nick=agent_nick,
                            principal=_Principal("sub-buyer-001"))

    sql, params = _insert_call(conn)
    assert "requested_by" in sql, "the draft row does not record who asked for it"
    persisted_payload = json.loads(params[11])
    assert persisted_payload["requested_by"] == "sub-buyer-001"
    assert "sub-buyer-001" in params, f"requested_by never reached the INSERT: {params}"


def test_a_draft_prepared_with_no_principal_names_nobody():
    """With ASK_AUTH_MODE=off there is no principal. The row must then say
    nobody asked -- NOT some placeholder that a later approver might match, and
    not a value taken from the request body."""
    conn = _FakeConn()
    agent_nick = _make_agent_nick(conn)

    payload = mod.EmailPrepareRequest(
        deal_id="DEAL-771",
        to=["supplier@example.com"],
        subject="Request for quote",
        body="Please quote.",
    )

    mod.prepare_email_draft(payload, agent_nick=agent_nick, principal=None)

    _, params = _insert_call(conn)
    persisted_payload = json.loads(params[11])
    assert persisted_payload.get("requested_by") in (None, "")
