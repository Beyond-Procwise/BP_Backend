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
