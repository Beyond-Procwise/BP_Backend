"""Draft write-back.

This is where the product claim is most exposed. Everything before it reads; this writes
into someone's mailbox, and the permission that allows a write sits next to the one that
allows a send. So the tests here are mostly about what the code and the credential
*cannot* do.

As with the rest of Phase 5, there is no Microsoft 365 tenant here. The write path is
verified against a stub, so what is proven is the request shape, the guard conditions and
the failure handling — not that Graph creates a draft when asked.
"""

from __future__ import annotations

import base64
import json
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

psycopg2 = pytest.importorskip("psycopg2")

from services.style.draft_writer import (
    SEND_PERMISSIONS,
    DraftWriteFailed,
    GraphDraftWriter,
    SendPermissionGranted,
    decode_token_roles,
    verify_no_send_permission,
)
from services.style.graph_source import (
    GraphCredentials,
    GraphExemplarSource,
    MailboxAccessDenied,
    MailboxUnreachable,
)
from services.style.mailbox import (
    PROVIDER_GRAPH,
    ROLE_BOTH,
    ROLE_DRAFT_TARGET,
    ROLE_EXEMPLAR_SOURCE,
    MailboxBindingRepository,
)
from tests.services.test_style_mailbox import ARN, _code_only, _connect


def _token(roles: List[str]) -> str:
    """A JWT-shaped token carrying a roles claim. Unsigned — nothing verifies it."""

    header = base64.urlsafe_b64encode(b'{"alg":"none"}').decode().rstrip("=")
    payload = base64.urlsafe_b64encode(
        json.dumps({"roles": roles, "aud": "graph"}).encode()
    ).decode().rstrip("=")
    return f"{header}.{payload}.signature"


READ_ONLY_TOKEN = _token(["Mail.Read", "User.Read.All"])
SEND_CAPABLE_TOKEN = _token(["Mail.Read", "Mail.ReadWrite", "Mail.Send"])


class WriteStub:
    """Captures what was POSTed so the request shape can be asserted."""

    def __init__(self, response: Tuple[int, Any] = (201, None), *,
                 token: str = READ_ONLY_TOKEN, raise_on_post: bool = False):
        self.response = response
        self.token = token
        self.raise_on_post = raise_on_post
        self.posts: List[Tuple[str, Any]] = []
        self.gets: List[str] = []

    def post_form(self, url, data):
        return 200, {"access_token": self.token, "expires_in": 3600}

    def get(self, url, headers):
        self.gets.append(url)
        return 200, {"value": []}

    def post_json(self, url, headers, payload):
        self.posts.append((url, payload))
        if self.raise_on_post:
            raise MailboxUnreachable("stub timeout")
        status, body = self.response
        if body is None:
            body = {"id": "AAMkAGDRAFT001", "webLink": "https://outlook.example/draft"}
        return status, body


class _Bench:
    def __init__(self, conn):
        self.conn = conn
        self.user_ref = f"test-{uuid.uuid4().hex[:12]}"
        self.repo = MailboxBindingRepository(conn)

    def binding(self, role=ROLE_DRAFT_TARGET, *, activate=True):
        b = self.repo.create(
            user_ref=self.user_ref, provider=PROVIDER_GRAPH,
            mailbox_address="nick@acme.example", role=role, credential_ref=ARN,
        )
        if activate:
            self.repo.verify_scope(
                b.binding_id, control_mailbox="cfo@acme.example",
                probe=lambda a: (False, "denied"),
            )
            return self.repo.get(b.binding_id)
        return b

    def writer(self, binding, stub=None, **kwargs):
        stub = stub or WriteStub()
        source = GraphExemplarSource(
            binding, transport=stub,
            credentials=GraphCredentials("t", "c", "s"),
        )
        return GraphDraftWriter(binding, source=source, **kwargs), stub

    def cleanup(self):
        # Drafts first: they carry a foreign key to bp_style_profile.
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM proc.draft_rfq_emails WHERE style_user_ref = %s",
                        (self.user_ref,))
            cur.execute("DELETE FROM proc.bp_mailbox_binding WHERE user_ref = %s",
                        (self.user_ref,))
            cur.execute("DELETE FROM proc.bp_style_exemplar WHERE user_ref = %s",
                        (self.user_ref,))
            cur.execute("DELETE FROM proc.bp_style_profile WHERE user_ref = %s",
                        (self.user_ref,))


@pytest.fixture
def bench():
    conn = _connect()
    b = _Bench(conn)
    try:
        yield b
    finally:
        b.cleanup()
        conn.close()


# --- the write ------------------------------------------------------------------------

def test_a_draft_is_created_and_its_identifier_returned(bench):
    binding = bench.binding()
    writer, stub = bench.writer(binding)

    written = writer.write_draft(subject="Chairs — RFQ-1", body="Hi Priya,\n\nPlease quote.")

    assert written.external_draft_ref == "AAMkAGDRAFT001"
    assert len(stub.posts) == 1


def test_the_request_creates_a_message_and_nothing_else(bench):
    """POST /users/{mailbox}/messages creates an unsent message. Nothing sets a send flag,
    and Graph cannot send as a side effect of creation."""
    binding = bench.binding()
    writer, stub = bench.writer(binding)
    writer.write_draft(subject="s", body="b", to=["sam@supplier.example"])

    url, payload = stub.posts[0]
    assert url.endswith("/users/nick@acme.example/messages")
    assert "sendMail" not in url and "/send" not in url
    assert set(payload) <= {"subject", "body", "toRecipients"}
    assert payload["body"]["contentType"] == "Text"
    assert payload["toRecipients"] == [{"emailAddress": {"address": "sam@supplier.example"}}]


def test_recipients_are_optional(bench):
    binding = bench.binding()
    writer, stub = bench.writer(binding)
    writer.write_draft(subject="s", body="b")
    assert "toRecipients" not in stub.posts[0][1]


# --- what it refuses to do ---------------------------------------------------------------

def test_a_credential_that_can_send_is_refused(bench):
    """The check that carries the claim. If a send permission is granted, 'this platform
    cannot send on your behalf' is false at the credential level, and no amount of care
    in this module makes it true again."""
    binding = bench.binding()
    writer, _ = bench.writer(binding, WriteStub(token=SEND_CAPABLE_TOKEN))

    with pytest.raises(SendPermissionGranted, match="Mail.Send"):
        writer.write_draft(subject="s", body="b")


def test_the_permission_check_runs_on_every_write_not_once(bench):
    """A permission added to the app registration next month would otherwise go unnoticed
    until someone asked why an email had been sent."""
    binding = bench.binding()
    stub = WriteStub(token=READ_ONLY_TOKEN)
    writer, _ = bench.writer(binding, stub)
    writer.write_draft(subject="s", body="b")

    stub.token = SEND_CAPABLE_TOKEN
    writer._source._token = None  # force a fresh token, as an expiry would
    with pytest.raises(SendPermissionGranted):
        writer.write_draft(subject="s", body="b")


@pytest.mark.parametrize("role", sorted(SEND_PERMISSIONS))
def test_every_send_capable_permission_is_caught(role):
    with pytest.raises(SendPermissionGranted):
        verify_no_send_permission(_token(["Mail.Read", role]))


def test_permission_matching_is_case_insensitive():
    with pytest.raises(SendPermissionGranted):
        verify_no_send_permission(_token(["mail.send"]))
    with pytest.raises(SendPermissionGranted):
        verify_no_send_permission(_token(["MAIL.SEND"]))


def test_a_read_only_credential_passes_and_reports_what_it_checked():
    roles = verify_no_send_permission(READ_ONLY_TOKEN)
    assert "Mail.Read" in roles


def test_delegated_scope_claims_are_read_too():
    """Delegated tokens put permissions in `scp` as a space-separated string rather than
    `roles`. A check that only read one would pass a token it had not actually inspected."""
    header = base64.urlsafe_b64encode(b'{"alg":"none"}').decode().rstrip("=")
    payload = base64.urlsafe_b64encode(
        json.dumps({"scp": "Mail.Read Mail.Send"}).encode()
    ).decode().rstrip("=")
    with pytest.raises(SendPermissionGranted):
        verify_no_send_permission(f"{header}.{payload}.sig")


def test_an_undecodable_token_raises_rather_than_passing():
    """Failing open here would mean an unreadable token counted as proof of no send
    permission."""
    for bad in ("not-a-jwt", "", "a.b"):
        with pytest.raises(ValueError):
            decode_token_roles(bad)


def test_a_binding_that_is_not_a_draft_target_is_refused(bench):
    binding = bench.binding(role=ROLE_EXEMPLAR_SOURCE)
    writer, _ = bench.writer(binding)
    with pytest.raises(DraftWriteFailed, match="not a draft target"):
        writer.write_draft(subject="s", body="b")


def test_the_both_role_may_receive_drafts(bench):
    binding = bench.binding(role=ROLE_BOTH)
    writer, _ = bench.writer(binding)
    assert writer.write_draft(subject="s", body="b").external_draft_ref


def test_an_inactive_binding_is_refused(bench):
    binding = bench.binding(activate=False)
    writer, _ = bench.writer(binding)
    with pytest.raises(DraftWriteFailed, match="inactive or revoked"):
        writer.write_draft(subject="s", body="b")


def test_an_empty_draft_is_refused(bench):
    binding = bench.binding()
    writer, _ = bench.writer(binding)
    for empty in ("", "   ", None):
        with pytest.raises(DraftWriteFailed, match="empty draft"):
            writer.write_draft(subject="s", body=empty)


def test_a_refusal_from_the_provider_surfaces(bench):
    binding = bench.binding()
    writer, _ = bench.writer(binding, WriteStub(response=(403, {})))
    with pytest.raises(MailboxAccessDenied):
        writer.write_draft(subject="s", body="b")


def test_a_missing_identifier_is_a_failure_not_a_success(bench):
    """A write that returned no id cannot be pointed at later, so it is not a write."""
    binding = bench.binding()
    writer, _ = bench.writer(binding, WriteStub(response=(201, {})))
    with pytest.raises(DraftWriteFailed, match="no draft identifier"):
        writer.write_draft(subject="s", body="b")


# --- invariant 2 -------------------------------------------------------------------------

def test_the_writer_has_no_send_capable_code():
    """Enforced by test, as the brief requires. Reads code, not prose: this module names
    the endpoints it refuses to call in order to explain why."""
    import inspect

    import services.style.draft_writer as mod

    code = _code_only(inspect.getsource(mod)).lower()
    assert "sendmail" not in code
    assert "/send" not in code
    assert "smtplib" not in code

    for name in dir(GraphDraftWriter):
        assert "send" not in name.lower() or name.startswith("_verify"), name


def test_no_send_probe_exists_anywhere():
    """A probe that tried to send and succeeded would have sent a real email to a real
    person. The permission claim is inspected instead."""
    import inspect

    import services.style.draft_writer as mod

    code = _code_only(inspect.getsource(mod))
    assert "post_json" in code, "the writer does POST — this test must stay meaningful"
    assert code.count("post_json") == 1, "only the draft-creation POST should exist"


# --- provenance -----------------------------------------------------------------------

def test_the_external_ref_is_persisted_with_the_draft(bench):
    from services.style.drafting import StyleDraftingService
    from services.style.exemplars import ExemplarService
    from services.style.repository import USER_LEVEL_INTENT, StyleProfileRepository
    from services.style.profile import parse_profile
    from tests.services.test_style_compiler import STUB_PROFILE

    profiles = StyleProfileRepository(bench.conn)
    rec = profiles.insert_version(
        user_ref=bench.user_ref, intent=USER_LEVEL_INTENT,
        profile=parse_profile(STUB_PROFILE), exemplar_count=5,
    )
    profiles.approve(rec.profile_id, "tester")

    binding = bench.binding()
    writer, _ = bench.writer(binding)

    draft = StyleDraftingService(
        bench.conn,
        exemplars=ExemplarService(bench.conn, generate=lambda p: "", embed=lambda t: [1.0] + [0.0] * 1023),
        generate=lambda p: "Subject: X\n\nHi Priya,\n\nAll fine.\n\nJo",
        draft_writer=writer,
    ).draft(user_ref=bench.user_ref, task="Ask Priya", recipient_email="priya@x.example")

    assert draft.external_draft_ref == "AAMkAGDRAFT001"
    with bench.conn.cursor() as cur:
        cur.execute("SELECT external_draft_ref, sent FROM proc.draft_rfq_emails WHERE id = %s",
                    (draft.draft_id,))
        row = cur.fetchone()
    assert row[0] == "AAMkAGDRAFT001"
    assert row[1] is False, "a written-back draft must never be marked sent"


def test_a_failed_write_back_does_not_lose_the_draft(bench):
    """The draft exists and is usable in the platform. Losing the mailbox copy is a
    degraded outcome, not a reason to throw away work the user is waiting for."""
    from services.style.drafting import StyleDraftingService
    from services.style.exemplars import ExemplarService

    binding = bench.binding()
    writer, _ = bench.writer(binding, WriteStub(raise_on_post=True))

    draft = StyleDraftingService(
        bench.conn,
        exemplars=ExemplarService(bench.conn, generate=lambda p: "", embed=lambda t: [1.0] + [0.0] * 1023),
        generate=lambda p: "Subject: X\n\nHi Priya,\n\nAll fine.\n\nJo",
        draft_writer=writer,
    ).draft(user_ref=bench.user_ref, task="Ask Priya")

    assert draft.body.strip()
    assert draft.external_draft_ref is None
    assert draft.draft_id is not None
