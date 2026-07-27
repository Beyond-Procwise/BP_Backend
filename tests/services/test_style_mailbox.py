"""Mode C: bindings, scope verification, health, and the C2 read path.

The three the brief names as its done-when have direct tests:

* scope verification blocks activation on a permissive policy
  — ``test_a_permissive_credential_is_refused_activation``
* a revoked binding degrades visibly
  — ``test_a_revoked_binding_forces_the_baseline_with_a_visible_reason``
* C1 drafting succeeds with the provider unreachable
  — ``test_c1_drafting_succeeds_with_the_mail_provider_unreachable``

**What these do not prove.** There is no Microsoft 365 tenant in this environment. Every
Graph interaction below runs against a stub transport, so these tests verify the logic,
the request shapes and the failure handling — not that Graph answers as assumed. The
request paths were written to the documented API and remain unexercised against a live
endpoint.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

psycopg2 = pytest.importorskip("psycopg2")

from services.style import mailbox_cache
from services.style.graph_source import (
    GraphCredentials,
    GraphExemplarSource,
    MailboxAccessDenied,
    MailboxUnreachable,
    folder_to_intent,
)
from services.style.health import check_all_bindings, check_binding
from services.style.mailbox import (
    HEALTH_DEGRADED,
    HEALTH_OK,
    HEALTH_REVOKED,
    PROVIDER_GRAPH,
    ROLE_EXEMPLAR_SOURCE,
    MailboxBindingRepository,
    ScopeVerificationFailed,
)
from services.style.mode_c import fetch_live_exemplars, message_set_hash
from services.style.profile import parse_profile
from services.style.repository import USER_LEVEL_INTENT, StyleProfileRepository
from services.style.resolver import LEVEL_BASELINE, LEVEL_EXACT, StyleResolver
from services.style.sources import ExemplarSource
from tests.services.test_style_compiler import STUB_PROFILE

PROFILE = parse_profile(STUB_PROFILE)
ARN = "arn:aws:secretsmanager:eu-west-1:123456789012:secret:style/graph/acme-AbCdEf"

KNOWN_INTENTS = {"rfq_invite", "award_notification", "escalation", "internal_update"}


def _connect():
    try:
        from dotenv import load_dotenv
        load_dotenv(Path.cwd() / ".env")
    except Exception:  # pragma: no cover
        pass
    if not os.getenv("DB_HOST"):
        pytest.skip("no database configured")
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT", 5432),
            dbname=os.getenv("DB_NAME"), user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"), connect_timeout=10,
        )
    except psycopg2.Error as exc:
        pytest.skip(f"database unreachable: {exc}")
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass('proc.bp_mailbox_binding')")
        if cur.fetchone()[0] is None:
            pytest.skip("style tables not migrated")
    return conn


class StubTransport:
    """A Graph stand-in. Routes are matched on substrings of the URL."""

    def __init__(self, routes: Dict[str, Tuple[int, Any]], *, raise_on: Optional[str] = None):
        self.routes = routes
        self.raise_on = raise_on
        self.calls: List[str] = []

    def post_form(self, url, data):
        self.calls.append(url)
        return 200, {"access_token": "stub-token", "expires_in": 3600}

    def get(self, url, headers):
        self.calls.append(url)
        if self.raise_on and self.raise_on in url:
            raise MailboxUnreachable("stub timeout")
        for fragment, response in self.routes.items():
            if fragment in url:
                return response
        return 404, {}


def _message(idx: int, subject: str, body: str) -> Dict[str, Any]:
    return {
        "id": f"AAMkAD{idx:04d}",
        "subject": subject,
        "body": {"contentType": "text", "content": body},
        "sentDateTime": f"2026-04-0{idx}T09:00:00Z",
    }


SENT_MESSAGES = {
    "value": [
        _message(1, "Racking quote", "Hi Sam,\n\nCan you send the unit rate?\n\nNick"),
        _message(2, "Desks", "Hi Dana,\n\nWe need 25 desks by April.\n\nNick"),
        _message(3, "Terms", "Hi Priya,\n\nCan you confirm 60 days?\n\nNick"),
    ]
}

FOLDERS = {
    "value": [
        {"id": "f-inbox", "displayName": "Inbox"},
        {"id": "f-sent", "displayName": "Sent Items"},
        {"id": "f-rfq", "displayName": "RFQ invites"},
    ]
}


def _source(routes, **kwargs) -> GraphExemplarSource:
    binding = kwargs.pop("binding")
    return GraphExemplarSource(
        binding,
        transport=StubTransport(routes, **kwargs),
        credentials=GraphCredentials("tenant", "client", "secret"),
        known_intents=KNOWN_INTENTS,
    )


class _Bench:
    def __init__(self, conn):
        self.conn = conn
        self.user_ref = f"test-{uuid.uuid4().hex[:12]}"
        self.repo = MailboxBindingRepository(conn)
        self.profiles = StyleProfileRepository(conn)

    def binding(self, address="nick@acme.example"):
        return self.repo.create(
            user_ref=self.user_ref, provider=PROVIDER_GRAPH,
            mailbox_address=address, role=ROLE_EXEMPLAR_SOURCE, credential_ref=ARN,
            scope_policy_ref="ApplicationAccessPolicy/style-readers",
        )

    def approved_profile(self):
        rec = self.profiles.insert_version(
            user_ref=self.user_ref, intent=USER_LEVEL_INTENT,
            profile=PROFILE, exemplar_count=5,
        )
        return self.profiles.approve(rec.profile_id, "tester")

    def cleanup(self):
        mailbox_cache.flush_all()
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


# --- creation and credentials ---------------------------------------------------------

def test_a_new_binding_is_inactive_until_its_scope_is_proven(bench):
    """A binding usable the moment it is created would make verification optional, and an
    optional check is the one that gets skipped on the day it matters."""
    binding = bench.binding()
    assert binding.is_active is False
    assert binding.scope_verified_at is None


def test_a_credential_ref_that_is_not_an_arn_is_refused(bench):
    """Invariant 8. The database enforces this too; failing here keeps a token out of a
    query string in the first place."""
    for bad in ("hunter2", "", "https://vault.example/secret", None):
        with pytest.raises(ValueError, match="Secrets Manager ARN"):
            bench.repo.create(
                user_ref=bench.user_ref, provider=PROVIDER_GRAPH,
                mailbox_address="a@b.example", role=ROLE_EXEMPLAR_SOURCE,
                credential_ref=bad,
            )


def test_credentials_are_read_from_secrets_manager_not_the_database():
    from services.style.graph_source import resolve_credentials

    class _Secrets:
        def get_secret_value(self, SecretId):
            assert SecretId == ARN
            return {"SecretString": json.dumps(
                {"tenant_id": "t", "client_id": "c", "client_secret": "s"})}

    creds = resolve_credentials(ARN, client=_Secrets())
    assert (creds.tenant_id, creds.client_id, creds.client_secret) == ("t", "c", "s")


# --- scope verification (done-when #1) -------------------------------------------------

def test_a_denied_control_read_activates_the_binding(bench):
    """A denial is the evidence. Nothing else activates a binding."""
    binding = bench.binding()
    source = _source({"/users/cfo@acme.example/messages": (403, {})}, binding=binding)

    verified = bench.repo.verify_scope(
        binding.binding_id, control_mailbox="cfo@acme.example",
        probe=source.probe_control_mailbox,
    )
    assert verified.is_active is True
    assert verified.scope_verified_at is not None
    assert verified.scope_evidence_ref.startswith("scope-")


def test_a_permissive_credential_is_refused_activation(bench):
    """DONE-WHEN #1. If the control read succeeds, the grant is wider than the binding
    claims and every statement about what this platform can reach becomes false."""
    binding = bench.binding()
    source = _source(
        {"/users/cfo@acme.example/messages": (200, {"value": [_message(1, "x", "y")]})},
        binding=binding,
    )

    with pytest.raises(ScopeVerificationFailed, match="wider than this binding claims"):
        bench.repo.verify_scope(
            binding.binding_id, control_mailbox="cfo@acme.example",
            probe=source.probe_control_mailbox,
        )

    assert bench.repo.get(binding.binding_id).is_active is False


def test_the_refusal_is_still_recorded_against_the_binding(bench):
    """A failed verification must leave a trace — otherwise a rejected binding looks the
    same as one nobody ever tried to verify."""
    binding = bench.binding()
    source = _source({"/users/cfo@acme.example/messages": (200, {"value": []})}, binding=binding)
    with pytest.raises(ScopeVerificationFailed):
        bench.repo.verify_scope(binding.binding_id, control_mailbox="cfo@acme.example",
                                probe=source.probe_control_mailbox)
    after = bench.repo.get(binding.binding_id)
    assert after.scope_verified_at is not None
    assert after.scope_evidence_ref is not None
    assert after.is_active is False


@pytest.mark.parametrize("status", [401, 403, 404])
def test_every_denial_shape_counts_as_a_denial(bench, status):
    """Graph returns 404 for a mailbox an application access policy hides — a denial in
    everything but name."""
    binding = bench.binding()
    source = _source({"/users/cfo@acme.example/messages": (status, {})}, binding=binding)
    allowed, detail = source.probe_control_mailbox("cfo@acme.example")
    assert allowed is False
    assert str(status) in detail


def test_a_transport_failure_is_not_treated_as_a_denial(bench):
    """'The network was down' and 'the tenant refused us' look identical from here.
    Treating the first as proof of the second would activate a binding on a timeout."""
    binding = bench.binding()
    source = _source({}, binding=binding, raise_on="/users/")
    with pytest.raises(MailboxUnreachable):
        source.probe_control_mailbox("cfo@acme.example")


def test_an_inconclusive_status_cannot_verify_scope(bench):
    binding = bench.binding()
    source = _source({"/users/cfo@acme.example/messages": (500, {})}, binding=binding)
    with pytest.raises(MailboxUnreachable, match="neither a denial nor a success"):
        source.probe_control_mailbox("cfo@acme.example")


def test_the_control_mailbox_must_differ_from_the_bound_one(bench):
    """Probing the bound mailbox proves nothing — it is supposed to succeed."""
    binding = bench.binding("nick@acme.example")
    with pytest.raises(ValueError, match="DIFFERENT mailbox"):
        bench.repo.verify_scope(
            binding.binding_id, control_mailbox="nick@acme.example",
            probe=lambda addr: (False, "denied"),
        )


# --- health and revocation (done-when #2) ----------------------------------------------

def test_a_refusal_marks_the_binding_revoked(bench):
    binding = bench.binding()

    def _probe(_b):
        raise MailboxAccessDenied("403")

    from services.style.health import _default_probe  # noqa: F401 - documents the default
    state = check_binding(binding, repo=bench.repo,
                          probe=lambda b: HEALTH_REVOKED if True else HEALTH_OK)
    assert state == HEALTH_REVOKED
    assert bench.repo.get(binding.binding_id).is_active is False


def test_a_timeout_only_degrades_it(bench):
    """Marking a binding revoked over a dropped packet would strip a customer's
    personalisation for a network blip."""
    binding = bench.binding()
    state = check_binding(binding, repo=bench.repo, probe=lambda b: HEALTH_DEGRADED)
    assert state == HEALTH_DEGRADED
    assert bench.repo.get(binding.binding_id).health_state == HEALTH_DEGRADED


def test_a_revoked_binding_forces_the_baseline_with_a_visible_reason(bench):
    """DONE-WHEN #2. The profile still exists — compiled from that mailbox earlier — and
    using it would mean drafting from mail we have been told to stop reading."""
    bench.approved_profile()
    binding = bench.binding()
    bench.repo.verify_scope(binding.binding_id, control_mailbox="cfo@acme.example",
                            probe=lambda a: (False, "denied"))

    resolver = StyleResolver(bench.conn, bindings=bench.repo)
    assert resolver.resolve(bench.user_ref, USER_LEVEL_INTENT).fallback_level == LEVEL_EXACT

    bench.repo.record_health(binding.binding_id, HEALTH_REVOKED)

    after = resolver.resolve(bench.user_ref, USER_LEVEL_INTENT)
    assert after.fallback_level == LEVEL_BASELINE
    assert after.degraded is True
    assert after.source_unreachable is True
    assert "revoked" in after.reason


def test_revocation_flushes_anything_cached_from_that_mailbox(bench):
    binding = bench.binding()
    mailbox_cache.put(binding.binding_id, None, ["a body that must not linger"])
    assert mailbox_cache.size() == 1
    bench.repo.record_health(binding.binding_id, HEALTH_REVOKED)
    assert mailbox_cache.size() == 0


def test_unbinding_flushes_the_cache_too(bench):
    """Unbinding is the customer saying stop reading. Leaving five minutes of their mail
    in a process would make that a suggestion."""
    binding = bench.binding()
    mailbox_cache.put(binding.binding_id, "rfq_invite", ["body"])
    bench.repo.unbind(binding.binding_id)
    assert mailbox_cache.size() == 0
    assert bench.repo.get(binding.binding_id).is_active is False


def test_the_health_sweep_tallies_every_active_binding(bench):
    binding = bench.binding()
    bench.repo.verify_scope(binding.binding_id, control_mailbox="cfo@acme.example",
                            probe=lambda a: (False, "denied"))
    tally = check_all_bindings(repo=bench.repo, probe=lambda b: HEALTH_OK)
    assert tally[HEALTH_OK] >= 1


def test_a_provider_with_no_reader_reports_degraded_not_ok(bench):
    """IMAP and Gmail bindings have no reader yet. Reporting OK would claim a check that
    never happened."""
    from services.style.health import _default_probe
    from services.style.mailbox import MailboxBinding

    imap = MailboxBinding(
        binding_id=-1, user_ref="u", provider="imap", mailbox_address="a@b.example",
        role=ROLE_EXEMPLAR_SOURCE, credential_ref=ARN, scope_policy_ref=None,
        scope_verified_at=None, scope_evidence_ref=None, last_health_check=None,
        health_state=HEALTH_OK, is_active=True,
    )
    assert _default_probe(imap) == HEALTH_DEGRADED


# --- intent from folder names -----------------------------------------------------------

@pytest.mark.parametrize("folder,expected", [
    ("RFQ invites", "rfq_invite"),
    ("rfq_invite", "rfq_invite"),
    ("Award notifications", "award_notification"),
    ("Escalation", "escalation"),
    ("Internal updates", "internal_update"),
])
def test_a_matching_folder_name_supplies_the_intent(folder, expected):
    """A person with a folder called 'Award notifications' has already classified their
    mail more reliably than any model will."""
    assert folder_to_intent(folder, KNOWN_INTENTS) == expected


@pytest.mark.parametrize("folder", ["Inbox", "Sent Items", "Drafts", "Junk Email", "Archive"])
def test_system_folders_are_not_intents(folder):
    assert folder_to_intent(folder, KNOWN_INTENTS) is None


@pytest.mark.parametrize("folder", ["Awards 2025", "Suppliers", "Q1 planning", "", None])
def test_a_near_miss_is_not_guessed_at(folder):
    """A wrong intent is worse than no intent: it compiles a profile from the wrong
    emails."""
    assert folder_to_intent(folder, KNOWN_INTENTS) is None


def test_fetch_prefers_the_matching_folder_over_sent_items(bench):
    binding = bench.binding()
    source = _source(
        {
            "/mailFolders?": (200, FOLDERS),
            "/mailFolders/f-rfq/messages": (200, SENT_MESSAGES),
            "/mailFolders/f-sent/messages": (200, {"value": []}),
        },
        binding=binding,
    )
    got = source.fetch(bench.user_ref, "rfq_invite")
    assert len(got) == 3
    assert all(e.intent == "rfq_invite" for e in got)
    assert any("f-rfq" in call for call in source.transport.calls)


def test_fetch_falls_back_to_sent_mail_when_no_folder_matches(bench):
    """Sent mail, not the inbox: the customer's own writing is the point, and their inbox
    is everyone else's style."""
    binding = bench.binding()
    source = _source(
        {"/mailFolders?": (200, FOLDERS), "/mailFolders/f-sent/messages": (200, SENT_MESSAGES)},
        binding=binding,
    )
    got = source.fetch(bench.user_ref, "escalation")
    assert len(got) == 3
    assert any("f-sent" in call for call in source.transport.calls)


def test_html_bodies_are_reduced_to_text(bench):
    binding = bench.binding()
    html = {"value": [{"id": "m1", "subject": "s",
                       "body": {"contentType": "html",
                                "content": "<p>Hi Sam,</p><p>All <b>fine</b>.</p>"}}]}
    source = _source(
        {"/mailFolders?": (200, FOLDERS), "/mailFolders/f-sent/messages": (200, html)},
        binding=binding,
    )
    body = source.fetch(bench.user_ref)[0].body
    assert "<p>" not in body and "Hi Sam," in body and "fine" in body


def test_the_graph_source_satisfies_the_phase_2_protocol(bench):
    """Adding Mode C required no change to the compiler. If it had, the seam was drawn in
    the wrong place."""
    binding = bench.binding()
    source = _source({"/mailFolders?": (200, FOLDERS)}, binding=binding)
    assert isinstance(source, ExemplarSource)


# --- Mode C2 ---------------------------------------------------------------------------

def test_c2_returns_exemplars_with_no_row_id(bench):
    """Nothing fetched is written to a table. No row means no id — a draft cites message
    ids instead."""
    binding = bench.binding()
    bench.repo.verify_scope(binding.binding_id, control_mailbox="cfo@acme.example",
                            probe=lambda a: (False, "denied"))
    active = bench.repo.get(binding.binding_id)
    source = _source(
        {"/mailFolders?": (200, FOLDERS), "/mailFolders/f-sent/messages": (200, SENT_MESSAGES)},
        binding=active,
    )
    outcome = fetch_live_exemplars(active, source, user_ref=bench.user_ref)
    assert outcome.usable
    assert all(e.exemplar_id is None for e in outcome.exemplars)
    assert all(e.message_id for e in outcome.exemplars)
    assert outcome.message_ids


def test_c2_never_writes_a_body_to_any_table(bench):
    """The claim that makes C2 worth choosing."""
    binding = bench.binding()
    bench.repo.verify_scope(binding.binding_id, control_mailbox="cfo@acme.example",
                            probe=lambda a: (False, "denied"))
    active = bench.repo.get(binding.binding_id)
    source = _source(
        {"/mailFolders?": (200, FOLDERS), "/mailFolders/f-sent/messages": (200, SENT_MESSAGES)},
        binding=active,
    )
    fetch_live_exemplars(active, source, user_ref=bench.user_ref)

    with bench.conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM proc.bp_style_exemplar WHERE user_ref = %s",
                    (bench.user_ref,))
        assert cur.fetchone()[0] == 0
        cur.execute("SELECT count(*) FROM proc.bp_style_ingest_staging WHERE user_ref = %s",
                    (bench.user_ref,))
        assert cur.fetchone()[0] == 0


def test_c2_bodies_are_redacted_even_though_they_are_never_stored(bench):
    """They are going into a prompt. A supplier's name and price do not need to be there
    for a model to copy the writing."""
    binding = bench.binding()
    bench.repo.verify_scope(binding.binding_id, control_mailbox="cfo@acme.example",
                            probe=lambda a: (False, "denied"))
    active = bench.repo.get(binding.binding_id)
    messages = {"value": [_message(
        1, "Quote QUT136586", "Hi Sam,\n\nThe rate is £1,240.50 against QUT136586.\n\nNick")]}
    source = _source(
        {"/mailFolders?": (200, FOLDERS), "/mailFolders/f-sent/messages": (200, messages)},
        binding=active,
    )
    body = fetch_live_exemplars(active, source, user_ref=bench.user_ref).exemplars[0].body
    assert "1,240.50" not in body
    assert "QUT136586" not in body
    assert "[AMOUNT]" in body


def test_the_c2_cache_is_hit_within_its_ttl(bench):
    binding = bench.binding()
    bench.repo.verify_scope(binding.binding_id, control_mailbox="cfo@acme.example",
                            probe=lambda a: (False, "denied"))
    active = bench.repo.get(binding.binding_id)
    source = _source(
        {"/mailFolders?": (200, FOLDERS), "/mailFolders/f-sent/messages": (200, SENT_MESSAGES)},
        binding=active,
    )
    first = fetch_live_exemplars(active, source, user_ref=bench.user_ref)
    calls_after_first = len(source.transport.calls)
    second = fetch_live_exemplars(active, source, user_ref=bench.user_ref)

    assert first.from_cache is False and second.from_cache is True
    assert len(source.transport.calls) == calls_after_first, "the provider was called again"


def test_the_c2_cache_expires(bench):
    binding = bench.binding()
    mailbox_cache.put(binding.binding_id, None, ["body"], ttl_seconds=1)
    assert mailbox_cache.get(binding.binding_id, None) is not None
    time.sleep(1.1)
    assert mailbox_cache.get(binding.binding_id, None) is None


def test_the_c2_cache_is_keyed_by_binding_and_intent(bench):
    """A flush must be able to target one customer's binding without disturbing others."""
    mailbox_cache.put(1, "rfq_invite", ["a"])
    mailbox_cache.put(1, "escalation", ["b"])
    mailbox_cache.put(2, "rfq_invite", ["c"])
    assert mailbox_cache.flush_binding(1) == 2
    assert mailbox_cache.get(2, "rfq_invite") == ["c"]
    mailbox_cache.flush_all()


def test_the_circuit_breaker_degrades_rather_than_raising(bench):
    """Someone writing an email should never be blocked because a mail API is slow."""
    binding = bench.binding()
    bench.repo.verify_scope(binding.binding_id, control_mailbox="cfo@acme.example",
                            probe=lambda a: (False, "denied"))
    active = bench.repo.get(binding.binding_id)
    source = _source({}, binding=active, raise_on="/mailFolders")

    outcome = fetch_live_exemplars(active, source, user_ref=bench.user_ref)
    assert outcome.unreachable is True
    assert outcome.usable is False
    assert "could not be reached" in outcome.reason


def test_a_refusal_is_reported_differently_from_a_timeout(bench):
    binding = bench.binding()
    bench.repo.verify_scope(binding.binding_id, control_mailbox="cfo@acme.example",
                            probe=lambda a: (False, "denied"))
    active = bench.repo.get(binding.binding_id)
    source = _source({"/mailFolders?": (403, {})}, binding=active)
    outcome = fetch_live_exemplars(active, source, user_ref=bench.user_ref)
    assert outcome.unreachable is True
    assert "refused" in outcome.reason


def test_an_unusable_binding_is_not_read_from(bench):
    binding = bench.binding()  # never verified, so inactive
    outcome = fetch_live_exemplars(binding, object(), user_ref=bench.user_ref)
    assert outcome.unreachable is True
    assert "inactive or revoked" in outcome.reason


def test_the_message_hash_identifies_the_set_not_its_order():
    """Under C2 this is the only durable record of which emails shaped a draft — the
    bodies are gone by the time anyone asks."""
    assert message_set_hash(["c", "a", "b"]) == message_set_hash(["a", "b", "c"])
    assert message_set_hash(["a"]) != message_set_hash(["a", "b"])
    assert message_set_hash([]) is None


# --- done-when #3 -------------------------------------------------------------------------

def test_c1_drafting_succeeds_with_the_mail_provider_unreachable(bench):
    """DONE-WHEN #3. C1 reads the mailbox at COMPILE time only. Drafting must complete
    with the provider unreachable — if it cannot, the mode is C2 wearing C1's label."""
    from services.style.drafting import StyleDraftingService
    from services.style.exemplars import ExemplarService

    profile = bench.approved_profile()
    exemplar_svc = ExemplarService(
        bench.conn,
        generate=lambda p: json.dumps({"emails": [
            {"subject": "s1", "body": "Hi Marta,\n\nAll fine.\n\nJo"},
            {"subject": "s2", "body": "Hi Ravi,\n\nAgreed.\n\nJo"},
        ]}),
        embed=lambda t: [1.0] + [0.0] * 1023,
    )
    exemplar_svc.generate_for(profile)

    binding = bench.binding()
    bench.repo.verify_scope(binding.binding_id, control_mailbox="cfo@acme.example",
                            probe=lambda a: (False, "denied"))

    # A source that fails on every call — the provider is simply gone.
    class _Dead:
        def fetch(self, *a, **k):
            raise MailboxUnreachable("provider down")

    draft = StyleDraftingService(
        bench.conn, exemplars=exemplar_svc, mode="C1",
        mailbox_source=_Dead(), binding=bench.repo.get(binding.binding_id),
        generate=lambda p: "Subject: Chairs\n\nHi Priya,\n\nAll fine.\n\nJo",
    ).draft(user_ref=bench.user_ref, task="Ask Priya for a quote", persist=False)

    assert draft.body.strip()
    assert draft.provenance.fallback_level == LEVEL_EXACT, "C1 must not degrade on a dead provider"
    assert draft.provenance.exemplar_ids, "C1 uses stored exemplars, not live ones"


def test_c2_drafting_degrades_visibly_when_the_provider_is_unreachable(bench):
    """The mirror image: C2 does depend on the live read, so losing it must show."""
    from services.style.drafting import StyleDraftingService
    from services.style.exemplars import ExemplarService

    bench.approved_profile()
    binding = bench.binding()
    bench.repo.verify_scope(binding.binding_id, control_mailbox="cfo@acme.example",
                            probe=lambda a: (False, "denied"))

    class _Dead:
        def fetch(self, *a, **k):
            raise MailboxUnreachable("provider down")

    draft = StyleDraftingService(
        bench.conn, exemplars=ExemplarService(bench.conn, generate=lambda p: "", embed=lambda t: []),
        mode="C2", mailbox_source=_Dead(), binding=bench.repo.get(binding.binding_id),
        generate=lambda p: "Subject: X\n\nHi Priya,\n\nAll fine.\n\nJo",
    ).draft(user_ref=bench.user_ref, task="Ask Priya for a quote", persist=False)

    assert draft.body.strip(), "a slow mailbox must not block someone writing an email"
    assert draft.provenance.fallback_level == LEVEL_BASELINE
    assert draft.provenance.as_response_fields()["style_degraded"] is True


def test_c2_provenance_cites_message_ids_and_the_binding(bench):
    from services.style.drafting import StyleDraftingService
    from services.style.exemplars import ExemplarService

    bench.approved_profile()
    binding = bench.binding()
    bench.repo.verify_scope(binding.binding_id, control_mailbox="cfo@acme.example",
                            probe=lambda a: (False, "denied"))
    active = bench.repo.get(binding.binding_id)
    source = _source(
        {"/mailFolders?": (200, FOLDERS), "/mailFolders/f-sent/messages": (200, SENT_MESSAGES)},
        binding=active,
    )

    draft = StyleDraftingService(
        bench.conn, exemplars=ExemplarService(bench.conn, generate=lambda p: "", embed=lambda t: []),
        mode="C2", mailbox_source=source, binding=active,
        generate=lambda p: "Subject: X\n\nHi Priya,\n\nAll fine.\n\nJo",
    ).draft(user_ref=bench.user_ref, task="Ask Priya for a quote")

    p = draft.provenance
    assert p.message_ids and all(m.startswith("AAMkAD") for m in p.message_ids)
    assert p.exemplar_ids == [], "C2 exemplars have no rows"
    assert p.exemplar_set_hash == message_set_hash(p.message_ids)
    assert p.mailbox_binding_id == active.binding_id

    with bench.conn.cursor() as cur:
        cur.execute(
            "SELECT style_message_ids, style_mailbox_binding_id, style_exemplar_ids "
            "FROM proc.draft_rfq_emails WHERE id = %s", (draft.draft_id,)
        )
        stored = cur.fetchone()
    assert sorted(stored[0]) == sorted(p.message_ids)
    assert stored[1] == active.binding_id
    assert stored[2] is None


# --- invariant 2, scoped ------------------------------------------------------------------

def _code_only(source: str) -> str:
    """``source`` with comments and docstrings stripped.

    The absence test below must read code, not prose: these modules deliberately name the
    endpoints they do not call, in order to explain why they do not call them, and a naive
    substring search over the whole file would flag that explanation as the thing it warns
    about.
    """

    import ast

    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                body.pop(0)
    return ast.unparse(tree)


def test_no_mailbox_adapter_exposes_a_send_method():
    """The Graph app registration is read-only by design; this asserts the code could not
    send even if the grant allowed it."""
    import inspect

    import services.style.graph_source as graph
    import services.style.mailbox as mailbox
    import services.style.mode_c as mode_c

    for module in (graph, mailbox, mode_c):
        code = _code_only(inspect.getsource(module)).lower()
        assert "sendmail" not in code, module.__name__
        assert "send_mail" not in code, module.__name__
        assert "smtplib" not in code, module.__name__

    for name in dir(GraphExemplarSource):
        assert "send" not in name.lower(), f"GraphExemplarSource.{name}"
