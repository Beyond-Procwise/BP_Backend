"""Prompt assembly, the fallback ladder, and provenance.

Two things are load-bearing here and both get direct tests:

* **Assembly order and precedence.** Rules before examples, and an explicit instruction
  that the rules win. A model shown three complete emails and a list of abstract rules
  will imitate the emails unless told otherwise.
* **Every rung of the ladder is reachable**, and lands with a reason the reader can see.
  Silent degradation is the failure this design exists to prevent.

Database tests run against real Postgres. Skipped, not failed, where none is reachable.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import List

import pytest

psycopg2 = pytest.importorskip("psycopg2")

from services.style.drafting import (
    PROMPT_NAME,
    SYSTEM_PROMPT_FALLBACK,
    StyleDraftingService,
    _exemplar_set_hash,
    _split_subject_and_body,
)
from services.style.exemplars import ExemplarRecord, ExemplarService
from services.style.profile import parse_profile
from services.style.repository import USER_LEVEL_INTENT, StyleProfileRepository
from services.style.resolver import (
    BASELINE_PROFILE,
    HOUSE_STYLE_USER_REF,
    LEVEL_BASELINE,
    LEVEL_EXACT,
    LEVEL_HOUSE,
    LEVEL_USER,
    StyleResolver,
)
from tests.services.test_style_compiler import STUB_PROFILE
from tests.services.test_style_exemplars import GENERATED, _fake_embed

PROFILE = parse_profile(STUB_PROFILE)

DRAFTED = "Subject: Chairs — RFQ-2201\n\nHi Marta,\n\nWe need 25 chairs by April.\n\nJo"


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
        cur.execute("SELECT to_regclass('proc.bp_style_profile')")
        if cur.fetchone()[0] is None:
            pytest.skip("style tables not migrated")
    return conn


class _Bench:
    def __init__(self, conn):
        self.conn = conn
        self.user_ref = f"test-{uuid.uuid4().hex[:12]}"
        self.repo = StyleProfileRepository(conn)
        self.exemplar_svc = ExemplarService(
            conn, generate=lambda p: json.dumps(GENERATED), embed=_fake_embed
        )

    def approve_profile(self, intent=USER_LEVEL_INTENT, user_ref=None):
        rec = self.repo.insert_version(
            user_ref=user_ref or self.user_ref, intent=intent,
            profile=PROFILE, exemplar_count=5,
        )
        return self.repo.approve(rec.profile_id, "tester")

    def service(self, generate=None):
        return StyleDraftingService(
            self.conn,
            exemplars=self.exemplar_svc,
            generate=generate or (lambda p: DRAFTED),
        )

    def cleanup(self):
        # Drafts first: draft_rfq_emails.style_profile_id references bp_style_profile, so
        # a profile cannot be removed while a draft still cites it as its provenance.
        # That constraint is the point — deleting it the other way round would leave a
        # draft claiming rules that no longer exist.
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM proc.draft_rfq_emails WHERE style_user_ref = %s",
                        (self.user_ref,))
            for user in (self.user_ref, HOUSE_STYLE_USER_REF):
                cur.execute("DELETE FROM proc.bp_style_exemplar WHERE user_ref = %s", (user,))
                cur.execute("DELETE FROM proc.bp_style_profile WHERE user_ref = %s", (user,))


@pytest.fixture
def bench():
    conn = _connect()
    b = _Bench(conn)
    try:
        yield b
    finally:
        b.cleanup()
        conn.close()


# --- the fallback ladder ------------------------------------------------------------

def test_level_0_exact_profile_for_this_intent(bench):
    bench.approve_profile("rfq_invite")
    resolved = StyleResolver(bench.conn).resolve(bench.user_ref, "rfq_invite")
    assert resolved.fallback_level == LEVEL_EXACT
    assert resolved.degraded is False
    assert resolved.scope_intent == "rfq_invite"


def test_level_1_falls_back_to_the_users_general_style(bench):
    bench.approve_profile(USER_LEVEL_INTENT)
    resolved = StyleResolver(bench.conn).resolve(bench.user_ref, "escalation")
    assert resolved.fallback_level == LEVEL_USER
    assert resolved.degraded is True
    assert resolved.scope_intent == USER_LEVEL_INTENT
    assert "general style" in resolved.reason


def test_level_2_falls_back_to_the_house_style(bench):
    bench.approve_profile(USER_LEVEL_INTENT, user_ref=HOUSE_STYLE_USER_REF)
    resolved = StyleResolver(bench.conn).resolve(bench.user_ref, "rfq_invite")
    assert resolved.fallback_level == LEVEL_HOUSE
    assert resolved.scope_user_ref == HOUSE_STYLE_USER_REF
    assert "house style" in resolved.reason


def test_level_3_falls_back_to_the_platform_baseline(bench):
    resolved = StyleResolver(bench.conn).resolve(bench.user_ref, "rfq_invite")
    assert resolved.fallback_level == LEVEL_BASELINE
    assert resolved.profile == BASELINE_PROFILE
    assert resolved.record is None
    assert resolved.source_unreachable is False


def test_level_3_when_the_source_is_unreachable(bench):
    """A database problem must not stop someone writing an email, but it must be visible."""

    class _Broken:
        def get_active(self, *a, **k):
            raise RuntimeError("database unreachable")

    resolved = StyleResolver(repo=_Broken()).resolve("anyone", "rfq_invite")
    assert resolved.fallback_level == LEVEL_BASELINE
    assert resolved.source_unreachable is True
    assert "temporarily unavailable" in resolved.reason


def test_asking_for_the_user_level_scope_does_not_report_a_fallback_that_did_not_happen(bench):
    """Level 1 IS the user-level lookup. Requesting it directly and finding it is a hit."""
    bench.approve_profile(USER_LEVEL_INTENT)
    resolved = StyleResolver(bench.conn).resolve(bench.user_ref, USER_LEVEL_INTENT)
    assert resolved.fallback_level == LEVEL_EXACT


def test_the_exact_profile_wins_over_the_general_one(bench):
    bench.approve_profile(USER_LEVEL_INTENT)
    exact = bench.approve_profile("rfq_invite")
    resolved = StyleResolver(bench.conn).resolve(bench.user_ref, "rfq_invite")
    assert resolved.fallback_level == LEVEL_EXACT
    assert resolved.profile_id == exact.profile_id


def test_a_deleted_profile_reverts_to_the_next_rung_visibly(bench):
    """Deleting a profile must revert drafting to the ladder with a visible flag."""
    profile = bench.approve_profile("rfq_invite")
    assert StyleResolver(bench.conn).resolve(bench.user_ref, "rfq_invite").fallback_level == 0
    bench.repo.deactivate(bench.user_ref, "rfq_invite")
    after = StyleResolver(bench.conn).resolve(bench.user_ref, "rfq_invite")
    assert after.fallback_level == LEVEL_BASELINE
    assert after.degraded is True


def test_a_stored_profile_that_no_longer_validates_does_not_get_drafted_against(bench):
    """A schema that moved under an approved profile is a reason to fall back, not to
    write against rules nobody can parse."""
    profile = bench.approve_profile("rfq_invite")
    with bench.conn.cursor() as cur:
        # DRAFT rows are still editable; approved ones are frozen by the trigger.
        cur.execute(
            "INSERT INTO proc.bp_style_profile "
            "(user_ref, intent, version, state, profile_json, exemplar_count, is_active, "
            " approved_by, approved_at) "
            "VALUES (%s, 'escalation', 1, 'APPROVED', %s, 5, TRUE, 'tester', NOW())",
            (bench.user_ref, '{"nonsense": true}'),
        )
    resolved = StyleResolver(bench.conn).resolve(bench.user_ref, "escalation")
    assert resolved.fallback_level == LEVEL_BASELINE
    assert resolved.source_unreachable is True


# --- prompt assembly ----------------------------------------------------------------

def test_the_order_is_rules_then_examples_then_task(bench):
    """Not incidental. Rules first because they govern; examples second because they
    illustrate; task last because it is what to act on."""
    profile = bench.approve_profile(USER_LEVEL_INTENT)
    bench.exemplar_svc.generate_for(profile)

    seen: List[str] = []
    bench.service(generate=lambda p: (seen.append(p), DRAFTED)[1]).draft(
        user_ref=bench.user_ref, task="Ask for a quote on 25 chairs", persist=False
    )
    prompt = seen[0]
    assert prompt.index("STYLE SPECIFICATION") < prompt.index("EXAMPLES")
    assert prompt.index("EXAMPLES") < prompt.index("TASK")


def test_the_precedence_instruction_is_present(bench):
    """A specification that loses to its own illustrations is not governing anything."""
    seen: List[str] = []
    bench.service(generate=lambda p: (seen.append(p), DRAFTED)[1]).draft(
        user_ref=bench.user_ref, task="Anything", persist=False
    )
    prompt = seen[0]
    assert "follow the style specification" in prompt
    assert "they do not override it" in prompt


def test_the_rules_are_rendered_as_prose_not_json(bench):
    seen: List[str] = []
    bench.approve_profile(USER_LEVEL_INTENT)
    bench.service(generate=lambda p: (seen.append(p), DRAFTED)[1]).draft(
        user_ref=bench.user_ref, task="Anything", persist=False
    )
    prompt = seen[0]
    assert '"structural"' not in prompt
    assert "Use contractions naturally" in prompt


def test_the_draft_is_told_the_examples_are_fiction(bench):
    """The exemplars carry invented suppliers and amounts. A model that lifts them would
    put fictional facts into a real email."""
    profile = bench.approve_profile(USER_LEVEL_INTENT)
    bench.exemplar_svc.generate_for(profile)
    seen: List[str] = []
    bench.service(generate=lambda p: (seen.append(p), DRAFTED)[1]).draft(
        user_ref=bench.user_ref, task="Anything", persist=False
    )
    assert "invented" in seen[0]
    assert "do not reuse them" in seen[0]


def test_drafting_works_with_no_exemplars_at_all(bench):
    seen: List[str] = []
    draft = bench.service(generate=lambda p: (seen.append(p), DRAFTED)[1]).draft(
        user_ref=bench.user_ref, task="Anything", persist=False
    )
    assert "none available" in seen[0]
    assert draft.body


def test_exemplars_come_from_the_scope_that_supplied_the_profile(bench):
    """Showing a user's own examples next to the house style would illustrate the wrong
    rules."""
    house = bench.approve_profile(USER_LEVEL_INTENT, user_ref=HOUSE_STYLE_USER_REF)
    bench.exemplar_svc.generate_for(house)

    draft = bench.service().draft(
        user_ref=bench.user_ref, task="Anything", intent="rfq_invite", persist=False
    )
    assert draft.provenance.fallback_level == LEVEL_HOUSE
    assert len(draft.provenance.exemplar_ids) == 3


# --- the governed system prompt ------------------------------------------------------

def test_the_system_prompt_comes_from_bp_prompt(bench):
    prompt, version = bench.service()._system_prompt()
    assert version >= 1, "the governed row should have been used"
    assert "follow the style specification" in prompt


def test_the_governed_prompt_and_the_python_fallback_are_identical(bench):
    """They must not drift: a draft generated on the fallback should read the same as one
    generated on the governed row."""
    with bench.conn.cursor() as cur:
        cur.execute(
            "SELECT prompts_desc->>'prompt_template' FROM proc.bp_prompt "
            "WHERE prompt_name = %s AND COALESCE(prompts_status, 1) = 1",
            (PROMPT_NAME,),
        )
        row = cur.fetchone()
    assert row, "the style_draft_system prompt is not seeded"
    assert row[0] == SYSTEM_PROMPT_FALLBACK


def test_an_unreachable_prompt_registry_reports_version_zero():
    """A draft must never be credited to a governed version it did not run under."""

    class _Broken:
        def cursor(self):
            raise RuntimeError("unreachable")

    prompt, version = StyleDraftingService(_Broken(), generate=lambda p: "")._system_prompt()
    assert prompt == SYSTEM_PROMPT_FALLBACK
    assert version == 0


# --- provenance ----------------------------------------------------------------------

def test_provenance_is_complete_and_persisted(bench):
    profile = bench.approve_profile(USER_LEVEL_INTENT)
    exemplars = bench.exemplar_svc.generate_for(profile)

    draft = bench.service().draft(
        user_ref=bench.user_ref, task="Ask for a quote on 25 chairs",
        workflow_id="WF-1", supplier_id="SUP-1", recipient_email="sam@example.com",
    )
    assert draft.draft_id is not None

    with bench.conn.cursor() as cur:
        cur.execute(
            "SELECT style_user_ref, style_intent, style_mode, style_profile_id, "
            "       style_profile_version, style_fallback_level, style_exemplar_ids, "
            "       style_exemplar_set_hash, style_model_id, style_prompt_version, "
            "       style_retrieved_at, subject, body, sent "
            "  FROM proc.draft_rfq_emails WHERE id = %s",
            (draft.draft_id,),
        )
        row = cur.fetchone()

    assert row[0] == bench.user_ref
    assert row[1] == USER_LEVEL_INTENT
    assert row[2] == "A"
    assert row[3] == profile.profile_id
    assert row[4] == profile.version
    assert row[5] == LEVEL_EXACT
    assert sorted(row[6]) == sorted(e.exemplar_id for e in exemplars)
    assert row[7] and len(row[7]) == 64          # sha256 hex
    assert row[8] == "injected"
    assert row[9] >= 1
    assert row[10] is not None
    assert row[11] and row[12]
    assert row[13] is False, "a style draft must never be persisted as already sent"


def test_the_exemplar_hash_identifies_the_set_not_its_order():
    """Two drafts built from the same three examples should fingerprint identically even
    when retrieval returned them in a different order."""
    assert _exemplar_set_hash([3, 1, 2]) == _exemplar_set_hash([1, 2, 3])
    assert _exemplar_set_hash([1, 2]) != _exemplar_set_hash([1, 2, 3])
    assert _exemplar_set_hash([]) is None


def test_fallback_level_is_returned_for_the_ui(bench):
    """Invariant 4 — levels 1-3 surface in the API response, not just the database."""
    draft = bench.service().draft(user_ref=bench.user_ref, task="Anything", persist=False)
    fields = draft.provenance.as_response_fields()
    assert fields["fallback_level"] == LEVEL_BASELINE
    assert fields["style_degraded"] is True
    assert fields["fallback_reason"]


def test_a_level_zero_draft_is_not_flagged_as_degraded(bench):
    bench.approve_profile(USER_LEVEL_INTENT)
    draft = bench.service().draft(user_ref=bench.user_ref, task="Anything", persist=False)
    assert draft.provenance.as_response_fields()["style_degraded"] is False


def test_a_baseline_draft_records_no_profile_id(bench):
    """Level 3 has no profile behind it, and must not claim one."""
    draft = bench.service().draft(user_ref=bench.user_ref, task="Anything")
    with bench.conn.cursor() as cur:
        cur.execute(
            "SELECT style_profile_id, style_profile_version, style_fallback_level "
            "FROM proc.draft_rfq_emails WHERE id = %s", (draft.draft_id,)
        )
        assert cur.fetchone() == (None, None, LEVEL_BASELINE)


def test_provenance_failure_is_raised_not_swallowed(bench):
    """A draft the user can see but that failed to record its provenance looks
    authoritative and cannot be explained."""

    class _Broken:
        def cursor(self):
            raise RuntimeError("write failed")

    svc = StyleDraftingService(
        _Broken(), resolver=StyleResolver(bench.conn),
        exemplars=bench.exemplar_svc, generate=lambda p: DRAFTED,
    )
    with pytest.raises(RuntimeError):
        svc.draft(user_ref=bench.user_ref, task="Anything")


# --- output shaping -------------------------------------------------------------------

@pytest.mark.parametrize("raw,subject,body_start", [
    ("Subject: Chairs — RFQ-1\n\nHi Sam,\n\nAll fine.", "Chairs — RFQ-1", "Hi Sam,"),
    ("subject: Lower case\n\nHi Sam,", "Lower case", "Hi Sam,"),
    ("Chairs — RFQ-1\n\nHi Sam,\n\nAll fine.", "Chairs — RFQ-1", "Hi Sam,"),
    ("Hi Sam,\nAll fine.", None, "Hi Sam,"),
    ("", None, ""),
])
def test_subject_is_split_off_the_body(raw, subject, body_start):
    got_subject, got_body = _split_subject_and_body(raw)
    assert got_subject == subject
    assert got_body.startswith(body_start)


# --- one drafting path ----------------------------------------------------------------

def test_the_style_path_has_no_send_capability():
    """Invariant 2, scoped to this subsystem. The platform's SES path is deliberately
    retained but must not be reachable from here."""
    import inspect

    import services.style.drafting as mod

    source = inspect.getsource(mod)
    for forbidden in ("EmailService", "EmailDispatchService", "send_email", "smtplib"):
        assert forbidden not in source, forbidden


def test_drafts_are_written_to_the_existing_table_not_a_second_one():
    """Two draft tables would mean two provenance stories and only one could be right."""
    import inspect

    import services.style.drafting as mod

    source = inspect.getsource(mod)
    assert "proc.draft_rfq_emails" in source
    assert "bp_style_draft" not in source


@pytest.mark.slow
def test_the_local_model_drafts_in_the_resolved_voice(bench):
    """End to end on the real model. Skipped where Ollama is unreachable."""
    import requests

    from services.ollama_client import OLLAMA_BASE_URL

    try:
        requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5).raise_for_status()
    except Exception as exc:
        pytest.skip(f"Ollama unreachable: {exc}")

    profile = bench.approve_profile(USER_LEVEL_INTENT)
    bench.exemplar_svc.generate_for(profile)

    draft = StyleDraftingService(bench.conn, exemplars=bench.exemplar_svc).draft(
        user_ref=bench.user_ref,
        task="Ask Meridian Supplies to quote for 25 height-adjustable desks, "
             "delivered by 30 April. We need unit rate and volume tiers.",
    )
    assert draft.body.strip()
    assert draft.provenance.fallback_level == LEVEL_EXACT
    assert draft.provenance.exemplar_ids
    assert draft.draft_id is not None
    # The examples are fiction; their invented facts must not surface in a real draft.
    for invented in ("RFQ-8812", "Q-7732", "Marta", "Ravi", "Ellen"):
        assert invented not in draft.body, f"{invented!r} leaked from an exemplar"


# --- template slots -------------------------------------------------------------------

def test_the_sender_slot_is_filled_when_a_name_is_supplied(bench):
    bench.approve_profile(USER_LEVEL_INTENT)
    draft = bench.service(
        generate=lambda p: "Subject: X\n\nHi Sam,\n\nAll fine.\n\n{sender_first_name}"
    ).draft(user_ref=bench.user_ref, task="Anything", sender_name="Nick", persist=False)
    assert draft.body.rstrip().endswith("Nick")
    assert "{" not in draft.body


def test_an_unfilled_slot_becomes_a_visible_gap_not_a_curly_brace(bench):
    """A square bracket reads as a gap to fill; a curly brace reads as a bug."""
    bench.approve_profile(USER_LEVEL_INTENT)
    draft = bench.service(
        generate=lambda p: "Subject: X\n\nHi {first_name},\n\nAll fine.\n\n{sender_first_name}"
    ).draft(user_ref=bench.user_ref, task="Anything", persist=False)
    assert "{" not in draft.body
    assert "[your name]" in draft.body
    assert "[contact name]" in draft.body


def test_the_grounding_guard_runs_on_every_draft(bench):
    """The prompt asks the model to invent nothing and it does anyway. The guard is what
    actually holds the line."""
    bench.approve_profile(USER_LEVEL_INTENT)
    draft = bench.service(
        generate=lambda p: "Subject: Quote — PO-99999\n\nHi Sam,\n\nBy 12 April.\n\nJo"
    ).draft(user_ref=bench.user_ref, task="Ask for a quote on 25 chairs", persist=False)
    assert "PO-99999" not in (draft.subject or "")
    assert "[reference]" in (draft.subject or "")
    assert draft.provenance.ungrounded_replacements
    assert draft.provenance.as_response_fields()["needs_review"] is True


@pytest.mark.parametrize("emitted", [
    "{sender_first_name}", "[sender_first_name]", "{sender_name}", "[Sender_First_Name]",
])
def test_the_sender_slot_is_filled_in_either_bracket_style(bench, emitted):
    """The profile stores {curly}, but a model that has just been told square brackets
    mark a gap frequently switches to [square] when copying the slot out. A substitution
    matching only the stored form leaves '[sender_first_name]' in a finished email."""
    bench.approve_profile(USER_LEVEL_INTENT)
    draft = bench.service(
        generate=lambda p: f"Subject: X\n\nHi Priya,\n\nAll fine.\n\n{emitted}"
    ).draft(user_ref=bench.user_ref, task="Priya: anything", sender_name="Nick", persist=False)
    assert draft.body.rstrip().endswith("Nick")
    assert "sender" not in draft.body.lower()
