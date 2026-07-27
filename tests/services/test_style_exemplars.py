"""Exemplar generation, replacement and retrieval.

The load-bearing claim is that synthetic exemplars are safe to store indefinitely and put
in every prompt, because they are fiction generated from the profile alone. That is a
structural argument, so it gets a structural test: the generation prompt is captured and
checked to contain nothing but the profile.

Retrieval runs against real pgvector. Skipped, not failed, where no database is reachable.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import List

import pytest

psycopg2 = pytest.importorskip("psycopg2")

from services.style.exemplars import (
    ORIGIN_SYNTHETIC,
    ExemplarGenerationError,
    ExemplarService,
    approve_and_generate,
)
from services.style.profile import parse_profile
from services.style.rendering import render_profile_rules
from services.style.repository import USER_LEVEL_INTENT, StyleProfileRepository
from tests.services.test_style_compiler import EMAILS, STUB_PROFILE

PROFILE = parse_profile(STUB_PROFILE)

GENERATED = {
    "emails": [
        {"subject": "Pallet racking — RFQ-8812",
         "body": "Hi Marta,\n\nWe're fitting out the Leeds unit and need 60 pallet "
                 "positions by the end of May. Could you send unit rate and volume "
                 "tiers?\n\nCan we talk Wednesday at 11?\n\nJo"},
        {"subject": "Cleaning contract — renewal",
         "body": "Hi Ravi,\n\nOur cleaning contract runs out in August and I'd like to "
                 "test the market. Can you quote on the same scope?\n\nI need it back by "
                 "Friday.\n\nJo"},
        {"subject": "Forklift servicing — outcome",
         "body": "Hi Ellen,\n\nThanks for quoting. We've gone with another supplier this "
                 "time — response times decided it.\n\nWorth revisiting in the "
                 "autumn?\n\nJo"},
    ]
}


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
        cur.execute("SELECT to_regclass('proc.bp_style_exemplar')")
        if cur.fetchone()[0] is None:
            pytest.skip("style tables not migrated")
    return conn


def _fake_embed(text: str) -> List[float]:
    """A deterministic stand-in: direction driven by a couple of marker words, so
    'nearer' and 'further' are meaningful without loading a real model."""
    vector = [0.0] * 1024
    vector[0] = 1.0 if "racking" in text.lower() else 0.0
    vector[1] = 1.0 if "cleaning" in text.lower() else 0.0
    vector[2] = 1.0 if "forklift" in text.lower() else 0.0
    if not any(vector[:3]):
        vector[3] = 1.0
    return vector


class _Bench:
    def __init__(self, conn):
        self.conn = conn
        self.user_ref = f"test-{uuid.uuid4().hex[:12]}"
        self.repo = StyleProfileRepository(conn)

    def approved_profile(self, intent=USER_LEVEL_INTENT):
        rec = self.repo.insert_version(
            user_ref=self.user_ref, intent=intent, profile=PROFILE, exemplar_count=5
        )
        return self.repo.approve(rec.profile_id, "tester")

    def service(self, generate=None):
        return ExemplarService(
            self.conn,
            generate=generate or (lambda p: json.dumps(GENERATED)),
            embed=_fake_embed,
        )

    def active_exemplars(self):
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT exemplar_id, profile_version_ref, origin, is_active "
                "FROM proc.bp_style_exemplar WHERE user_ref = %s ORDER BY exemplar_id",
                (self.user_ref,),
            )
            return cur.fetchall()

    def cleanup(self):
        with self.conn.cursor() as cur:
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


# --- generation --------------------------------------------------------------------

def test_approval_produces_exemplars(bench):
    profile = bench.approved_profile()
    stored = bench.service().generate_for(profile)
    assert len(stored) == 3
    assert all(e.origin == ORIGIN_SYNTHETIC for e in stored)
    assert all(e.is_active for e in stored)


def test_exemplars_cite_the_profile_version_they_demonstrate(bench):
    profile = bench.approved_profile()
    stored = bench.service().generate_for(profile)
    assert all(e.profile_version_ref == profile.version for e in stored)


def test_the_embedding_is_stored_and_the_model_recorded(bench):
    profile = bench.approved_profile()
    bench.service().generate_for(profile)
    with bench.conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM proc.bp_style_exemplar "
            "WHERE user_ref = %s AND embedding IS NOT NULL AND embedding_model IS NOT NULL",
            (bench.user_ref,),
        )
        assert cur.fetchone()[0] == 3


# --- invariant 1, structurally -----------------------------------------------------

def test_the_generator_is_shown_the_profile_and_nothing_else(bench):
    """This is what makes a synthetic exemplar safe to keep forever and put in every
    prompt. It is fiction derived from rules, not a copy of anyone's mail."""
    seen: List[str] = []

    def _capture(prompt: str) -> str:
        seen.append(prompt)
        return json.dumps(GENERATED)

    profile = bench.approved_profile()
    bench.service(generate=_capture).generate_for(profile)

    prompt = seen[0]
    for secret in ("Sam", "Dana", "Priya", "Ade", "1,240.50", "QUT136586", "racking"):
        assert secret not in prompt, f"{secret!r} reached the exemplar generator"
    # It did get the rules.
    assert "Hi {first_name}," in prompt
    assert "INVENTED" in prompt


def test_the_generation_prompt_is_derived_only_from_profile_json(bench):
    """Belt and braces: whatever the renderer emits must be reconstructible from the
    profile alone."""
    profile = bench.approved_profile()
    rules = render_profile_rules(profile.as_profile())
    source_text = " ".join(f"{s} {b}" for s, b in EMAILS).lower()
    for line in rules.splitlines():
        # No rule may be a phrase lifted from the corpus. Blank lines are skipped: an
        # empty string is trivially "in" anything and would make this assert nothing.
        payload = line.split(". ", 1)[-1].strip().lower()
        if len(payload) < 20:
            continue
        assert payload[:40] not in source_text, line


# --- replacement on recompile ------------------------------------------------------

def test_recompiling_replaces_the_previous_exemplar_set(bench):
    """A prompt must never mix exemplars generated against two versions of the rules."""
    v1 = bench.approved_profile()
    first = bench.service().generate_for(v1)

    v2_draft = bench.repo.insert_version(
        user_ref=bench.user_ref, intent=USER_LEVEL_INTENT, profile=PROFILE, exemplar_count=7
    )
    v2 = bench.repo.approve(v2_draft.profile_id, "tester")
    second = bench.service().generate_for(v2)

    rows = {r[0]: r for r in bench.active_exemplars()}
    for e in first:
        assert rows[e.exemplar_id][3] is False, "an old exemplar is still active"
    for e in second:
        assert rows[e.exemplar_id][3] is True
    assert {r[1] for r in rows.values() if r[3]} == {v2.version}


def test_retrieval_only_ever_sees_the_current_generation(bench):
    v1 = bench.approved_profile()
    bench.service().generate_for(v1)
    v2_draft = bench.repo.insert_version(
        user_ref=bench.user_ref, intent=USER_LEVEL_INTENT, profile=PROFILE, exemplar_count=7
    )
    v2 = bench.repo.approve(v2_draft.profile_id, "tester")
    bench.service().generate_for(v2)

    found = bench.service().retrieve(bench.user_ref, USER_LEVEL_INTENT, "anything")
    assert {e.profile_version_ref for e in found} == {v2.version}


# --- retrieval ---------------------------------------------------------------------

def test_retrieval_is_ordered_by_proximity_to_the_task(bench):
    profile = bench.approved_profile()
    svc = bench.service()
    svc.generate_for(profile)

    found = svc.retrieve(bench.user_ref, USER_LEVEL_INTENT, "quote for pallet racking")
    assert found
    assert "racking" in (found[0].subject or "").lower()
    distances = [e.distance for e in found]
    assert distances == sorted(distances), distances


def test_retrieval_is_scoped_to_the_intent_asked_for(bench):
    """Exact match only. Choosing which intent to ask for is the fallback ladder's job."""
    user_level = bench.approved_profile(USER_LEVEL_INTENT)
    bench.service().generate_for(user_level)

    per_intent_draft = bench.repo.insert_version(
        user_ref=bench.user_ref, intent="rfq_invite", profile=PROFILE, exemplar_count=3
    )
    per_intent = bench.repo.approve(per_intent_draft.profile_id, "tester")
    bench.service().generate_for(per_intent)

    assert all(e.intent == "rfq_invite"
               for e in bench.service().retrieve(bench.user_ref, "rfq_invite", "x"))
    assert all(e.intent == USER_LEVEL_INTENT
               for e in bench.service().retrieve(bench.user_ref, USER_LEVEL_INTENT, "x"))


def test_retrieval_is_capped_at_three(bench):
    profile = bench.approved_profile()
    bench.service().generate_for(profile)
    assert len(bench.service().retrieve(bench.user_ref, USER_LEVEL_INTENT, "x")) <= 3


def test_a_scope_with_no_exemplars_returns_nothing_rather_than_raising(bench):
    assert bench.service().retrieve(bench.user_ref, "escalation", "x") == []


def test_an_unreachable_embedder_degrades_to_recency_rather_than_failing(bench):
    """Drafting must not go down because the embedding model is unavailable."""
    profile = bench.approved_profile()
    bench.service().generate_for(profile)

    def _broken(text):
        raise RuntimeError("model unavailable")

    svc = ExemplarService(bench.conn, generate=lambda p: "", embed=_broken)
    found = svc.retrieve(bench.user_ref, USER_LEVEL_INTENT, "quote for pallet racking")
    assert len(found) == 3
    assert all(e.distance is None for e in found)


def test_retrieval_without_a_task_still_returns_exemplars(bench):
    profile = bench.approved_profile()
    bench.service().generate_for(profile)
    assert len(bench.service().retrieve(bench.user_ref, USER_LEVEL_INTENT)) == 3


# --- failure handling --------------------------------------------------------------

def test_junk_from_the_model_raises_rather_than_storing_nothing_silently(bench):
    profile = bench.approved_profile()
    with pytest.raises(ExemplarGenerationError, match="did not return JSON"):
        bench.service(generate=lambda p: "sorry").generate_for(profile)


def test_an_empty_email_list_raises(bench):
    profile = bench.approved_profile()
    with pytest.raises(ExemplarGenerationError, match="no emails"):
        bench.service(generate=lambda p: json.dumps({"emails": []})).generate_for(profile)


def test_a_failed_generation_leaves_the_previous_set_alone(bench):
    """Deactivating the old set before knowing the new one is usable would leave the
    scope with no exemplars at all."""
    profile = bench.approved_profile()
    first = bench.service().generate_for(profile)
    with pytest.raises(ExemplarGenerationError):
        bench.service(generate=lambda p: "not json").generate_for(profile)
    rows = {r[0]: r for r in bench.active_exemplars()}
    assert all(rows[e.exemplar_id][3] is True for e in first)


def test_approval_survives_a_generation_failure(bench):
    """Approval and generation are separate concerns. A profile that cannot be
    illustrated is still a profile."""
    draft = bench.repo.insert_version(
        user_ref=bench.user_ref, intent=USER_LEVEL_INTENT, profile=PROFILE, exemplar_count=5
    )
    record, exemplars = approve_and_generate(
        draft.profile_id, "tester", conn=bench.conn,
        service=bench.service(generate=lambda p: "not json"),
    )
    assert record.is_active is True
    assert exemplars == []


def test_approve_and_generate_does_both_on_the_happy_path(bench):
    draft = bench.repo.insert_version(
        user_ref=bench.user_ref, intent=USER_LEVEL_INTENT, profile=PROFILE, exemplar_count=5
    )
    record, exemplars = approve_and_generate(
        draft.profile_id, "tester", conn=bench.conn, service=bench.service()
    )
    assert record.is_active is True
    assert len(exemplars) == 3


# --- the rendered rules ------------------------------------------------------------

def test_the_rules_read_as_instructions_not_as_json():
    rules = render_profile_rules(PROFILE)
    assert "{" not in rules.replace("{first_name}", "")
    assert "Hi {first_name}," in rules
    assert "50-90 words" in rules
    assert "Use contractions" in rules
    assert "Never use these phrases" in rules


def test_every_profile_field_reaches_the_rules():
    """A field that never makes it into the prompt is a field the model is not held to."""
    rules = render_profile_rules(PROFILE).lower()
    for expected in (
        "topic — reference",          # subject_pattern
        "hi {first_name},",           # greeting
        "context first",              # opening_move
        "short sentences",            # body_form
        "50-90 words",                # target_words
        "direct",                     # directness
        "hedge rarely",               # hedging
        "first person singular",      # person
        "contractions",               # contractions
        "numerals",                   # number_format
        "weekday",                    # date_format
        "proposing a specific time",  # cta_form
        "softly",                     # deadline_phrasing
        "sign off with: nick",        # sign_off
        "do not add a signature",     # signature_block
        "unit rate",                  # preferred_terms
        "i hope this email finds",    # banned_phrases
        "neutral -> firm",            # escalation_ladder
    ):
        assert expected in rules, expected


@pytest.mark.slow
def test_the_local_model_generates_usable_exemplars(bench):
    """The case that can actually go wrong. Skipped where Ollama is unreachable."""
    import requests

    from services.ollama_client import OLLAMA_BASE_URL

    try:
        requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5).raise_for_status()
    except Exception as exc:
        pytest.skip(f"Ollama unreachable: {exc}")

    profile = bench.approved_profile()
    svc = ExemplarService(bench.conn, embed=_fake_embed)
    stored = svc.generate_for(profile)
    assert 2 <= len(stored) <= 3
    for e in stored:
        assert e.body.strip()
        assert e.origin == ORIGIN_SYNTHETIC


def test_placeholder_guidance_is_kept_off_the_template_lines():
    """An earlier version put the explanation as a parenthetical next to the sign-off
    template, and the model copied it into the email — output read "Alex  (Alex)".
    Guidance sitting inside a template gets treated as part of the template."""
    from services.style.profile import parse_profile

    payload = json.loads(json.dumps(STUB_PROFILE))
    payload["structural"]["sign_off"] = "{sender_first_name}"
    rules = render_profile_rules(parse_profile(payload))

    sign_off_line = next(l for l in rules.splitlines() if l.startswith("14."))
    assert sign_off_line.strip() == "14. Sign off with: {sender_first_name}"
    assert "Placeholder meanings" in rules
    assert "never output the placeholder itself" in rules


def test_generated_exemplars_carry_no_placeholders(bench):
    """A finished example email with {first_name} still in it is not an example."""
    profile = bench.approved_profile()
    stored = bench.service().generate_for(profile)
    for e in stored:
        assert "{" not in e.body, e.body
        assert "[NAME]" not in e.body
