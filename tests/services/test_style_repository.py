"""Profile versioning, approval and immutability — against a real Postgres.

These are integration tests on purpose. The guarantees under test are enforced by the
schema, not by Python: the partial unique index is what makes approval atomic, and the
freeze trigger is what makes an approved profile immutable. A mocked connection would
assert that this module issues the SQL it issues, which is not the same claim.

Skipped, not failed, where no database is reachable.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest

psycopg2 = pytest.importorskip("psycopg2")

from services.style.profile import parse_profile
from services.style.repository import (
    STATE_APPROVED,
    STATE_DRAFT,
    STATE_SUPERSEDED,
    USER_LEVEL_INTENT,
    ProfileNotApprovable,
    ProfileNotFound,
    StyleProfileRepository,
)

PROFILE = parse_profile({
    "structural": {
        "subject_pattern": "topic — reference",
        "greeting": "Hi {first_name},",
        "opening_move": "context_before_ask",
        "body_form": "short_prose",
        "target_words": [60, 110],
        "sign_off": "Nick",
        "signature_block": False,
    },
    "register": {
        "formality": 3, "directness": 4, "hedging": "low",
        "contractions": True, "person": "first_singular",
    },
    "lexical": {
        "preferred_terms": ["unit rate"],
        "banned_phrases": ["I hope this email finds you well"],
        "number_format": "bare_numerals", "date_format": "weekday_name",
    },
    "behavioural": {
        "cta_form": "proposes_specific_time",
        "deadline_phrasing": "soft_by_date",
        "escalation_ladder": ["neutral", "firm", "formal"],
    },
})


def _connect():
    try:
        from dotenv import load_dotenv
        load_dotenv(Path.cwd() / ".env")
    except Exception:  # pragma: no cover - dotenv is optional
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
            pytest.skip("bp_style_profile not migrated")
    return conn


@pytest.fixture
def repo():
    """A repository bound to a throwaway user_ref, cleaned up afterwards."""

    conn = _connect()
    user_ref = f"test-{uuid.uuid4().hex[:12]}"
    r = StyleProfileRepository(conn)
    r.test_user_ref = user_ref  # type: ignore[attr-defined]
    try:
        yield r
    finally:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM proc.bp_style_profile WHERE user_ref = %s", (user_ref,))
        conn.close()


# --- versioning --------------------------------------------------------------------

def test_a_compiled_profile_lands_in_draft_and_is_not_active(repo):
    """Invariant 6 — nothing auto-activates."""
    rec = repo.insert_version(
        user_ref=repo.test_user_ref, intent=USER_LEVEL_INTENT,
        profile=PROFILE, exemplar_count=5, source_batch_id="batch-1",
    )
    assert rec.version == 1
    assert rec.state == STATE_DRAFT
    assert rec.is_active is False
    assert rec.approved_by is None
    assert repo.get_active(repo.test_user_ref, USER_LEVEL_INTENT) is None


def test_recompiling_inserts_a_new_version_rather_than_editing(repo):
    first = repo.insert_version(user_ref=repo.test_user_ref, intent=USER_LEVEL_INTENT,
                                profile=PROFILE, exemplar_count=5)
    second = repo.insert_version(user_ref=repo.test_user_ref, intent=USER_LEVEL_INTENT,
                                 profile=PROFILE, exemplar_count=7)
    assert (first.version, second.version) == (1, 2)
    assert first.profile_id != second.profile_id
    assert len(repo.list_versions(repo.test_user_ref, USER_LEVEL_INTENT)) == 2


def test_versions_are_numbered_per_scope_not_globally(repo):
    user_level = repo.insert_version(user_ref=repo.test_user_ref, intent=USER_LEVEL_INTENT,
                                     profile=PROFILE, exemplar_count=5)
    per_intent = repo.insert_version(user_ref=repo.test_user_ref, intent="rfq_invite",
                                     profile=PROFILE, exemplar_count=3)
    assert user_level.version == 1 and per_intent.version == 1


def test_the_stored_json_round_trips_back_into_a_profile(repo):
    rec = repo.insert_version(user_ref=repo.test_user_ref, intent=USER_LEVEL_INTENT,
                              profile=PROFILE, exemplar_count=5)
    assert repo.get(rec.profile_id).as_profile() == PROFILE


# --- approval ----------------------------------------------------------------------

def test_approval_activates_and_records_who_did_it(repo):
    rec = repo.insert_version(user_ref=repo.test_user_ref, intent=USER_LEVEL_INTENT,
                              profile=PROFILE, exemplar_count=5)
    approved = repo.approve(rec.profile_id, "nick@example.com")
    assert approved.state == STATE_APPROVED
    assert approved.is_active is True
    assert approved.approved_by == "nick@example.com"
    assert approved.approved_at is not None
    assert repo.get_active(repo.test_user_ref, USER_LEVEL_INTENT).profile_id == rec.profile_id


def test_approving_n_plus_1_deactivates_n_atomically(repo):
    """The named test for invariant 5/6. The partial unique index on
    (user_ref, intent) WHERE is_active is what makes this atomic: a transaction that
    failed to stand v1 down could not have committed v2."""
    v1 = repo.insert_version(user_ref=repo.test_user_ref, intent=USER_LEVEL_INTENT,
                             profile=PROFILE, exemplar_count=5)
    repo.approve(v1.profile_id, "nick@example.com")

    v2 = repo.insert_version(user_ref=repo.test_user_ref, intent=USER_LEVEL_INTENT,
                             profile=PROFILE, exemplar_count=8)
    repo.approve(v2.profile_id, "nick@example.com")

    assert repo.get(v1.profile_id).state == STATE_SUPERSEDED
    assert repo.get(v1.profile_id).is_active is False
    assert repo.get(v2.profile_id).is_active is True

    # Exactly one active version, ever.
    actives = [p for p in repo.list_versions(repo.test_user_ref, USER_LEVEL_INTENT) if p.is_active]
    assert len(actives) == 1 and actives[0].profile_id == v2.profile_id


def test_two_active_profiles_for_one_scope_are_impossible(repo):
    """Belt and braces: even a direct UPDATE cannot produce a second active profile."""
    v1 = repo.insert_version(user_ref=repo.test_user_ref, intent=USER_LEVEL_INTENT,
                             profile=PROFILE, exemplar_count=5)
    repo.approve(v1.profile_id, "nick@example.com")
    v2 = repo.insert_version(user_ref=repo.test_user_ref, intent=USER_LEVEL_INTENT,
                             profile=PROFILE, exemplar_count=5)

    with repo._conn.cursor() as cur:
        with pytest.raises(psycopg2.errors.UniqueViolation):
            cur.execute(
                "UPDATE proc.bp_style_profile SET state='APPROVED', is_active=TRUE, "
                "approved_by='forced', approved_at=NOW() WHERE profile_id=%s",
                (v2.profile_id,),
            )


def test_only_a_draft_can_be_approved(repo):
    rec = repo.insert_version(user_ref=repo.test_user_ref, intent=USER_LEVEL_INTENT,
                              profile=PROFILE, exemplar_count=5)
    repo.approve(rec.profile_id, "nick@example.com")
    with pytest.raises(ProfileNotApprovable):
        repo.approve(rec.profile_id, "someone@example.com")


def test_approving_something_that_does_not_exist_raises(repo):
    with pytest.raises(ProfileNotFound):
        repo.approve(-1, "nick@example.com")


def test_approval_requires_a_named_approver(repo):
    rec = repo.insert_version(user_ref=repo.test_user_ref, intent=USER_LEVEL_INTENT,
                              profile=PROFILE, exemplar_count=5)
    for empty in ("", "   ", None):
        with pytest.raises(ValueError):
            repo.approve(rec.profile_id, empty)


# --- immutability (invariant 5) ----------------------------------------------------

def test_updating_profile_json_on_an_approved_row_raises(repo):
    """The named test for invariant 5. Enforced by a trigger, not by this repository,
    because a support script at 2am does not import Python."""
    rec = repo.insert_version(user_ref=repo.test_user_ref, intent=USER_LEVEL_INTENT,
                              profile=PROFILE, exemplar_count=5)
    repo.approve(rec.profile_id, "nick@example.com")

    with repo._conn.cursor() as cur:
        with pytest.raises(psycopg2.Error, match="immutable"):
            cur.execute(
                "UPDATE proc.bp_style_profile SET profile_json = %s WHERE profile_id = %s",
                ('{"tampered": true}', rec.profile_id),
            )


def test_a_superseded_profile_is_immutable_too(repo):
    """A superseded version was approved once, and the drafts citing it must keep
    pointing at the rules that actually produced them."""
    v1 = repo.insert_version(user_ref=repo.test_user_ref, intent=USER_LEVEL_INTENT,
                             profile=PROFILE, exemplar_count=5)
    repo.approve(v1.profile_id, "nick@example.com")
    v2 = repo.insert_version(user_ref=repo.test_user_ref, intent=USER_LEVEL_INTENT,
                             profile=PROFILE, exemplar_count=5)
    repo.approve(v2.profile_id, "nick@example.com")

    with repo._conn.cursor() as cur:
        with pytest.raises(psycopg2.Error, match="immutable"):
            cur.execute(
                "UPDATE proc.bp_style_profile SET profile_json = %s WHERE profile_id = %s",
                ('{"tampered": true}', v1.profile_id),
            )


def test_a_draft_is_still_editable(repo):
    """The freeze applies from approval onward; a draft is still being worked on."""
    rec = repo.insert_version(user_ref=repo.test_user_ref, intent=USER_LEVEL_INTENT,
                              profile=PROFILE, exemplar_count=5)
    with repo._conn.cursor() as cur:
        cur.execute(
            "UPDATE proc.bp_style_profile SET profile_json = %s WHERE profile_id = %s",
            ('{"still": "drafting"}', rec.profile_id),
        )
    assert repo.get(rec.profile_id).profile_json == {"still": "drafting"}


def test_an_approved_profile_cannot_be_walked_back_to_draft(repo):
    rec = repo.insert_version(user_ref=repo.test_user_ref, intent=USER_LEVEL_INTENT,
                              profile=PROFILE, exemplar_count=5)
    repo.approve(rec.profile_id, "nick@example.com")
    with repo._conn.cursor() as cur:
        with pytest.raises(psycopg2.Error, match="cannot return to"):
            cur.execute(
                "UPDATE proc.bp_style_profile SET state='DRAFT' WHERE profile_id=%s",
                (rec.profile_id,),
            )


def test_the_identity_of_an_approved_profile_is_frozen(repo):
    rec = repo.insert_version(user_ref=repo.test_user_ref, intent=USER_LEVEL_INTENT,
                              profile=PROFILE, exemplar_count=5)
    repo.approve(rec.profile_id, "nick@example.com")
    with repo._conn.cursor() as cur:
        with pytest.raises(psycopg2.Error, match="identity of an approved profile"):
            cur.execute(
                "UPDATE proc.bp_style_profile SET version = 99 WHERE profile_id = %s",
                (rec.profile_id,),
            )


# --- deactivation ------------------------------------------------------------------

def test_deactivating_leaves_no_active_profile(repo):
    """Deleting a profile must revert drafting to the fallback ladder, visibly."""
    rec = repo.insert_version(user_ref=repo.test_user_ref, intent=USER_LEVEL_INTENT,
                              profile=PROFILE, exemplar_count=5)
    repo.approve(rec.profile_id, "nick@example.com")
    assert repo.deactivate(repo.test_user_ref, USER_LEVEL_INTENT) == 1
    assert repo.get_active(repo.test_user_ref, USER_LEVEL_INTENT) is None
    assert repo.get(rec.profile_id).state == STATE_SUPERSEDED


def test_the_user_level_scope_is_flagged_as_such(repo):
    rec = repo.insert_version(user_ref=repo.test_user_ref, intent=USER_LEVEL_INTENT,
                              profile=PROFILE, exemplar_count=5)
    assert rec.is_user_level is True
    per_intent = repo.insert_version(user_ref=repo.test_user_ref, intent="rfq_invite",
                                     profile=PROFILE, exemplar_count=3)
    assert per_intent.is_user_level is False
