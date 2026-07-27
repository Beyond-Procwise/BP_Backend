"""The feedback loop: measure the drift, suggest, never act.

Two claims carry this phase and both get direct tests:

* **The sent text is never stored** — ``test_the_sent_text_reaches_no_column``, backed by
  a schema-level check that no column could hold it.
* **Nothing recompiles itself** — ``test_accepting_a_suggestion_changes_no_profile``.

The rest is arithmetic on the score and the sustained-drift rule.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest

psycopg2 = pytest.importorskip("psycopg2")

from services.style.diffing import diff_profiles, render_diff
from services.style.feedback import (
    DIVERGENCE_THRESHOLD,
    MIN_OBSERVATIONS,
    OBSERVED_MANUAL,
    StyleFeedbackService,
    divergence_score,
)
from services.style.profile import parse_profile
from services.style.repository import USER_LEVEL_INTENT, StyleProfileRepository
from tests.services.test_style_compiler import STUB_PROFILE

PROFILE = parse_profile(STUB_PROFILE)

DRAFTED = ("Hi Priya,\n\nWe're requesting a quote for 25 height-adjustable desks under "
           "PO-4471. Please provide the unit rate and volume tier pricing. Delivery must "
           "be by 30 April.\n\nNick")


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
        cur.execute("SELECT to_regclass('proc.bp_style_divergence')")
        if cur.fetchone()[0] is None:
            pytest.skip("feedback tables not migrated")
    return conn


class _Bench:
    def __init__(self, conn):
        self.conn = conn
        self.user_ref = f"test-{uuid.uuid4().hex[:12]}"
        self.svc = StyleFeedbackService(conn)
        self.profiles = StyleProfileRepository(conn)
        self.profile = None

    def approved_profile(self):
        if self.profile is None:
            rec = self.profiles.insert_version(
                user_ref=self.user_ref, intent=USER_LEVEL_INTENT,
                profile=PROFILE, exemplar_count=5,
            )
            self.profile = self.profiles.approve(rec.profile_id, "tester")
        return self.profile

    def draft(self, body=DRAFTED, *, with_style=True):
        profile = self.approved_profile() if with_style else None
        unique = f"FB-{uuid.uuid4().hex[:12]}"
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO proc.draft_rfq_emails "
                "(rfq_id, unique_id, subject, body, sent, style_user_ref, style_intent, "
                " style_profile_id, style_profile_version, style_fallback_level) "
                "VALUES (%s, %s, 'Desks', %s, FALSE, %s, %s, %s, %s, 0) RETURNING id",
                (unique, unique, body,
                 self.user_ref if with_style else None,
                 USER_LEVEL_INTENT if with_style else None,
                 profile.profile_id if profile else None,
                 profile.version if profile else None),
            )
            return cur.fetchone()[0]

    def cleanup(self):
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM proc.bp_style_recompile_suggestion WHERE user_ref = %s",
                        (self.user_ref,))
            cur.execute(
                "DELETE FROM proc.bp_style_divergence WHERE user_ref = %s", (self.user_ref,))
            cur.execute("DELETE FROM proc.draft_rfq_emails WHERE style_user_ref = %s",
                        (self.user_ref,))
            cur.execute("DELETE FROM proc.draft_rfq_emails WHERE rfq_id LIKE 'FB-%%'")
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


# --- the score ---------------------------------------------------------------------

def test_an_unedited_draft_scores_zero():
    assert divergence_score(DRAFTED, DRAFTED) == 0.0


def test_a_completely_rewritten_draft_scores_one():
    assert divergence_score("alpha beta gamma", "delta epsilon zeta") == 1.0


def test_a_small_correction_scores_small():
    """Filling in a placeholder or fixing a date is ordinary editing, not a wrong profile."""
    edited = DRAFTED.replace("30 April", "1 May")
    assert 0 < divergence_score(DRAFTED, edited) < 0.15


def test_the_score_is_word_level_not_character_level():
    """'30 April' -> '1 May' should read as one edit, not seven."""
    a, b = "delivery by 30 April please", "delivery by 1 May please"
    assert divergence_score(a, b) <= 0.4


def test_case_and_punctuation_do_not_count_as_edits():
    assert divergence_score("Hi Priya, all fine.", "hi priya all fine") == 0.0


@pytest.mark.parametrize("a,b,expected", [
    ("", "", 0.0),
    ("something", "", 1.0),
    ("", "something", 1.0),
])
def test_degenerate_comparisons(a, b, expected):
    assert divergence_score(a, b) == expected


def test_the_score_is_symmetric_in_magnitude():
    """Deleting half an email and rewriting it from scratch both mean the draft was
    wrong; normalising by the longer side keeps them comparable."""
    long_text = " ".join(["word"] * 100)
    short_text = "word word"
    assert divergence_score(long_text, short_text) == divergence_score(short_text, long_text)


# --- capture -----------------------------------------------------------------------

def test_a_divergence_is_recorded_against_the_draft(bench):
    draft_id = bench.draft()
    observed = bench.svc.record_divergence(
        draft_id=draft_id, sent_body=DRAFTED.replace("30 April", "1 May"))

    assert observed is not None
    assert observed.draft_id == draft_id
    assert observed.user_ref == bench.user_ref
    assert observed.style_profile_version == bench.profile.version
    assert 0 < observed.score < 0.2


def test_the_sent_text_reaches_no_column(bench):
    """The claim that makes this loop acceptable. The sent body is a parameter, never a
    column, and does not outlive the call."""
    secret = ("Hi Priya, CONFIDENTIAL BOARD MATTER, the acquisition price is "
              "£4.2 million and Meridian must not hear of it.")
    draft_id = bench.draft()
    bench.svc.record_divergence(draft_id=draft_id, sent_body=secret)

    with bench.conn.cursor() as cur:
        # Every text-ish column in both feedback tables, concatenated.
        cur.execute(
            "SELECT COALESCE(user_ref,'') || COALESCE(intent,'') || COALESCE(observed_via,'') "
            "  FROM proc.bp_style_divergence WHERE draft_id = %s", (draft_id,))
        blob = "".join(r[0] for r in cur.fetchall())
    for fragment in ("CONFIDENTIAL", "acquisition", "4.2", "Meridian"):
        assert fragment not in blob, fragment


def test_no_column_in_the_feedback_tables_could_hold_an_email(bench):
    """Belt and braces at the schema level: a future column that could hold a body should
    fail this test before it ships."""
    with bench.conn.cursor() as cur:
        cur.execute(
            "SELECT table_name, column_name FROM information_schema.columns "
            " WHERE table_schema = 'proc' "
            "   AND table_name IN ('bp_style_divergence','bp_style_recompile_suggestion') "
            "   AND data_type IN ('text','character varying')")
        columns = {(t, c) for t, c in cur.fetchall()}

    allowed = {"user_ref", "intent", "observed_via", "status", "actioned_by", "reason"}
    unexpected = {(t, c) for t, c in columns if c not in allowed}
    assert not unexpected, f"unexpected free-text columns: {unexpected}"


def test_a_draft_with_no_style_provenance_is_not_scored(bench):
    """Someone typed it. There is no profile to hold responsible."""
    draft_id = bench.draft(with_style=False)
    assert bench.svc.record_divergence(draft_id=draft_id, sent_body="anything") is None


def test_a_missing_draft_is_not_scored(bench):
    assert bench.svc.record_divergence(draft_id=-1, sent_body="anything") is None


def test_a_draft_is_only_observed_once(bench):
    """A draft cannot be sent twice, and a second row would double-count it in every
    average built from this table."""
    draft_id = bench.draft()
    bench.svc.record_divergence(draft_id=draft_id, sent_body="one version")
    bench.svc.record_divergence(draft_id=draft_id, sent_body="a totally different version")

    with bench.conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM proc.bp_style_divergence WHERE draft_id = %s",
                    (draft_id,))
        assert cur.fetchone()[0] == 1


def test_how_the_sent_version_was_observed_is_recorded(bench):
    """A score from a mailbox read and one a user typed in are different kinds of
    evidence."""
    draft_id = bench.draft()
    bench.svc.record_divergence(draft_id=draft_id, sent_body="x", observed_via=OBSERVED_MANUAL)
    with bench.conn.cursor() as cur:
        cur.execute("SELECT observed_via FROM proc.bp_style_divergence WHERE draft_id = %s",
                    (draft_id,))
        assert cur.fetchone()[0] == OBSERVED_MANUAL


# --- sustained drift ----------------------------------------------------------------

def _rewrite(n: int) -> str:
    return " ".join(f"completely different sentence {i}" for i in range(n))


def test_one_rewritten_draft_is_not_a_pattern(bench):
    """A bad afternoon, not a wrong profile."""
    bench.svc.record_divergence(draft_id=bench.draft(), sent_body=_rewrite(20))
    assert bench.svc.suggest_if_sustained(bench.user_ref, USER_LEVEL_INTENT) is None


def test_many_lightly_edited_drafts_are_a_profile_working(bench):
    """Twenty small corrections are not evidence of anything wrong."""
    for _ in range(6):
        draft_id = bench.draft()
        bench.svc.record_divergence(
            draft_id=draft_id, sent_body=DRAFTED.replace("30 April", "1 May"))
    assert bench.svc.suggest_if_sustained(bench.user_ref, USER_LEVEL_INTENT) is None


def test_sustained_rewriting_raises_a_suggestion(bench):
    for _ in range(MIN_OBSERVATIONS):
        bench.svc.record_divergence(draft_id=bench.draft(), sent_body=_rewrite(20))

    suggestion = bench.svc.suggest_if_sustained(bench.user_ref, USER_LEVEL_INTENT)
    assert suggestion is not None
    assert suggestion.status == "pending"
    assert suggestion.observation_count >= MIN_OBSERVATIONS
    assert suggestion.mean_score >= DIVERGENCE_THRESHOLD
    assert "edited" in suggestion.reason


def test_only_one_suggestion_is_open_at_a_time(bench):
    """A repeated sweep should tell the user once, not build a queue."""
    for _ in range(MIN_OBSERVATIONS):
        bench.svc.record_divergence(draft_id=bench.draft(), sent_body=_rewrite(20))

    first = bench.svc.suggest_if_sustained(bench.user_ref, USER_LEVEL_INTENT)
    second = bench.svc.suggest_if_sustained(bench.user_ref, USER_LEVEL_INTENT)
    assert first is not None and second is None

    with bench.conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM proc.bp_style_recompile_suggestion "
                    "WHERE user_ref = %s AND status = 'pending'", (bench.user_ref,))
        assert cur.fetchone()[0] == 1


def test_the_open_suggestion_can_be_read_back(bench):
    for _ in range(MIN_OBSERVATIONS):
        bench.svc.record_divergence(draft_id=bench.draft(), sent_body=_rewrite(20))
    bench.svc.suggest_if_sustained(bench.user_ref, USER_LEVEL_INTENT)

    open_one = bench.svc.open_suggestion(bench.user_ref, USER_LEVEL_INTENT)
    assert open_one is not None and open_one.status == "pending"


# --- no auto-recompilation ------------------------------------------------------------

def test_accepting_a_suggestion_changes_no_profile(bench):
    """The named test. Accepting records a decision — it does not recompile, and it does
    not activate anything. Silently improving someone's writing profile would change how
    their mail reads without them agreeing to it."""
    for _ in range(MIN_OBSERVATIONS):
        bench.svc.record_divergence(draft_id=bench.draft(), sent_body=_rewrite(20))
    suggestion = bench.svc.suggest_if_sustained(bench.user_ref, USER_LEVEL_INTENT)

    before = bench.profiles.list_versions(bench.user_ref, USER_LEVEL_INTENT)
    resolved = bench.svc.resolve_suggestion(
        suggestion.suggestion_id, status="accepted", actioned_by="nick@example.com")
    after = bench.profiles.list_versions(bench.user_ref, USER_LEVEL_INTENT)

    assert resolved.status == "accepted"
    assert len(after) == len(before), "a new profile version appeared without a compile"
    assert [p.profile_id for p in after if p.is_active] == \
           [p.profile_id for p in before if p.is_active], "the active profile changed"


def test_the_feedback_service_cannot_compile_or_approve():
    """Structural: the module has no route to a profile write at all."""
    import inspect

    import services.style.feedback as mod

    source = inspect.getsource(mod)
    for forbidden in ("StyleCompiler", "compile_for", "insert_version", "approve("):
        assert forbidden not in source, forbidden


def test_dismissing_suppresses_the_suggestion(bench):
    for _ in range(MIN_OBSERVATIONS):
        bench.svc.record_divergence(draft_id=bench.draft(), sent_body=_rewrite(20))
    suggestion = bench.svc.suggest_if_sustained(bench.user_ref, USER_LEVEL_INTENT)

    bench.svc.resolve_suggestion(
        suggestion.suggestion_id, status="dismissed", actioned_by="nick@example.com")
    assert bench.svc.open_suggestion(bench.user_ref, USER_LEVEL_INTENT) is None


def test_resolving_requires_a_named_person(bench):
    for _ in range(MIN_OBSERVATIONS):
        bench.svc.record_divergence(draft_id=bench.draft(), sent_body=_rewrite(20))
    suggestion = bench.svc.suggest_if_sustained(bench.user_ref, USER_LEVEL_INTENT)

    for empty in ("", "   ", None):
        with pytest.raises(ValueError):
            bench.svc.resolve_suggestion(
                suggestion.suggestion_id, status="accepted", actioned_by=empty)


def test_a_suggestion_is_accepted_or_dismissed_and_nothing_else(bench):
    for _ in range(MIN_OBSERVATIONS):
        bench.svc.record_divergence(draft_id=bench.draft(), sent_body=_rewrite(20))
    suggestion = bench.svc.suggest_if_sustained(bench.user_ref, USER_LEVEL_INTENT)

    with pytest.raises(ValueError, match="accepted or dismissed"):
        bench.svc.resolve_suggestion(
            suggestion.suggestion_id, status="applied", actioned_by="nick")


# --- the profile diff -----------------------------------------------------------------

def test_an_identical_profile_shows_no_changes():
    assert diff_profiles(PROFILE, PROFILE) == []
    assert "No changes" in render_diff(PROFILE, PROFILE)


def test_a_changed_field_is_described_in_plain_language():
    import json

    after = json.loads(json.dumps(STUB_PROFILE))
    after["register"]["formality"] = 4
    after["structural"]["target_words"] = [90, 140]

    text = render_diff(PROFILE, parse_profile(after))
    assert "Formality (1-5): 2 → 4" in text
    assert "Typical length" in text
    assert "structural.target_words" not in text, "the raw schema path leaked into the UI"


def test_list_changes_report_what_was_added_and_removed():
    import json

    after = json.loads(json.dumps(STUB_PROFILE))
    after["lexical"]["preferred_terms"] = ["unit rate", "lead time", "payment terms"]

    change = next(c for c in diff_profiles(PROFILE, parse_profile(after))
                  if c.path == "lexical.preferred_terms")
    assert change.added == ["payment terms"]
    assert change.removed == ["volume tiers"]
    assert "added “payment terms”" in change.describe()
    assert "removed “volume tiers”" in change.describe()


def test_a_tuple_and_a_list_of_the_same_values_are_not_a_change():
    """target_words round-trips through JSON as a list and comes back from Pydantic as a
    tuple. Reporting that as a change would put a phantom line in every diff."""
    assert diff_profiles(PROFILE.to_json_dict(), PROFILE) == []


def test_booleans_read_as_yes_and_no():
    import json

    after = json.loads(json.dumps(STUB_PROFILE))
    after["structural"]["signature_block"] = True
    assert "Signature block: no → yes" in render_diff(PROFILE, parse_profile(after))


def test_the_diff_is_what_makes_approval_a_real_gate(bench):
    """Invariant 6 says nothing auto-activates. A review nobody can perform activates
    everything, so the diff between the active version and a new DRAFT is the review."""
    import json

    active = bench.approved_profile()

    changed = json.loads(json.dumps(STUB_PROFILE))
    changed["register"]["directness"] = 2
    draft = bench.profiles.insert_version(
        user_ref=bench.user_ref, intent=USER_LEVEL_INTENT,
        profile=parse_profile(changed), exemplar_count=6,
    )

    text = render_diff(active.as_profile(), draft.as_profile())
    assert "Directness (1-5): 5 → 2" in text
    assert draft.state == "DRAFT" and draft.is_active is False


# --- capturing the sent version ---------------------------------------------------------

class _Reader:
    """Stands in for the mailbox. Returns whatever the test says is there."""

    def __init__(self, messages):
        self.messages = messages
        self.asked = []

    def get_message(self, message_id):
        self.asked.append(message_id)
        return self.messages.get(message_id)


def _written_back(bench, external_ref):
    draft_id = bench.draft()
    with bench.conn.cursor() as cur:
        cur.execute("UPDATE proc.draft_rfq_emails SET external_draft_ref = %s WHERE id = %s",
                    (external_ref, draft_id))
    return draft_id


def test_a_draft_that_was_sent_is_scored(bench):
    from services.style.feedback import capture_sent_drafts
    from services.style.mailbox import MailboxBinding

    draft_id = _written_back(bench, "MSG-1")
    reader = _Reader({"MSG-1": {
        "id": "MSG-1", "isDraft": False, "sentDateTime": "2026-04-02T09:00:00Z",
        "body": {"contentType": "text", "content": _rewrite(20)},
    }})
    binding = MailboxBinding(
        binding_id=1, user_ref=bench.user_ref, provider="graph",
        mailbox_address="a@b.example", role="both", credential_ref="arn:aws:secretsmanager:x",
        scope_policy_ref=None, scope_verified_at=None, scope_evidence_ref=None,
        last_health_check=None, health_state="OK", is_active=True,
    )

    tally = capture_sent_drafts(reader, binding=binding, conn=bench.conn)
    assert tally["scored"] == 1
    with bench.conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM proc.bp_style_divergence WHERE draft_id = %s",
                    (draft_id,))
        assert cur.fetchone()[0] == 1


def test_a_draft_still_sitting_in_drafts_is_not_scored(bench):
    """Nothing has been learned yet — they have not sent it."""
    from services.style.feedback import capture_sent_drafts
    from services.style.mailbox import MailboxBinding

    _written_back(bench, "MSG-2")
    reader = _Reader({"MSG-2": {"id": "MSG-2", "isDraft": True,
                                "body": {"content": "unchanged"}}})
    binding = MailboxBinding(
        binding_id=1, user_ref=bench.user_ref, provider="graph",
        mailbox_address="a@b.example", role="both", credential_ref="arn:aws:secretsmanager:x",
        scope_policy_ref=None, scope_verified_at=None, scope_evidence_ref=None,
        last_health_check=None, health_state="OK", is_active=True,
    )
    tally = capture_sent_drafts(reader, binding=binding, conn=bench.conn)
    assert tally["still_draft"] == 1 and tally["scored"] == 0


def test_a_deleted_message_is_not_scored(bench):
    from services.style.feedback import capture_sent_drafts
    from services.style.mailbox import MailboxBinding

    _written_back(bench, "MSG-3")
    binding = MailboxBinding(
        binding_id=1, user_ref=bench.user_ref, provider="graph",
        mailbox_address="a@b.example", role="both", credential_ref="arn:aws:secretsmanager:x",
        scope_policy_ref=None, scope_verified_at=None, scope_evidence_ref=None,
        last_health_check=None, health_state="OK", is_active=True,
    )
    tally = capture_sent_drafts(_Reader({}), binding=binding, conn=bench.conn)
    assert tally["gone"] == 1 and tally["scored"] == 0


def test_an_already_scored_draft_is_not_re_read(bench):
    """The sweep should not keep pulling bodies out of a mailbox it has already learned
    from."""
    from services.style.feedback import capture_sent_drafts
    from services.style.mailbox import MailboxBinding

    draft_id = _written_back(bench, "MSG-4")
    bench.svc.record_divergence(draft_id=draft_id, sent_body="already seen")
    binding = MailboxBinding(
        binding_id=1, user_ref=bench.user_ref, provider="graph",
        mailbox_address="a@b.example", role="both", credential_ref="arn:aws:secretsmanager:x",
        scope_policy_ref=None, scope_verified_at=None, scope_evidence_ref=None,
        last_health_check=None, health_state="OK", is_active=True,
    )
    reader = _Reader({})
    capture_sent_drafts(reader, binding=binding, conn=bench.conn)
    assert "MSG-4" not in reader.asked


def test_the_sweep_is_registered_as_a_scheduled_job():
    from pathlib import Path as _P

    source = _P("src/services/backend_scheduler.py").read_text()
    assert "STYLE_FEEDBACK_JOB_NAME" in source
    assert "self._register_style_feedback_job()" in source
