"""Compilation: staging in, DRAFT profile out, staging gone.

Invariant 1 has its named test here — ``test_no_field_of_a_compiled_profile_echoes_its_source``
— and it is applied twice: once to a profile from a stub model, and once to a profile
compiled by the real local model, which is the case that can actually go wrong.

The database tests run against real Postgres. Skipped, not failed, where none is reachable.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from pathlib import Path
from typing import Iterable, List

import pytest

psycopg2 = pytest.importorskip("psycopg2")

from services.style.compiler import CompilationError, StyleCompiler, sweep_expired_staging
from services.style.profile import parse_profile
from services.style.repository import STATE_DRAFT, STATE_UNCOMPILED, USER_LEVEL_INTENT
from services.style.sources import PastedExemplarSource, RawExemplar

# Five emails in one recognisable voice: short, direct, contraction-heavy, context before
# the ask, first-name greeting, bare sign-off.
EMAILS = [
    ("Racking quote — QUT136586",
     "Hi Sam,\n\nThanks for turning that around quickly. The unit rate on line 3 is "
     "£1,240.50 against £1,180 last time, and lead time's gone from 10 to 15 days. "
     "What's driving both?\n\nCan we talk Thursday at 10?\n\nNick"),
    ("Desks — 25 units",
     "Hi Dana,\n\nWe're refreshing the second floor and need 25 desks by end of April. "
     "Could you send unit rate and volume tiers?\n\nHappy to talk Tuesday morning if "
     "that's easier.\n\nNick"),
    ("Payment terms",
     "Hi Priya,\n\nFinance have flagged that we're on 30 days with you and 60 with "
     "everyone else. I'd like to align them.\n\nCan you confirm 60 by Friday?\n\nNick"),
    ("Framework outcome",
     "Hi Ade,\n\nThanks for taking part. We've gone with another supplier this round — "
     "pricing was close but lead times decided it.\n\nI'd like to keep you on the list "
     "for the next cycle. Worth a call in June?\n\nNick"),
    ("Chasing the racking revision",
     "Hi Sam,\n\nI haven't seen the revised quote yet and finance need it Monday.\n\n"
     "Can you get it over by close of play tomorrow?\n\nNick"),
]

STUB_PROFILE = {
    "structural": {
        "subject_pattern": "topic — reference",
        "greeting": "Hi {first_name},",
        "opening_move": "context_before_ask",
        "body_form": "short_prose",
        "target_words": [50, 90],
        "sign_off": "Nick",
        "signature_block": False,
    },
    "register": {
        "formality": 2, "directness": 5, "hedging": "low",
        "contractions": True, "person": "first_singular",
    },
    "lexical": {
        "preferred_terms": ["unit rate", "volume tiers", "lead time"],
        "banned_phrases": ["I hope this email finds you well"],
        "number_format": "bare_numerals", "date_format": "weekday_name",
    },
    "behavioural": {
        "cta_form": "proposes_specific_time",
        "deadline_phrasing": "soft_by_date",
        "escalation_ladder": ["neutral", "firm"],
    },
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
        cur.execute("SELECT to_regclass('proc.bp_style_ingest_staging')")
        if cur.fetchone()[0] is None:
            pytest.skip("style tables not migrated")
    return conn


class _Bench:
    """A user_ref with its own staging rows, cleaned up afterwards."""

    def __init__(self, conn):
        self.conn = conn
        self.user_ref = f"test-{uuid.uuid4().hex[:12]}"
        self.batch_id = uuid.uuid4().hex
        self.source = PastedExemplarSource(conn)

    def stage(self, emails=EMAILS, intent=None):
        for subject, body in emails:
            self.source.stage(
                user_ref=self.user_ref, batch_id=self.batch_id,
                submitted_by="tester", body=body, subject=subject, intent=intent,
            )

    def staging_count(self) -> int:
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM proc.bp_style_ingest_staging WHERE user_ref = %s",
                (self.user_ref,),
            )
            return cur.fetchone()[0]

    def compiler(self, generate=None, min_exemplars=3) -> StyleCompiler:
        from services.style.config import StyleConfig

        return StyleCompiler(
            self.source,
            conn=self.conn,
            config=StyleConfig(min_exemplars=min_exemplars, loaded_from_db=True),
            generate=generate or (lambda prompt: json.dumps(STUB_PROFILE)),
        )

    def cleanup(self):
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM proc.bp_style_ingest_staging WHERE user_ref = %s",
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


# --- the happy path ----------------------------------------------------------------

def test_five_pasted_emails_produce_a_draft_profile(bench):
    bench.stage()
    result = bench.compiler().compile_for(bench.user_ref)
    assert result.compiled
    assert result.state == STATE_DRAFT
    assert result.profile.version == 1
    assert result.profile.is_active is False
    assert result.exemplar_count == 5


def test_staging_is_empty_afterwards(bench):
    """Invariant 9 — staging is a queue, not a store. A profile must never coexist with
    the emails it came from."""
    bench.stage()
    assert bench.staging_count() == 5
    bench.compiler().compile_for(bench.user_ref)
    assert bench.staging_count() == 0


def test_the_profile_cites_the_batch_it_came_from(bench):
    """The batch is gone, but which batch it was survives for audit."""
    bench.stage()
    result = bench.compiler().compile_for(bench.user_ref)
    assert result.profile.source_batch_id == result.batch_id


# --- the min_exemplars gate --------------------------------------------------------

def test_below_min_exemplars_nothing_is_produced(bench):
    """Invariant 7. A profile inferred from two emails is a guess in the costume of a
    specification."""
    bench.stage(EMAILS[:2])
    result = bench.compiler().compile_for(bench.user_ref)
    assert result.state == STATE_UNCOMPILED
    assert result.profile is None
    assert "2 email(s) available" in result.reason


def test_the_gate_does_not_consume_the_queue(bench):
    """Someone who pastes two emails and comes back with a third must still have the
    first two."""
    bench.stage(EMAILS[:2])
    bench.compiler().compile_for(bench.user_ref)
    assert bench.staging_count() == 2


def test_emails_that_redact_to_nothing_do_not_count(bench):
    """Three emails that are pure quoted chains are not three exemplars."""
    quoted = [("Re: x", "On Tue, 14 Mar 2026 at 09:12, Dana <d@x.example> wrote:\n"
                        "> our price is £500\n") for _ in range(3)]
    bench.stage(quoted)
    result = bench.compiler().compile_for(bench.user_ref)
    assert result.state == STATE_UNCOMPILED
    assert "after redaction" in result.reason


# --- invariant 1 -------------------------------------------------------------------

def _words(text: str) -> List[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def _ngrams(text: str, n: int = 6) -> set:
    words = _words(text)
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def _strings_in(payload) -> Iterable[str]:
    if isinstance(payload, str):
        yield payload
    elif isinstance(payload, dict):
        for value in payload.values():
            yield from _strings_in(value)
    elif isinstance(payload, list):
        for value in payload:
            yield from _strings_in(value)


def assert_no_source_ngrams(profile_json: dict, sources: List[str], n: int = 6) -> None:
    """No field of the profile may reproduce an n-gram of length >= n from its inputs.

    This is the named test for invariant 1. It is a function so it can be applied to both
    the stubbed and the real-model profiles.
    """

    source_ngrams: set = set()
    for text in sources:
        source_ngrams |= _ngrams(text, n)

    for value in _strings_in(profile_json):
        overlap = _ngrams(value, n) & source_ngrams
        assert not overlap, (
            f"profile field {value!r} reproduces source text: {sorted(overlap)[:3]}"
        )


def test_no_field_of_a_compiled_profile_echoes_its_source(bench):
    bench.stage()
    result = bench.compiler().compile_for(bench.user_ref)
    assert_no_source_ngrams(
        result.profile.profile_json, [f"{s}\n{b}" for s, b in EMAILS]
    )


def test_the_ngram_check_would_actually_catch_a_leak():
    """A test that cannot fail proves nothing. This asserts the detector detects."""
    leaked = json.loads(json.dumps(STUB_PROFILE))
    leaked["structural"]["subject_pattern"] = "thanks for turning that around quickly"
    with pytest.raises(AssertionError, match="reproduces source text"):
        assert_no_source_ngrams(leaked, [f"{s}\n{b}" for s, b in EMAILS], n=5)


def test_the_model_never_sees_an_unredacted_email(bench):
    """Redaction happens before the prompt is built, and the prompt is built only from
    redacted text. This captures the actual prompt and looks for what must not be in it."""
    seen: List[str] = []

    def _capture(prompt: str) -> str:
        seen.append(prompt)
        return json.dumps(STUB_PROFILE)

    bench.stage()
    bench.compiler(generate=_capture).compile_for(bench.user_ref)

    assert len(seen) == 1
    prompt = seen[0]
    for secret in ("Sam", "Dana", "Priya", "Ade", "1,240.50", "QUT136586", "Nick"):
        assert secret not in prompt, f"{secret!r} reached the model"
    assert "[NAME]" in prompt and "[AMOUNT]" in prompt


# --- failure handling --------------------------------------------------------------

def test_a_model_that_returns_junk_raises_and_keeps_the_queue(bench):
    """A failed compile must not eat the user's emails."""
    bench.stage()
    with pytest.raises(CompilationError, match="did not return JSON"):
        bench.compiler(generate=lambda p: "sorry, I can't do that").compile_for(bench.user_ref)
    assert bench.staging_count() == 5


def test_a_model_that_returns_a_non_conforming_profile_raises(bench):
    bench.stage()
    bad = json.dumps({"structural": {"greeting": "Hi"}})
    with pytest.raises(CompilationError, match="did not validate"):
        bench.compiler(generate=lambda p: bad).compile_for(bench.user_ref)
    assert bench.staging_count() == 5


def test_no_profile_row_is_left_behind_by_a_failed_compile(bench):
    bench.stage()
    with pytest.raises(CompilationError):
        bench.compiler(generate=lambda p: "{}").compile_for(bench.user_ref)
    with bench.conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM proc.bp_style_profile WHERE user_ref = %s",
                    (bench.user_ref,))
        assert cur.fetchone()[0] == 0


# --- scoping -----------------------------------------------------------------------

def test_the_user_level_profile_learns_from_every_intent(bench):
    """The primary artifact. However each email was tagged, it counts."""
    bench.stage(EMAILS[:3], intent="rfq_invite")
    bench.stage(EMAILS[3:], intent="award_notification")
    result = bench.compiler().compile_for(bench.user_ref, USER_LEVEL_INTENT)
    assert result.exemplar_count == 5


def test_a_per_intent_profile_only_sees_its_own_intent(bench):
    bench.stage(EMAILS[:3], intent="rfq_invite")
    bench.stage(EMAILS[3:], intent="award_notification")
    result = bench.compiler().compile_for(bench.user_ref, "rfq_invite")
    assert result.exemplar_count == 3


def test_a_per_intent_scope_below_the_floor_stays_uncompiled(bench):
    bench.stage(EMAILS[:2], intent="rfq_invite")
    bench.stage(EMAILS[2:], intent="award_notification")
    assert bench.compiler().compile_for(bench.user_ref, "rfq_invite").state == STATE_UNCOMPILED


# --- the TTL sweep -----------------------------------------------------------------

def test_the_sweep_purges_expired_rows_and_leaves_live_ones(bench):
    """Invariant 9's second half: what the compiler never consumed still has to go."""
    bench.stage()
    with bench.conn.cursor() as cur:
        cur.execute(
            "UPDATE proc.bp_style_ingest_staging SET purge_after = NOW() - INTERVAL '1 hour' "
            "WHERE user_ref = %s AND ingest_id IN ("
            "  SELECT ingest_id FROM proc.bp_style_ingest_staging WHERE user_ref = %s "
            "  ORDER BY ingest_id LIMIT 2)",
            (bench.user_ref, bench.user_ref),
        )
    sweep_expired_staging(bench.conn)
    assert bench.staging_count() == 3


def test_the_sweep_is_safe_when_there_is_nothing_to_do(bench):
    bench.stage()
    assert sweep_expired_staging(bench.conn) == 0
    assert bench.staging_count() == 5


def test_the_sweep_is_registered_as_a_scheduled_job():
    """A purge that depends on someone remembering to run it is not a purge.

    Reads the scheduler source rather than importing it: BackendScheduler pulls in the
    whole orchestrator graph, and this assertion is about wiring, not behaviour.
    """
    source = Path("src/services/backend_scheduler.py").read_text()
    assert "STYLE_STAGING_SWEEP_JOB_NAME" in source
    assert "self._register_style_staging_sweep_job()" in source
    assert "sweep_expired_staging" in source


# --- the source seam ---------------------------------------------------------------

def test_no_exemplar_source_can_send_mail():
    """Invariant 2, scoped to this subsystem per the Phase -1 decision. The platform's SES
    path is deliberately retained, but it must not be reachable from here."""
    import services.style.sources as sources_module

    for name in dir(PastedExemplarSource):
        assert "send" not in name.lower(), f"PastedExemplarSource.{name} looks like a send path"
    assert "email_service" not in inspect_source(sources_module)
    assert "EmailDispatchService" not in inspect_source(sources_module)


def inspect_source(module) -> str:
    import inspect

    return inspect.getsource(module)


def test_the_compiler_depends_only_on_the_protocol(bench):
    """Mode C sources arrive as further implementations of ExemplarSource. If adding one
    requires touching the compiler, the seam was drawn in the wrong place."""

    class FakeMailbox:
        def fetch(self, user_ref, intent=None):
            return [RawExemplar(body=b, subject=s) for s, b in EMAILS]

    from services.style.config import StyleConfig

    compiler = StyleCompiler(
        FakeMailbox(), conn=bench.conn,
        config=StyleConfig(min_exemplars=3, loaded_from_db=True),
        generate=lambda p: json.dumps(STUB_PROFILE),
    )
    result = compiler.compile_for(bench.user_ref)
    assert result.compiled
    # Nothing was staged, so nothing should have been purged.
    assert bench.staging_count() == 0


# --- the real model ----------------------------------------------------------------

@pytest.mark.slow
def test_the_local_model_compiles_a_conforming_profile_that_leaks_nothing(bench):
    """The case that can actually go wrong. Skipped where Ollama is unreachable."""
    import requests

    from services.ollama_client import OLLAMA_BASE_URL

    try:
        requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5).raise_for_status()
    except Exception as exc:
        pytest.skip(f"Ollama unreachable: {exc}")

    bench.stage()
    result = StyleCompiler(
        bench.source, conn=bench.conn,
        config=__import__("services.style.config", fromlist=["StyleConfig"]).StyleConfig(
            min_exemplars=3, loaded_from_db=True
        ),
    ).compile_for(bench.user_ref)

    assert result.compiled
    parse_profile(result.profile.profile_json)  # conforms
    assert_no_source_ngrams(result.profile.profile_json, [f"{s}\n{b}" for s, b in EMAILS])
    assert bench.staging_count() == 0


# --- measured, not generated -------------------------------------------------------

def test_target_words_is_counted_not_asked_of_the_model(bench):
    """Length is arithmetic over the exemplars. The first live run against the local model
    returned [1, 2] for emails averaging sixty words — a figure that can be computed should
    never be generated."""
    absurd = json.loads(json.dumps(STUB_PROFILE))
    absurd["structural"]["target_words"] = [1, 2]

    bench.stage()
    result = bench.compiler(generate=lambda p: json.dumps(absurd)).compile_for(bench.user_ref)

    low, high = result.profile.profile_json["structural"]["target_words"]
    assert (low, high) != (1, 2)
    assert 20 < low <= high < 200, (low, high)


def test_measured_length_reflects_the_actual_emails():
    from services.style.compiler import _measure_length
    from services.style.redaction import redact

    redacted = [redact(b, s) for s, b in EMAILS]
    low, high = _measure_length(redacted)
    actual = sorted(len(r.body_text.split()) for r in redacted)
    assert (low, high) == (actual[0], actual[-1])


def test_signature_block_is_observed_during_redaction_not_guessed(bench):
    """Redaction is what removes the signature block, so by the time the model sees the
    text the evidence is gone. Asking anyway invites a confident guess about something
    unknowable from the input."""
    from services.style.compiler import _had_signature_block
    from services.style.redaction import redact

    with_sig = redact("Hi Sam,\n\nAll fine.\n\nKind regards,\n\nNicholas Geelen\n"
                      "Procurement Lead | Techworld Ltd")
    without = redact("Hi Sam,\n\nAll fine.\n\nNick")
    assert _had_signature_block([with_sig]) is True
    assert _had_signature_block([without]) is False

    claims_none = json.loads(json.dumps(STUB_PROFILE))
    claims_none["structural"]["signature_block"] = True
    bench.stage()  # these emails have no signature block
    result = bench.compiler(generate=lambda p: json.dumps(claims_none)).compile_for(bench.user_ref)
    assert result.profile.profile_json["structural"]["signature_block"] is False


def test_an_echoed_placeholder_becomes_a_template_slot(bench):
    """The redacted text ends in [NAME], so the model sometimes hands it straight back.
    That is a redaction artefact, not a habit."""
    echoed = json.loads(json.dumps(STUB_PROFILE))
    echoed["structural"]["sign_off"] = "[NAME]"
    bench.stage()
    result = bench.compiler(generate=lambda p: json.dumps(echoed)).compile_for(bench.user_ref)
    assert result.profile.profile_json["structural"]["sign_off"] == "{sender_first_name}"


def test_the_sender_slot_is_distinct_from_the_greeting_slot(bench):
    """Given {first_name} in both the greeting and the sign-off, a model reads the
    sign-off as the person being written TO and signs the email with the recipient's
    name. Observed on the first live run: an email to Sarah signed 'Sarah'."""
    collided = json.loads(json.dumps(STUB_PROFILE))
    collided["structural"]["sign_off"] = "{first_name}"
    bench.stage()
    result = bench.compiler(generate=lambda p: json.dumps(collided)).compile_for(bench.user_ref)
    structural = result.profile.profile_json["structural"]
    assert structural["sign_off"] == "{sender_first_name}"
    assert structural["greeting"] != structural["sign_off"]


def test_a_real_sign_off_is_left_alone(bench):
    kept = json.loads(json.dumps(STUB_PROFILE))
    kept["structural"]["sign_off"] = "Kind regards,"
    bench.stage()
    result = bench.compiler(generate=lambda p: json.dumps(kept)).compile_for(bench.user_ref)
    assert result.profile.profile_json["structural"]["sign_off"] == "Kind regards,"
