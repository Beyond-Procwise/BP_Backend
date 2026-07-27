"""Turn a batch of emails into a style profile, then destroy the batch.

The sequence is fixed:

    fetch (source) -> redact -> compile (LLM) -> validate -> insert DRAFT -> purge

Three properties are load-bearing:

* **The model never sees an unredacted email.** Redaction happens before the prompt is
  built, and the prompt is built only from redacted text.
* **Below ``min_exemplars`` nothing is produced.** The scope stays ``UNCOMPILED`` and
  drafting falls back visibly. A profile inferred from two emails would be a guess wearing
  the costume of a specification.
* **The batch is purged on success.** Staging is a queue. What the compiler consumed is
  deleted in the same transaction that records the profile's source batch, so a profile
  never coexists with the emails it came from.

Output is grammar-constrained to the profile JSON schema. Ollama masks non-conforming
tokens during decoding, so the model cannot emit a shape the Pydantic model would reject —
which matters more than usual here, because a free-text field the schema did not expect is
exactly how content would leak.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from typing import Any, List, Optional

from pydantic import ValidationError

from services.db import get_conn
from services.style.config import StyleConfig, load_style_config
from services.style.profile import StyleProfile, parse_profile
from services.style.redaction import RedactionResult, redact
from services.style.repository import (
    STATE_UNCOMPILED,
    USER_LEVEL_INTENT,
    ProfileRecord,
    StyleProfileRepository,
)
from services.style.sources import ExemplarSource, PastedExemplarSource, RawExemplar

logger = logging.getLogger(__name__)

# Audit phase for proc.bp_agent_actions. Written to the existing spine, never a parallel one.
PHASE_STYLE = "style"

_SYSTEM = """You are a writing-style analyst. You are given several emails written by \
one person, with all names, organisations, amounts and reference numbers already replaced \
by placeholders such as [NAME] and [AMOUNT].

Your job is to describe HOW this person writes — never WHAT they wrote about.

Describe habits only: how long their emails run, how they open and close, how formal and \
how direct they are, whether they use contractions, how they phrase a request or a \
deadline, and which short domain terms they favour or avoid.

You must NOT reproduce any sentence, clause or phrase from the emails. Do not quote them. \
Do not paraphrase their content. Do not mention suppliers, products, projects, prices or \
dates. If you find yourself copying words from an email, you are describing content \
instead of habit, and that is wrong.

The one exception is `banned_phrases`, which lists short generic stock phrases this writer \
avoids (for example "I hope this email finds you well"). These are phrases NOT present in \
their writing, so they cannot come from the emails.

Three fields are TEMPLATES, not descriptions. Write what the writer would actually type, \
with {placeholders} where a specific value would go — never a sentence about them:
  greeting        -> "Hi {first_name}," NOT "Direct, uses a first name"
  sign_off        -> "Kind regards," or their own first name alone — the literal words \
they close with, NOT "Single line with name only"
  subject_pattern -> "topic — reference" NOT "Short and action-oriented"

For the closed-choice fields, pick the option that fits:
  opening_move  context_before_ask (says why first, then asks) | ask_before_context \
(asks first, explains after) | greeting_only
  body_form     short_prose (a few sentences) | long_prose | bullets (actual bullet \
points) | mixed
  cta_form      proposes_specific_time | open_question | deadline_only | none

Reply with a single JSON object conforming to the given schema. No commentary."""

_USER_TEMPLATE = """Here are {count} emails written by the same person, already redacted.

{exemplars}

Describe this person's writing habits as a style specification."""


class CompilationError(Exception):
    """The model could not produce a valid profile."""


@dataclass
class CompilationResult:
    """What happened, in enough detail for the UI to explain it."""

    state: str
    profile: Optional[ProfileRecord] = None
    exemplar_count: int = 0
    batch_id: Optional[str] = None
    reason: Optional[str] = None
    redaction_counts: Optional[dict] = None

    @property
    def compiled(self) -> bool:
        return self.profile is not None


def _render_exemplars(redacted: List[RedactionResult]) -> str:
    return "\n\n".join(
        f"--- email {i} ---\n{r.text}" for i, r in enumerate(redacted, start=1)
    )


def _measure_length(redacted: List[RedactionResult]) -> tuple[int, int]:
    """The observed word-count range of the writer's own prose.

    Counted here rather than asked of the model. Length is arithmetic over the exemplars,
    and a language model asked to count words guesses — the first live run against the
    local model returned ``[1, 2]`` for emails averaging sixty words. A figure that can be
    computed should never be generated.

    Subject lines and placeholders are excluded: ``[NAME]`` stands in for a word that was
    there, so it counts, but the "Subject:" prefix the prompt adds does not.
    """

    counts = [len(r.body_text.split()) for r in redacted if r.body_text.strip()]
    if not counts:
        return (1, 1)
    return (max(1, min(counts)), max(counts))


def _had_signature_block(redacted: List[RedactionResult]) -> bool:
    """Whether these emails carried a signature block.

    Observed during redaction, not asked of the model — redaction is what removes the
    block, so by the time the model sees the text the evidence is gone. Asking anyway
    invites a confident guess about something unknowable from the input.
    """

    return any(
        "signature_block" in r.blocks_removed or "signature_delimiter" in r.blocks_removed
        for r in redacted
    )


# The sender's own name. Deliberately NOT {first_name}, which the greeting uses for the
# RECIPIENT: given the same token in both slots a model reads the sign-off as the person
# being written to, and signs the email with their name. Observed on the first live run.
SENDER_SLOT = "{sender_first_name}"


def _normalise_sign_off(sign_off: str) -> str:
    """Turn an echoed placeholder back into a template slot.

    The redacted text ends in ``[NAME]``, so a model asked for the sign-off will
    sometimes hand ``[NAME]`` straight back. That is a redaction artefact, not a habit;
    what it means is "signs off with their own first name".
    """

    if not sign_off:
        return sign_off
    cleaned = sign_off.replace("[NAME]", SENDER_SLOT).replace("{first_name}", SENDER_SLOT)
    cleaned = cleaned.strip()
    return cleaned or SENDER_SLOT


def _audit(action_type: str, *, status: str, summary: str, details: Any = None) -> None:
    """Write to the existing agent-actions spine. Best-effort, never raises."""

    try:
        from services.agent_actions import record_action

        record_action(
            phase=PHASE_STYLE,
            action_type=action_type,
            agent="style_compiler",
            status=status,
            summary=summary,
            details=details,
        )
    except Exception:  # pragma: no cover - audit must never break the caller
        logger.debug("style audit write failed", exc_info=True)


class StyleCompiler:
    """Compiles style profiles from whatever an ``ExemplarSource`` provides."""

    def __init__(
        self,
        source: Optional[ExemplarSource] = None,
        *,
        conn: Optional[Any] = None,
        config: Optional[StyleConfig] = None,
        generate: Optional[Any] = None,
    ) -> None:
        self._conn = conn
        self.source = source or PastedExemplarSource(conn)
        self.config = config or load_style_config(conn)
        self.repo = StyleProfileRepository(conn)
        # Injected so tests do not need a GPU. Defaults to the platform's local model.
        self._generate = generate

    # -- LLM ----------------------------------------------------------------------

    def _call_model(self, prompt: str, *, temperature: float = 0.0) -> str:
        if self._generate is not None:
            return self._generate(prompt)

        from services.ollama_client import ollama_generate

        # `format` is the JSON schema, so decoding is grammar-guided: the model cannot
        # emit a shape the Pydantic model would then reject.
        text = ollama_generate(
            prompt,
            temperature=temperature,
            format=StyleProfile.model_json_schema(by_alias=True),
            think=False,
        )
        if not text:
            raise CompilationError("the model returned nothing")
        return text

    def _attempt(self, prompt: str, *, temperature: float = 0.0) -> StyleProfile:
        raw = self._call_model(prompt, temperature=temperature)
        try:
            payload = json.loads(raw)
        except (TypeError, ValueError) as exc:
            raise CompilationError(f"the model did not return JSON: {exc}") from exc
        try:
            return parse_profile(payload)
        except ValidationError as exc:
            raise CompilationError(f"the model's profile did not validate: {exc}") from exc

    def _compile_profile(self, redacted: List[RedactionResult]) -> StyleProfile:
        """Compile, with one corrective retry.

        Grammar-constrained decoding guarantees the JSON *shape* but not its semantics:
        a JSON schema can say ``escalation_ladder`` is an array of enum values, and the
        model will still emit ``["neutral","neutral","neutral"]`` because nothing in the
        grammar forbids repeats. Rules of that kind live in the Pydantic validators, which
        the decoder cannot see.

        So a validation failure is treated as a correctable mistake rather than a dead
        end: the error is handed back once, naming what was wrong. A second failure is
        real, and raises — better an UNCOMPILED scope with a visible reason than a profile
        nobody can trust.

        The retry runs at a non-zero temperature. At 0 the decoder is deterministic, so a
        retry reproduces the rejected answer byte for byte and the second attempt is pure
        waste — observed, not theorised.
        """

        prompt = (
            f"{_SYSTEM}\n\n"
            + _USER_TEMPLATE.format(
                count=len(redacted), exemplars=_render_exemplars(redacted)
            )
        )
        measured = _measure_length(redacted)
        had_signature = _had_signature_block(redacted)

        def _with_measured_length(profile: StyleProfile) -> StyleProfile:
            """Overwrite the fields that were measured rather than inferred.

            Everything here is something redaction either computed or destroyed the
            evidence of, so the model's answer is at best a guess and at worst — as with
            ``target_words`` — confidently wrong.
            """

            return profile.model_copy(
                update={
                    "structural": profile.structural.model_copy(
                        update={
                            "target_words": measured,
                            "signature_block": had_signature,
                            "sign_off": _normalise_sign_off(profile.structural.sign_off),
                        }
                    )
                }
            )

        try:
            return _with_measured_length(self._attempt(prompt))
        except CompilationError as first:
            logger.info("Style compile rejected, retrying once with the error: %s", first)
            _audit(
                "compile_retry",
                status="warning",
                summary="first compile attempt was rejected; retrying",
                details={"error": str(first)[:2000]},
            )
            corrective = (
                f"{prompt}\n\n"
                "Your previous answer was rejected:\n"
                f"{first}\n\n"
                "Correct it. Every entry in a list must be distinct, every value must be "
                "one of the allowed options, and describe only habits — never content."
            )
            return _with_measured_length(self._attempt(corrective, temperature=0.3))

    # -- purge --------------------------------------------------------------------

    def _purge(self, staging_ids: List[int]) -> int:
        """Hard-delete the consumed batch. Invariant 9."""

        if not staging_ids:
            return 0

        def _run(conn):
            cur = conn.cursor()
            cur.execute(
                "DELETE FROM proc.bp_style_ingest_staging WHERE ingest_id = ANY(%s)",
                (staging_ids,),
            )
            count = cur.rowcount
            cur.close()
            if self._conn is None and hasattr(conn, "commit"):
                conn.commit()
            return count

        if self._conn is not None:
            return _run(self._conn)
        with get_conn() as conn:
            return _run(conn)

    # -- the pipeline -------------------------------------------------------------

    def compile_for(
        self, user_ref: str, intent: str = USER_LEVEL_INTENT
    ) -> CompilationResult:
        """Compile a DRAFT profile for one scope, or explain why not.

        ``intent`` defaults to the user-level scope, which is the primary artifact.
        Per-intent profiles are compiled only where that intent has enough emails of its
        own; the caller decides, this method just does what it is told.
        """

        batch_id = uuid.uuid4().hex
        # A user-level profile learns from everything the user pasted, however each email
        # happened to be tagged. A per-intent profile only sees its own intent.
        fetch_intent = None if intent == USER_LEVEL_INTENT else intent
        raw: List[RawExemplar] = self.source.fetch(user_ref, fetch_intent)

        if len(raw) < self.config.min_exemplars:
            reason = (
                f"{len(raw)} email(s) available; {self.config.min_exemplars} are needed "
                "before a profile can be described rather than guessed"
            )
            logger.info("Not compiling style profile for %s/%s: %s", user_ref, intent, reason)
            _audit(
                "compile",
                status="skipped",
                summary=f"below min_exemplars for {user_ref}/{intent}",
                details={"available": len(raw), "required": self.config.min_exemplars},
            )
            return CompilationResult(
                state=STATE_UNCOMPILED,
                exemplar_count=len(raw),
                reason=reason,
            )

        redacted = [redact(r.body, r.subject) for r in raw]
        # An email whose body was entirely a quoted reply chain has nothing of this
        # writer's own left in it. A surviving subject line does not make it a writing
        # sample, so emptiness is judged on the body alone.
        redacted = [r for r in redacted if r.has_body]
        if len(redacted) < self.config.min_exemplars:
            reason = (
                f"only {len(redacted)} email(s) had any content left after redaction; "
                f"{self.config.min_exemplars} are needed"
            )
            _audit("redact", status="skipped", summary=reason)
            return CompilationResult(
                state=STATE_UNCOMPILED, exemplar_count=len(redacted), reason=reason
            )

        totals: dict = {}
        for r in redacted:
            for key, count in r.counts.items():
                totals[key] = totals.get(key, 0) + count

        _audit(
            "redact",
            status="ok",
            summary=f"redacted {len(redacted)} emails for {user_ref}/{intent}",
            # Counts only. The text itself is never logged.
            details={"replacements": totals, "batch_id": batch_id},
        )

        try:
            profile = self._compile_profile(redacted)
        except CompilationError as exc:
            _audit("compile", status="error", summary=str(exc), details={"batch_id": batch_id})
            logger.warning("Style compilation failed for %s/%s: %s", user_ref, intent, exc)
            raise

        record = self.repo.insert_version(
            user_ref=user_ref,
            intent=intent,
            profile=profile,
            exemplar_count=len(redacted),
            source_batch_id=batch_id,
        )

        # Only now is the batch expendable: the profile exists and cites it.
        purged = self._purge([r.staging_id for r in raw if r.staging_id is not None])

        _audit(
            "compile",
            status="ok",
            summary=(
                f"compiled style profile v{record.version} for {user_ref}/{intent} "
                f"from {len(redacted)} exemplars (DRAFT)"
            ),
            details={
                "profile_id": record.profile_id,
                "version": record.version,
                "batch_id": batch_id,
                "staging_rows_purged": purged,
            },
        )
        logger.info(
            "Compiled style profile v%s for %s/%s; purged %s staging rows",
            record.version, user_ref, intent, purged,
        )

        return CompilationResult(
            state=record.state,
            profile=record,
            exemplar_count=len(redacted),
            batch_id=batch_id,
            redaction_counts=totals,
        )


def sweep_expired_staging(conn: Optional[Any] = None) -> int:
    """Delete staging rows past their TTL. Returns how many went.

    Invariant 9's second half: the compiler purges what it consumes, and this purges what
    never got consumed — an abandoned paste-and-navigate-away must not leave someone's
    emails sitting in the database indefinitely.
    """

    def _run(c):
        cur = c.cursor()
        cur.execute(
            "DELETE FROM proc.bp_style_ingest_staging WHERE purge_after <= NOW()"
        )
        count = cur.rowcount
        cur.close()
        if conn is None and hasattr(c, "commit"):
            c.commit()
        return count

    if conn is not None:
        deleted = _run(conn)
    else:
        with get_conn() as own:
            deleted = _run(own)

    # An audit row every run, including the quiet ones. A sweep that logged only when it
    # found something would be indistinguishable from a sweep that stopped running.
    _audit(
        "staging_sweep",
        status="ok",
        summary=f"purged {deleted} expired style staging row(s)",
        details={"deleted": deleted},
    )
    if deleted:
        logger.info("Style staging sweep purged %s expired row(s)", deleted)
    return deleted
