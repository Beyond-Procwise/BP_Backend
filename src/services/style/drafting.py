"""Assemble the prompt, generate the draft, record where every part of it came from.

The assembly order is fixed and deliberate: **profile rules, then exemplars, then the
task**. Rules first because they are what the model is held to; exemplars second because
they illustrate the rules and would otherwise be read as the specification; the task last
because it is the thing to act on and recency helps.

The system prompt carries an explicit precedence instruction. Without it a model shown
three complete emails and a list of rules will imitate the emails — they are concrete and
the rules are abstract — and a profile that loses to its own illustrations is not
governing anything.

Provenance is written for every draft: which profile version, which exemplars, which
model, which prompt version, and which rung of the fallback ladder it landed on. A draft
whose origins cannot be reconstructed is a draft nobody can defend later.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from services.db import get_conn
from services.style.exemplars import ExemplarRecord, ExemplarService
from services.style.grounding import ground_draft
from services.style.mode_c import message_set_hash
from services.style.rendering import render_profile_rules
from services.style.repository import USER_LEVEL_INTENT
from services.style.resolver import ResolvedStyle, StyleResolver

logger = logging.getLogger(__name__)

# Governed in proc.bp_prompt so the precedence wording can be changed without a deploy.
# The constant below is the fallback when the database is unreachable, and must stay in
# step with the seeded row — see deploy/sql/2026-07-27_bp_prompt_style_drafting.sql.
PROMPT_NAME = "style_draft_system"

SYSTEM_PROMPT_FALLBACK = """You are drafting an email on behalf of a specific person, in \
their voice.

You are given a style specification describing how they write, then two or three example \
emails, then the task.

Where the example emails and the style specification conflict, follow the style \
specification. The examples illustrate the specification; they do not override it.

The examples are fiction. Their suppliers, amounts, reference numbers, names and dates \
are invented and must never appear in your draft. Take only the manner of writing from \
them; take every fact from the task.

Invent nothing. Every name, figure, date, reference number and commitment in your draft \
must come from the task. If the task does not give you something the email seems to need \
— the contact's name, a reference number, a deadline — write a square-bracketed \
placeholder such as [contact name] or [reference] and carry on. A placeholder is correct. \
A plausible-looking invention is not, and is worse than leaving the gap visible.

Write only the email. Give a subject line, then a blank line, then the body, and end with \
the sign-off the specification requires. No preamble, no commentary, no explanation of \
your choices."""

_PROMPT_VERSION_FALLBACK = 0


@dataclass
class DraftProvenance:
    """Everything needed to reconstruct why a draft reads the way it does."""

    user_ref: str
    intent: str
    mode: str
    fallback_level: int
    fallback_reason: str
    style_profile_id: Optional[int] = None
    style_profile_version: Optional[int] = None
    exemplar_ids: List[int] = field(default_factory=list)
    exemplar_set_hash: Optional[str] = None
    model_id: Optional[str] = None
    prompt_version: Optional[int] = None
    retrieved_at: Optional[Any] = None
    # Facts the model invented and the guard replaced with visible placeholders.
    ungrounded_replacements: List[str] = field(default_factory=list)
    # Mode C2 only: the bodies are never stored, so the message ids ARE the record of
    # which emails shaped this draft.
    message_ids: List[str] = field(default_factory=list)
    mailbox_binding_id: Optional[int] = None

    def as_response_fields(self) -> Dict[str, Any]:
        """The subset the UI needs. ``fallback_level`` and its reason are surfaced so a
        borrowed voice is visible rather than merely recorded."""

        return {
            "fallback_level": self.fallback_level,
            "fallback_reason": self.fallback_reason,
            "style_degraded": self.fallback_level > 0,
            "style_profile_version": self.style_profile_version,
            "exemplar_count": len(self.exemplar_ids),
            # Surfaced so the reader knows the draft has gaps to fill, rather than
            # discovering later that a placeholder went out to a supplier.
            "placeholders_inserted": len(self.ungrounded_replacements),
            "needs_review": bool(self.ungrounded_replacements),
        }


@dataclass
class StyleDraft:
    """A generated draft and its full provenance."""

    subject: Optional[str]
    body: str
    provenance: DraftProvenance
    prompt: Optional[str] = None
    draft_id: Optional[int] = None
    # The provider's identifier for the copy in the customer's Drafts folder.
    external_draft_ref: Optional[str] = None


def _exemplar_set_hash(exemplar_ids: List[int]) -> Optional[str]:
    """A stable fingerprint of the exemplar set, over sorted ids.

    Sorted so the hash identifies *which* exemplars were used, not the order retrieval
    happened to return them in — two drafts built from the same three examples should
    fingerprint identically even when the task reordered them.
    """

    if not exemplar_ids:
        return None
    joined = ",".join(str(i) for i in sorted(exemplar_ids))
    return hashlib.sha256(joined.encode()).hexdigest()


def _force_baseline(resolved: ResolvedStyle, reason: Optional[str]) -> ResolvedStyle:
    """Drop to the platform baseline with the reason attached.

    Used when Mode C2 could not read the mailbox. The profile is not the problem — the
    exemplars are missing and the source is unavailable — but a draft that silently used
    the stored rules without saying the live read failed would be claiming a currency it
    does not have.
    """

    from services.style.resolver import BASELINE_PROFILE, LEVEL_BASELINE

    return ResolvedStyle(
        profile=BASELINE_PROFILE,
        fallback_level=LEVEL_BASELINE,
        reason=reason or "your mailbox could not be reached — using the platform default",
        source_unreachable=True,
    )


def _fill_template_slots(text: str, sender_name: Optional[str]) -> str:
    """Substitute the profile's placeholder slots, or make them visibly empty.

    The profile stores ``sign_off: "{sender_first_name}"`` because the compiler could not
    know the writer's name — the redactor had already removed it. The drafter does know,
    if the caller passed it. When it does not, the slot becomes ``[your name]`` rather
    than being left as ``{sender_first_name}``: a square bracket reads as a gap to fill,
    a curly brace reads as a bug.
    """

    if not text:
        return text
    replacement = sender_name.strip() if sender_name and sender_name.strip() else "[your name]"
    # Both bracket styles. The profile stores {curly}, but a model copying a slot into its
    # output frequently switches to [square] — it has just been told that square brackets
    # are how to mark a gap — and a substitution that only matched the stored form would
    # leave "[sender_first_name]" in a finished email.
    text = re.sub(r"[\{\[]\s*sender_(?:first_)?name\s*[\}\]]", replacement, text,
                  flags=re.IGNORECASE)
    text = re.sub(r"[\{\[]\s*(?:first_name|recipient_name|contact_name)\s*[\}\]]",
                  "[contact name]", text, flags=re.IGNORECASE)
    return text


def _split_subject_and_body(text: str) -> tuple[Optional[str], str]:
    """Pull a leading subject line off the generated text."""

    cleaned = (text or "").strip()
    if not cleaned:
        return None, ""

    lines = cleaned.split("\n")
    first = lines[0].strip()
    lowered = first.lower()
    if lowered.startswith("subject:"):
        return first[len("subject:"):].strip() or None, "\n".join(lines[1:]).lstrip("\n")

    # A short opening line followed by a blank one is a subject even without the label.
    if len(lines) > 2 and not lines[1].strip() and len(first) < 120:
        return first, "\n".join(lines[2:]).lstrip("\n")

    return None, cleaned


class StyleDraftingService:
    """Generates drafts in a person's voice, and records how."""

    def __init__(
        self,
        conn: Optional[Any] = None,
        *,
        resolver: Optional[StyleResolver] = None,
        exemplars: Optional[ExemplarService] = None,
        generate: Optional[Any] = None,
        mode: str = "A",
        mailbox_source: Optional[Any] = None,
        binding: Optional[Any] = None,
        draft_writer: Optional[Any] = None,
    ) -> None:
        self._conn = conn
        self._mailbox_source = mailbox_source
        self._binding = binding
        self._draft_writer = draft_writer
        self.resolver = resolver or StyleResolver(conn)
        self.exemplars = exemplars or ExemplarService(conn)
        self._generate = generate
        self.mode = mode

    # -- the governed system prompt ------------------------------------------------

    def _system_prompt(self) -> tuple[str, int]:
        """The system prompt and its version, from proc.bp_prompt.

        Governed rather than hardcoded so the precedence wording — the sentence the whole
        design rests on — can be corrected without a deploy. Falls back to the constant
        when the database is unreachable, and reports version 0 so a draft is never
        credited to a governed version it did not actually use.
        """

        def _run(conn):
            cur = conn.cursor()
            cur.execute(
                "SELECT prompts_desc->>'prompt_template', version FROM proc.bp_prompt "
                "WHERE prompt_name = %s AND COALESCE(prompts_status, 1) = 1 "
                "ORDER BY version DESC LIMIT 1",
                (PROMPT_NAME,),
            )
            row = cur.fetchone()
            cur.close()
            return row

        try:
            if self._conn is not None:
                row = _run(self._conn)
            else:
                with get_conn() as conn:
                    row = _run(conn)
            if row and row[0]:
                return row[0], int(row[1] or 1)
        except Exception:
            logger.warning("Could not read the governed style prompt; using the fallback",
                           exc_info=True)
        return SYSTEM_PROMPT_FALLBACK, _PROMPT_VERSION_FALLBACK

    # -- assembly ------------------------------------------------------------------

    def build_prompt(
        self,
        resolved: ResolvedStyle,
        exemplars: List[ExemplarRecord],
        task: str,
        *,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Rules, then examples, then the task. The order is not incidental."""

        parts = [system_prompt or SYSTEM_PROMPT_FALLBACK]

        parts.append("STYLE SPECIFICATION — these rules govern.\n"
                     + render_profile_rules(resolved.profile))

        if exemplars:
            rendered = []
            for i, ex in enumerate(exemplars, start=1):
                header = f"--- example {i} ---"
                subject = f"Subject: {ex.subject}\n" if ex.subject else ""
                rendered.append(f"{header}\n{subject}{ex.body}")
            parts.append(
                "EXAMPLES — these illustrate the specification above. Their facts are "
                "invented; do not reuse them.\n" + "\n\n".join(rendered)
            )
        else:
            parts.append(
                "EXAMPLES — none available. Follow the specification directly."
            )

        parts.append(f"TASK\n{task.strip()}")
        return "\n\n".join(parts)

    def _call_model(self, prompt: str) -> tuple[str, str]:
        if self._generate is not None:
            return self._generate(prompt), "injected"

        from services.ollama_client import DEFAULT_MODEL, ollama_generate

        text = ollama_generate(prompt, temperature=0.3, think=False)
        if not text:
            raise RuntimeError("the model returned nothing")
        return text, DEFAULT_MODEL

    # -- the pipeline --------------------------------------------------------------

    def draft(
        self,
        *,
        user_ref: str,
        task: str,
        intent: str = USER_LEVEL_INTENT,
        persist: bool = True,
        sender_name: Optional[str] = None,
        workflow_id: Optional[str] = None,
        supplier_id: Optional[str] = None,
        recipient_email: Optional[str] = None,
    ) -> StyleDraft:
        """Generate one draft, with provenance."""

        resolved = self.resolver.resolve(user_ref, intent)

        # Exemplars come from the scope that actually supplied the profile — showing a
        # user's own examples alongside the house style would illustrate the wrong rules.
        exemplars: List[ExemplarRecord] = []
        retrieved_at = None
        message_ids: List[str] = []
        binding_id: Optional[int] = None

        if self.mode == "C2" and self._mailbox_source is not None and self._binding is not None:
            # Mode C2 reads the mailbox now and keeps none of it. A failure here degrades
            # the draft rather than delaying or failing it — see mode_c.fetch_live_exemplars.
            from services.style.mode_c import fetch_live_exemplars

            outcome = fetch_live_exemplars(
                self._binding, self._mailbox_source, user_ref=user_ref, intent=intent
            )
            binding_id = self._binding.binding_id
            if outcome.usable:
                exemplars = outcome.exemplars
                message_ids = outcome.message_ids
                retrieved_at = _now(self._conn)
            else:
                resolved = _force_baseline(resolved, outcome.reason)
        elif resolved.scope_user_ref and resolved.scope_intent:
            try:
                exemplars = self.exemplars.retrieve(
                    resolved.scope_user_ref, resolved.scope_intent, task
                )
                retrieved_at = _now(self._conn)
            except Exception:
                logger.warning("Could not retrieve exemplars; drafting from rules alone",
                               exc_info=True)

        system_prompt, prompt_version = self._system_prompt()
        prompt = self.build_prompt(resolved, exemplars, task, system_prompt=system_prompt)

        raw, model_id = self._call_model(prompt)
        subject, body = _split_subject_and_body(raw)

        # The prompt asks the model to invent nothing. It does anyway — reliably, and
        # fluently enough that the result looks ready to send. The guard is what actually
        # holds the line; see services/style/grounding.py.
        grounded = ground_draft(subject, body, task)
        subject = _fill_template_slots(grounded.subject, sender_name)
        body = _fill_template_slots(grounded.body, sender_name)

        exemplar_ids = [e.exemplar_id for e in exemplars if e.exemplar_id is not None]
        provenance = DraftProvenance(
            user_ref=user_ref,
            intent=intent,
            mode=self.mode,
            fallback_level=resolved.fallback_level,
            fallback_reason=resolved.reason,
            style_profile_id=resolved.profile_id,
            style_profile_version=resolved.profile_version,
            exemplar_ids=exemplar_ids,
            exemplar_set_hash=(
                message_set_hash(message_ids) if message_ids
                else _exemplar_set_hash(exemplar_ids)
            ),
            message_ids=message_ids,
            mailbox_binding_id=binding_id,
            model_id=model_id,
            prompt_version=prompt_version,
            retrieved_at=retrieved_at,
            ungrounded_replacements=[original for original, _ in grounded.replacements],
        )

        draft = StyleDraft(subject=subject, body=body, provenance=provenance, prompt=prompt)

        # Write-back happens BEFORE the row is written, so external_draft_ref lands in the
        # same INSERT. A failure here is logged and swallowed: the draft exists and is
        # usable in this platform, and losing the mailbox copy is a degraded outcome, not
        # a reason to throw away work the user is waiting for.
        if self._draft_writer is not None:
            try:
                written = self._draft_writer.write_draft(
                    subject=draft.subject, body=draft.body,
                    to=[recipient_email] if recipient_email else None,
                )
                draft.external_draft_ref = written.external_draft_ref
            except Exception:
                logger.exception(
                    "Could not write the draft back to the mailbox; it remains available "
                    "in the platform"
                )

        if persist:
            draft.draft_id = self._persist(
                draft,
                workflow_id=workflow_id,
                supplier_id=supplier_id,
                recipient_email=recipient_email,
            )
        return draft

    # -- provenance ----------------------------------------------------------------

    def _persist(
        self,
        draft: StyleDraft,
        *,
        workflow_id: Optional[str],
        supplier_id: Optional[str],
        recipient_email: Optional[str],
    ) -> Optional[int]:
        """Write the draft and its provenance to proc.draft_rfq_emails.

        The existing draft table, extended in Phase 0, rather than a second one. Two draft
        tables would mean two provenance stories and only one of them could be right.
        """

        p = draft.provenance
        unique_id = f"STYLE-{_short_id()}"

        def _run(conn):
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO proc.draft_rfq_emails "
                "(rfq_id, unique_id, workflow_id, supplier_id, recipient_email, subject, "
                " body, sent, style_user_ref, style_intent, style_mode, style_profile_id, "
                " style_profile_version, style_fallback_level, style_exemplar_ids, "
                " style_exemplar_set_hash, style_retrieved_at, style_model_id, "
                " style_prompt_version, style_message_ids, style_mailbox_binding_id, "
                " external_draft_ref) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, FALSE, %s, %s, %s, %s, %s, %s, %s, "
                "        %s, %s, %s, %s, %s, %s, %s) "
                "RETURNING id",
                (
                    # rfq_id is NOT NULL and carries the same value as unique_id on this
                    # table — the existing prepare path does the same.
                    unique_id, unique_id, workflow_id, supplier_id, recipient_email,
                    # subject is NOT NULL. An empty one is stored rather than invented:
                    # a reply legitimately has none, and a fabricated subject is a fact
                    # the model was never given.
                    draft.subject or "", draft.body,
                    p.user_ref, p.intent, p.mode, p.style_profile_id,
                    p.style_profile_version, p.fallback_level,
                    p.exemplar_ids or None, p.exemplar_set_hash, p.retrieved_at,
                    p.model_id, p.prompt_version, p.message_ids or None,
                    p.mailbox_binding_id, draft.external_draft_ref,
                ),
            )
            row = cur.fetchone()
            cur.close()
            if self._conn is None and hasattr(conn, "commit"):
                conn.commit()
            return row[0] if row else None

        try:
            if self._conn is not None:
                return _run(self._conn)
            with get_conn() as conn:
                return _run(conn)
        except Exception:
            # A draft the user can see but that failed to record its provenance is worse
            # than useless — it looks authoritative and cannot be explained. Surface it.
            logger.exception("Failed to persist style draft provenance")
            raise


def _short_id() -> str:
    import uuid

    return uuid.uuid4().hex[:16].upper()


def _now(conn: Optional[Any]):
    from datetime import datetime, timezone

    return datetime.now(timezone.utc)
