"""Exemplars: the two or three emails shown alongside the profile.

A profile tells a model what the rules are. An exemplar shows it what obeying them looks
like. Both go in the prompt, and where they disagree the profile governs — the examples
illustrate the specification, they do not override it.

Under Modes A and B these are **fiction**. They are generated from the approved profile
alone, with invented suppliers, amounts and references, and the generator is never shown
the emails the profile was compiled from. That is not a precaution bolted on afterwards:
it is what lets an exemplar be stored indefinitely and put in every prompt without any of
it being the customer's correspondence. The safety argument is structural, and there is a
test asserting the generation prompt contains nothing but the profile.

Mode C's ``customer_retained`` exemplars are a different proposition — real emails, read
from a mailbox the customer controls, retained by their choice. Those arrive in Phase 5.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, List, Optional

from services.db import get_conn
from services.style.profile import StyleProfile
from services.style.rendering import render_profile_rules
from services.style.repository import ProfileRecord

logger = logging.getLogger(__name__)

# Matches config/settings.py (embedding_model / vector_size). One model across the
# platform: a profile embedded under one and retrieved under another returns confident
# nonsense, and the failure is silent.
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
EMBEDDING_DIM = 1024

DEFAULT_EXEMPLAR_COUNT = 3

ORIGIN_SYNTHETIC = "synthetic"
ORIGIN_CUSTOMER_RETAINED = "customer_retained"

_GENERATION_SCHEMA = {
    "type": "object",
    "properties": {
        "emails": {
            "type": "array",
            "minItems": 2,
            "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string"},
                    "body": {"type": "string"},
                },
                "required": ["subject", "body"],
            },
        }
    },
    "required": ["emails"],
}

_GENERATION_PROMPT = """You write example emails that demonstrate a writing style.

Below is a style specification describing how one person writes. Write {count} short \
procurement emails that follow it exactly. They will be shown to another model as \
examples of what obeying this specification looks like.

Everything in them must be INVENTED. Make up the supplier names, the products, the \
amounts, the reference numbers and the dates. Do not use real company names. These are \
illustrations of form, not records of anything.

Vary the situation across the {count} emails so they show the style in different \
circumstances, but keep the voice identical.

Where the specification uses a placeholder such as {{first_name}}, substitute an invented \
name and write the finished email. Never leave a placeholder in the output, and never \
repeat the specification's own explanatory notes.

STYLE SPECIFICATION
{rules}

Reply with a single JSON object: {{"emails": [{{"subject": "...", "body": "..."}}]}}"""


@dataclass(frozen=True)
class ExemplarRecord:
    """One row of ``proc.bp_style_exemplar``."""

    # Optional because a Mode C2 exemplar is read at draft time and never stored, so it
    # has no row and therefore no id — only the provider's message id.
    exemplar_id: Optional[int]
    user_ref: str
    intent: str
    origin: str
    profile_version_ref: int
    subject: Optional[str]
    body: str
    is_active: bool
    distance: Optional[float] = None
    message_id: Optional[str] = None

    @property
    def is_synthetic(self) -> bool:
        return self.origin == ORIGIN_SYNTHETIC


class ExemplarGenerationError(Exception):
    """The model could not produce usable exemplars."""


_embedder = None


def _default_embedder():
    """Lazily load the platform's embedding model, once per process."""

    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model %s for style exemplars", EMBEDDING_MODEL)
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


def _to_vector_literal(values) -> str:
    return "[" + ",".join(f"{float(x):.6f}" for x in values) + "]"


class ExemplarService:
    """Generates, stores and retrieves style exemplars."""

    def __init__(
        self,
        conn: Optional[Any] = None,
        *,
        generate: Optional[Any] = None,
        embed: Optional[Any] = None,
    ) -> None:
        self._conn = conn
        # Both injectable so tests need neither a GPU nor a model download.
        self._generate = generate
        self._embed = embed

    # -- plumbing -----------------------------------------------------------------

    def _with_conn(self, fn):
        if self._conn is not None:
            return fn(self._conn)
        with get_conn() as conn:
            result = fn(conn)
            if hasattr(conn, "commit"):
                conn.commit()
            return result

    def embed(self, text: str) -> List[float]:
        if self._embed is not None:
            return list(self._embed(text))
        vector = _default_embedder().encode(text, normalize_embeddings=True)
        return [float(x) for x in vector]

    def _call_model(self, prompt: str, *, temperature: float = 0.4) -> str:
        if self._generate is not None:
            return self._generate(prompt)

        from services.ollama_client import ollama_generate

        # Warmer than compilation on purpose: three exemplars generated at temperature 0
        # come back near-identical, which defeats the point of having three.
        text = ollama_generate(
            prompt, temperature=temperature, format=_GENERATION_SCHEMA, think=False
        )
        if not text:
            raise ExemplarGenerationError("the model returned nothing")
        return text

    # -- generation ---------------------------------------------------------------

    def generate_for(
        self,
        profile_record: ProfileRecord,
        *,
        count: int = DEFAULT_EXEMPLAR_COUNT,
    ) -> List[ExemplarRecord]:
        """Generate and store a fresh exemplar set for an approved profile.

        The previous set for this scope is deactivated in the same transaction, so a
        prompt can never mix exemplars generated against two different versions of the
        rules.
        """

        profile: StyleProfile = profile_record.as_profile()
        rules = render_profile_rules(profile)
        prompt = _GENERATION_PROMPT.format(count=count, rules=rules)

        raw = self._call_model(prompt)
        try:
            payload = json.loads(raw)
        except (TypeError, ValueError) as exc:
            raise ExemplarGenerationError(f"the model did not return JSON: {exc}") from exc

        emails = payload.get("emails") if isinstance(payload, dict) else None
        if not isinstance(emails, list) or not emails:
            raise ExemplarGenerationError("the model returned no emails")

        cleaned = [
            (str(e.get("subject") or "").strip(), str(e.get("body") or "").strip())
            for e in emails
            if isinstance(e, dict) and str(e.get("body") or "").strip()
        ]
        if not cleaned:
            raise ExemplarGenerationError("the model returned no usable email bodies")

        vectors = [self.embed(f"{subject}\n{body}") for subject, body in cleaned]

        def _run(conn):
            cur = conn.cursor()
            # Stand the previous generation down first — a mixed set would be showing the
            # model two different specifications at once.
            cur.execute(
                "UPDATE proc.bp_style_exemplar SET is_active = FALSE "
                "WHERE user_ref = %s AND intent = %s AND is_active",
                (profile_record.user_ref, profile_record.intent),
            )
            replaced = cur.rowcount

            stored: List[ExemplarRecord] = []
            for (subject, body), vector in zip(cleaned, vectors):
                cur.execute(
                    "INSERT INTO proc.bp_style_exemplar "
                    "(user_ref, intent, origin, profile_version_ref, subject, body, "
                    " embedding, embedding_model) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s::vector, %s) "
                    "RETURNING exemplar_id, is_active",
                    (
                        profile_record.user_ref,
                        profile_record.intent,
                        ORIGIN_SYNTHETIC,
                        profile_record.version,
                        subject or None,
                        body,
                        _to_vector_literal(vector),
                        EMBEDDING_MODEL,
                    ),
                )
                row = cur.fetchone()
                stored.append(
                    ExemplarRecord(
                        exemplar_id=row[0],
                        user_ref=profile_record.user_ref,
                        intent=profile_record.intent,
                        origin=ORIGIN_SYNTHETIC,
                        profile_version_ref=profile_record.version,
                        subject=subject or None,
                        body=body,
                        is_active=row[1],
                    )
                )
            cur.close()
            return stored, replaced

        stored, replaced = self._with_conn(_run)
        logger.info(
            "Generated %s synthetic exemplars for %s/%s v%s (deactivated %s previous)",
            len(stored), profile_record.user_ref, profile_record.intent,
            profile_record.version, replaced,
        )
        return stored

    # -- retrieval ----------------------------------------------------------------

    def retrieve(
        self,
        user_ref: str,
        intent: str,
        task_text: Optional[str] = None,
        *,
        limit: int = DEFAULT_EXEMPLAR_COUNT,
    ) -> List[ExemplarRecord]:
        """The active exemplars for a scope, nearest to the task first.

        Exact intent match only. Deciding *which* intent to ask for when a scope has no
        exemplars is the fallback ladder's job, and that belongs to the drafting path —
        this method answers the question it was asked.

        Without ``task_text`` the ordering falls back to newest-first: with a handful of
        exemplars per scope the difference is usually academic, and an unreachable
        embedding model must not take drafting down with it.
        """

        vector = None
        if task_text and task_text.strip():
            try:
                vector = _to_vector_literal(self.embed(task_text))
            except Exception:
                logger.warning(
                    "Could not embed the task; falling back to recency ordering",
                    exc_info=True,
                )

        def _run(conn):
            cur = conn.cursor()
            if vector is not None:
                cur.execute(
                    "SELECT exemplar_id, user_ref, intent, origin, profile_version_ref, "
                    "       subject, body, is_active, (embedding <=> %s::vector) AS distance "
                    "  FROM proc.bp_style_exemplar "
                    " WHERE user_ref = %s AND intent = %s AND is_active "
                    "   AND embedding IS NOT NULL "
                    " ORDER BY embedding <=> %s::vector "
                    " LIMIT %s",
                    (vector, user_ref, intent, vector, limit),
                )
            else:
                cur.execute(
                    "SELECT exemplar_id, user_ref, intent, origin, profile_version_ref, "
                    "       subject, body, is_active, NULL AS distance "
                    "  FROM proc.bp_style_exemplar "
                    " WHERE user_ref = %s AND intent = %s AND is_active "
                    " ORDER BY created_at DESC, exemplar_id DESC "
                    " LIMIT %s",
                    (user_ref, intent, limit),
                )
            rows = cur.fetchall()
            cur.close()
            return rows

        rows = self._with_conn(_run)
        return [
            ExemplarRecord(
                exemplar_id=r[0], user_ref=r[1], intent=r[2], origin=r[3],
                profile_version_ref=r[4], subject=r[5], body=r[6], is_active=r[7],
                distance=float(r[8]) if r[8] is not None else None,
            )
            for r in rows
        ]

    def deactivate_all(self, user_ref: str, intent: str) -> int:
        """Stand every exemplar for a scope down. Used when a profile is deleted."""

        def _run(conn):
            cur = conn.cursor()
            cur.execute(
                "UPDATE proc.bp_style_exemplar SET is_active = FALSE "
                "WHERE user_ref = %s AND intent = %s AND is_active",
                (user_ref, intent),
            )
            count = cur.rowcount
            cur.close()
            return count

        return self._with_conn(_run)


def approve_and_generate(
    profile_id: int,
    approved_by: str,
    *,
    conn: Optional[Any] = None,
    service: Optional[ExemplarService] = None,
) -> tuple[ProfileRecord, List[ExemplarRecord]]:
    """Approve a profile and give it a fresh exemplar set.

    Deliberately a function rather than a method on the repository: the repository is
    data access and should not acquire a dependency on an LLM. Approval succeeds or fails
    on its own; if generation then falls over, the profile is still active and drafting
    still works — it just runs without illustrations until this is retried.
    """

    from services.style.repository import StyleProfileRepository

    record = StyleProfileRepository(conn).approve(profile_id, approved_by)

    svc = service or ExemplarService(conn)
    try:
        exemplars = svc.generate_for(record)
    except Exception:
        logger.exception(
            "Approved profile %s but could not generate exemplars; drafting will run "
            "without them until this is retried",
            profile_id,
        )
        exemplars = []
    return record, exemplars
