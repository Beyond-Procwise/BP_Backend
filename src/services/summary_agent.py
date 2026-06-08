"""Persona-driven summaries over the final (_trgt) procurement tables.

Mirrors ``deal_summary`` (direct SQL, cloud LLM, no local GPU). A persona is
resolved from the ``bp_prompt`` governance table (``prompt_type='summary_persona'``)
with a raw-string fallback. Results are cached and versioned in ``proc.bp_summary``.
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from src.services.db import get_conn
from src.services.ollama_client import ollama_cloud_generate
from src.services.deal_summary import gather_deal_context, _build_prompt

log = logging.getLogger(__name__)

# Summaries run on the Ollama Cloud API (remote), keeping the local GPU free.
_SUMMARY_MODEL = os.getenv("PROCWISE_SUMMARY_MODEL", "gpt-oss:120b")


class SummarizationError(RuntimeError):
    """Raised when the LLM returns no usable summary."""


class SnapshotNotFound(RuntimeError):
    """Raised when an as_of request finds no snapshot at/before the datetime."""


def resolve_persona(persona: str, conn: Any) -> tuple[str, str]:
    """Return (framing_text, persona_source).

    Looks up ``persona`` in bp_prompt (prompt_type='summary_persona'). On a hit
    returns the stored template and 'bp_prompt'; on a miss returns the persona
    string itself and 'raw'.
    """
    cur = conn.cursor()
    row = None
    try:
        cur.execute(
            "SELECT prompts_desc FROM proc.bp_prompt "
            "WHERE prompt_type = 'summary_persona' AND prompt_name = %s "
            "AND COALESCE(prompts_status, 1) = 1 LIMIT 1",
            (persona,),
        )
        row = cur.fetchone()
    except Exception:  # pragma: no cover - defensive
        log.exception("persona lookup failed for %s", persona)
    if row and row[0]:
        payload = row[0] if isinstance(row[0], dict) else json.loads(row[0])
        if isinstance(payload, dict):
            template = payload.get("prompt_template") or payload.get("template")
            if template:
                return str(template), "bp_prompt"
    return persona, "raw"
