"""Which reader produced each value we kept.

An extraction is several readers arguing: regex patterns (L1), an engineered/NER gap-filler
(L2), a grounded judge (L3), and the context layer (AgentNick), which is the authoritative
gate and frequently writes a value no pattern found. When a human later corrects a field,
"the currency was wrong" tells us nothing unless we know WHO said it — so this records the
producer of every value at the moment it is persisted.

Deliberately cheap: one INSERT per non-null column, best-effort, never able to fail a
promotion. A provenance row is evidence, not part of the document.
"""
from __future__ import annotations

import json
import logging
from decimal import Decimal
from typing import Any

log = logging.getLogger(__name__)

AI_SOURCE = "context_layer"

# anchor_ref is jsonb on proc.bp_extraction_provenance (shared with the structural
# extractor and extraction_v2 writers) — the pattern_name must be cast/encoded as JSON,
# not passed as a bare string, or Postgres rejects it as invalid JSON input.
_INSERT = """
    INSERT INTO proc.bp_extraction_provenance
        (parent_table, parent_pk, field_name, source, anchor_ref, confidence, attempt)
    VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s)
"""


def _comparable(value: Any) -> str:
    """One shape for comparing a captured string against a bound column value.

    The candidate holds what was literally on the page ("1,234.50"); the column holds what
    the type binder made of it (Decimal('1234.50')). Same value, different shapes.
    """
    if value is None:
        return ""
    if isinstance(value, (int, float, Decimal)):
        return format(Decimal(str(value)).normalize(), "f")
    text = str(value).strip().replace(",", "")
    try:
        return format(Decimal(text).normalize(), "f")
    except Exception:            # noqa: BLE001 — not a number, compare as text
        return str(value).strip().lower()


def producer_of(field: str, value: Any, candidates: list) -> tuple[str, str | None, float | None]:
    """(source, pattern_name, confidence) for whoever produced this value.

    Attribution is on the VALUE, not the field: several patterns offer a currency and only
    one of them is what got kept. No match means no reader we track offered it, which on
    this pipeline means the context layer wrote it — recorded as such rather than guessed
    at, because crediting a pattern for the AI layer's work would poison its score.
    """
    target = _comparable(value)
    for c in candidates or []:
        if getattr(c, "field", None) == field and _comparable(getattr(c, "value", None)) == target:
            return (getattr(c, "source", None) or AI_SOURCE,
                    getattr(c, "pattern_name", None),
                    getattr(c, "confidence", None))
    return AI_SOURCE, None, None


def record(cur, *, parent_table: str, parent_pk: str, columns: dict,
           candidates: list, attempt: int = 1) -> int:
    """One provenance row per non-null column. Returns rows written."""
    if not parent_pk:
        return 0
    written = 0
    for field, value in (columns or {}).items():
        if value is None or value == "":
            continue          # a field we did not fill has no producer
        source, pattern_name, confidence = producer_of(field, value, candidates)
        anchor_ref = json.dumps(pattern_name) if pattern_name is not None else None
        cur.execute(_INSERT, (parent_table, str(parent_pk), field, source,
                              anchor_ref, confidence, attempt))
        written += 1
    return written
