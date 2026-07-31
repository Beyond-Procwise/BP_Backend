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
from typing import Any, Iterable

log = logging.getLogger(__name__)

AI_SOURCE = "context_layer"
HITL_SOURCE = "hitl"

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


def snapshot(columns: dict, candidates: list) -> dict[str, dict[str, Any]]:
    """Freeze producer_of() for every non-null column, for replay later.

    promote() — the one funnel every promotion path goes through (dispatch's own
    inline call, the HITL NOTIFY listener, and the promote_pending catch-up sweep)
    — only ever sees the persisted _raw row, not the in-memory Candidate objects
    that produced it. This captures the attribution once, here, while candidates
    are still in scope, in a JSON-serialisable shape meant to be embedded in
    parser_snapshot and read back at promotion time by ``record(..., snapshot=...)``.
    """
    out: dict[str, dict[str, Any]] = {}
    for field, value in (columns or {}).items():
        if value is None or value == "":
            continue
        source, pattern_name, confidence = producer_of(field, value, candidates)
        out[field] = {
            "source": source,
            "pattern_name": pattern_name,
            "confidence": float(confidence) if confidence is not None else None,
        }
    return out


def record(cur, *, parent_table: str, parent_pk: str, columns: dict,
           candidates: list | None = None, attempt: int = 1,
           snapshot: dict[str, dict[str, Any]] | None = None,
           hitl_fields: Iterable[str] | None = None) -> int:
    """One provenance row per non-null column. Returns rows written.

    Attribution priority per field:
      1. ``hitl_fields`` — a human corrected this field; record it as such, not as
         whatever produced the PRE-correction value.
      2. ``snapshot`` — a producer_of() result frozen at extraction time (see
         :func:`snapshot`), used when the live ``candidates`` list that produced
         these values is no longer in scope (i.e. every call from promote()).
      3. ``candidates`` — matched live via producer_of(), as at first-pass
         dispatch time when the Candidate objects are still in memory.
    """
    if not parent_pk:
        return 0
    hitl = set(hitl_fields or ())
    written = 0
    for field, value in (columns or {}).items():
        if value is None or value == "":
            continue          # a field we did not fill has no producer
        if field in hitl:
            source, pattern_name, confidence = HITL_SOURCE, None, None
        elif snapshot is not None:
            entry = snapshot.get(field)
            if entry:
                source = entry.get("source") or AI_SOURCE
                pattern_name = entry.get("pattern_name")
                confidence = entry.get("confidence")
            else:
                source, pattern_name, confidence = AI_SOURCE, None, None
        else:
            source, pattern_name, confidence = producer_of(field, value, candidates)
        anchor_ref = json.dumps(pattern_name) if pattern_name is not None else None
        cur.execute(_INSERT, (parent_table, str(parent_pk), field, source,
                              anchor_ref, confidence, attempt))
        written += 1
    return written
