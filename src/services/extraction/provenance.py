"""Which reader produced each value we kept.

An extraction is several readers arguing: regex patterns (L1), an engineered/NER gap-filler
(L2), a grounded judge (L3), and the context layer (AgentNick), which is the authoritative
gate and frequently writes a value no pattern found. When a human later corrects a field,
"the currency was wrong" tells us nothing unless we know WHO said it — so this records the
producer of every value at the moment it is persisted.

Deliberately cheap: one INSERT per non-null column, best-effort, never able to fail a
promotion. A provenance row is evidence, not part of the document.

READ CONTRACT for a HITL-corrected field (this is what Tasks 2-4 must follow — no join to
_raw.parser_snapshot is needed or supported):

  A field a human corrected gets TWO rows in this table, not one:
    - source != 'hitl'  — the reader that produced the value BEFORE the human overrode
      it (the one being judged for getting it wrong). Absent only when the field was
      genuinely empty pre-correction (a human filled a gap, not a fix).
    - source = 'hitl'   — always present for a corrected field; what is actually stored
      now is the human's, and a row claiming otherwise would misrepresent the document.

  To find "which reader produced the value a human rejected" for (parent_pk, field_name):
      SELECT * FROM proc.bp_extraction_provenance
       WHERE parent_table=... AND parent_pk=... AND field_name=... AND source != 'hitl';
  To find the field's current, human-confirmed value's provenance: filter source='hitl'.
  A field that was NEVER corrected has exactly one row and both queries agree trivially.
"""
from __future__ import annotations

import calendar
import json
import logging
import re
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Iterable

log = logging.getLogger(__name__)

AI_SOURCE = "context_layer"
HITL_SOURCE = "hitl"

_MONTH_NAME_DATE = re.compile(
    r"^(\d{1,2})[\s\-]+([A-Za-z]{3,9})[\s\-,]+(\d{4})$"          # 15 January 2024
    r"|^([A-Za-z]{3,9})[\s\-]+(\d{1,2}),?[\s\-]+(\d{4})$",       # January 15, 2024 / Jan 15 2024
)
_MONTH_LOOKUP = {
    name.lower(): i for i, name in enumerate(calendar.month_name) if name
} | {
    abbr.lower(): i for i, abbr in enumerate(calendar.month_abbr) if abbr
}
_SLASH_OR_DASH_DATE = re.compile(r"^(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})$")


def _comparable_date(text: str) -> str | None:
    """Normalise an unambiguous date STRING to ISO 'YYYY-MM-DD', else None.

    Deliberately conservative: only formats where the day cannot be confused with the
    month are handled — a month spelled out ("15 Jan 2024", "January 15, 2024"), or a
    numeric D/M/Y or M/D/Y form where one part is unambiguously > 12. Plain "01/02/2024"
    is genuinely ambiguous (2 Jan vs 1 Feb) and is left unmatched rather than guessed —
    a wrong match would mis-attribute a value with false confidence, which is worse than
    an honest "unattributed" (context_layer bucket). Genuinely ambiguous numeric dates
    (both parts <= 12, e.g. "01/02/2024") are documented residue, not a bug: resolving
    them needs locale/format context this function does not have, and guessing risks a
    silently wrong attribution — out of scope for what's a comparison helper, not a date
    parser.
    """
    m = _MONTH_NAME_DATE.match(text)
    if m:
        if m.group(1):
            day, month_name, year = m.group(1), m.group(2), m.group(3)
        else:
            month_name, day, year = m.group(4), m.group(5), m.group(6)
        month = _MONTH_LOOKUP.get(month_name.lower())
        if month:
            try:
                return date(int(year), month, int(day)).isoformat()
            except ValueError:
                return None
        return None
    m = _SLASH_OR_DASH_DATE.match(text)
    if m:
        a, b, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        # Unambiguous only when exactly one of the two parts cannot be a month.
        a_is_day_only, b_is_day_only = a > 12, b > 12
        if a_is_day_only and not b_is_day_only:
            day, month = a, b
        elif b_is_day_only and not a_is_day_only:
            day, month = b, a
        else:
            return None                     # ambiguous (or both >12: invalid) — don't guess
        try:
            return date(year, month, day).isoformat()
        except ValueError:
            return None
    return None


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
    # bool is a subclass of int in Python — isinstance(True, int) is True — so this must
    # be checked BEFORE the numeric branch, or Decimal(str(True)) raises InvalidOperation.
    # No boolean columns exist on the four _stg tables today, so this was latent, but a
    # function whose whole job is tolerating column shapes should not crash on one.
    if isinstance(value, bool):
        return str(value).strip().lower()
    if isinstance(value, (int, float, Decimal)):
        return format(Decimal(str(value)).normalize(), "f")
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = str(value).strip().replace(",", "")
    try:
        return format(Decimal(text).normalize(), "f")
    except Exception:            # noqa: BLE001 — not a number, try a date, then plain text
        as_date = _comparable_date(text)
        if as_date is not None:
            return as_date
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


def _write_row(cur, parent_table: str, parent_pk: str, field: str,
               source: str, pattern_name: str | None, confidence: float | None,
               attempt: int) -> None:
    anchor_ref = json.dumps(pattern_name) if pattern_name is not None else None
    cur.execute(_INSERT, (parent_table, str(parent_pk), field, source,
                          anchor_ref, confidence, attempt))


def _original_claim(
    field: str, value: Any, snapshot: dict[str, dict[str, Any]] | None, candidates: list | None,
) -> tuple[str, str | None, float | None] | None:
    """What the pipeline claimed for this field before any HITL correction, or None
    if nothing did (a human filled a genuine gap, not a fix — no reader to blame)."""
    if snapshot is not None:
        entry = snapshot.get(field)
        if not entry:
            return None
        return entry.get("source") or AI_SOURCE, entry.get("pattern_name"), entry.get("confidence")
    if candidates is not None:
        # NOTE: only meaningful when `value` is still the PRE-correction value, i.e. the
        # first-pass dispatch call. Every real HITL caller (promote()) passes `snapshot`
        # instead, precisely because by promotion time `value` is already the human's,
        # and matching a candidate against a value it never offered would misattribute.
        return producer_of(field, value, candidates)
    return None


def record(cur, *, parent_table: str, parent_pk: str, columns: dict,
           candidates: list | None = None, attempt: int = 1,
           snapshot: dict[str, dict[str, Any]] | None = None,
           hitl_fields: Iterable[str] | None = None) -> int:
    """Provenance rows for this promotion. Returns rows written.

    One row per non-null column normally. TWO rows for a field in ``hitl_fields``: the
    reader that produced the value BEFORE the human overrode it (source != 'hitl'; omitted
    only when nothing did — a gap filled, not a correction) PLUS a source='hitl' row for
    what is actually stored now. See the module docstring for the read contract this gives
    Tasks 2-4: filter on source to get either answer with no join to _raw required.

    A HITL field with a NULL current value (keep_null: a human explicitly cleared a bad
    value) still gets its rejected-producer + 'hitl' rows — the correction event is real
    even though nothing is left to attribute a "current" producer to.
    """
    if not parent_pk:
        return 0
    hitl = set(hitl_fields or ())
    cols = columns or {}
    non_null_fields = {f for f, v in cols.items() if not (v is None or v == "")}
    written = 0
    for field in non_null_fields | (hitl & set(cols.keys())):
        value = cols.get(field)
        if field in hitl:
            claim = _original_claim(field, value, snapshot, candidates)
            if claim is not None:
                _write_row(cur, parent_table, parent_pk, field, *claim, attempt)
                written += 1
            _write_row(cur, parent_table, parent_pk, field, HITL_SOURCE, None, None, attempt)
            written += 1
        else:
            claim = _original_claim(field, value, snapshot, candidates) or (AI_SOURCE, None, None)
            _write_row(cur, parent_table, parent_pk, field, *claim, attempt)
            written += 1
    return written
