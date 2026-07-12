"""Grounding guard: block hallucinated field values from promoting.

A committed field is *grounded* when its value can be located in the source
document by a format-tolerant match. This deliberately does NOT require a
byte-exact `evidence_text in full_text` (the check used by the offline
hallucination audit), because that over-reports: legitimate reformatting —
newlines in the evidence span, dates normalized to ISO, amounts normalized to a
plain decimal — trips exact matching even when the value is correct (see
FINDINGS.md F1, where ~96%+ of "hallucinations" were correct-but-reformatted).

The guard blocks only values that cannot be found in the document by ANY
tolerant strategy, so it removes genuine fabrications without discarding correct
data. Ungrounded fields are demoted to residuals (→ NULL / manual review) rather
than promoted to the final tables.
"""
from __future__ import annotations

import re
from datetime import date, datetime

from .schemas.result import CommittedField, ResidualField

# Models whose values are deterministically derived by our own pipeline code
# (not read/claimed from the document) — they need not appear verbatim.
_DERIVATION_MODELS = frozenset({"pipeline_recovery"})

# Evidence markers the pipeline uses for deliberately synthetic values.
_SYNTHETIC_MARKERS = ("synthetic_", "syn-")


def _norm(s: str) -> str:
    """Collapse whitespace and lowercase for tolerant substring matching."""
    return re.sub(r"\s+", " ", s or "").strip().lower()


def _digits(s: str) -> str:
    return re.sub(r"\D", "", s or "")


def _date_renderings(value: str) -> list[str]:
    """If ``value`` parses as an ISO date, return common textual renderings of it.

    Lets an ISO-normalized value (``2024-02-15``) match a document that writes
    the date differently (``15 Feb 2024``, ``5/2/2024``, ``15.02.2024``, ...).
    Renderings cover both zero-padded and non-padded day/month and the common
    separators, so a correctly-extracted date is not blocked over formatting.
    """
    v = (value or "").strip()
    if not v:
        return []
    # parse_date handles ISO, day-first (15/02/2024), and textual (3 July 2026)
    # formats, so a value stored in ANY of them cross-renders to the others.
    d: date | None = None
    try:
        from src.services.extraction_v2.parsers.dates import parse_date as _pd
        parsed = _pd(v)
        if parsed is not None:
            d = date(parsed.year, parsed.month, parsed.day)
    except Exception:
        d = None
    if d is None:
        try:
            d = datetime.fromisoformat(v[:10]).date()
        except (ValueError, TypeError):
            return []

    mon_abbr = d.strftime("%b")   # "Feb"
    mon_full = d.strftime("%B")   # "February"
    yr = d.year
    day, mon = d.day, d.month     # non-padded ints
    out: list[str] = []
    # textual month forms (padded + non-padded day, DMY and MDY)
    for dd in (f"{day:02d}", str(day)):
        for mon_name in (mon_abbr, mon_full):
            out.append(f"{dd} {mon_name} {yr}")
            out.append(f"{dd} {mon_name}, {yr}")
            out.append(f"{mon_name} {dd}, {yr}")
            out.append(f"{mon_name} {dd} {yr}")
    # numeric forms across separators, padded + non-padded, DMY / MDY / YMD
    for sep in ("/", "-", "."):
        for dd, mm in ((f"{day:02d}", f"{mon:02d}"), (str(day), str(mon))):
            out.append(f"{dd}{sep}{mm}{sep}{yr}")   # DMY
            out.append(f"{mm}{sep}{dd}{sep}{yr}")   # MDY
            out.append(f"{yr}{sep}{mm}{sep}{dd}")   # YMD
    return out


def is_value_grounded(
    value: str,
    evidence_text: str,
    full_text: str,
    *,
    model: str = "",
) -> bool:
    """Return True if ``value`` is locatable in ``full_text`` by any tolerant match.

    Grounded (keep) when ANY of:
      - the field is a deterministic pipeline derivation, or synthetic;
      - the document is unavailable (cannot verify → do not block);
      - normalized ``evidence_text`` is a substring of the document;
      - normalized ``value`` is a substring of the document;
      - ``value`` parses as a date and any common rendering appears; or
      - the digit-signature of ``value`` (≥3 digits) appears in the document.
    Otherwise the value is treated as ungrounded (a hallucination) → block.
    """
    if model in _DERIVATION_MODELS:
        return True

    ev = evidence_text or ""
    if ev and ev.strip().lower().startswith(_SYNTHETIC_MARKERS):
        return True

    nft = _norm(full_text)
    if not nft:  # no document text to verify against — never block
        return True

    if ev and _norm(ev) in nft:
        return True

    val = value or ""
    if val and _norm(val) in nft:
        return True

    for rendering in _date_renderings(val):
        if _norm(rendering) in nft:
            return True

    dv = _digits(val)
    if len(dv) >= 3 and dv in _digits(full_text):
        return True

    return False


def ground_committed_fields(
    committed: list[CommittedField],
    residuals: list[ResidualField],
    full_text: str,
) -> tuple[list[CommittedField], list[ResidualField]]:
    """Demote ungrounded committed fields to residuals.

    Returns ``(kept_committed, residuals)`` where each removed field gains a
    ``ResidualField(reason="ungrounded_value")`` so it is routed to review
    instead of promoted.
    """
    kept: list[CommittedField] = []
    for cf in committed:
        if is_value_grounded(cf.value, cf.evidence_text, full_text, model=cf.model):
            kept.append(cf)
        else:
            residuals.append(
                ResidualField(field_path=cf.field_path, reason="ungrounded_value", candidates=[])
            )
    return kept, residuals
