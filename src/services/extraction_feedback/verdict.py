"""The human's judgement on an extracted value, attached to the reader that produced it.

This is the signal the feedback loop has been missing. The existing proposer counts how
often a field FAILED per vendor; it never reads what the human said the right answer was.
A correction is the most valuable datum the system can get — somebody looked at the
document and the value and told us we were wrong — and it was being thrown away.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

from src.services.extraction.provenance import HITL_SOURCE

log = logging.getLogger(__name__)

# The stg table for a doc_type, matching provenance's `parent_table`. Sourced from
# promotion.py's _RAW_TO_STG rather than reconstructed as f"proc.bp_{doc_type}_stg" —
# contract diverges from that pattern (proc.bp_contracts, not proc.bp_contract_stg), so a
# formulaic name would silently mismatch every contract-field verdict.
try:
    from src.services.extraction.promotion import _RAW_TO_STG as _STG_TABLES
except Exception:  # pragma: no cover - defensive; promotion.py should always import cleanly
    log.warning("verdict: could not import _RAW_TO_STG from promotion.py", exc_info=True)
    _STG_TABLES = {}


def _stg_table(doc_type: str) -> str:
    mapped = _STG_TABLES.get(doc_type)
    return mapped[1] if mapped else f"proc.bp_{doc_type}_stg"


# READ CONTRACT (see the module docstring of src/services/extraction/provenance.py): a
# HITL-corrected field gets TWO rows in bp_extraction_provenance — the pre-correction
# reader's claim (source != 'hitl') and a source='hitl' row for what is now stored, in
# that order. Plain `ORDER BY id DESC LIMIT 1` returns the 'hitl' row itself, which would
# blame "hitl" for the mistake and lose the reader that actually made it — so 'hitl' rows
# are excluded explicitly.
#
# `attempt`: a re-promoted document (a field corrected once, then the same or another
# field corrected again on a later promotion pass) can leave more than one non-hitl claim
# for the same field, one per attempt. The verdict being recorded is about whatever the
# human is overriding *right now*, which is always the most recent prior claim — so this
# orders by attempt DESC (then id DESC as a tiebreaker within an attempt) rather than by
# id DESC alone, which would be correct for a single-attempt document but wrong the moment
# a second attempt exists.
_PROVENANCE_SQL = """
    SELECT source, anchor_ref, confidence
      FROM proc.bp_extraction_provenance
     WHERE parent_table = %s AND parent_pk = %s AND field_name = %s AND source != %s
     ORDER BY attempt DESC, id DESC
     LIMIT 1
"""

_INSERT = """
    INSERT INTO proc.bp_extraction_verdict
        (doc_type, doc_pk, field_name, source, pattern_name, prior_confidence,
         verdict, extracted_value, corrected_value, decided_by)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""


def _same(a: Any, b: Any) -> bool:
    return str(a if a is not None else "").strip() == str(b if b is not None else "").strip()


def _decode_pattern_name(value: Any) -> Any:
    """anchor_ref is jsonb; Task 1 writes it as json.dumps(pattern_name) — a JSON string.

    A standard psycopg2 connection has the jsonb adapter registered and hands this back
    already decoded (a plain Python str). Be defensive anyway: if some caller's cursor
    doesn't have that adapter registered, the raw '"dollar_symbol"' JSON literal would
    otherwise land verbatim (quotes and all) in bp_extraction_verdict.pattern_name.
    """
    if not isinstance(value, str):
        return value
    try:
        decoded = json.loads(value)
    except (ValueError, TypeError):
        return value
    return decoded if isinstance(decoded, str) else value


def verdict_for(action: Optional[str], *, resolved_value: Any,
                extracted_value: Any) -> Optional[str]:
    """What a resolution says about the VALUE (not about the finding).

    'dismiss' reads as 'rejected' — the finding was rejected, which is agreement with what
    was extracted. Scoring it as a negative would teach the system to distrust precisely the
    readers people keep agreeing with.
    """
    verb = (action or "").strip().lower()
    if verb == "dismiss":
        return "rejected"
    if verb == "keep_null":
        return "corrected" if extracted_value not in (None, "") else "confirmed"
    if verb == "apply_value":
        return "confirmed" if _same(resolved_value, extracted_value) else "corrected"
    return None


def record_verdict(cur, *, doc_type: str, doc_pk: str, field_name: str,
                   action: Optional[str], resolved_value: Any, extracted_value: Any,
                   resolved_by: Optional[str]) -> Optional[str]:
    """Write one verdict row. Returns the verdict, or None when there is nothing to say."""
    verdict = verdict_for(action, resolved_value=resolved_value,
                          extracted_value=extracted_value)
    if verdict is None:
        return None
    source = pattern_name = prior = None
    try:
        cur.execute(_PROVENANCE_SQL,
                    (_stg_table(doc_type), str(doc_pk), field_name, HITL_SOURCE))
        row = cur.fetchone()
        if row:
            source, pattern_name, prior = row[0], _decode_pattern_name(row[1]), row[2]
    except Exception:
        # Provenance is an optimisation for attribution; a verdict without it is still worth
        # keeping. Documents extracted before Task 1 shipped have none at all.
        log.debug("verdict: no provenance for %s.%s", doc_pk, field_name, exc_info=True)
    cur.execute(_INSERT, (doc_type, str(doc_pk), field_name, source, pattern_name, prior,
                          verdict,
                          None if extracted_value is None else str(extracted_value),
                          None if resolved_value is None else str(resolved_value),
                          resolved_by))
    return verdict
