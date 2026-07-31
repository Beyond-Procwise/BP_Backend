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


# Principals that are NOT people. Two of them already write
# status='resolved', resolution_action='dismiss' rows in bulk:
#   - 'dedup-migration'     (deploy/sql/2026-07-30_discrepancy_dedup.sql, lines 26 and 51)
#   - 'session_postprocess' (src/services/session_postprocess.py:34-38)
# On the live corpus that is 41 dismissals nothing human ever looked at, 20 of them with
# blocks_promotion=TRUE — i.e. they reach apply_hitl_fixes_and_promote and would each mint
# a 'rejected' verdict, which accuracy._AGREES counts as "the reader was right". Eight of
# them crosses MIN_SAMPLE on their own and would mask a demotion that real corrections had
# earned, or manufacture a perfect score for a reader no person ever endorsed.
#
# The skip lives HERE (do not record) rather than in accuracy._LOAD_SQL (do not count)
# because bp_extraction_verdict is read by more than one consumer: supplier_currency's
# _LOAD_SQL joins the same table for the learned-currency vote and never goes through
# _LOAD_SQL, so filtering at read time would leave that consumer poisoned. Not writing the
# row keeps the table's own stated meaning — "one row per human judgement" — literally true
# for every reader of it, present and future.
MACHINE_PRINCIPALS = frozenset({"dedup-migration", "session_postprocess"})


def is_machine_principal(decided_by: Any) -> bool:
    """True when this resolution was written by a process, not a person."""
    return str(decided_by or "").strip().lower() in MACHINE_PRINCIPALS


# READ CONTRACT (see the module docstring of src/services/extraction/provenance.py): a
# HITL-corrected field gets TWO rows in bp_extraction_provenance — the pre-correction
# reader's claim (source != 'hitl') and a source='hitl' row for what is now stored, in
# that order. Plain `ORDER BY id DESC LIMIT 1` returns the 'hitl' row itself, which would
# blame "hitl" for the mistake and lose the reader that actually made it — so 'hitl' rows
# are excluded explicitly.
#
# `attempt`: a re-promoted document can leave more than one non-hitl claim for the same
# field. The verdict being recorded is about whatever the human is overriding *right now*,
# i.e. the most recent prior claim — and `id` alone identifies it. `attempt` deliberately
# does NOT appear in the ORDER BY: it is numbered per raw_id, not per document, so a
# re-extraction (a NEW raw_id for the same parent_pk) starts again at attempt=1 and
# `ORDER BY attempt DESC` would then hand back the OLDER raw_id's attempt=2 row in
# preference to the newer claim. id is a monotonic BIGSERIAL on a single table, so
# ORDER BY id DESC is the correct — and only correct — "most recent" here.
_PROVENANCE_SQL = """
    SELECT source, anchor_ref, confidence
      FROM proc.bp_extraction_provenance
     WHERE parent_table = %s AND parent_pk = %s AND field_name = %s AND source != %s
     ORDER BY id DESC
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
                   resolved_by: Optional[str],
                   snapshot: Optional[dict] = None) -> Optional[str]:
    """Write one verdict row. Returns the verdict, or None when nothing was recorded.

    ``snapshot`` — the ``_field_provenance`` map dispatch froze into
    ``_raw.parser_snapshot`` at extraction time (``provenance.snapshot()``), shaped
    ``{field: {"source", "pattern_name", "confidence"}}``. It is the attribution fallback,
    and on this pipeline it is the one that actually fires.

    Why a fallback is not optional: proc.bp_extraction_provenance is written only inside
    promotion.promote(), and the population that produces verdicts is by construction
    documents that have NOT been promoted — a blocking discrepancy sets
    promotion_status='discrepancy', dispatch skips the inline promote(), and promote_pending
    only scans 'pending'. So at the moment a human resolves a blocking finding there is not
    one provenance row for that document, and reading only that table would write
    source/pattern_name/prior_confidence NULL on every real verdict. A NULL reader matches
    nothing in PatternRegistry.apply_observed and nothing in _compute_accuracy_score, so the
    whole feedback loop would be decorative. The snapshot holds exactly the same three
    values, is written before any discrepancy is raised, and is what promote() itself falls
    back to — so it is used here too, and the provenance table stays authoritative when it
    does have a row (a re-promoted document whose reader has since changed).
    """
    verdict = verdict_for(action, resolved_value=resolved_value,
                          extracted_value=extracted_value)
    if verdict is None:
        return None
    if is_machine_principal(resolved_by):
        # Not a human judgement — see MACHINE_PRINCIPALS above. Recording it would let a
        # bulk dismissal vote "the reader was right" eight times in one migration.
        log.debug("verdict: skipping machine-written resolution by %r on %s.%s",
                  resolved_by, doc_pk, field_name)
        return None
    source = pattern_name = prior = None
    try:
        cur.execute(_PROVENANCE_SQL,
                    (_stg_table(doc_type), str(doc_pk), field_name, HITL_SOURCE))
        row = cur.fetchone()
        if row:
            source, pattern_name, prior = row[0], _decode_pattern_name(row[1]), row[2]
    except Exception:
        # A provenance read failure must not cost us the verdict itself. The snapshot below
        # still attributes it in the ordinary case.
        log.debug("verdict: provenance lookup failed for %s.%s", doc_pk, field_name,
                  exc_info=True)
    if source is None:
        entry = (snapshot or {}).get(field_name)
        if isinstance(entry, dict):
            source = entry.get("source")
            pattern_name = entry.get("pattern_name")
            prior = entry.get("confidence")
    cur.execute(_INSERT, (doc_type, str(doc_pk), field_name, source, pattern_name, prior,
                          verdict,
                          None if extracted_value is None else str(extracted_value),
                          None if resolved_value is None else str(resolved_value),
                          resolved_by))
    return verdict
