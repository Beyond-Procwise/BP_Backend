"""Observed accuracy per reader, from what humans actually did.

A rate here is a measurement, not a setting. It answers one question: when this reader
produced this field, how often did a person let the answer stand?
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Any, Optional

log = logging.getLogger(__name__)

# Below this many judgements a rate is noise, and acting on noise is worse than acting on
# the hand-set prior — one bad afternoon of corrections would otherwise switch a reader off.
MIN_SAMPLE = 8

# A verdict that counts as the reader having been RIGHT. 'rejected' means the human dismissed
# the finding, i.e. agreed with the extracted value.
_AGREES = {"confirmed", "rejected"}

# NOTE on who is counted: every row in this table is a HUMAN judgement, because
# verdict.record_verdict refuses to write one for a machine principal (see
# verdict.MACHINE_PRINCIPALS — 'dedup-migration' and 'session_postprocess' both bulk-write
# resolved/dismiss rows). The filter is applied at write time rather than here so that
# every consumer of the table gets it, not just this query.
_LOAD_SQL = """
    SELECT doc_type, field_name, pattern_name, source, verdict
      FROM proc.bp_extraction_verdict
     WHERE decided_at >= now() - (%s || ' days')::interval
"""


def _key(row: dict) -> tuple[str, str, Optional[str]]:
    # A pattern is the finest attribution we have; the AI layer has no pattern, so it is
    # scored under its source. Never pool the two — they are different readers.
    return (row.get("doc_type"), row.get("field_name"),
            row.get("pattern_name") or row.get("source"))


def observed_accuracy(rows: list[dict], *, min_sample: int = MIN_SAMPLE) -> dict:
    """Map (doc_type, field, reader) -> agreement rate, for readers with enough evidence.

    A key with fewer than ``min_sample`` judgements is ABSENT, not zero: callers must fall
    back to the static prior rather than read silence as failure.
    """
    agree: dict = defaultdict(int)
    total: dict = defaultdict(int)
    for row in rows or []:
        k = _key(row)
        total[k] += 1
        if row.get("verdict") in _AGREES:
            agree[k] += 1
    return {k: round(agree[k] / n, 4) for k, n in total.items() if n >= min_sample}


def load_accuracy(conn=None, *, window_days: int = 180,
                  min_sample: int = MIN_SAMPLE) -> dict:
    """observed_accuracy over the last ``window_days`` of verdicts.

    Windowed because a reader that was fixed six months ago should not be judged forever on
    what it did before the fix.
    """
    def _run(c):
        cur = c.cursor()
        cur.execute(_LOAD_SQL, (str(window_days),))
        cols = [d[0] for d in (cur.description or [])]
        return observed_accuracy([dict(zip(cols, r)) for r in cur.fetchall()],
                                 min_sample=min_sample)
    if conn is not None:
        return _run(conn)
    from src.services.db import get_conn
    with get_conn() as own:
        return _run(own)


# ---------------------------------------------------------------------------
# Process-wide cache
# ---------------------------------------------------------------------------
# A rate computed over a 180-day window does not move minute to minute, but
# load_accuracy() is anything but free: src.services.db.get_conn has no pool (every call
# is a fresh psycopg2.connect) and _LOAD_SQL filters on decided_at, which has no index —
# so an uncached call is a TCP connection plus a sequential scan of the whole verdict
# table. Both callers (extraction dispatch, once per document; promotion.promote, once per
# promoted document) were paying that per document. One cache, shared, so the two of them
# also agree on what has been learned within a run.
CACHE_TTL_SECONDS = 900

_CACHE: dict = {}
_CACHED_AT: float = 0.0


def clear_cache() -> None:
    """Drop the cached map (tests, and anything that wants a forced refresh)."""
    global _CACHE, _CACHED_AT
    _CACHE, _CACHED_AT = {}, 0.0


def cached_accuracy(conn=None, *, ttl_seconds: float = CACHE_TTL_SECONDS) -> dict:
    """load_accuracy(), refreshed at most once per ``ttl_seconds``.

    ``conn`` — an open connection to refresh ON, when one is already in hand. A warm cache
    never touches it at all. A refresh runs inside a SAVEPOINT so that a failed read (a
    missing table, a stale column) cannot abort the caller's transaction: promote() calls
    this in the middle of the transaction that writes _stg, and nothing about measuring
    accuracy may cost a document its promotion.

    A failed refresh returns whatever was cached before and does NOT mark the cache fresh,
    so the next document retries rather than pinning an empty map for 15 minutes.
    """
    global _CACHE, _CACHED_AT
    now = time.monotonic()
    if _CACHED_AT and (now - _CACHED_AT) <= ttl_seconds:
        return _CACHE
    try:
        if conn is None:
            _CACHE = load_accuracy()
        elif getattr(conn, "autocommit", False):
            # No open transaction to poison; a savepoint is impossible here anyway.
            _CACHE = load_accuracy(conn)
        else:
            from src.services.agent_actions import _write_on_shared_conn
            box: dict = {}
            _write_on_shared_conn(conn, lambda _c: box.update(v=load_accuracy(conn)))
            _CACHE = box.get("v") or {}
        _CACHED_AT = now
    except Exception:  # noqa: BLE001 — measurement is never worth a caller's transaction
        log.warning("accuracy: refresh failed; keeping the previous map", exc_info=True)
    return _CACHE
