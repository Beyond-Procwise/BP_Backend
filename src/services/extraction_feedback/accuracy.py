"""Observed accuracy per reader, from what humans actually did.

A rate here is a measurement, not a setting. It answers one question: when this reader
produced this field, how often did a person let the answer stand?
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Optional

log = logging.getLogger(__name__)

# Below this many judgements a rate is noise, and acting on noise is worse than acting on
# the hand-set prior — one bad afternoon of corrections would otherwise switch a reader off.
MIN_SAMPLE = 8

# A verdict that counts as the reader having been RIGHT. 'rejected' means the human dismissed
# the finding, i.e. agreed with the extracted value.
_AGREES = {"confirmed", "rejected"}

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
