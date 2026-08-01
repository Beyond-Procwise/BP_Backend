"""Strip findings the sweep captured too late to be true.

Before analysis_store gained its history rule, sweep pass 1 created events for
sessions that had resolved days earlier and left them 'running'. Pass 2 then
froze them by running analysis_findings.capture() — reading the data as it
stood THAT day and filing it as what the July analysis found. Those numbers are
a reconstruction wearing a historical date.

This nulls them, so the UI's honest "Findings were not captured for this
analysis" path renders instead, and re-dates the close to when the session
really ended.

What is NOT touched:
  * document_count — session_document_outcome is a durable record of what was
    genuinely uploaded in that session, not a reconstruction.
  * any analysis frozen promptly after its session resolved — those findings
    are real. Lateness of the close is the whole discriminator.
  * status — an analysis that legitimately failed stays failed.

Idempotent: once findings IS NULL the predicate no longer matches.

Usage:
    ./venv/bin/python -m scripts.repair_backfilled_analysis_findings --dry-run
    ./venv/bin/python -m scripts.repair_backfilled_analysis_findings
"""
from __future__ import annotations

import argparse
import logging
from typing import Any, Optional

from src.services.db import get_conn

log = logging.getLogger(__name__)

# How long after a session's true end a freeze is still believable. Matches
# analysis_store.sweep's history_after_minutes: the same judgement, applied
# after the fact rather than at creation time.
GRACE_MINUTES = 60

_FIND = """
    SELECT a.analysis_id
      FROM proc.bp_analysis a
      JOIN (SELECT session_id, MAX(end_ts AT TIME ZONE 'UTC') AS resolved_at
              FROM proc.process_monitor
             WHERE session_id IS NOT NULL
             GROUP BY session_id) s
        ON s.session_id = a.session_id
     WHERE a.findings IS NOT NULL
       AND s.resolved_at IS NOT NULL
       AND a.completed_at > s.resolved_at + (%s || ' minutes')::interval
"""

_REPAIR = """
    UPDATE proc.bp_analysis a
       SET findings = NULL,
           value_found = NULL,
           currency = NULL,
           completed_at = s.resolved_at
      FROM (SELECT session_id, MAX(end_ts AT TIME ZONE 'UTC') AS resolved_at
              FROM proc.process_monitor
             WHERE session_id IS NOT NULL
             GROUP BY session_id) s
     WHERE a.session_id = s.session_id
       AND a.findings IS NOT NULL
       AND s.resolved_at IS NOT NULL
       AND a.completed_at > s.resolved_at + (%s || ' minutes')::interval
    RETURNING a.analysis_id
"""


def repair(*, dry_run: bool = False, grace_minutes: int = GRACE_MINUTES,
           conn: Optional[Any] = None) -> dict:
    """Null the late-captured findings. Returns {"repaired": int}."""
    def _run(c: Any) -> dict:
        cur = c.cursor()
        cur.execute(_FIND if dry_run else _REPAIR, (str(int(grace_minutes)),))
        rows = cur.fetchall() or []
        return {"repaired": len(rows),
                "analysis_ids": [str(r[0]) for r in rows]}

    if conn is not None:
        return _run(conn)
    # get_conn() is a context manager, not a bare connection.
    with get_conn() as c:
        got = _run(c)
        if not dry_run:
            c.commit()
        return got


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true",
                   help="report what would change without writing")
    p.add_argument("--grace-minutes", type=int, default=GRACE_MINUTES)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    got = repair(dry_run=args.dry_run, grace_minutes=args.grace_minutes)
    verb = "would repair" if args.dry_run else "repaired"
    log.info("%s %s analyses %s", verb, got["repaired"], got["analysis_ids"])


if __name__ == "__main__":
    main()
