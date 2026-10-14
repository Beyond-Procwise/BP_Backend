"""Retry approved actions whose replay never happened (user ruling, Task 7 review).

The approve endpoint commits the decision and then hands the replay to a background worker. A
restart in between, or no agent runtime at that moment, would lose it. Every sweep (once a
minute, from the approval sweep job) looks for approved cases with no agent_policy_replay row
naming them whose approval is older than RETRY_AFTER, and runs replay.run for them.

Bounded: each case records its attempts in facts.replayAttempts, bumped BEFORE the run with a
compare-and-set (two sweeps never both take the same attempt), and is never tried after
MAX_ATTEMPTS. Approvals older than LOOKBACK are left alone. Safe to repeat: replay.run is
exactly-once per group (a replay row is claimed before the tool runs), and a group that is still
waiting for another approval, or was rejected, simply answers so.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

from services.agent_policy import approvals

logger = logging.getLogger(__name__)

RETRY_AFTER = timedelta(minutes=2)
LOOKBACK = timedelta(days=7)
MAX_ATTEMPTS = 3
REPLAY_SUBJECT_TYPE = "agent_policy_replay"
_BATCH = 50

_CANDIDATES = """
    SELECT c.decision_id, COALESCE((c.facts->>'replayAttempts')::int, 0) AS attempts
      FROM proc.bp_decision c
     WHERE c.subject_type = %(case_type)s AND c.decision = 'approve_or_reject' AND c.status <> 'open'
       AND COALESCE((c.facts->>'replayAttempts')::int, 0) < %(max)s
       AND EXISTS (SELECT 1 FROM proc.bp_decision a
                    WHERE a.subject_type = c.subject_type AND a.subject_id = c.subject_id
                      AND a.decision = 'approve' AND a.actioned_by IS NOT NULL
                      AND a.actioned_at <= %(before)s AND a.actioned_at >= %(since)s)
       AND NOT EXISTS (SELECT 1 FROM proc.bp_decision r
                        WHERE r.subject_type = %(replay_type)s
                          AND r.facts->'caseIds' @> to_jsonb(ARRAY[c.decision_id]))
       {narrow}
     ORDER BY c.decision_id
     LIMIT %(batch)s
"""


def _take_attempt(conn, decision_id: int, seen: int, now: datetime) -> bool:
    """Bump replayAttempts from `seen` to seen+1; False if another sweep took it first."""
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE proc.bp_decision SET facts = facts || jsonb_build_object("
            "'replayAttempts', %s::int, 'replayLastAttemptAt', %s::text) "
            "WHERE decision_id = %s AND subject_type = %s "
            "AND COALESCE((facts->>'replayAttempts')::int, 0) = %s",
            (seen + 1, now.isoformat(), decision_id, approvals.SUBJECT_TYPE, seen))
        return cur.rowcount == 1


def retry_lost_replays(conn, now: datetime, *, run: Optional[Callable[[int], Any]] = None,
                       decision_ids: Optional[List[int]] = None) -> Dict[str, int]:
    """Run replay.run for approved cases that were never replayed. `decision_ids` narrows (tests)."""
    if run is None:
        from services.agent_policy import replay  # local: pulls in the orchestrator tools
        run = replay.run
    params = {"case_type": approvals.SUBJECT_TYPE, "replay_type": REPLAY_SUBJECT_TYPE,
              "max": MAX_ATTEMPTS, "before": now - RETRY_AFTER, "since": now - LOOKBACK, "batch": _BATCH}
    narrow = ""
    if decision_ids is not None:
        narrow = "AND c.decision_id = ANY(%(ids)s)"
        params["ids"] = list(decision_ids)
    with conn.cursor() as cur:
        cur.execute(_CANDIDATES.format(narrow=narrow), params)
        rows = [(int(r[0]), int(r[1])) for r in cur.fetchall()]
    counts = {"retried": 0, "skipped": 0, "errors": 0}
    for did, seen in rows:
        if not _take_attempt(conn, did, seen, now):
            counts["skipped"] += 1
            continue
        try:
            out = run(did)
            counts["retried"] += 1
            logger.info("replay retry for decision %s (attempt %s): %s", did, seen + 1,
                        (out or {}).get("status") if isinstance(out, dict) else "done")
        except Exception as exc:  # noqa: BLE001
            counts["errors"] += 1
            logger.error("replay retry for decision %s failed: %s", did, type(exc).__name__)
    return counts
