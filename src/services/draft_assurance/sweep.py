"""Close the drafts nobody sent and nobody abandoned.

A person closing a draft without sending it is recorded when they do it, best-effort. A draft that is simply left
sits forever with no outcome, so "how many drafts were never used" cannot be answered. This sweep gives such a draft an
``abandoned`` outcome (by ``system:draft-sweep``) once it has been quiet for a governed number of days.

The one mistake that matters is a FALSE abandon: recording a send is best-effort too, so a draft with no outcome may well
have gone out. So the sweep abandons a draft only when the product tables CONFIRM it was not sent:

    sent        draft_rfq_emails says sent / has a sent_on, or workflow_email_tracking has a row for it  -> left alone
    unsent      a draft_rfq_emails row exists and nothing says it went                                     -> abandoned
    unknown     the product has no record of it                                                            -> left alone

If that check cannot be made at all, nothing is abandoned. Only the LATEST capture of a draft is considered (an earlier
one was replaced by a regeneration, not abandoned). Config is a policy row; a missing or bad value makes the sweep refuse.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)
SLUG = "email_draft_sweep_rules"
ACTOR = "system:draft-sweep"


class SweepRulesUnavailable(RuntimeError):
    """The period or batch size is missing or unusable: the sweep does not run on a number it made up."""


def check_rules(rules: Any) -> Dict[str, int]:
    if not isinstance(rules, dict):
        raise SweepRulesUnavailable("no sweep rules are defined")
    out = {}
    for key in ("abandon_after_days", "batch_size"):
        v = rules.get(key)
        if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
            raise SweepRulesUnavailable(f"{key} is missing or not a positive whole number")
        out[key] = v
    return out


def load_rules(policy_engine: Any) -> Dict[str, int]:
    if policy_engine is None:
        raise SweepRulesUnavailable("no policy engine available")
    try:
        policy = policy_engine.get_policy(SLUG)
    except Exception as exc:  # noqa: BLE001
        raise SweepRulesUnavailable(f"policy store unreadable: {exc}") from exc
    rules = ((policy or {}).get("details") or {}).get("rules") if isinstance(policy, dict) else None
    return check_rules(rules)


def find_stale(conn: Any, days: int, now: datetime, limit: int) -> List[Dict[str, Any]]:
    """The latest capture of each draft that has NO outcome and was captured more than ``days`` ago, oldest first."""

    with conn.cursor() as cur:
        cur.execute(
            """SELECT c.capture_id, c.unique_id, c.captured_at FROM email_agent.bp_draft_capture c
               WHERE c.captured_at < %s
                 AND c.capture_id = (SELECT max(c2.capture_id) FROM email_agent.bp_draft_capture c2 WHERE c2.unique_id = c.unique_id)
                 AND NOT EXISTS (SELECT 1 FROM email_agent.bp_draft_outcome o WHERE o.capture_id = c.capture_id)
               ORDER BY c.captured_at, c.capture_id LIMIT %s""",
            (now - timedelta(days=days), limit))
        return [{"capture_id": r[0], "unique_id": r[1], "captured_at": r[2]} for r in cur.fetchall()]


def send_status(conn: Any, unique_ids: Iterable[str]) -> Dict[str, str]:
    """{unique_id: 'sent' | 'unsent' | 'unknown'} from the product tables. Raises if the tables cannot be read."""

    ids = sorted({str(u) for u in unique_ids if u})
    if not ids:
        return {}
    status = {u: "unknown" for u in ids}
    with conn.cursor() as cur:
        cur.execute("SELECT unique_id, bool_or(COALESCE(sent, false) OR sent_on IS NOT NULL) FROM proc.draft_rfq_emails "
                    "WHERE unique_id = ANY(%s) GROUP BY unique_id", (ids,))
        for uid, went in cur.fetchall():
            status[uid] = "sent" if went else "unsent"
        cur.execute("SELECT DISTINCT unique_id FROM proc.workflow_email_tracking WHERE unique_id = ANY(%s)", (ids,))
        for (uid,) in cur.fetchall():
            status[uid] = "sent"                                    # a dispatch was tracked: it went, whatever the draft row says
    return status


def abandon(conn: Any, capture_id: int, days: int) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO email_agent.bp_draft_outcome (capture_id, outcome, abandoned_by, abandon_reason)
               SELECT %s, 'abandoned', %s, %s
               WHERE NOT EXISTS (SELECT 1 FROM email_agent.bp_draft_outcome WHERE capture_id = %s)
               RETURNING outcome_id""",
            (capture_id, ACTOR, f"no send or decision within {days} days", capture_id))
        return cur.fetchone() is not None


def sweep(writer_conn: Any, reader_conn: Any, rules: Dict[str, Any], now: Optional[datetime] = None) -> Dict[str, Any]:
    """One pass. ``writer_conn`` reads and writes email_agent; ``reader_conn`` reads the product tables."""

    rules = check_rules(rules)
    now = now or datetime.now(timezone.utc)
    stale = find_stale(writer_conn, rules["abandon_after_days"], now, rules["batch_size"])
    report: Dict[str, Any] = {"examined": len(stale), "abandoned": 0, "skipped_sent": 0, "skipped_unverifiable": 0}
    if not stale:
        return report
    try:
        status = send_status(reader_conn, [s["unique_id"] for s in stale])
    except Exception as exc:  # noqa: BLE001 - not being able to check is not a reason to abandon anything
        logger.exception("draft sweep could not check what was sent")
        return {**report, "error": f"{type(exc).__name__}: {exc}"}
    for item in stale:
        state = status.get(item["unique_id"], "unknown")
        if state == "sent":
            report["skipped_sent"] += 1
        elif state != "unsent":
            report["skipped_unverifiable"] += 1
        elif abandon(writer_conn, item["capture_id"], rules["abandon_after_days"]):
            report["abandoned"] += 1
    return report
