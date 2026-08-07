"""Operational counts and alert conditions for mail intake.

Derived from the tables rather than from a counter library, matching how the
rest of this codebase reports (src/api/routers/metrics.py). Alerts go through
the existing send_alert helper.

The quarantine-rate alert deliberately requires a minimum sample. A rate is
meaningless over a handful of messages, and an alert that cries wolf during a
quiet hour is an alert people learn to close without reading — which costs more
than not having it.

Spec §9.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from src.services.mail_intake.states import ArtifactState, BundleState

log = logging.getLogger(__name__)

ALERT_ADAPTER_FAILURE = "mail_intake.adapter_failure"
ALERT_QUARANTINE_RATE = "mail_intake.quarantine_rate"

MIN_SAMPLE_FOR_RATE = 20
STALE_BINDING_SECONDS = 3600


def _scalar(cur, sql: str, params: tuple) -> int:
    cur.execute(sql, params)
    row = cur.fetchone()
    if not row:
        return 0
    return int(row[0] if isinstance(row, (tuple, list)) else list(row.values())[0])


def collect(cur: Any, *, window_hours: int = 24) -> Dict[str, Any]:
    """Counts over the trailing window."""
    window = (window_hours,)

    bundles = _scalar(
        cur,
        "SELECT count(*) FROM proc.bp_ingest_bundle "
        "WHERE created_at > now() - (%s || ' hours')::interval",
        window,
    )
    artifacts = _scalar(
        cur,
        "SELECT count(*) FROM proc.bp_ingest_artifact "
        "WHERE created_at > now() - (%s || ' hours')::interval AND state = %s",
        (window_hours, ArtifactState.EXTRACTION_COMPLETE.value),
    )

    cur.execute(
        "SELECT quarantine_reason, count(*) FROM proc.bp_ingest_artifact "
        "WHERE quarantine_reason IS NOT NULL "
        "AND created_at > now() - (%s || ' hours')::interval "
        "GROUP BY quarantine_reason",
        window,
    )
    # Plain fetchall, deliberately. This previously carried an unwrap branch
    # that existed only because the test double drained its whole queue —
    # production code contorted to fit a fake. The fake was fixed instead.
    by_reason = {str(r[0]): int(r[1]) for r in (cur.fetchall() or [])}

    duplicates = _scalar(
        cur,
        "SELECT count(*) FROM proc.bp_ingest_artifact "
        "WHERE state = %s AND created_at > now() - (%s || ' hours')::interval",
        (ArtifactState.DUPLICATE.value, window_hours),
    )
    auth_failures = _scalar(
        cur,
        "SELECT count(*) FROM proc.bp_ingest_sender_auth "
        "WHERE (spf = 'fail' OR dkim = 'fail' OR dmarc = 'fail' "
        "       OR alignment = 'misaligned') "
        "AND created_at > now() - (%s || ' hours')::interval",
        window,
    )
    unmatched = _scalar(
        cur,
        "SELECT count(*) FROM proc.bp_ingest_sender_auth "
        "WHERE supplier_match_state <> 'matched' "
        "AND created_at > now() - (%s || ' hours')::interval",
        window,
    )

    return {
        "window_hours": window_hours,
        "bundles_received": bundles,
        "artifacts_ingested": artifacts,
        "quarantine_by_reason": by_reason,
        "quarantine_total": sum(by_reason.values()),
        "duplicate_count": duplicates,
        "sender_auth_failures": auth_failures,
        "unmatched_domains": unmatched,
    }


def check_alerts(
    cur: Any,
    *,
    quarantine_baseline: float = 0.15,
    window_hours: int = 24,
    alert: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> List[str]:
    """Evaluate the Plan 1 alert conditions. Returns the codes that fired."""
    if alert is None:
        from src.services.email_watcher import send_alert as alert  # type: ignore

    fired: List[str] = []

    total = _scalar(
        cur,
        "SELECT count(*) FROM proc.bp_ingest_artifact "
        "WHERE created_at > now() - (%s || ' hours')::interval",
        (window_hours,),
    )
    quarantined = _scalar(
        cur,
        "SELECT count(*) FROM proc.bp_ingest_artifact "
        "WHERE quarantine_reason IS NOT NULL "
        "AND created_at > now() - (%s || ' hours')::interval",
        (window_hours,),
    )

    if total >= MIN_SAMPLE_FOR_RATE:
        rate = quarantined / total
        if rate > quarantine_baseline:
            fired.append(ALERT_QUARANTINE_RATE)
            alert(
                ALERT_QUARANTINE_RATE,
                {"rate": round(rate, 3), "baseline": quarantine_baseline,
                 "total": total, "quarantined": quarantined},
            )

    cur.execute(
        "SELECT binding_id, EXTRACT(EPOCH FROM (now() - last_sync_at))::int "
        "  FROM proc.bp_mailbox_binding "
        " WHERE is_active AND role = 'intake' "
        "   AND (last_sync_at IS NULL OR last_sync_at < now() - interval '1 hour')"
    )
    stale = list(cur.fetchall() or [])
    if stale:
        fired.append(ALERT_ADAPTER_FAILURE)
        alert(
            ALERT_ADAPTER_FAILURE,
            {"stale_bindings": [str(r[0]) for r in stale]},
        )

    return fired
