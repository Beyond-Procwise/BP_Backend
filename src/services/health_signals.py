"""The two liveness signals /health reports about the product's output.

Both exist because their silence has already gone unnoticed for weeks: findings
writes were rejected in bp_sqldb for 65 days, and the document vector store sat
empty for five weeks after the old vector service died. From every existing surface, both
looked exactly like "nothing to report". These helpers are pure functions over
a cursor or a client so they can be tested without a database; /health owns the
connections and turns any failure into the string "unavailable".
"""
from __future__ import annotations

from typing import Any, Optional


def last_finding_written(cur) -> Optional[str]:
    """ISO-8601 timestamp of the newest detection finding, or None when there
    has never been one. ``detected_at`` is the write time of a finding row."""
    cur.execute("SELECT max(detected_at) FROM proc.bp_detection_finding")
    row = cur.fetchone()
    ts = row[0] if row else None
    return ts.isoformat() if ts is not None else None


def vector_store_points(store: Any) -> int:
    """Exact number of document vectors the vector store holds (the one tenant there is today). A zero is a real zero."""
    return int(store.count("default", collection="documents"))


def last_triage_run(cur) -> Optional[dict]:
    """When triage last finished, and how many deals that run failed; None when
    no run has ever finished.

    This is the liveness signal, not last_finding_written: triage dedupes by
    fingerprint, so a finding it already holds is updated in place and never
    written again, and an unchanged corpus produces no new rows for weeks while
    triage is perfectly healthy. A rolled-back run did not happen."""
    cur.execute(
        "SELECT finished_at, failed_deals FROM proc.bp_triage_run"
        " WHERE finished_at IS NOT NULL AND rolled_back_at IS NULL"
        " ORDER BY finished_at DESC LIMIT 1"
    )
    row = cur.fetchone()
    if not row:
        return None
    finished_at, failed = row
    return {"finished_at": finished_at.isoformat(), "failed_deals": len(failed or {})}
