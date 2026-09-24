"""Report jobs: the record behind "start a report now, collect it later".

One row per request in ``proc.bp_report_job`` (deploy/sql/2026-09-24_bp_report_job.sql).
The table holds the two rules that matter, so a bug here cannot break them:
one active job per report, scope and day; and a deck only on a released job.

OWNER, AND WHY A RESTART FAILS A JOB RATHER THAN RESUMING IT

The worker is a thread in this process. A job accepted by a process that has
since exited will never finish, and a status that reads "running" forever is
worse than one that says what happened. Each process mints ``OWNER`` once; a
queued or running job owned by anyone else is healed to failed when it is next
read, and before a new request for the same report so it cannot block one.
Re-running it instead would be a second model call nobody asked for.

That rule assumes one API process -- procwise.service runs ``--workers 1``. A
second worker would heal the first's live jobs; move the queue out of process
before adding one.

Every statement here is a single statement on an autocommit connection, which
is what ``get_conn`` hands out (see reference_get_conn_is_autocommit): the
conditional UPDATEs are atomic on their own and need no transaction around them.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any, Dict, List, Optional, Tuple

from src.services.db import get_conn

OWNER = f"proc-{uuid.uuid4().hex[:12]}"

ACTIVE = ("queued", "running")
_RESTARTED = ("The report was interrupted by a server restart before it finished. "
              "Please run it again.")

# Everything but the deck: a status read must never haul the file.
_COLUMNS = ("job_id", "report_type", "scope", "as_of", "status", "owner",
            "requested_by", "requested_at", "started_at", "finished_at", "run_id",
            "stage_reached", "blocking", "error")
_SELECT = f"SELECT {', '.join(_COLUMNS)} FROM proc.bp_report_job"


def dedup_key(report_type: str, scope: Dict[str, Any], as_of: str) -> str:
    canonical = json.dumps({"report_type": report_type, "scope": scope,
                            "as_of": as_of}, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _row(row: Optional[tuple]) -> Optional[Dict[str, Any]]:
    if row is None:
        return None
    job = dict(zip(_COLUMNS, row))
    for key in ("requested_at", "started_at", "finished_at", "as_of"):
        if job.get(key) is not None:
            job[key] = job[key].isoformat()
    return job


def _heal(cur: Any, where: str, param: str) -> None:
    cur.execute(
        "UPDATE proc.bp_report_job SET status = 'failed', error = %s, finished_at = now() "
        f"WHERE {where} = %s AND status IN %s AND owner <> %s",
        (_RESTARTED, param, ACTIVE, OWNER))


def create(report_type: str, *, scope: Dict[str, Any], as_of: str,
           requested_by: Optional[str]) -> Tuple[Dict[str, Any], bool]:
    """File a job, or return the active one for the same request.

    Returns ``(job, created)``. ``created`` is False when an identical request
    is already queued or running -- the caller must not hand that one to the
    worker again.
    """
    key = dedup_key(report_type, scope, as_of)
    with get_conn() as conn, conn.cursor() as cur:
        _heal(cur, "dedup_key", key)
        # Twice at most: the active job can finish between the conflict and the
        # read, in which case the key is free again and the insert will land.
        for _ in range(2):
            cur.execute(
                "INSERT INTO proc.bp_report_job "
                "  (job_id, report_type, scope, as_of, dedup_key, owner, requested_by) "
                "VALUES (%s, %s, %s::jsonb, %s, %s, %s, %s) "
                "ON CONFLICT (dedup_key) WHERE status IN ('queued', 'running') DO NOTHING "
                "RETURNING job_id",
                (f"rpt-{uuid.uuid4().hex[:12]}", report_type, json.dumps(scope), as_of,
                 key, OWNER, requested_by))
            inserted = cur.fetchone()
            if inserted:
                cur.execute(f"{_SELECT} WHERE job_id = %s", (inserted[0],))
                return _row(cur.fetchone()), True
            cur.execute(f"{_SELECT} WHERE dedup_key = %s AND status IN %s",
                        (key, ACTIVE))
            active = cur.fetchone()
            if active:
                return _row(active), False
    raise RuntimeError(f"could not file or find a report job for {report_type}")


def get(job_id: str) -> Optional[Dict[str, Any]]:
    with get_conn() as conn, conn.cursor() as cur:
        _heal(cur, "job_id", job_id)
        cur.execute(f"{_SELECT} WHERE job_id = %s", (job_id,))
        return _row(cur.fetchone())


def claim(job_id: str) -> bool:
    """Move a queued job to running. True for exactly one caller."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE proc.bp_report_job SET status = 'running', started_at = now() "
            "WHERE job_id = %s AND status = 'queued' AND owner = %s",
            (job_id, OWNER))
        return cur.rowcount == 1


def _finish(job_id: str, status: str, **fields: Any) -> None:
    sets = ", ".join(f"{k} = %s" for k in fields)
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            f"UPDATE proc.bp_report_job SET status = %s, finished_at = now(), {sets} "
            "WHERE job_id = %s AND status IN %s",
            (status, *fields.values(), job_id, ACTIVE))


def finish_released(job_id: str, *, run_id: str, stage_reached: str, deck: bytes,
                    media_type: str, filename: str) -> None:
    _finish(job_id, "released", run_id=run_id, stage_reached=stage_reached,
            deck=deck, media_type=media_type, filename=filename)


def finish_blocked(job_id: str, *, run_id: str, stage_reached: str,
                   blocking: List[Dict[str, Any]]) -> None:
    _finish(job_id, "blocked", run_id=run_id, stage_reached=stage_reached,
            blocking=json.dumps(blocking))


def finish_failed(job_id: str, error: str, *, run_id: Optional[str] = None) -> None:
    _finish(job_id, "failed", error=error, run_id=run_id)


def deck(job_id: str) -> Optional[Tuple[bytes, str, str]]:
    """The released deck as ``(bytes, media_type, filename)``, or None."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT deck, media_type, filename FROM proc.bp_report_job "
                    "WHERE job_id = %s AND status = 'released'", (job_id,))
        row = cur.fetchone()
    if row is None or row[0] is None:
        return None
    return bytes(row[0]), row[1], row[2]
