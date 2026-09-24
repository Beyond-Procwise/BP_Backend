"""Report jobs: the record behind "start a report now, collect it later".

One row per request in ``proc.bp_report_job`` (deploy/sql/2026-09-24_bp_report_job.sql).
The table holds the two rules that matter, so a bug here cannot break them:
one active job per report, scope and day; and a deck only on a released job.

WHY A RESTART FAILS A JOB RATHER THAN RESUMING IT, AND HOW ONE IS SPOTTED

The worker is a thread in the process that accepted the job. If that process
exits, the job will never finish, and a status that reads "running" forever is
worse than one that says what happened. Re-running it instead would be a second
model call nobody asked for.

A job is spotted as stranded by its HEARTBEAT, not by who is reading it: the
worker stamps ``heartbeat_at`` on every job its process holds every 30 seconds
(``beat``), and a queued or running job whose stamp is older than
``STALE_SECONDS`` is healed to failed when next read or listed, and before a
repeat request for the same report so it cannot block one. The first version
healed by owner, which meant any OTHER process -- a test run, a second server --
failed the main server's live report the moment it listed the jobs.

``OWNER`` still decides who may CLAIM a job: a job runs in the process whose
queue it was put on, never in another that happens to read it.

Every statement here is a single statement on an autocommit connection, which
is what ``get_conn`` hands out (see reference_get_conn_is_autocommit): the
conditional UPDATEs are atomic on their own and need no transaction around them.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.services.db import get_conn
from src.services.rga import audit
from src.services.rga.factpack import pack_id_for

OWNER = f"proc-{uuid.uuid4().hex[:12]}"

ACTIVE = ("queued", "running")
#: Four missed beats. Long enough that a busy GPU never starves the heartbeat
#: thread into a false alarm; short enough that a restart shows within minutes.
STALE_SECONDS = 120
_RESTARTED = ("The report was interrupted by a server restart before it finished. "
              "Please run it again.")

# Everything but the deck: a status read must never haul the file.
_COLUMNS = ("job_id", "report_type", "scope", "as_of", "status", "owner",
            "requested_by", "requested_at", "started_at", "finished_at", "run_id",
            "stage_reached", "blocking", "error", "entitlement",
            "dismissed_at", "dismissed_by", "dismiss_reason", "has_page",
            "title", "current_version", "last_edited_by", "editable")
# Computed columns, by name: whether a printable page is stored, without hauling it.
_COMPUTED = {"has_page": "({p}page IS NOT NULL)",
             # Editable = released with its Fact Pack stored (2026-09-24 onwards).
             "editable": "({p}fact_pack IS NOT NULL)"}


def _select_list(prefix: str = "") -> str:
    return ", ".join(_COMPUTED[c].format(p=prefix) if c in _COMPUTED else f"{prefix}{c}"
                     for c in _COLUMNS)


_SELECT = f"SELECT {_select_list()} FROM proc.bp_report_job"


def dedup_key(report_type: str, scope: Dict[str, Any], as_of: str) -> str:
    canonical = json.dumps({"report_type": report_type, "scope": scope,
                            "as_of": as_of}, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _row(row: Optional[tuple]) -> Optional[Dict[str, Any]]:
    if row is None:
        return None
    job = dict(zip(_COLUMNS, row))
    for key in ("requested_at", "started_at", "finished_at", "as_of", "dismissed_at"):
        if job.get(key) is not None:
            job[key] = job[key].isoformat()
    return job


_STRANDED = ("status IN %s AND heartbeat_at < now() - make_interval(secs => %s)")


def _emit_healed(rows: List[Dict[str, Any]]) -> None:
    """Close each healed job's trail: without this, a stranded run's events just
    stop, and read exactly like a run still in progress."""
    for row in rows:
        with audit.run_context(job_id=row["job_id"], requested_by=row["requested_by"]):
            audit.emit(audit.RUN_FAILED, run_id=row["run_id"] or row["job_id"],
                       agent="rga_job_store", status="failed", summary=_RESTARTED,
                       details={"reason": "stranded", "stale_after_seconds": STALE_SECONDS})


_on_healed = _emit_healed


def _heal(cur: Any, where: Optional[str] = None, param: Optional[str] = None) -> None:
    """Fail the stranded jobs -- all of them, or those where ``where = param``.

    RETURNING names exactly the rows this statement moved, so two processes
    healing at once each report only their own: the second finds nothing to move.
    """
    scoped = f"{where} = %s AND " if where else ""
    cur.execute(
        "UPDATE proc.bp_report_job SET status = 'failed', error = %s, finished_at = now() "
        f"WHERE {scoped}{_STRANDED} RETURNING job_id, run_id, requested_by",
        (_RESTARTED, *((param,) if where else ()), ACTIVE, STALE_SECONDS))
    healed = [dict(zip(("job_id", "run_id", "requested_by"), r)) for r in cur.fetchall()]
    if healed:
        try:
            _on_healed(healed)
        except Exception:  # noqa: BLE001 - a missing event must not undo the heal
            import logging
            logging.getLogger(__name__).exception("rga: could not audit healed report jobs")


def beat() -> None:
    """Vouch for every job this process holds. Called by the worker's heartbeat."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("UPDATE proc.bp_report_job SET heartbeat_at = now() "
                    "WHERE owner = %s AND status IN %s", (OWNER, ACTIVE))


def create(report_type: str, *, scope: Dict[str, Any], as_of: str,
           requested_by: Optional[str],
           entitlement: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], bool]:
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
                "  (job_id, report_type, scope, as_of, dedup_key, owner, requested_by, "
                "   run_id, entitlement) "
                "VALUES (%s, %s, %s::jsonb, %s, %s, %s, %s, %s, %s::jsonb) "
                "ON CONFLICT (dedup_key) WHERE status IN ('queued', 'running') DO NOTHING "
                "RETURNING job_id",
                (f"rpt-{uuid.uuid4().hex[:12]}", report_type, json.dumps(scope), as_of,
                 key, OWNER, requested_by, pack_id_for(report_type, scope, as_of),
                 json.dumps(entitlement) if entitlement is not None else None))
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


def recent(limit: int) -> List[Dict[str, Any]]:
    """The newest jobs, everyone's, without their decks. Heals the stranded first
    so the list never shows a job as running that no process is running."""
    with get_conn() as conn, conn.cursor() as cur:
        _heal(cur)
        cur.execute(f"{_SELECT} ORDER BY requested_at DESC LIMIT %s", (limit,))
        return [_row(r) for r in cur.fetchall()]


# A later run of the SAME report and scope. `scope` is jsonb, so equality is by
# value, not by key order.
_LATER_SAME = ("r.report_type = j.report_type AND r.scope = j.scope "
               "AND r.requested_at > j.requested_at")


def needs_attention(limit: int) -> List[Dict[str, Any]]:
    """Blocked or failed jobs nobody has dealt with -- the Action Centre's
    Reports items. Dealt with means dismissed by a person, or settled by a later
    RELEASED run of the same report and scope. A rerun still in progress does
    not settle it; it is reported as ``rerun_status`` so the item can say so."""
    cols = _select_list("j.")
    with get_conn() as conn, conn.cursor() as cur:
        _heal(cur)
        cur.execute(
            f"SELECT {cols}, "
            f"  (SELECT r.status FROM proc.bp_report_job r WHERE {_LATER_SAME} "
            "     AND r.status IN %s ORDER BY r.requested_at DESC LIMIT 1) AS rerun_status "
            "  FROM proc.bp_report_job j "
            # The newest sign-off decision on the job, if any (bp_approval, newest wins).
            "  LEFT JOIN LATERAL (SELECT a.status FROM proc.bp_approval a "
            "                      WHERE a.grounding->>'report_job_id' = j.job_id "
            "                      ORDER BY a.created_date DESC, a.approval_id DESC LIMIT 1) d "
            "         ON true "
            " WHERE j.dismissed_at IS NULL "
            # Blocked or failed; or released and not signed off (awaiting or refused -- the
            # caller drops report types the policy says need no sign-off).
            "   AND (j.status IN ('blocked', 'failed') "
            "        OR (j.status = 'released' AND COALESCE(d.status, '') <> 'approved')) "
            f"  AND NOT EXISTS (SELECT 1 FROM proc.bp_report_job r WHERE {_LATER_SAME} "
            "                     AND r.status = 'released') "
            " ORDER BY j.requested_at DESC LIMIT %s",
            (ACTIVE, limit))
        out = []
        for row in cur.fetchall():
            job = _row(row[:-1])
            job["rerun_status"] = row[-1]
            out.append(job)
        return out


def dismiss(job_id: str, *, by: Optional[str], reason: Optional[str],
            allow_released: bool = False) -> bool:
    """Take a blocked or failed job off the Action Centre. True for exactly one
    caller; False for any other status, or if someone already dismissed it.
    ``allow_released`` admits a released job too -- the caller passes it only for a
    deck whose sign-off was refused."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE proc.bp_report_job "
            "   SET dismissed_at = now(), dismissed_by = %s, dismiss_reason = %s "
            " WHERE job_id = %s AND dismissed_at IS NULL "
            "   AND (status IN ('blocked', 'failed') OR (%s AND status = 'released'))",
            (by, reason, job_id, bool(allow_released)))
        return cur.rowcount == 1


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
                    media_type: str, filename: str, page: Optional[bytes] = None,
                    page_media_type: Optional[str] = None,
                    fact_pack: Optional[Dict[str, Any]] = None,
                    ast: Optional[Dict[str, Any]] = None,
                    title: Optional[str] = None) -> None:
    """Release a job. With its Fact Pack and AST it also becomes version 1, editable: an edit
    re-renders from exactly this pack (never a re-query). Without them -- a caller from before
    the editor -- it is released as before and cannot be edited."""
    if fact_pack is None or ast is None:
        _finish(job_id, "released", run_id=run_id, stage_reached=stage_reached,
                deck=deck, media_type=media_type, filename=filename,
                page=page, page_media_type=page_media_type)
        return

    title = title or "Executive procurement summary"
    with get_conn() as conn:
        conn.autocommit = False
        try:
            cur = conn.cursor()
            cur.execute(
                "UPDATE proc.bp_report_job SET status = 'released', finished_at = now(), "
                "  run_id = %s, stage_reached = %s, deck = %s, media_type = %s, filename = %s, "
                "  page = %s, page_media_type = %s, fact_pack = %s::jsonb, title = %s, "
                "  current_version = 1 "
                "WHERE job_id = %s AND status IN %s RETURNING job_id",
                (run_id, stage_reached, deck, media_type, filename, page, page_media_type,
                 json.dumps(fact_pack), title, job_id, ACTIVE))
            if cur.fetchone():
                cur.execute(
                    "INSERT INTO proc.bp_report_version (job_id, version, title, ast, deck, page, "
                    "  deck_sha256, page_sha256, edited_by, summary) "
                    "VALUES (%s, 1, %s, %s::jsonb, %s, %s, %s, %s, NULL, %s)",
                    (job_id, title, json.dumps(ast), deck, page,
                     hashlib.sha256(deck).hexdigest(),
                     hashlib.sha256(page).hexdigest() if page is not None else None,
                     "released by the reporting agent"))
            conn.commit()
        except Exception:
            conn.rollback()
            raise


def draft(job_id: str) -> Optional[Dict[str, Any]]:
    """The current version's title and AST, with the job's Fact Pack -- what the editor edits.
    None for a job that has no versions (not released, or released before the editor)."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT j.current_version, v.title, v.ast, j.fact_pack "
            "  FROM proc.bp_report_job j "
            "  JOIN proc.bp_report_version v ON v.job_id = j.job_id AND v.version = j.current_version "
            " WHERE j.job_id = %s", (job_id,))
        row = cur.fetchone()
    if row is None:
        return None
    return {"version": row[0], "title": row[1], "ast": row[2], "fact_pack": row[3]}


class StaleVersion(Exception):
    """The edit was made on a version that is no longer the current one (someone else saved
    first), or the job cannot take an edit at all. ``current`` is the version now current."""

    def __init__(self, current: Optional[int]) -> None:
        super().__init__(f"the report is now at version {current}")
        self.current = current


def save_version(job_id: str, *, base_version: int, title: str, ast: Dict[str, Any],
                 deck: bytes, page: bytes, by: str, summary: Optional[str] = None,
                 before_commit: Optional[Callable[[int], None]] = None) -> int:
    """Make an edit the next version and the job's current files, in one transaction.

    The job row is locked first, so two people saving the same version at once cannot both
    win: the second finds ``current_version`` moved and gets StaleVersion. ``before_commit``
    runs inside the transaction with the new version number -- the audit write; if it
    raises, nothing is saved.
    """
    with get_conn() as conn:
        conn.autocommit = False
        try:
            cur = conn.cursor()
            cur.execute("SELECT current_version FROM proc.bp_report_job "
                        "WHERE job_id = %s AND status = 'released' AND fact_pack IS NOT NULL "
                        "FOR UPDATE", (job_id,))
            row = cur.fetchone()
            current = row[0] if row else None
            if current is None or current != base_version:
                raise StaleVersion(current)
            version = current + 1
            deck_sha = hashlib.sha256(deck).hexdigest()
            page_sha = hashlib.sha256(page).hexdigest()
            cur.execute(
                "INSERT INTO proc.bp_report_version (job_id, version, title, ast, deck, page, "
                "  deck_sha256, page_sha256, edited_by, summary) "
                "VALUES (%s, %s, %s, %s::jsonb, %s, %s, %s, %s, %s, %s)",
                (job_id, version, title, json.dumps(ast), deck, page, deck_sha, page_sha,
                 by, summary))
            cur.execute(
                "UPDATE proc.bp_report_job SET deck = %s, page = %s, title = %s, "
                "  current_version = %s, last_edited_by = %s WHERE job_id = %s",
                (deck, page, title, version, by, job_id))
            if before_commit is not None:
                before_commit(version)
            conn.commit()
            return version
        except Exception:
            conn.rollback()
            raise


def page(job_id: str) -> Optional[Tuple[bytes, str]]:
    """The released job's printable page as ``(bytes, media_type)``, or None."""
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT page, page_media_type FROM proc.bp_report_job "
                    "WHERE job_id = %s AND status = 'released'", (job_id,))
        row = cur.fetchone()
    if row is None or row[0] is None:
        return None
    return bytes(row[0]), row[1]


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
