"""Extraction runs: the record behind "start now, watch it fill in, collect it later".

One row per run in ``proc.bp_policy_extraction_run`` and one per finding in
``proc.bp_policy_extraction_item`` (deploy/sql/2026-10-09_bp_agent_policy_extraction.sql).
Same shape as services.rga.job_store.

A run is spotted as stranded by its HEARTBEAT: the runner stamps ``heartbeat_at``
every 30 seconds, and a running run whose stamp is older than STALE_SECONDS is
healed to failed when next read or listed. A restart fails a run rather than
resuming it: items already appended stay, and a second model pass nobody asked
for is worse than saying what happened.

``get_conn()`` is AUTOCOMMIT, so single statements are atomic on their own. The
one multi-statement step, ``append_item``'s seq allocation, runs in an explicit
transaction that locks the run row, and puts the connection's autocommit back.
"""
from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List, Optional

OWNER = f"proc-{uuid.uuid4().hex[:12]}"

STALE_SECONDS = 120
HEALED_ERROR = ("The server restarted while this run was working. "
                "Start it again; the policies already listed were saved.")

_RUN_COLS = ("run_id", "kind", "status", "request", "owner", "heartbeat_at", "counts",
             "error", "started_by", "created_at", "finished_at")
_ITEM_COLS = ("seq", "kind", "document_id", "document_version", "reference", "payload",
              "policy_key", "decision", "saved_version", "created_at")
_RUN_SELECT = f"SELECT {', '.join(_RUN_COLS)} FROM proc.bp_policy_extraction_run"


def _iso(d: Dict[str, Any], keys) -> Dict[str, Any]:
    for k in keys:
        if d.get(k) is not None:
            d[k] = d[k].isoformat()
    return d


def _run(row) -> Optional[Dict[str, Any]]:
    if row is None:
        return None
    return _iso(dict(zip(_RUN_COLS, row)), ("heartbeat_at", "created_at", "finished_at"))


def heal(conn: Any, run_id: Optional[int] = None) -> int:
    """Fail running runs whose heartbeat has lapsed -- one run, or all. Returns how many."""
    scoped = "run_id = %s AND " if run_id is not None else ""
    params = ((HEALED_ERROR,) + ((run_id,) if run_id is not None else ()) + (STALE_SECONDS,))
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE proc.bp_policy_extraction_run "
            "   SET status = 'failed', error = %s, finished_at = now() "
            f" WHERE {scoped}status = 'running' "
            "   AND (heartbeat_at IS NULL OR heartbeat_at < now() - make_interval(secs => %s))",
            params)
        return cur.rowcount


def create(conn: Any, *, kind: str, request: Dict[str, Any], actor: str) -> Dict[str, Any]:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO proc.bp_policy_extraction_run (kind, status, request, owner, started_by) "
            "VALUES (%s, 'queued', %s::jsonb, %s, %s) RETURNING run_id",
            (kind, json.dumps(request), OWNER, actor))
        run_id = cur.fetchone()[0]
        cur.execute(f"{_RUN_SELECT} WHERE run_id = %s", (run_id,))
        return _run(cur.fetchone())


def claim(conn: Any, run_id: int, owner: str) -> bool:
    """queued -> running, for exactly one caller, and only for the owning process."""
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE proc.bp_policy_extraction_run "
            "   SET status = 'running', heartbeat_at = now() "
            " WHERE run_id = %s AND status = 'queued' AND owner = %s",
            (run_id, owner))
        return cur.rowcount == 1


def beat(conn: Any, run_id: int, owner: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE proc.bp_policy_extraction_run SET heartbeat_at = now() "
            " WHERE run_id = %s AND owner = %s AND status = 'running'", (run_id, owner))


def append_item(conn: Any, run_id: int, *, kind: str, payload: Dict[str, Any],
                document_id: Optional[int] = None, document_version: Optional[int] = None,
                reference: Optional[str] = None, policy_key: Optional[str] = None,
                decision: Optional[str] = None, saved_version: Optional[int] = None) -> int:
    """Append a finding and return its seq. Seqs per run are consecutive from 1.

    The run row is locked for the transaction, so two writers queue up and each
    reads the other's committed MAX(seq).
    """
    previous = conn.autocommit
    conn.autocommit = False
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT run_id FROM proc.bp_policy_extraction_run "
                        "WHERE run_id = %s FOR UPDATE", (run_id,))
            if cur.fetchone() is None:
                raise LookupError(f"no extraction run {run_id}")
            cur.execute("SELECT COALESCE(MAX(seq), 0) + 1 FROM proc.bp_policy_extraction_item "
                        "WHERE run_id = %s", (run_id,))
            seq = cur.fetchone()[0]
            cur.execute(
                "INSERT INTO proc.bp_policy_extraction_item "
                "  (run_id, seq, kind, document_id, document_version, reference, payload, "
                "   policy_key, decision, saved_version) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s)",
                (run_id, seq, kind, document_id, document_version, reference,
                 json.dumps(payload), policy_key, decision, saved_version))
        conn.commit()
        return seq
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.autocommit = previous


def finish(conn: Any, run_id: int, status: str, *, counts: Dict[str, Any],
           error: Optional[str] = None) -> None:
    """Close a queued or running run as done or failed."""
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE proc.bp_policy_extraction_run "
            "   SET status = %s, counts = %s::jsonb, error = %s, finished_at = now() "
            " WHERE run_id = %s AND status IN ('queued', 'running')",
            (status, json.dumps(counts or {}), error, run_id))


def get(conn: Any, run_id: int, *, after_seq: int = 0) -> Optional[Dict[str, Any]]:
    """The run with its items newer than ``after_seq``; heals it first."""
    heal(conn, run_id)
    with conn.cursor() as cur:
        cur.execute(f"{_RUN_SELECT} WHERE run_id = %s", (run_id,))
        run = _run(cur.fetchone())
        if run is None:
            return None
        cur.execute(
            f"SELECT {', '.join(_ITEM_COLS)} FROM proc.bp_policy_extraction_item "
            "WHERE run_id = %s AND seq > %s ORDER BY seq", (run_id, after_seq))
        items: List[Dict[str, Any]] = [_iso(dict(zip(_ITEM_COLS, r)), ("created_at",))
                                       for r in cur.fetchall()]
    run["items"] = items
    return run


def list_recent(conn: Any, limit: int = 20) -> List[Dict[str, Any]]:
    heal(conn)
    with conn.cursor() as cur:
        cur.execute(f"{_RUN_SELECT} ORDER BY created_at DESC, run_id DESC LIMIT %s", (limit,))
        return [_run(r) for r in cur.fetchall()]
