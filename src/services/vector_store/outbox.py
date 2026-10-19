"""The embedding outbox (proc.bp_embedding_job): enqueue in the writer's transaction, claim without collisions, retry with backoff.

A job says "this source changed" (upsert) or "this source is gone" (delete). It carries no text: the worker re-reads the source row, so a
job is always processed against the source as it is NOW and running one twice is harmless.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

ACTIONS = ("upsert", "delete")
BACKOFF_BASE_SECONDS = 30


@dataclass
class Job:
    job_id: int
    tenant_id: str
    collection: str
    source_id: str
    action: str
    profile: Optional[str]
    attempts: int
    max_attempts: int


def enqueue(cur: Any, tenant_id: str, collection: str, source_id: str, action: str = "upsert", profile: Optional[str] = None) -> bool:
    """Queue a job on the CALLER'S cursor, so it commits or rolls back with the write that caused it. True if newly queued, False if an
    identical job was already pending (coalesced)."""
    if not isinstance(tenant_id, str) or not tenant_id.strip():
        raise ValueError("an embedding job needs a tenant_id")
    if action not in ACTIONS:
        raise ValueError(f"action must be one of {ACTIONS}")
    cur.execute("INSERT INTO proc.bp_embedding_job (tenant_id, collection, source_id, action, profile) VALUES (%s, %s, %s, %s, %s) "
                "ON CONFLICT (tenant_id, collection, source_id, action, (COALESCE(profile, ''))) WHERE status = 'pending' DO NOTHING",
                (tenant_id, collection, str(source_id), action, profile))
    return cur.rowcount == 1


def claim(cur: Any, limit: int, visibility_seconds: int = 300) -> List[Job]:
    """Take up to ``limit`` due jobs. SKIP LOCKED means two workers never take the same one; a job left 'processing' by a worker that died
    is taken again once ``visibility_seconds`` have passed."""
    cur.execute(
        "UPDATE proc.bp_embedding_job SET status = 'processing', locked_at = now(), attempts = attempts + 1, updated_at = now() "
        "WHERE job_id IN (SELECT job_id FROM proc.bp_embedding_job "
        "WHERE (status = 'pending' AND next_attempt_at <= now()) "
        "OR (status = 'processing' AND locked_at < now() - make_interval(secs => %s)) "
        "ORDER BY job_id FOR UPDATE SKIP LOCKED LIMIT %s) "
        "RETURNING job_id, tenant_id, collection, source_id, action, profile, attempts, max_attempts", (visibility_seconds, limit))
    return [Job(*row) for row in cur.fetchall()]


def complete(cur: Any, job: Job) -> None:
    cur.execute("UPDATE proc.bp_embedding_job SET status = 'done', done_at = now(), locked_at = NULL, last_error = NULL, updated_at = now() "
                "WHERE job_id = %s", (job.job_id,))


def fail(cur: Any, job: Job, error: str) -> str:
    """Retry later with exponential backoff, or give up at ``max_attempts``. Returns the status the job now has."""
    error = (error or "")[:2000]
    if job.attempts >= job.max_attempts:
        cur.execute("UPDATE proc.bp_embedding_job SET status = %s, last_error = %s, locked_at = NULL, updated_at = now() WHERE job_id = %s",
                    ("failed", error, job.job_id))
        return "failed"
    cur.execute("UPDATE proc.bp_embedding_job SET status = %s, next_attempt_at = now() + make_interval(secs => %s), last_error = %s, "
                "locked_at = NULL, updated_at = now() WHERE job_id = %s AND NOT EXISTS ("
                "SELECT 1 FROM proc.bp_embedding_job o WHERE o.status = 'pending' AND o.job_id <> %s AND o.tenant_id = %s "
                "AND o.collection = %s AND o.source_id = %s AND o.action = %s AND COALESCE(o.profile, '') = %s)",
                ("pending", BACKOFF_BASE_SECONDS * 2 ** job.attempts, error, job.job_id, job.job_id, job.tenant_id, job.collection,
                 job.source_id, job.action, job.profile or ""))
    if cur.rowcount == 0:                           # a newer pending job already covers this source: this one is superseded
        cur.execute("UPDATE proc.bp_embedding_job SET status = 'done', done_at = now(), last_error = %s, locked_at = NULL, updated_at = now() "
                    "WHERE job_id = %s", ("superseded by a newer pending job: " + error[:200], job.job_id))
        return "done"
    return "pending"
