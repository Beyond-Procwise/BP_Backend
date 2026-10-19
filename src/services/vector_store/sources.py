"""Writing the SOURCE rows the vectors are derived from (proc.bp_vector_source) - what callers use instead of embedding and upserting inline.

A write here is a Postgres write. A trigger on the table queues the embedding job in the same transaction, and the worker embeds it. A
caller that needs to know the vectors are searchable (an upload the user is about to ask about) calls ``wait_until_embedded`` - it waits
for the worker, it never embeds in the request.
"""
from __future__ import annotations

import json
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union

from .pgvector_store import DISALLOWED_METADATA_KEYS


def _need(value: Any, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{what} is required")
    return value


def _conn(connect: Optional[Callable[[], Any]]):
    if connect is not None:
        return connect()
    from src.services.db import get_conn
    return get_conn()


def put_source(tenant_id: str, collection: str, source_id: str, content: str, metadata: Optional[Mapping[str, Any]] = None, *,
               connect: Optional[Callable[[], Any]] = None) -> None:
    """Insert or update a source row. Idempotent; the trigger queues one embedding job (coalesced) if the content or metadata changed."""
    _need(tenant_id, "tenant_id"), _need(collection, "collection"), _need(source_id, "source_id"), _need(content, "content")
    meta: Dict[str, Any] = {k: v for k, v in dict(metadata or {}).items() if k not in DISALLOWED_METADATA_KEYS}
    with _conn(connect) as conn, conn.cursor() as cur:
        cur.execute("INSERT INTO proc.bp_vector_source (tenant_id, collection, source_id, content, metadata) VALUES (%s, %s, %s, %s, %s::jsonb) "
                    "ON CONFLICT (tenant_id, collection, source_id) DO UPDATE SET content = EXCLUDED.content, "
                    "metadata = EXCLUDED.metadata, updated_at = now()",
                    (tenant_id, collection, str(source_id), content, json.dumps(meta, sort_keys=True, default=str)))


def delete_source(tenant_id: str, collection: str, source_id: str, *, connect: Optional[Callable[[], Any]] = None) -> int:
    """Delete a source row; the trigger queues the delete job that removes its vectors."""
    _need(tenant_id, "tenant_id"), _need(collection, "collection"), _need(source_id, "source_id")
    with _conn(connect) as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM proc.bp_vector_source WHERE tenant_id = %s AND collection = %s AND source_id = %s",
                    (tenant_id, collection, str(source_id)))
        return cur.rowcount


def delete_sources_matching(tenant_id: str, collection: str, metadata: Optional[Mapping[str, Any]], *,
                            connect: Optional[Callable[[], Any]] = None) -> int:
    """Delete every source whose metadata contains ``metadata`` (e.g. all chunks of one policy file). An empty match is refused, so this can
    never delete a whole collection."""
    _need(tenant_id, "tenant_id"), _need(collection, "collection")
    if not metadata:
        raise ValueError("a non-empty metadata match is required")
    with _conn(connect) as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM proc.bp_vector_source WHERE tenant_id = %s AND collection = %s AND metadata @> %s::jsonb",
                    (tenant_id, collection, json.dumps(dict(metadata), sort_keys=True, default=str)))
        return cur.rowcount


def find_source_metadata(tenant_id: str, collection: str, metadata: Optional[Mapping[str, Any]], *,
                         connect: Optional[Callable[[], Any]] = None) -> Optional[Dict[str, Any]]:
    """The metadata of one source matching ``metadata`` (what a Qdrant scroll with a payload filter and limit 1 returned), or None."""
    _need(tenant_id, "tenant_id"), _need(collection, "collection")
    if not metadata:
        raise ValueError("a non-empty metadata match is required")
    with _conn(connect) as conn, conn.cursor() as cur:
        cur.execute("SELECT metadata FROM proc.bp_vector_source WHERE tenant_id = %s AND collection = %s AND metadata @> %s::jsonb "
                    "ORDER BY source_id LIMIT 1", (tenant_id, collection, json.dumps(dict(metadata), sort_keys=True, default=str)))
        row = cur.fetchone()
    if not row:
        return None
    meta = row[0]
    return json.loads(meta) if isinstance(meta, str) else dict(meta or {})


def wait_until_embedded(tenant_id: str, collection: str, source_id: Union[str, Sequence[str]], *, timeout_s: float = 30.0, poll_s: float = 0.5,
                        connect: Optional[Callable[[], Any]] = None) -> str:
    """'done' when no job for the source is waiting or running, 'pending' if still queued at the timeout, 'failed' if the only outcome is a failure."""
    deadline = time.monotonic() + max(0.0, timeout_s)
    while True:
        ids: List[str] = [str(i) for i in ([source_id] if isinstance(source_id, str) else source_id)]
        with _conn(connect) as conn, conn.cursor() as cur:
            cur.execute("SELECT status FROM proc.bp_embedding_job WHERE tenant_id = %s AND collection = %s AND source_id = ANY(%s) "
                        "AND status IN ('pending', 'processing', 'failed') ORDER BY job_id DESC LIMIT 50", (tenant_id, collection, ids))
            statuses = [r[0] for r in cur.fetchall() if r[0] in ("pending", "processing", "failed")]
        if not statuses:
            return "done"
        if "pending" not in statuses and "processing" not in statuses:
            return "failed"
        if time.monotonic() >= deadline:
            return "pending"
        time.sleep(poll_s)
