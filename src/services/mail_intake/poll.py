"""Stage 0: fetch new mail for every active intake binding and ingest it.

Plan 1 built the source adapter and the pipeline and connected neither: no code
called RelayMailSource.iter_messages or IngestPipeline.ingest, so mail sat in the
relay prefix and nothing read it. This is that connection.

Deterministic, like the rest of the Ingestion Plane. One transaction per
message, so one bad message cannot roll back the others, and the binding's
cursor only moves past a message once it is committed.

The cursor (bp_mailbox_binding.last_sync_at) is the newest arrival time seen,
not the clock: a message that lands while a poll is running is picked up by the
next one. Arrival times are compared with `<`, so a message stamped exactly at
the cursor is fetched again, and insert_bundle's idempotency makes that harmless.
A binding whose cursor is NULL starts from now, so switching on a binding does
not ingest everything the relay prefix has ever held.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Mapping, Optional

log = logging.getLogger(__name__)


def active_bindings(cur: Any, provider: str = "relay") -> List[Dict[str, Any]]:
    cur.execute(
        """
        SELECT binding_id, user_ref, mailbox_address, folder_scope, last_sync_at
          FROM proc.bp_mailbox_binding
         WHERE role = 'intake' AND provider = %s AND is_active
         ORDER BY binding_id
        """,
        (provider,),
    )
    out = []
    for row in cur.fetchall() or ():
        if not isinstance(row, Mapping):
            row = dict(zip(
                ("binding_id", "user_ref", "mailbox_address", "folder_scope", "last_sync_at"),
                row,
            ))
        binding = dict(row)
        # The binding OWNER is the dedup scope (migration header, spec §5.3).
        binding["owner_ref"] = binding["user_ref"]
        out.append(binding)
    return out


def _require_transactional(conn: Any) -> None:
    # get_conn() is AUTOCOMMIT: rollback() would do nothing, and a crash after
    # insert_bundle would leave a bundle row that makes every re-poll skip the
    # message as already ingested — lost, with nothing to investigate.
    if getattr(conn, "autocommit", False):
        raise RuntimeError("mail_intake.poll needs a non-autocommit connection")


def _start_cursor(cur: Any, binding_id: Any) -> datetime:
    now = datetime.now(timezone.utc)
    cur.execute(
        "UPDATE proc.bp_mailbox_binding SET last_sync_at = %s "
        "WHERE binding_id = %s AND last_sync_at IS NULL",
        (now, binding_id),
    )
    return now


def poll_binding(conn: Any, binding: Mapping[str, Any], source: Any,
                 pipeline: Any) -> Dict[str, int]:
    """Ingest every message newer than the binding's cursor. Returns counts."""
    _require_transactional(conn)
    counts = {"messages": 0, "ready": 0, "quarantined": 0, "duplicates": 0, "failed": 0}
    since = binding.get("last_sync_at")
    if since is None:
        with conn.cursor() as cur:
            since = _start_cursor(cur, binding["binding_id"])
        conn.commit()
        return counts

    # Arrival order, not listing order: the relay lists by key, and the cursor
    # is "newest committed arrival", so committing a later message before an
    # earlier one would carry the cursor past the earlier one.
    messages = sorted(
        source.iter_messages(binding, since),
        key=lambda m: m.received_at or since,
    )
    for message in messages:
        try:
            with conn.cursor() as cur:
                outcome = pipeline.ingest(cur, binding, message)
                if message.received_at is not None:
                    cur.execute(
                        "UPDATE proc.bp_mailbox_binding SET last_sync_at = %s "
                        "WHERE binding_id = %s AND last_sync_at < %s",
                        (message.received_at, binding["binding_id"], message.received_at),
                    )
            conn.commit()
        except Exception:
            # A failure here is the pipeline or the database, not the message:
            # a message it cannot read is quarantined inside ingest(). Roll back
            # so the cursor does not move past it, and say which one.
            conn.rollback()
            counts["failed"] += 1
            log.exception(
                "mail_intake.poll failed binding=%s message=%s; stopping this "
                "binding's pass so the cursor stays before it and it is retried",
                binding["binding_id"], message.provider_message_id,
            )
            # Stop, do not continue: a later message committing would move the
            # cursor past this one and it would never be fetched again. A
            # message that keeps failing holds the binding and says so every
            # pass, which is loud; skipping it would be silent.
            break
        counts["messages"] += 1
        counts["ready"] += len(outcome.artifact_ids)
        counts["quarantined"] += len(outcome.quarantined)
        counts["duplicates"] += len(outcome.duplicates)
    return counts


def poll_all(conn: Any, source_for: Callable[[Mapping[str, Any]], Any],
             pipeline: Any) -> Dict[str, int]:
    """Poll every active relay intake binding. A failing binding does not stop the rest."""
    _require_transactional(conn)
    with conn.cursor() as cur:
        bindings = active_bindings(cur)
    conn.commit()
    total: Dict[str, int] = {"bindings": len(bindings)}
    for binding in bindings:
        try:
            counts = poll_binding(conn, binding, source_for(binding), pipeline)
        except Exception:
            conn.rollback()
            log.exception("mail_intake.poll binding=%s failed", binding["binding_id"])
            total["binding_errors"] = total.get("binding_errors", 0) + 1
            continue
        for key, value in counts.items():
            total[key] = total.get(key, 0) + value
    return total
