"""Every database read and write for mail intake. Nothing else touches the DB.

Two responsibilities beyond plain SQL.

**Transitions are validated before they are written.** `assert_transition`
runs first and raises, so an illegal move never reaches the database at all.
The CHECK constraints in the migration are the second line of defence, not the
first.

**Redaction happens here, at the write boundary.** Spec §9 forbids logging
attachment content, credentials or full body text. Enforcing that in every
caller means one forgetful caller leaks; enforcing it in `safe_log_fields`
means the unsafe fields are not available to leak. Callers log what this
returns, never the row.

Spec §5, §9.
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.services.mail_intake.states import (
    ArtifactState,
    BundleState,
    QuarantineReason,
    assert_transition,
)

log = logging.getLogger(__name__)

# Fields that must never reach a log line, in any environment.
_NEVER_LOG = frozenset(
    {"body_text", "body_html", "subject", "credential_ref", "content", "sha256"}
)
_CORRELATION = ("bundle_id", "artifact_id", "rfc_message_id", "provider_message_id",
                "binding_id", "state", "quarantine_reason", "raw_mime_key")


def _is_production() -> bool:
    return os.environ.get("ENVIRONMENT", "").strip().lower() in ("prod", "production")


def safe_log_fields(row: Dict[str, Any]) -> Dict[str, Any]:
    """Reduce a row to the fields that are safe to log."""
    out: Dict[str, Any] = {
        key: row[key] for key in _CORRELATION if key in row and key not in _NEVER_LOG
    }
    address = row.get("sender_address")
    if address:
        if _is_production():
            out["sender_address"] = address
        elif "@" in address:
            out["sender_address"] = f"<redacted>@{address.rsplit('@', 1)[1]}"
        else:
            out["sender_address"] = "<redacted>"
    return out


def _applied(cur, what: str, row_id: str, current, target) -> bool:
    """Did the guarded UPDATE actually move the row?

    Every transition here carries `AND state = %s`, which correctly stops a
    stale writer clobbering a row that has since moved on. But a guarded UPDATE
    matching zero rows is indistinguishable from a successful one to a caller
    that ignores rowcount — so a lost race looked exactly like a completed
    transition. Surfacing it costs one branch and turns a silent no-op into a
    visible event.
    """
    applied = getattr(cur, "rowcount", 1) != 0
    if not applied:
        log.warning(
            "mail_intake.transition_skipped op=%s id=%s expected_state=%s target=%s",
            what, row_id, getattr(current, "value", current),
            getattr(target, "value", target),
        )
    return applied


def insert_bundle(cur, **fields: Any) -> Optional[str]:
    """Insert a bundle, or return None if this message was already ingested.

    The ON CONFLICT clause repeats the index's WHERE predicate, and must. The
    supporting index is PARTIAL — unique on (binding_id, provider_message_id)
    only where provider_message_id IS NOT NULL — and Postgres will not infer a
    partial index from column names alone. Without the predicate this raises
    InvalidColumnReference on every insert, so the idempotency mechanism fails
    the first time it runs rather than the first time it is needed. Verified by
    executing both forms against the live schema.

    The None return is the idempotency contract, and it matters: the unique
    index on (binding_id, provider_message_id) makes a re-delivered provider
    event or an IMAP re-poll a no-op. Returning the freshly generated UUID on
    conflict would be worse than useless — no row would exist under it, every
    later UPDATE would silently match zero rows, and the artifact inserts would
    fail their foreign key. Callers must treat None as "already done" and stop.
    """
    # The caller may supply the id: the raw-MIME key is derived from it and the
    # column is NOT NULL, so the pipeline needs the id before the insert.
    bundle_id = fields.get("bundle_id") or str(uuid.uuid4())
    cur.execute(
        """
        INSERT INTO proc.bp_ingest_bundle (
            bundle_id, binding_id, owner_ref, provider_message_id, rfc_message_id,
            thread_id, sender_address, sender_display_name, sender_domain,
            recipients, subject, received_at, body_text, body_html,
            raw_mime_key, state
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (binding_id, provider_message_id)
            WHERE provider_message_id IS NOT NULL
            DO NOTHING
        RETURNING bundle_id
        """,
        (
            bundle_id,
            fields["binding_id"],
            fields["owner_ref"],
            fields.get("provider_message_id"),
            fields.get("rfc_message_id"),
            fields.get("thread_id"),
            fields.get("sender_address"),
            fields.get("sender_display_name"),
            fields.get("sender_domain"),
            json.dumps(fields.get("recipients") or []),
            fields.get("subject"),
            fields.get("received_at"),
            fields.get("body_text"),
            fields.get("body_html"),
            fields["raw_mime_key"],
            BundleState.RECEIVED.value,
        ),
    )
    row = cur.fetchone()
    if not row:
        return None
    return str(row[0] if isinstance(row, (tuple, list)) else row["bundle_id"])


def transition_bundle(cur, bundle_id: str, current: BundleState,
                      target: BundleState,
                      reason: Optional[QuarantineReason] = None) -> bool:
    assert_transition(current, target)
    cur.execute(
        """
        UPDATE proc.bp_ingest_bundle
           SET state = %s, quarantine_reason = %s, updated_at = now()
         WHERE bundle_id = %s AND state = %s
        """,
        (target.value, reason.value if reason else None, bundle_id, current.value),
    )
    return _applied(cur, "transition_bundle", bundle_id, current, target)


def insert_artifact(cur, **fields: Any) -> str:
    artifact_id = str(uuid.uuid4())
    cur.execute(
        """
        INSERT INTO proc.bp_ingest_artifact (
            artifact_id, bundle_id, binding_id, owner_ref, declared_filename,
            declared_mime, detected_mime, size_bytes, sha256, object_key,
            nesting_depth, parent_artifact_id, disposition, content_id, state
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            artifact_id,
            fields["bundle_id"],
            fields["binding_id"],
            fields["owner_ref"],
            fields.get("declared_filename"),
            fields.get("declared_mime"),
            fields.get("detected_mime"),
            fields.get("size_bytes"),
            fields.get("sha256"),
            fields.get("object_key"),
            fields.get("nesting_depth", 0),
            fields.get("parent_artifact_id"),
            fields.get("disposition"),
            fields.get("content_id"),
            ArtifactState.DETACHED.value,
        ),
    )
    return artifact_id


def transition_artifact(cur, artifact_id: str, current: ArtifactState,
                        target: ArtifactState,
                        reason: Optional[QuarantineReason] = None,
                        detail: Optional[Dict[str, Any]] = None) -> bool:
    assert_transition(current, target)
    cur.execute(
        """
        UPDATE proc.bp_ingest_artifact
           SET state = %s, quarantine_reason = %s, quarantine_detail = %s,
               updated_at = now()
         WHERE artifact_id = %s AND state = %s
        """,
        (
            target.value,
            reason.value if reason else None,
            json.dumps(detail) if detail else None,
            artifact_id,
            current.value,
        ),
    )
    return _applied(cur, "transition_artifact", artifact_id, current, target)


def mark_duplicate(cur, artifact_id: str, duplicate_of: str,
                   current: ArtifactState) -> bool:
    """Mark an artifact as a duplicate of another.

    `current` is a required parameter, not a literal and not defaulted. An
    earlier version called assert_transition(STORED, DUPLICATE) with two
    hardcoded constants — which can never raise whatever state the row is
    actually in, so it validated nothing while looking exactly like the
    sibling functions that do. It also omitted their `AND state = %s` guard,
    so a concurrent transition could overwrite silently. A later fix added
    `current` as a parameter but defaulted it to STORED — the exact value the
    original tautology hardcoded — so any caller that omitted it silently
    reproduced the assumption the fix was meant to remove. There is no safe
    default for "what state do you believe this row is in"; the caller must
    say.
    """
    assert_transition(current, ArtifactState.DUPLICATE)
    cur.execute(
        """
        UPDATE proc.bp_ingest_artifact
           SET state = %s, duplicate_of_artifact_id = %s, updated_at = now()
         WHERE artifact_id = %s AND state = %s
        """,
        (ArtifactState.DUPLICATE.value, duplicate_of, artifact_id, current.value),
    )
    return _applied(cur, "mark_duplicate", artifact_id, current,
                    ArtifactState.DUPLICATE)


def insert_sender_auth(cur, bundle_id: str, auth: Any, bind_result: Any) -> None:
    cur.execute(
        """
        INSERT INTO proc.bp_ingest_sender_auth (
            bundle_id, spf, dkim, dmarc, envelope_from, header_from, alignment,
            asserted_by, supplier_match_state, matched_supplier_id,
            candidate_supplier_ids
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (bundle_id) DO NOTHING
        """,
        (
            bundle_id,
            auth.spf,
            auth.dkim,
            auth.dmarc,
            auth.envelope_from,
            auth.header_from,
            auth.alignment,
            auth.asserted_by,
            bind_result.state,
            bind_result.supplier_id,
            json.dumps(list(bind_result.candidates)),
        ),
    )


def record_arrival(cur, artifact_id: str, bundle_id: str) -> None:
    cur.execute(
        """
        INSERT INTO proc.bp_ingest_arrival (artifact_id, bundle_id)
        VALUES (%s, %s)
        """,
        (artifact_id, bundle_id),
    )


def find_existing_artifact(cur, owner_ref: str, digest: str,
                           logical: Optional[str]) -> Optional[str]:
    """Content-hash lookup within the dedup scope (spec §5.3: the binding owner)."""
    cur.execute(
        """
        SELECT artifact_id FROM proc.bp_ingest_artifact
         WHERE owner_ref = %s AND sha256 = %s AND duplicate_of_artifact_id IS NULL
         LIMIT 1
        """,
        (owner_ref, digest),
    )
    row = cur.fetchone()
    if not row:
        return None
    return str(row[0] if isinstance(row, (tuple, list)) else row["artifact_id"])


def claim_ready_artifacts(cur, limit: int = 50) -> List[Dict[str, Any]]:
    """The emit queue. A durable row, not an in-memory event."""
    cur.execute(
        """
        SELECT artifact_id, bundle_id, object_key, detected_mime, owner_ref
          FROM proc.bp_ingest_artifact
         WHERE state = %s
         ORDER BY created_at
         LIMIT %s
        """,
        (ArtifactState.READY_FOR_EXTRACTION.value, limit),
    )
    return list(cur.fetchall() or [])


def append_derivation(cur, artifact_id: str, pipeline_version: str,
                      run_ref: Optional[str] = None) -> None:
    """Append-only: reprocessing adds a row and never updates one."""
    cur.execute(
        """
        INSERT INTO proc.bp_artifact_derivation (
            artifact_id, pipeline_version, extraction_run_ref
        ) VALUES (%s, %s, %s)
        """,
        (artifact_id, pipeline_version, run_ref),
    )


def supplier_candidates(cur, domain: str) -> List[Tuple[str, str]]:
    """Supplier records whose website domain matches. See spec §7.3 on ambiguity."""
    cur.execute(
        """
        SELECT supplier_id,
               lower(regexp_replace(website_url, '^https?://(www\\.)?', '')) AS domain
          FROM proc.bp_supplier
         WHERE website_url IS NOT NULL
           AND lower(regexp_replace(website_url, '^https?://(www\\.)?', '')) = %s
        """,
        (domain,),
    )
    return [(str(r[0]), str(r[1])) for r in (cur.fetchall() or [])]
