"""The translation audit trail: what was created, what reached a person, and who changed it.

Records go to proc.bp_agent_actions (phase='translation'), the trail the rest of the product
reports from. Nothing here is ever deleted.

  translation.served           dynamic text translated for a person: who asked, the exact
                               source and the exact text shown (translated or English).
                               MUST be written: a translation that cannot be audited is not
                               shown (the caller serves English instead).
  translation.generated        the model created new translations (best-effort: the rows in
                               proc.bp_translation are themselves immutable and dated).
  translation.reviewed_import  human translations loaded; any reviewed text it replaced is
                               kept here as old -> new.
"""
from __future__ import annotations

from typing import Any, Iterable, Optional

from src.services.agent_actions import AuditWriteError, record_action, record_action_or_fail
from src.services.i18n.store import source_hash

PHASE = "translation"
AGENT = "translator"
SERVED = "translation.served"
GENERATED = "translation.generated"
REVIEWED_IMPORT = "translation.reviewed_import"
PUBLIC_KEYS = "translation.public_keys_published"

__all__ = ["AuditWriteError", "record_served", "record_generated", "record_reviewed_import",
           "record_public_keys", "read_events"]

# Indirection so tests capture the rows without a database.
_record_best_effort = record_action
_record_or_fail = record_action_or_fail


def record_served(*, lang: str, requested_by: Optional[str], model: str, prompt_version: str,
                  items: list[dict[str, str]]) -> None:
    """Raises AuditWriteError when the record cannot be written."""
    who = requested_by or "anonymous (auth off)"
    _record_or_fail(
        phase=PHASE, action_type=SERVED, agent=AGENT, status="ok",
        summary=f"{len(items)} text(s) into {lang} for {who}",
        details={
            "lang": lang, "requested_by": who, "model": model, "prompt_version": prompt_version,
            "items": [{**i, "source_hash": source_hash(i["source"])} for i in items],
        },
    )


def record_generated(*, lang: str, model: str, prompt_version: str, hashes: Iterable[str]) -> None:
    hashes = list(hashes)
    _record_best_effort(
        phase=PHASE, action_type=GENERATED, agent=AGENT, status="ok",
        summary=f"{len(hashes)} new translation(s) into {lang}",
        details={"lang": lang, "model": model, "prompt_version": prompt_version, "hashes": hashes},
    )


def record_reviewed_import(*, lang: str, imported_by: str, added: int,
                           changed: list[tuple[str, str, str]]) -> None:
    """Raises AuditWriteError: an import that cannot be audited must not run."""
    _record_or_fail(
        phase=PHASE, action_type=REVIEWED_IMPORT, agent=AGENT, status="ok",
        summary=f"{added} reviewed translation(s) into {lang} by {imported_by}; {len(changed)} replaced",
        details={"lang": lang, "imported_by": imported_by, "added": added,
                 "changed": [{"source_hash": h, "old": old, "new": new} for h, old, new in changed]},
    )


def record_public_keys(*, published_by: str, total: int, added: list[str], removed: list[str]) -> None:
    """Raises AuditWriteError: what a signed-out visitor can read must not change untraced."""
    _record_or_fail(
        phase=PHASE, action_type=PUBLIC_KEYS, agent=AGENT, status="ok",
        summary=f"signed-out key list set to {total} key(s) by {published_by} "
                f"(+{len(added)} / -{len(removed)})",
        details={"published_by": published_by, "total": total, "added": added, "removed": removed},
    )


def read_events(*, lang: Optional[str] = None, requested_by: Optional[str] = None,
                action_type: Optional[str] = None, since: Optional[str] = None,
                until: Optional[str] = None, limit: int = 200) -> list[dict[str, Any]]:
    """The trail, newest first, for the audit report."""
    from src.services.db import get_conn

    where, params = ["phase = %s"], [PHASE]
    if lang:
        where.append("details->>'lang' = %s"); params.append(lang)
    if requested_by:
        where.append("details->>'requested_by' = %s"); params.append(requested_by)
    if action_type:
        where.append("action_type = %s"); params.append(action_type)
    if since:
        where.append("created_at >= %s"); params.append(since)
    if until:
        where.append("created_at < %s"); params.append(until)
    params.append(max(1, min(limit, 1000)))
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT action_id, created_at, action_type, status, summary, details "
            "FROM proc.bp_agent_actions WHERE " + " AND ".join(where)
            + " ORDER BY created_at DESC, action_id DESC LIMIT %s",
            params,
        )
        return [{"action_id": r[0], "at": r[1].isoformat(), "action_type": r[2], "status": r[3],
                 "summary": r[4], "details": r[5]} for r in cur.fetchall()]
