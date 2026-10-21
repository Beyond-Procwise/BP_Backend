"""The activity store: observed behaviour, as plain append-only metadata. Not the graph.

No message content, ever: callers pass ids and verbs. Retention is configurable (default 12 months).
Per-user erase deletes the user's rows and every derived edge touching them; asserted decision edges stay
for audit and can be pseudonymised.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from services.people.admin import TENANT

logger = logging.getLogger(__name__)
VERBS = ("message.sent", "mention", "comment", "view", "edit", "suggestion.shown", "suggestion.accepted",
         "suggestion.dismissed", "notification.opened", "notification.ignored")
DEFAULT_RETENTION_DAYS = 365
_FORBIDDEN_META = {"body", "text", "content", "message", "subject_line"}


def record(cur, verb: str, actor: str, *, target_user_id: Optional[str] = None, entity_type: Optional[str] = None,
           entity_id: Optional[str] = None, conversation_id: Optional[int] = None, duration_ms: Optional[int] = None,
           occurred_at: Optional[datetime] = None, **metadata: Any) -> bool:
    """Append one row on the caller's cursor (so it commits or rolls back with the caller). Never raises:
    a failure to log behaviour must not break the action being logged."""
    if verb not in VERBS:
        raise ValueError(f"unknown activity verb {verb!r}")
    clash = _FORBIDDEN_META & set(metadata)
    if clash:
        raise ValueError(f"activity stores metadata only, not content ({', '.join(sorted(clash))})")
    try:
        cur.execute("SAVEPOINT activity_row")
    except Exception:
        pass
    try:
        cur.execute("INSERT INTO proc.bp_activity (tenant_id, actor_id, verb, target_user_id, entity_type, entity_id, "
                    "conversation_id, occurred_at, duration_ms, metadata) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)",
                    (TENANT, actor, verb, target_user_id, entity_type, entity_id, conversation_id,
                     occurred_at or datetime.now(timezone.utc), duration_ms, json.dumps(metadata, default=str)))
        try:
            cur.execute("RELEASE SAVEPOINT activity_row")
        except Exception:
            pass
        return True
    except Exception:
        logger.warning("activity row %s not recorded", verb, exc_info=True)
        try:
            cur.execute("ROLLBACK TO SAVEPOINT activity_row")
        except Exception:
            pass
        return False


def prune(conn, retention_days: int = DEFAULT_RETENTION_DAYS, now: Optional[datetime] = None) -> int:
    cutoff = (now or datetime.now(timezone.utc)) - timedelta(days=retention_days)
    with conn.cursor() as cur:
        cur.execute("DELETE FROM proc.bp_activity WHERE tenant_id=%s AND occurred_at < %s", (TENANT, cutoff))
        return cur.rowcount


def erase_user(conn, store, tenant: str, subject: str, *, pseudonymise: bool = False) -> Dict[str, Any]:
    """GDPR erase (``tenant`` addresses the graph; the activity table is scoped by the app tenant): the person's activity rows (as actor or target) and every derived/inferred edge touching them.
    Asserted edges (decisions, org, RBAC) are kept for audit; ``pseudonymise`` replaces the person's identity
    on them with an opaque id instead of leaving the name."""
    from services.people.graph.model import NodeRef
    import hashlib
    u = NodeRef("User", subject)
    with conn.cursor() as cur:
        cur.execute("DELETE FROM proc.bp_activity WHERE tenant_id=%s AND (actor_id=%s OR target_user_id=%s)", (TENANT, subject, subject))
        rows = cur.rowcount
    derived = 0
    for src in ("derived", "inferred"):
        derived += store.delete_edges(tenant, source=src, src=u)
        derived += store.delete_edges(tenant, source=src, dst=u)
    alias = None
    if pseudonymise:
        alias = "erased:" + hashlib.sha256(f"{tenant}|{subject}".encode()).hexdigest()[:16]
        store.rename_node(tenant, u, alias, drop_props=("display_name",))
    return {"activity_rows": rows, "derived_edges": derived, "pseudonymised_as": alias}
