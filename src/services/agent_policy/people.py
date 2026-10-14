"""Who a sign-in id is (stage 5b): the email a person's sign-in carried, kept so a screen can show
"Approved by sam@example.com" instead of a Cognito subject.

Nothing else records the email of whoever DECIDED an approval or SAVED a version (the decision row,
the firing row and the audit record hold the sign-in id only). So every agent-policy WRITE a
signed-in person makes through the gateway remembers them here (`remember`, called from the
router's `_require` for every action that is not a read), and the reads resolve the ids they
already show with ONE query per answer (`names`).

Rules
  * Remembering never fails a person's action: `remember` never raises; a failure is logged (one
    line, no address) and ignored.
  * A sign-in with no email writes nothing, so it never erases a stored one. A `system:*` actor is
    not a person and is never stored or looked up.
  * A read never writes: nothing here is called for an `agent_policy.read`.
  * Privacy is the reader's: a name is only ever put beside an id the answer already carried.

Table proc.bp_agent_policy_person (deploy/sql/2026-10-17_bp_agent_policy_person.sql).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, Optional, Tuple

logger = logging.getLogger(__name__)

#: The longest email kept (the longest an address can be: 64 + 1 + 255).
MAX_EMAIL = 320

_REMEMBER_SQL = """
    INSERT INTO proc.bp_agent_policy_person (subject, email) VALUES (%s, %s)
    ON CONFLICT (subject) DO UPDATE SET email = EXCLUDED.email, last_seen_at = now()
"""
_NAMES_SQL = "SELECT subject, email FROM proc.bp_agent_policy_person WHERE subject = ANY(%s)"
_SAVEPOINT = "agent_policy_person"


def _is_person(subject: Any) -> bool:
    return isinstance(subject, str) and bool(subject.strip()) and not subject.strip().lower().startswith("system:")


def identity(principal: Any) -> Optional[Tuple[str, str]]:
    """(sign-in id, email) as they would be stored: both trimmed, the email capped; or None when
    there is nothing to remember (no id, a system actor, or a sign-in that carries no email)."""
    subject, email = getattr(principal, "subject", None), getattr(principal, "email", None)
    if not _is_person(subject) or not isinstance(email, str) or not email.strip():
        return None
    return subject.strip(), email.strip()[:MAX_EMAIL]


def remember(conn_or_cur: Any, principal: Any) -> None:
    """Keep the person's email under their sign-in id (insert, or update the email and when they
    were last seen). Nothing is written when `identity` is None. Never raises.

    Given a connection in autocommit (get_conn()) the statement stands alone. On a connection that
    is inside a transaction it runs in a savepoint, so a failure here cannot spoil the caller's
    own work."""
    who = identity(principal)
    if who is None:
        return
    try:
        cur = conn_or_cur.cursor() if hasattr(conn_or_cur, "cursor") else conn_or_cur
        conn = conn_or_cur if hasattr(conn_or_cur, "cursor") else getattr(conn_or_cur, "connection", None)
        if conn is None or getattr(conn, "autocommit", True):
            cur.execute(_REMEMBER_SQL, who)
            return
        cur.execute(f"SAVEPOINT {_SAVEPOINT}")
        try:
            cur.execute(_REMEMBER_SQL, who)
        except Exception:
            cur.execute(f"ROLLBACK TO SAVEPOINT {_SAVEPOINT}")
            raise
        finally:
            cur.execute(f"RELEASE SAVEPOINT {_SAVEPOINT}")
    except Exception as exc:  # noqa: BLE001 - remembering a name must never fail a person's action
        logger.warning("agent policy people: %s could not be remembered (%s); their id is shown instead",
                       who[0], type(exc).__name__)


def names(cur, subjects: Iterable[Any]) -> Dict[str, str]:
    """{sign-in id: email} for those of `subjects` that are remembered: ONE query for the whole
    set, and none at all when the set holds no person. Call it once per answer, never per row.

    Never raises: names that cannot be read are no names (the ids are shown), said once in the
    log. Meant for the reads, whose connection is in autocommit."""
    wanted = sorted({s.strip() for s in subjects if _is_person(s)})
    if not wanted:
        return {}
    try:
        cur.execute(_NAMES_SQL, (wanted,))
        return {subject: email for subject, email in cur.fetchall() if str(email or "").strip()}
    except Exception as exc:  # noqa: BLE001 - a page is never lost for want of a name
        logger.warning("agent policy people: names could not be read (%s); sign-in ids are shown", type(exc).__name__)
        return {}
