"""Which state moves are legal, for findings and opportunities.

The rules live in ONE place, proc.bp_lifecycle_transition, and the database enforces
them with a trigger (deploy/sql/2026-09-17_bp_lifecycle_transitions.sql) -- because the
Node gateway writes these tables too, and a Python-only check would not bind it.

This module is the Python face of that: ask before acting (``can_apply``), and turn the
trigger's refusal into something a person can read (``refusal``). It holds no copy of
the rules, so it cannot drift from what the database enforces.
"""
from __future__ import annotations

from typing import Any, Optional

# The SQLSTATE the trigger raises. Class "BP" is unused by Postgres.
REFUSED_SQLSTATE = "BP409"


class IllegalTransition(Exception):
    """A move the lifecycle table does not allow. The message says what moved where."""


def can_apply(cur: Any, object_type: str, frm: str, to: str) -> bool:
    """Is ``frm -> to`` a legal move for ``object_type``? Staying put always is."""
    if frm == to:
        return True
    cur.execute(
        "SELECT 1 FROM proc.bp_lifecycle_transition "
        "WHERE object_type = %s AND from_state = %s AND to_state = %s",
        (object_type, frm, to),
    )
    return cur.fetchone() is not None


def refusal(exc: BaseException) -> Optional[str]:
    """The trigger's reason if ``exc`` is a refused move, else None (a real failure)."""
    if getattr(exc, "pgcode", None) != REFUSED_SQLSTATE:
        return None
    diag = getattr(exc, "diag", None)
    return getattr(diag, "message_primary", None) or str(exc).strip()
