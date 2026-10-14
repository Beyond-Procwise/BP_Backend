"""Who may see which rows. A Buyer, Viewer or Approver sees only the deals assigned to them;
an Admin sees all. No assignment means NO rows, never all rows (fail closed).

The assignment lives in proc.bp_user_buyer_scope, granted and revoked only by an Admin; it is
read on every request (no cached scope), so a revoke takes effect on the next call.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List, Optional

from src.services import rbac

logger = logging.getLogger(__name__)
ADMIN = "Admin"


@dataclass(frozen=True)
class Scope:
    role: str
    all_rows: bool                 # Admin
    buyers: tuple                  # the buyer_id codes a non-admin may see (may be empty)
    subject: Optional[str] = None

    @property
    def assigned_nothing(self) -> bool:
        return not self.all_rows and not self.buyers


def _read_buyers(subject: str) -> List[str]:
    from src.services.db import get_conn
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT buyer_id FROM proc.bp_user_buyer_scope WHERE subject = %s AND revoked_at IS NULL "
                    "ORDER BY buyer_id", (subject,))
        return [r[0] for r in cur.fetchall()]


def resolve(principal: Any) -> Scope:
    role = rbac.effective_role(principal)
    subject = getattr(principal, "subject", None)
    if role == ADMIN:
        return Scope(role, True, (), subject)
    if not subject:
        return Scope(role, False, (), None)
    try:
        return Scope(role, False, tuple(_read_buyers(subject)), subject)
    except Exception:
        # An unreadable assignment must never widen access: no rows, and the failure is logged.
        logger.error("report scope: could not read bp_user_buyer_scope for %s", subject, exc_info=True)
        return Scope(role, False, (), subject)
