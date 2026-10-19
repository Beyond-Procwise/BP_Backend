"""The caller context for a request, built from the authenticated principal. Called by the router/orchestrator only."""
from __future__ import annotations

import logging
from typing import Any, List

from .context import CallerContext

logger = logging.getLogger(__name__)
_warned_link_table = False


def _read_reseller_accounts(subject: str) -> List[str]:
    from src.services.db import get_conn
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT reseller_account_id FROM proc.bp_user_reseller_link WHERE subject = %s AND revoked_at IS NULL "
                    "ORDER BY reseller_account_id", (subject,))
        return [r[0] for r in cur.fetchall()]


def caller_from_principal(principal: Any, *, request_id: str = "", session_id: str = "") -> CallerContext:
    """Role, buyer scope and reseller accounts from the session. Buyer scope uses the resolver the reports use.

    Fails closed throughout: no principal, no assignment, or an unreadable assignment (including the reseller link table not
    existing yet) gives an empty scope, which sees nothing. An Admin is unrestricted (``None``).
    """
    global _warned_link_table
    from src.services.report_data import scope as report_scope
    s = report_scope.resolve(principal)
    resellers: Any = ()
    if s.all_rows:
        resellers = None
    elif s.subject:
        try:
            resellers = tuple(_read_reseller_accounts(s.subject))
        except Exception:                                   # noqa: BLE001 - never widen on a failed read
            if not _warned_link_table:
                logger.warning("could not read proc.bp_user_reseller_link (migration not applied?); reseller scope is empty", exc_info=True)
                _warned_link_table = True
            resellers = ()
    return CallerContext(principal_subject=s.subject or "", role=s.role, buyer_scope=None if s.all_rows else tuple(s.buyers),
                         reseller_account_ids=resellers, request_id=request_id, session_id=session_id, principal=principal)
