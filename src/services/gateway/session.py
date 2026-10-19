"""The caller context for a request, built from the authenticated principal. Called by the router/orchestrator only."""
from __future__ import annotations

from typing import Any

from .context import CallerContext


def caller_from_principal(principal: Any, *, request_id: str = "", session_id: str = "") -> CallerContext:
    """Role and buyer scope from the session, using the same resolver the reports use (``report_data.scope.resolve``).

    Fails closed: no principal, no assignment, or an unreadable assignment gives an empty scope, which sees nothing.
    An Admin is unrestricted (``buyer_scope=None``).
    """
    from src.services.report_data import scope as report_scope
    s = report_scope.resolve(principal)
    return CallerContext(principal_subject=s.subject or "", role=s.role, buyer_scope=None if s.all_rows else tuple(s.buyers),
                         request_id=request_id, session_id=session_id)
