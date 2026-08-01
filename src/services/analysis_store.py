"""Writes for analysis events (proc.bp_analysis + its two child tables).

This is the ONLY module that inserts into bp_analysis*. Three entry points:

    start()   an upload begins           -> status 'running'
    freeze()  its session resolves       -> status 'complete', findings captured
    sweep()   safety net on a timer      -> creates missed events, freezes stuck
                                            ones, fails ones that never resolved

See docs/superpowers/specs/2026-08-01-analysis-events-design.md
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Iterator, Optional

from src.services.db import get_conn

log = logging.getLogger(__name__)

MODES = ("new", "amend", "bulk")


@contextmanager
def _txn(conn: Optional[Any]) -> Iterator[Any]:
    """Run inside the caller's connection, or own one and commit/rollback.

    Matches the pattern in src/services/deal_lifecycle.py: a caller that is
    already inside a transaction keeps control of it.
    """
    if conn is not None:
        yield conn
        return
    with get_conn() as own:
        own.autocommit = False
        try:
            yield own
            own.commit()
        except Exception:
            own.rollback()
            raise


def start(*, session_id: str, name: Optional[str] = None, mode: str = "new",
          created_by: Optional[str] = None, conn: Optional[Any] = None) -> str:
    """Create (or return) the analysis event for an upload session.

    Idempotent on session_id. A second call NEVER overwrites a name that is
    already set — the UI's call carries the user's chosen name, the sweep's
    fallback call does not, and whichever lands second must not win.
    """
    sid = (session_id or "").strip()
    if not sid:
        raise ValueError("session_id is required")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")

    with _txn(conn) as c:
        cur = c.cursor()
        cur.execute(
            """
            INSERT INTO proc.bp_analysis (session_id, name, mode, created_by)
            VALUES (%s, NULLIF(%s, ''), %s, %s)
            ON CONFLICT (session_id) DO UPDATE
               SET name = COALESCE(proc.bp_analysis.name, EXCLUDED.name)
            RETURNING analysis_id
            """,
            (sid, (name or "").strip(), mode, created_by),
        )
        return str(cur.fetchone()[0])
