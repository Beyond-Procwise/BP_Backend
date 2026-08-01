"""Writes for analysis events (proc.bp_analysis + its two child tables).

This is the ONLY module that inserts into bp_analysis*. Three entry points:

    start()   an upload begins           -> status 'running'
    freeze()  its session resolves       -> status 'complete', findings captured
    sweep()   safety net on a timer      -> creates missed events, freezes stuck
                                            ones, fails ones that never resolved

See docs/superpowers/specs/2026-08-01-analysis-events-design.md
"""
from __future__ import annotations

import json
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


def _allocate_version(cur: Any, deal_id: str) -> int:
    """The next version number for THIS deal.

    Per deal_id, never global — one analysis run can touch several deals and
    each of them is at a different point in its own history.
    """
    cur.execute(
        "SELECT COALESCE(MAX(version), 0) + 1 FROM proc.bp_analysis_deal "
        "WHERE deal_id = %s",
        (str(deal_id),),
    )
    row = cur.fetchone()
    return int(row[0]) if row else 1


def _link_deals(cur: Any, analysis_id: Any, deal_ids: list) -> int:
    """Link an analysis to the deals it produced, allocating each deal's next
    version. Returns how many links were written.

    Version is allocated PER deal_id, never globally — one analysis run can
    touch several deals, and each of them is at a different point in its own
    history, so the same run may be v3 for one deal and v1 for another.
    """
    count = 0
    for deal_id in deal_ids:
        cur.execute(
            "UPDATE proc.bp_analysis_deal SET is_latest = false "
            "WHERE deal_id = %s",
            (str(deal_id),),
        )
        cur.execute(
            "INSERT INTO proc.bp_analysis_deal "
            "       (analysis_id, deal_id, version, is_latest) "
            "VALUES (%s, %s, %s, true) "
            "ON CONFLICT (analysis_id, deal_id) DO NOTHING",
            (analysis_id, str(deal_id), _allocate_version(cur, deal_id)),
        )
        count += 1
    return count


def freeze(session_id: str, *, findings: Optional[dict] = None,
           document_count: Optional[int] = None, value_found: Any = None,
           currency: Optional[str] = None,
           conn: Optional[Any] = None) -> Optional[str]:
    """Freeze the running analysis for a resolved upload session.

    Returns the analysis_id, or None when there is nothing in 'running' for
    this session — which is the normal outcome of a second call, and is what
    makes this safe for both the listener and the sweep to invoke.
    """
    sid = (session_id or "").strip()
    if not sid:
        return None

    with _txn(conn) as c:
        cur = c.cursor()

        cur.execute(
            "SELECT analysis_id FROM proc.bp_analysis "
            "WHERE session_id = %s AND status = 'running' FOR UPDATE",
            (sid,),
        )
        row = cur.fetchone()
        if not row:
            return None
        analysis_id = row[0]

        # 1. What this analysis read. Copied from the trigger-written outcome
        #    table, which is the authoritative record of the session.
        cur.execute(
            """
            INSERT INTO proc.bp_analysis_document
                   (analysis_id, doc_type, file_path, file_name, outcome)
            SELECT %s, sdo.document_type, sdo.file_path,
                   regexp_replace(sdo.file_path, '^.*/', ''), sdo.outcome
              FROM proc.session_document_outcome sdo
             WHERE sdo.session_id = %s
            ON CONFLICT (analysis_id, file_path) DO NOTHING
            """,
            (analysis_id, sid),
        )

        # 2. Which deals it produced. process_monitor already carries deal_id.
        cur.execute(
            "SELECT DISTINCT deal_id FROM proc.process_monitor "
            "WHERE session_id = %s AND deal_id IS NOT NULL ORDER BY deal_id",
            (sid,),
        )
        deal_ids = [deal_id for (deal_id,) in (cur.fetchall() or [])]
        _link_deals(cur, analysis_id, deal_ids)

        # 3. Freeze.
        cur.execute(
            """
            UPDATE proc.bp_analysis
               SET status = 'complete', completed_at = now(),
                   findings = %s, document_count = %s,
                   value_found = %s, currency = %s
             WHERE analysis_id = %s
            """,
            (json.dumps(findings) if findings is not None else None,
             document_count, value_found, currency, analysis_id),
        )
        return str(analysis_id)
