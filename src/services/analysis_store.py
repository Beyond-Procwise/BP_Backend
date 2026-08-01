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


def _json_default(obj: Any) -> Any:
    """Encoder fallback for json.dumps(findings).

    findings carries raw DB rows straight through analysis_findings.capture(),
    so two non-JSON-native types show up: Decimal (NUMERIC columns like
    quote_total, financial_impact_gbp) and datetime/date (deal_date,
    detected_on, ...).

    Decimal -> str() preserves the exact monetary value; this is a frozen
    audit snapshot, and a float would risk silent precision drift.
    datetime/date -> .isoformat(), NOT str(): str() on a tz-aware datetime
    gives a space-separated, non-canonical form ("2026-07-29 14:46:22.691203
    +00:00") that not every JavaScript Date implementation parses reliably.
    isoformat() is the canonical, universally-parseable form.
    """
    import datetime as _dt
    import decimal as _decimal

    if isinstance(obj, (_dt.datetime, _dt.date)):
        return obj.isoformat()
    if isinstance(obj, _decimal.Decimal):
        return str(obj)
    return str(obj)


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


def deal_ids_for_session(session_id: str, *, conn: Optional[Any] = None) -> list:
    """Which deals this session's documents landed on.

    process_monitor already carries deal_id, so this is a direct lookup rather
    than a walk through the _trgt tables.
    """
    with _txn(conn) as c:
        cur = c.cursor()
        cur.execute(
            "SELECT DISTINCT deal_id FROM proc.process_monitor "
            "WHERE session_id = %s AND deal_id IS NOT NULL",
            (session_id,),
        )
        return [r[0] for r in (cur.fetchall() or [])]


def document_count_for_session(session_id: str, *, conn: Optional[Any] = None) -> Optional[int]:
    """How many documents this analysis read. None on failure or zero — zero
    would read as 'it read nothing' when the truth is 'we don't know'."""
    try:
        with _txn(conn) as c:
            cur = c.cursor()
            cur.execute(
                "SELECT COUNT(*) FROM proc.session_document_outcome "
                "WHERE session_id = %s",
                (session_id,),
            )
            row = cur.fetchone()
            return int(row[0]) if row and row[0] else None
    except Exception:
        log.exception("document count failed for session=%s", session_id)
        return None


def file_paths_for_session(session_id: str, *, conn: Optional[Any] = None) -> list:
    """The file paths this session's documents were read from — discrepancies
    are scoped by file path, not deal_id (bp_extraction_discrepancy has no
    deal_id column)."""
    with _txn(conn) as c:
        cur = c.cursor()
        cur.execute(
            "SELECT file_path FROM proc.session_document_outcome "
            "WHERE session_id = %s",
            (session_id,),
        )
        return [r[0] for r in (cur.fetchall() or [])]


def start(*, session_id: str, name: Optional[str] = None, mode: str = "new",
          created_by: Optional[str] = None, started_at: Optional[Any] = None,
          conn: Optional[Any] = None) -> str:
    """Create (or return) the analysis event for an upload session.

    Idempotent on session_id. A second call NEVER overwrites a name that is
    already set — the UI's call carries the user's chosen name, the sweep's
    fallback call does not, and whichever lands second must not win.

    started_at defaults to None, which leaves the column's own DEFAULT now()
    in charge — correct for the live listener path, where "now" IS the true
    start. The sweep passes an explicit historical timestamp (recovered from
    process_monitor / session_document_outcome) when it creates an event
    after the fact, so a browser-closed session is dated by when the upload
    actually happened, not by when the sweep noticed it. Once set, started_at
    is never overwritten on conflict, same principle as name — it is simply
    left out of the UPDATE's SET clause.
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
            INSERT INTO proc.bp_analysis (session_id, name, mode, created_by, started_at)
            VALUES (%s, NULLIF(%s, ''), %s, %s, COALESCE(%s, now()))
            ON CONFLICT (session_id) DO UPDATE
               SET name = COALESCE(proc.bp_analysis.name, EXCLUDED.name)
            RETURNING analysis_id
            """,
            (sid, (name or "").strip(), mode, created_by, started_at),
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
            (json.dumps(findings, default=_json_default) if findings is not None else None,
             document_count, value_found, currency, analysis_id),
        )
        return str(analysis_id)


def _freeze_one(session_id: str) -> Optional[str]:
    """Freeze one session exactly the way the live listener does, findings and all.

    Returns whatever freeze() returned: the analysis_id if something was
    actually frozen, or None when there was nothing 'running' for this
    session. The caller (sweep) must not count this as a freeze unless the
    result is not None.
    """
    from src.services import analysis_findings  # noqa: PLC0415

    deal_ids = deal_ids_for_session(session_id)
    file_paths = file_paths_for_session(session_id)
    findings = analysis_findings.capture(deal_ids, file_paths=file_paths)
    value_found, currency = analysis_findings.headline(findings)
    return freeze(session_id, findings=findings,
                  document_count=document_count_for_session(session_id),
                  value_found=value_found, currency=currency)


_SWEEP_SAVEPOINT = "sp_analysis_sweep_create"


def sweep(*, stale_minutes: int = 60, conn: Optional[Any] = None) -> dict:
    """Safety net for everything the listener path can miss.

    Three passes:
      1. sessions with documents but no analysis event at all (the browser
         closed before the UI's POST fired)
      2. analyses still 'running' whose session HAS resolved (the listener died
         mid-session)
      3. analyses 'running' past the stale cap whose session never resolved

    stale_minutes defaults to 60 - deliberately far beyond the UI's 6-minute
    patience cap, because a large upload legitimately takes minutes and must
    never be declared failed while it is still working.

    Pass 1 runs start() on the SAME transaction/cursor as the rest of the
    sweep (it needs to, so a session created in this run is visible to a
    same-run freeze). That means a SQL-level error in one iteration would
    otherwise poison the whole Postgres transaction — catching the Python
    exception does not un-abort it, so every other pass-1 row and all of
    passes 2 and 3 would be silently discarded when _txn rolls back. Each
    pass-1 iteration therefore runs inside its own SAVEPOINT, so one bad
    session is skipped without harming the rest. Pass 2 does not need this:
    _freeze_one() opens its own connection, so a failure there cannot poison
    this transaction in the first place.
    """
    result = {"created": 0, "frozen": 0, "failed": 0}
    with _txn(conn) as c:
        cur = c.cursor()

        cur.execute(
            """
            SELECT pm.session_id,
                   MAX(pm.deal_name),
                   COALESCE(
                       MIN(pm.start_ts AT TIME ZONE 'UTC'),
                       MIN(pm.created_date AT TIME ZONE 'UTC'),
                       MIN(sdo.created_at)
                   ) AS true_started_at
              FROM proc.process_monitor pm
              JOIN proc.session_document_outcome sdo
                ON sdo.session_id = pm.session_id
             WHERE pm.session_id IS NOT NULL
               AND NOT EXISTS (SELECT 1 FROM proc.bp_analysis a
                                WHERE a.session_id = pm.session_id)
             GROUP BY pm.session_id
            """
        )
        for session_id, deal_name, true_started_at in (cur.fetchall() or []):
            try:
                cur.execute(f"SAVEPOINT {_SWEEP_SAVEPOINT}")
                start(session_id=session_id, name=deal_name,
                      started_at=true_started_at, conn=c)
                cur.execute(f"RELEASE SAVEPOINT {_SWEEP_SAVEPOINT}")
                result["created"] += 1
            except Exception:
                log.exception("sweep could not create event for %s", session_id)
                cur.execute(f"ROLLBACK TO SAVEPOINT {_SWEEP_SAVEPOINT}")

        cur.execute(
            """
            SELECT a.session_id FROM proc.bp_analysis a
             WHERE a.status = 'running'
               AND EXISTS (SELECT 1 FROM proc.process_monitor pm
                            WHERE pm.session_id = a.session_id
                              AND pm.action_status IS NOT NULL)
            """
        )
        for (session_id,) in (cur.fetchall() or []):
            try:
                if _freeze_one(session_id) is not None:
                    result["frozen"] += 1
            except Exception:
                log.exception("sweep could not freeze %s", session_id)

        cur.execute(
            """
            UPDATE proc.bp_analysis
               SET status = 'failed', completed_at = now(),
                   failure_reason = %s
             WHERE status = 'running'
               AND started_at < now() - (%s || ' minutes')::interval
             RETURNING session_id
            """,
            ("session did not resolve", str(int(stale_minutes))),
        )
        result["failed"] = len(cur.fetchall() or [])
    return result
