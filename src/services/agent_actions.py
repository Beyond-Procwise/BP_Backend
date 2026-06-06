"""Best-effort writer for the proc.agent_actions event log.

One row per action/step. Writes are best-effort: any failure is logged and
swallowed so a logging problem can never break extraction. Callers may pass an
existing ``conn`` to fold the write into their own transaction (the caller then
owns commit/rollback); otherwise a short autonomous transaction is used.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Iterable, Mapping

from src.services.db import get_conn

log = logging.getLogger(__name__)

# Canonical phase values (stored as free text in the DB).
PHASE_EXTRACTION = "extraction"
PHASE_VALIDATION = "validation"
PHASE_CONSOLIDATION = "consolidation"

# Canonical action_type values (documentation; DB column is free text).
#   extraction:  parse, regex_extract, engineered_extract, ner_gapfill, judge,
#                context_synthesize, line_recovery, grounding_gate, persist
#   validation:  discrepancy
#   consolidation: reconcile_match, reconcile_mismatch  (writer slot only)

_COLUMNS = (
    "deal_id", "document_id", "doc_pk", "doc_type", "process_monitor_id",
    "trace_id", "phase", "action_type", "agent", "field_name", "status",
    "summary", "details", "confidence", "pipeline_version",
)

_INSERT = (
    "INSERT INTO proc.agent_actions ("
    + ", ".join(_COLUMNS)
    + ") VALUES (" + ", ".join(["%s"] * len(_COLUMNS)) + ")"
)


def _as_text(value: Any) -> Any:
    return str(value) if value is not None else None


def _row_params(fields: Mapping[str, Any]) -> tuple:
    """Build the positional params tuple for one row. Requires phase + action_type."""
    if not fields.get("phase") or not fields.get("action_type"):
        raise ValueError("agent_actions row requires phase and action_type")
    details = fields.get("details")
    if details is not None and not isinstance(details, str):
        details = json.dumps(details, default=str)
    return (
        fields.get("deal_id"),
        fields.get("document_id"),
        fields.get("doc_pk"),
        fields.get("doc_type"),
        fields.get("process_monitor_id"),
        _as_text(fields.get("trace_id")),
        fields["phase"],
        fields["action_type"],
        fields.get("agent"),
        fields.get("field_name"),
        fields.get("status", "ok"),
        fields.get("summary"),
        details,
        fields.get("confidence"),
        fields.get("pipeline_version"),
    )


_SAVEPOINT = "agent_actions_sp"


def _write_on_shared_conn(conn: Any, run) -> None:
    """Run an INSERT on a caller-owned conn inside a SAVEPOINT.

    A logging write must never poison the caller's transaction. Without a
    savepoint, a failed INSERT into proc.agent_actions (e.g. the table is
    missing in some environment) would leave psycopg2 in an aborted-transaction
    state, so the caller's subsequent commit would silently discard its own rows
    (e.g. the discrepancy rows in write_discrepancies). The savepoint confines
    any failure to the action write alone.
    """
    cur = conn.cursor()
    cur.execute(f"SAVEPOINT {_SAVEPOINT}")
    try:
        run(cur)
        cur.execute(f"RELEASE SAVEPOINT {_SAVEPOINT}")
    except Exception:
        cur.execute(f"ROLLBACK TO SAVEPOINT {_SAVEPOINT}")
        raise


def record_action(*, phase: str, action_type: str, conn: Any = None, **fields: Any) -> None:
    """Insert one action row. Best-effort: errors are logged, never raised.

    If ``conn`` is provided, the row is written on that connection's cursor
    (inside a SAVEPOINT) and the caller owns commit/rollback. Otherwise a short
    autonomous transaction is opened and committed here.
    """
    try:
        fields["phase"] = phase
        fields["action_type"] = action_type
        params = _row_params(fields)
        if conn is not None:
            _write_on_shared_conn(conn, lambda cur: cur.execute(_INSERT, params))
            return
        with get_conn() as own:
            own.autocommit = False
            cur = own.cursor()
            try:
                cur.execute(_INSERT, params)
                own.commit()
            except Exception:
                own.rollback()
                raise
    except Exception as exc:  # best-effort: never break the caller
        log.warning("agent_actions.record_action failed (%s/%s): %s", phase, action_type, exc)


def bulk_record(rows: Iterable[Mapping[str, Any]], *, conn: Any = None) -> None:
    """Insert many rows (each mapping must include phase + action_type).

    Best-effort: errors are logged, never raised.
    """
    try:
        params = [_row_params(r) for r in rows]
        if not params:
            return
        if conn is not None:
            _write_on_shared_conn(conn, lambda cur: cur.executemany(_INSERT, params))
            return
        with get_conn() as own:
            own.autocommit = False
            cur = own.cursor()
            try:
                cur.executemany(_INSERT, params)
                own.commit()
            except Exception:
                own.rollback()
                raise
    except Exception as exc:  # best-effort
        log.warning("agent_actions.bulk_record failed: %s", exc)
