"""One audit row per gateway call: kept in memory, logged, put on the /ask trace, and (when switched on) written to the database.

The database trail is ``proc.bp_gateway_call`` (deploy/sql/2026-10-19_bp_gateway_audit.sql, append-only). It is behind
``ASK_GATEWAY_AUDIT_DB`` (opt-in) until the migration is promoted to production. A failed database write is logged and never
breaks the call: a read that cannot be audited is still a read, and the row remains in memory and in the log.
"""
from __future__ import annotations

import json
import logging
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional

logger = logging.getLogger(__name__)

_MAX_IDS = 200       # an aggregate can name thousands of records; the audit row keeps the first 200 and row_count says the rest

_INSERT = (
    "INSERT INTO proc.bp_gateway_call (call_id, ts, caller_subject, role, tenant_id, agent, request_id, function, function_version, "
    "params, result_record_ids, row_count, truncated, ok, error_code, latency_ms, scope_violation) "
    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s, %s, %s, %s, %s, %s) ON CONFLICT (call_id) DO NOTHING")


class DbAuditWriter:
    """Writes one audit row. ``connect`` returns a context-managed DB connection (default: the shared ``get_conn``)."""

    def __init__(self, connect: Optional[Callable[[], Any]] = None) -> None:
        self._connect = connect

    def __call__(self, row: Dict[str, Any]) -> None:
        connect = self._connect
        if connect is None:
            from src.services.db import get_conn
            connect = get_conn
        params = (row.get("call_id"), row.get("ts"), row.get("caller_subject") or "", row.get("role"), row.get("tenant_id"), row.get("agent"),
                  row.get("request_id"), row.get("function"), row.get("function_version"),
                  json.dumps(row.get("params") or {}, default=str, sort_keys=True),
                  json.dumps(list(row.get("result_record_ids") or [])[:_MAX_IDS], default=str),
                  row.get("row_count"), bool(row.get("truncated")), bool(row.get("ok")), row.get("error_code"), row.get("latency_ms"),
                  bool(row.get("scope_violation")))
        with connect() as conn, conn.cursor() as cur:
            cur.execute(_INSERT, params)


class AuditSink:
    def __init__(self, keep: int = 1000, db_writer: Optional[Callable[[Dict[str, Any]], None]] = None) -> None:
        self._rows: Deque[Dict[str, Any]] = deque(maxlen=keep)
        self._db_writer = db_writer

    @property
    def rows(self) -> List[Dict[str, Any]]:
        return list(self._rows)

    def record(self, row: Dict[str, Any]) -> None:
        self._rows.append(row)
        logger.info("gateway call %s fn=%s agent=%s ok=%s code=%s rows=%s ms=%s", row.get("call_id"), row.get("function"),
                    row.get("agent"), row.get("ok"), row.get("error_code"), row.get("row_count"), row.get("latency_ms"))
        try:
            from src.services import ask_trace
            ask_trace.current().add_gateway_call(row)
        except Exception:  # noqa: BLE001 - the trace is a debugging aid, never a reason to fail a call
            pass
        if self._db_writer is not None:
            try:
                self._db_writer(row)
            except Exception:  # noqa: BLE001 - see module docstring
                logger.error("gateway audit row %s could not be written to the database", row.get("call_id"), exc_info=True)


def default_audit_sink() -> AuditSink:
    """The sink a gateway gets when none is passed: database trail on only when ASK_GATEWAY_AUDIT_DB is."""
    from src.services import ask_flags
    return AuditSink(db_writer=DbAuditWriter() if ask_flags.enabled(ask_flags.GATEWAY_AUDIT) else None)
