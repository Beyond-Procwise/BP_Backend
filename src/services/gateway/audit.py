"""One audit row per gateway call. In-process until ``proc.bp_gateway_call`` is approved (a schema migration).

Rows go to a bounded in-memory list (what tests read), the log, and the /ask trace when one is active. Nothing here
writes to the database.
"""
from __future__ import annotations

import logging
from collections import deque
from typing import Any, Deque, Dict, List

logger = logging.getLogger(__name__)


class AuditSink:
    def __init__(self, keep: int = 1000) -> None:
        self._rows: Deque[Dict[str, Any]] = deque(maxlen=keep)

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
