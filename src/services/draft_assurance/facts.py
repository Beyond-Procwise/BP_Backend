"""Resolve facts from Postgres, with provenance. Read-only; parameterised; identifiers whitelisted.

A fact is only a fact if a row says so. Each one comes back with the table, column,
row id and the time it was read, so a later step can re-read exactly that row and
see whether it moved.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional

from .family import FactSource

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResolvedFact:
    key: str
    value: Any
    table: str
    column: str
    row_id: str
    retrieved_at: str

    def provenance(self) -> Dict[str, Any]:
        return {"source": "postgres", "table": self.table, "column": self.column,
                "row_id": self.row_id, "retrieved_at": self.retrieved_at}


@dataclass(frozen=True)
class Unresolved:
    key: str
    reason: str  # "none" | "multiple" | "no_lookup_key" | "error"
    detail: str = ""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def coerce(value: Any, value_type: str) -> Any:
    if value is None:
        return None
    if value_type == "number":
        try:
            return Decimal(str(value).replace(",", "").strip())
        except (InvalidOperation, ValueError):
            return None
    if value_type == "date":
        return value.isoformat() if hasattr(value, "isoformat") else str(value)
    text = str(value).strip()
    return text or None


def _select(src: FactSource) -> str:
    # Identifiers were validated against a plain-identifier pattern when the family
    # was parsed; every value is a bound parameter.
    where = " AND ".join(f"{col} = %s" for col in src.lookup)
    order = f" ORDER BY {src.order_by}" if src.order_by else ""
    return (f"SELECT {src.row_id}, {src.column} FROM proc.{src.table} "
            f"WHERE {where}{order} LIMIT 2")


class FactResolver:
    def __init__(self, conn: Any) -> None:
        self._conn = conn

    def resolve(self, src: FactSource, lookup_keys: Dict[str, Any]) -> Any:
        params: List[Any] = []
        for name in src.lookup.values():
            value = lookup_keys.get(name)
            if value in (None, ""):
                return Unresolved(src.key, "no_lookup_key", name)
            params.append(value)
        try:
            rows = self._fetch(_select(src), params)
        except Exception as exc:  # noqa: BLE001
            logger.exception("fact lookup failed for %s", src.key)
            return Unresolved(src.key, "error", type(exc).__name__)
        rows = [r for r in rows if coerce(r[1], src.value_type) is not None]
        if not rows:
            return Unresolved(src.key, "none")
        if len(rows) > 1 and not src.order_by:
            return Unresolved(src.key, "multiple")
        row_id, raw = rows[0]
        return ResolvedFact(src.key, coerce(raw, src.value_type), src.table, src.column,
                            str(row_id), _now())

    def reread(self, fact: ResolvedFact, src: FactSource) -> Any:
        """The value now held by the exact row a fact was first read from."""

        sql = f"SELECT {src.column} FROM proc.{src.table} WHERE {src.row_id} = %s"
        rows = self._fetch(sql, [fact.row_id])
        return coerce(rows[0][0], src.value_type) if rows else None

    def _fetch(self, sql: str, params: List[Any]) -> List[Any]:
        # get_conn() connections are AUTOCOMMIT, where a bare SET TRANSACTION does
        # nothing. An explicit read-only block makes a write here fail in Postgres
        # itself, whatever role the login holds.
        autocommit = bool(getattr(self._conn, "autocommit", False))
        with self._conn.cursor() as cur:
            if autocommit:
                cur.execute("BEGIN READ ONLY")
            try:
                cur.execute(sql, params)
                return list(cur.fetchall())
            finally:
                if autocommit:
                    cur.execute("ROLLBACK")
