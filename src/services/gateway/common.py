"""Helpers shared by the gateway function modules."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict

from .context import CallerContext
from .envelope import GatewayFault
from .gateway import Backends

_PO_TABLE = "bp_purchase_order_trgt"


class _P(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _prov(table: str, ids: List[str], store: str = "postgres") -> Dict[str, Any]:
    return {"store": store, "table": table, "record_ids": [str(i) for i in ids]}


def _connect(backends: Optional[Backends]):
    if backends is None or backends.connect is None:
        raise GatewayFault("UNAVAILABLE", "no database connection is configured for the gateway", retryable=True)
    return backends.connect


def _rows(cur: Any, sql: str, params: tuple) -> List[Dict[str, Any]]:
    cur.execute(sql, params)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _scope_sql(ctx: CallerContext, column: str) -> Tuple[str, tuple]:
    """The buyer clause for this caller, or nothing for an unrestricted one."""
    if ctx.buyer_scope is None:
        return "", ()
    return f" AND {column}::text = ANY(%s)", (list(ctx.buyer_scope),)


def _blind(ctx: CallerContext) -> bool:
    """An empty scope sees no purchase orders at all; no query is needed to know that."""
    return ctx.buyer_scope is not None and len(ctx.buyer_scope) == 0
