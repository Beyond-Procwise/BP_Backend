"""Sales quotes through the gateway (ruled 2026-10-19: Sales and Admin only; margin only through the one existing permit).

Three layers, all of which must pass: (1) the function's role rule from config/gateway/functions.yaml, enforced by the gateway
before this code runs; (2) the existing ``quote.read`` / ``margin.read`` permit checks, asked about the signed-in user carried on
the caller context; (3) for margin, the shared ``sell_side.margin_permit`` rule, which the sales router also calls. The gateway has
no copy of any of these rules.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import Field

from src.services.sell_side import quotes
from src.services.sell_side._db import NotFound, StateConflict, transactional_conn

from .common import _P, _prov
from .context import CallerContext
from .envelope import GatewayFault, Raw
from .gateway import Backends

_TABLE = "bp_sales_quote"
# The list shows the customer record's header only: no totals of cost, no margin, no lines.
_LIST_COLS = ("sales_quote_id, quote_ref, quote_number, account_id, currency, quote_date, valid_until, total_ex_tax, status, "
              "created_date")


class GetQuoteParams(_P):
    quote_id: int = Field(ge=1)


class ListQuotesParams(_P):
    status: Optional[str] = Field(default=None, max_length=20)
    limit: int = Field(default=10, ge=1, le=25)

    def model_post_init(self, _ctx: Any) -> None:
        if self.status is not None and self.status not in quotes.QUOTE_STATUSES:
            raise ValueError(f"status must be one of {sorted(quotes.QUOTE_STATUSES)}")


def _connect(backends: Optional[Backends]):
    # The sell-side services need a transactional connection (they refuse autocommit); tests inject their own.
    return (backends.extra.get("sales_connect") if backends else None) or transactional_conn


def _ask_permit(ctx: CallerContext, action: str, context: Dict[str, Any]) -> None:
    import api.endpoint_gate as eg
    try:
        eg.require(action, ctx.principal, agent=ctx.agent or "Gateway", context=context)
    except eg.NotPermitted:
        raise GatewayFault("NOT_ALLOWED", f"{action} is not permitted for this user")


def get_quote(ctx: CallerContext, p: GetQuoteParams, backends: Optional[Backends]) -> Raw:
    _ask_permit(ctx, "quote.read", {"sales_quote_id": p.quote_id})
    with _connect(backends)() as conn:
        try:
            record = quotes.get_quote(conn, p.quote_id)
        except NotFound:
            return Raw(data={}, provenance=[_prov(_TABLE, [])])
    return Raw(data=record, provenance=[_prov(_TABLE, [p.quote_id])])


def get_quote_margin(ctx: CallerContext, p: GetQuoteParams, backends: Optional[Backends]) -> Raw:
    from src.services.sell_side import margin_permit      # imported here: it needs the api package on the path, the catalogue does not
    try:
        margin_permit.check(ctx.principal, {"sales_quote_id": p.quote_id}, agent=ctx.agent or "Gateway")
    except margin_permit.NotPermitted:
        raise GatewayFault("NOT_ALLOWED", "margin.read is not permitted for this user")
    with _connect(backends)() as conn:
        try:
            record = quotes.get_quote_internal(conn, p.quote_id)
        except NotFound:
            return Raw(data={}, provenance=[_prov(_TABLE, [])])
        except StateConflict as exc:                          # a missing cost record is refused, never read as "cost unknown"
            raise GatewayFault("UNAVAILABLE", str(exc))
    return Raw(data={**record, "margin_note": margin_permit.MARGIN_NOTE}, provenance=[_prov(_TABLE, [p.quote_id])])


def list_quotes(ctx: CallerContext, p: ListQuotesParams, backends: Optional[Backends]) -> Raw:
    _ask_permit(ctx, "quote.read", {"scope": "list"})
    where, params = ("WHERE status = %s", (p.status,)) if p.status else ("", ())
    with _connect(backends)() as conn:
        from src.services.sell_side._db import dict_cursor
        cur = dict_cursor(conn)
        cur.execute(f"SELECT {_LIST_COLS} FROM proc.{_TABLE} {where} ORDER BY created_date DESC, sales_quote_id DESC LIMIT %s",
                    params + (p.limit,))
        rows = [dict(r) for r in cur.fetchall()]
    return Raw(data=rows, provenance=[_prov(_TABLE, [r["sales_quote_id"] for r in rows])])


DECLARED = {
    "get_quote": (GetQuoteParams, get_quote),
    "list_quotes": (ListQuotesParams, list_quotes),
    "get_quote_margin": (GetQuoteParams, get_quote_margin),
}
