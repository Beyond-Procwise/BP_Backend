"""Backfill bp_opportunity.deal_id from its anchoring quote (currently always NULL)."""
from __future__ import annotations
from typing import Any
from src.services.db import get_conn

_LINK_SQL = (
    "update proc.bp_opportunity o "
    "set deal_id = d.deal_id, updated_at = now() "
    "from proc.bp_deal_documents d "
    "where d.doc_type = 'quote' and d.doc_pk = o.quote_id "
    "and o.quote_id is not null and o.deal_id is null"
)


def _run(c: Any) -> int:
    cur = c.cursor()
    cur.execute(_LINK_SQL)
    return int(getattr(cur, "rowcount", 0) or 0)


def link_opportunities_to_deals(conn: Any = None) -> int:
    if conn is None:
        with get_conn() as own:
            own.autocommit = False
            try:
                n = _run(own); own.commit(); return n
            except Exception:
                own.rollback(); raise
    return _run(conn)
