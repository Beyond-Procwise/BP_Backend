"""Assemble and summarize everything known about a procurement deal.

A "deal" is the entity keyed by ``deal_id`` across the final (_trgt) document
tables. ``gather_deal_context`` is read-only and returns a plain dict;
``summarize_deal`` turns that dict into clear text via a grounded LLM call.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

from src.services.db import get_conn

log = logging.getLogger(__name__)

# General model for summarization — deliberately NOT the extraction adapter,
# so the GPU/AgentNick stays free for extraction. Swappable to llm_router later.
_SUMMARY_MODEL = "qwen2.5:7b"

# (final table, line-items table or None, primary-key column)
_DOC_SOURCES = {
    "invoices": ("proc.bp_invoice_trgt", "proc.bp_invoice_line_items_trgt", "invoice_id"),
    "purchase_orders": ("proc.bp_purchase_order_trgt", "proc.bp_po_line_items_trgt", "po_id"),
    "quotes": ("proc.bp_quote_trgt", "proc.bp_quote_line_items_trgt", "quote_id"),
    "contracts": ("proc.bp_contracts", None, "contract_id"),
}


def _fetch_dicts(cur, sql: str, params: tuple = ()) -> list[dict]:
    cur.execute(sql, params)
    cols = [d[0] for d in (cur.description or [])]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def _gather(conn, deal_id: str) -> Optional[dict]:
    cur = conn.cursor()
    documents: dict[str, list[dict]] = {}
    total_docs = 0
    deal_name: Optional[str] = None

    for key, (table, lines_table, pk) in _DOC_SOURCES.items():
        rows = _fetch_dicts(cur, f"SELECT * FROM {table} WHERE deal_id = %s", (deal_id,))
        for r in rows:
            if not deal_name and r.get("deal_name"):
                deal_name = r["deal_name"]
            if lines_table is not None and r.get(pk) is not None:
                r["line_items"] = _fetch_dicts(
                    cur, f"SELECT * FROM {lines_table} WHERE {pk} = %s", (r[pk],)
                )
            else:
                r["line_items"] = []
        documents[key] = rows
        total_docs += len(rows)

    if total_docs == 0:
        return None

    actions = _fetch_dicts(
        cur,
        "SELECT * FROM proc.agent_actions WHERE deal_id = %s ORDER BY created_at ASC",
        (deal_id,),
    )
    discrepancies = _fetch_dicts(
        cur,
        "SELECT * FROM proc.bp_extraction_discrepancy WHERE doc_pk_candidate IN ("
        " SELECT invoice_id::text FROM proc.bp_invoice_trgt WHERE deal_id = %s"
        " UNION SELECT po_id::text FROM proc.bp_purchase_order_trgt WHERE deal_id = %s"
        " UNION SELECT quote_id::text FROM proc.bp_quote_trgt WHERE deal_id = %s)",
        (deal_id, deal_id, deal_id),
    )

    return {
        "deal_id": deal_id,
        "deal_name": deal_name,
        "documents": documents,
        "actions": actions,
        "discrepancies": discrepancies,
        "sources": {
            "invoices": len(documents["invoices"]),
            "purchase_orders": len(documents["purchase_orders"]),
            "quotes": len(documents["quotes"]),
            "contracts": len(documents["contracts"]),
            "actions": len(actions),
            "discrepancies": len(discrepancies),
        },
    }


def gather_deal_context(deal_id: str, conn: Any = None) -> Optional[dict]:
    """Read-only assembly of a deal. Returns None if no final records exist."""
    if conn is not None:
        return _gather(conn, deal_id)
    with get_conn() as own:
        return _gather(own, deal_id)


def _build_prompt(ctx: dict) -> str:
    facts = json.dumps(ctx, indent=2, default=str)
    return (
        "You are a procurement analyst. Write a clear, plain-English summary of "
        "the deal described by the JSON facts below.\n\n"
        "Rules:\n"
        "- Use ONLY the data provided. Do not fabricate or infer values that are "
        "not present.\n"
        "- If something is absent or null, simply leave it out — do not guess.\n"
        "- Cover: the documents involved (invoices, purchase orders, quotes, "
        "contracts), key amounts and currencies, suppliers, and any "
        "discrepancies or notable actions in the trail.\n"
        "- Be concise and factual; no marketing language.\n\n"
        f"Deal facts (JSON):\n{facts}\n\n"
        "Summary:"
    )
