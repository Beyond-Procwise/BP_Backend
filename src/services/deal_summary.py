"""Assemble and summarize everything known about a procurement deal.

A "deal" is the entity keyed by ``deal_id`` across the final (_trgt) document
tables. ``gather_deal_context`` is read-only and returns a plain dict;
``summarize_deal`` turns that dict into clear text via a grounded LLM call.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

from src.services.db import get_conn
from src.services.ollama_client import ollama_generate

log = logging.getLogger(__name__)

# Single-brain setup: summaries run on the local AgentNick:unified reasoning
# model — the same brain used for analysis, negotiation and the agentic flow.
# Extraction keeps its own specialist. Env-configurable via PROCWISE_SUMMARY_MODEL.
_SUMMARY_MODEL = os.getenv("PROCWISE_SUMMARY_MODEL", "BeyondProcwise/AgentNick:unified")

# (final table, line-items table or None, primary-key column)
# Contracts are intentionally excluded: proc.bp_contracts has no deal_id column
# (contracts are not deal-scoped in the current schema), so they cannot be
# linked to a deal via deal_id. A deal is composed of the deal_id-bearing docs.
_DOC_SOURCES = {
    "invoices": ("proc.bp_invoice_trgt", "proc.bp_invoice_line_items_trgt", "invoice_id"),
    "purchase_orders": ("proc.bp_purchase_order_trgt", "proc.bp_po_line_items_trgt", "po_id"),
    "quotes": ("proc.bp_quote_trgt", "proc.bp_quote_line_items_trgt", "quote_id"),
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
        "SELECT * FROM proc.bp_agent_actions WHERE deal_id = %s ORDER BY created_at ASC",
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


def _strip_fx(obj):
    """Drop USD-conversion fields (e.g. converted_amount_usd, exchange_rate_to_usd)
    so the summary reports the deal in its NATIVE currency, matching the GBP
    dashboards. The stored conversion is also internally inconsistent, so feeding
    it to the LLM produced summaries headlined in USD."""
    if isinstance(obj, dict):
        return {k: _strip_fx(v) for k, v in obj.items() if not k.lower().endswith("_usd")}
    if isinstance(obj, list):
        return [_strip_fx(v) for v in obj]
    return obj


_DOC_SINGULAR = {"quotes": "quote", "purchase_orders": "purchase_order", "invoices": "invoice"}


def _doc_facts(typ: str, d: dict) -> dict:
    """One compact, native-currency fact record per document."""
    total = d.get("total_amount")
    if total is None:
        total = d.get("invoice_amount")
    incl = d.get("total_amount_incl_tax")
    if incl is None:
        incl = d.get("invoice_total_incl_tax")
    return {
        "type": typ,
        "supplier": d.get("supplier_name") or d.get("supplier_id"),
        "buyer": d.get("buyer_name") or d.get("buyer_id"),
        "currency": d.get("currency"),
        "total_amount": total,
        "tax_percent": d.get("tax_percent"),
        "total_incl_tax": incl,
        "date": d.get("quote_date") or d.get("order_date") or d.get("invoice_date"),
        "line_item_count": len(d.get("line_items") or []),
    }


def _summary_facts(ctx: dict) -> dict:
    """Curate a small, native-currency fact sheet for the LLM.

    The raw context (SELECT * rows + the agent-actions audit log) contains
    USD-converted amounts and internal event blobs that made the model headline
    figures in USD and sometimes describe the JSON instead of the deal. We feed
    only the fields a deal summary needs, in native currency, and drop the
    audit-log noise entirely.
    """
    docs = []
    for key, typ in _DOC_SINGULAR.items():
        for d in ctx.get("documents", {}).get(key, []):
            docs.append(_doc_facts(typ, d))
    discrepancies = [_strip_fx(x) for x in (ctx.get("discrepancies") or [])][:20]
    return {
        "deal_id": ctx.get("deal_id"),
        "deal_name": ctx.get("deal_name"),
        "document_count": len(docs),
        "documents": docs,
        "discrepancies": discrepancies,
    }


def _build_prompt(ctx: dict) -> str:
    facts = json.dumps(_summary_facts(ctx), indent=2, default=str)
    return (
        "You are a procurement analyst. Using ONLY the JSON facts below, write a "
        "SHORT, precise summary of the deal. Do not fabricate or infer values that "
        "are not present; if something is absent, leave it out. Report every "
        "monetary value in the deal's native currency (the `currency` and "
        "`total_amount`/`invoice_amount` fields); never convert currencies.\n\n"
        "Respond in EXACTLY this format and keep it tight:\n"
        "<one or two plain-English sentences: supplier, buyer, the documents "
        "involved (quote/PO/invoice), and total value with currency>\n"
        "Key Outcomes:\n"
        "• <Label>: <value>\n"
        "• <Label>: <value>\n"
        "(3-5 bullets maximum, each one short fact — e.g. Supplier, Total Value, "
        "Items, Price vs Quote, Discrepancies. Each bullet MUST start with '• ' "
        "and be of the form 'Label: value'.)\n"
        "Conclusion:\n"
        "<exactly one short sentence stating whether the deal is consistent and "
        "complete, and flagging any single issue worth noting>\n\n"
        f"Deal facts (JSON):\n{facts}\n"
    )


class SummarizationError(RuntimeError):
    """Raised when the LLM returns no usable summary."""


def summarize_deal(deal_id: str, conn: Any = None) -> Optional[dict]:
    """Summarize a deal into clear text. Returns None if the deal is unknown.

    Raises SummarizationError if the LLM returns nothing.
    """
    # Reconcile first (best-effort) so the action trail the summary reads back
    # includes the fresh consolidation matches/mismatches. A reconcile failure
    # must never block the summary. Imported lazily to avoid an import cycle
    # (reconciliation imports gather_deal_context from this module).
    try:
        from src.services.reconciliation import reconcile_deal
        reconcile_deal(deal_id, conn=conn)
    except Exception as exc:
        log.warning("reconcile before summary failed for %s: %s", deal_id, exc)

    ctx = gather_deal_context(deal_id, conn=conn)
    if ctx is None:
        return None
    prompt = _build_prompt(ctx)
    text = ollama_generate(
        prompt,
        model=_SUMMARY_MODEL,
        temperature=0.0,
        num_predict=400,
        timeout=120,
        retries=2,
        # AgentNick:unified is a reasoning model — keep the answer in `response`
        # instead of a separate `thinking` field, otherwise the summary comes
        # back empty ("Summary not available") or as a salvaged JSON blob.
        think=False,
    )
    if not text or not text.strip():
        raise SummarizationError(f"empty summary for deal {deal_id}")
    return {
        "deal_id": deal_id,
        "deal_name": ctx.get("deal_name"),
        "summary": text.strip(),
        "sources": ctx["sources"],
    }
