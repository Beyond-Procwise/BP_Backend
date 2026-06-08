"""Persona-driven summaries over the final (_trgt) procurement tables.

Mirrors ``deal_summary`` (direct SQL, cloud LLM, no local GPU). A persona is
resolved from the ``bp_prompt`` governance table (``prompt_type='summary_persona'``)
with a raw-string fallback. Results are cached and versioned in ``proc.bp_summary``.
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from src.services.db import get_conn
from src.services.ollama_client import ollama_cloud_generate
from src.services.deal_summary import gather_deal_context, _build_prompt

log = logging.getLogger(__name__)

# Summaries run on the Ollama Cloud API (remote), keeping the local GPU free.
_SUMMARY_MODEL = os.getenv("PROCWISE_SUMMARY_MODEL", "gpt-oss:120b")


class SummarizationError(RuntimeError):
    """Raised when the LLM returns no usable summary."""


class SnapshotNotFound(RuntimeError):
    """Raised when an as_of request finds no snapshot at/before the datetime."""


def resolve_persona(persona: str, conn: Any) -> tuple[str, str]:
    """Return (framing_text, persona_source).

    Looks up ``persona`` in bp_prompt (prompt_type='summary_persona'). On a hit
    returns the stored template and 'bp_prompt'; on a miss returns the persona
    string itself and 'raw'.
    """
    cur = conn.cursor()
    row = None
    try:
        cur.execute(
            "SELECT prompts_desc FROM proc.bp_prompt "
            "WHERE prompt_type = 'summary_persona' AND prompt_name = %s "
            "AND COALESCE(prompts_status, 1) = 1 LIMIT 1",
            (persona,),
        )
        row = cur.fetchone()
    except Exception:  # pragma: no cover - defensive
        log.exception("persona lookup failed for %s", persona)
    if row and row[0]:
        payload = row[0] if isinstance(row[0], dict) else json.loads(row[0])
        if isinstance(payload, dict):
            template = payload.get("prompt_template") or payload.get("template")
            if template:
                return str(template), "bp_prompt"
    return persona, "raw"


def gather_portfolio_context(conn: Any) -> Optional[dict]:
    """Aggregate the final (_trgt) tables into a compact portfolio fact dict.

    Returns None when all three document tables are empty.
    """
    cur = conn.cursor()

    def _scalar(sql: str) -> Any:
        cur.execute(sql)
        r = cur.fetchone()
        return r[0] if r else None

    inv = _scalar("SELECT count(*) FROM proc.bp_invoice_trgt") or 0
    pos = _scalar("SELECT count(*) FROM proc.bp_purchase_order_trgt") or 0
    quotes = _scalar("SELECT count(*) FROM proc.bp_quote_trgt") or 0
    if (inv + pos + quotes) == 0:
        return None

    inv_usd = _scalar(
        "SELECT COALESCE(SUM(converted_amount_usd),0) FROM proc.bp_invoice_trgt"
    ) or 0
    po_usd = _scalar(
        "SELECT COALESCE(SUM(converted_amount_usd),0) FROM proc.bp_purchase_order_trgt t"
    ) or 0

    cur.execute(
        "SELECT supplier_id, COALESCE(SUM(converted_amount_usd),0) AS usd "
        "FROM proc.bp_invoice_trgt WHERE supplier_id IS NOT NULL "
        "GROUP BY supplier_id ORDER BY usd DESC LIMIT 10"
    )
    top_suppliers = [
        {"supplier_id": s, "invoice_usd": float(u or 0)} for s, u in cur.fetchall()
    ]

    cur.execute("SELECT currency, COUNT(*) AS n FROM proc.bp_invoice_trgt GROUP BY currency")
    currency_mix = {str(c): int(n) for c, n in cur.fetchall() if c is not None}

    disc = _scalar("SELECT count(*) FROM proc.bp_extraction_discrepancy") or 0
    actions = _scalar("SELECT count(*) FROM proc.bp_agent_actions") or 0

    return {
        "scope": "portfolio",
        "totals": {
            "invoices": int(inv),
            "purchase_orders": int(pos),
            "quotes": int(quotes),
            "invoice_spend_usd": float(inv_usd or 0),
            "po_value_usd": float(po_usd or 0),
        },
        "top_suppliers": top_suppliers,
        "currency_mix": currency_mix,
        "sources": {
            "invoices": int(inv),
            "purchase_orders": int(pos),
            "quotes": int(quotes),
            "discrepancies": int(disc),
            "actions": int(actions),
        },
    }


def _build_persona_prompt(framing: str, facts: dict) -> str:
    """Persona framing + the grounded base rules + the fact JSON.

    Reuses ``deal_summary._build_prompt`` for the no-fabrication base so the
    grounding rules stay identical across both summary paths.
    """
    base = _build_prompt(facts)
    return f"{framing.strip()}\n\n{base}"
