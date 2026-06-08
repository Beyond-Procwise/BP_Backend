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


def _store_summary(
    conn: Any,
    *,
    persona: str,
    persona_source: str,
    scope: str,
    deal_id: Optional[str],
    summary: str,
    data_snapshot: Any,
    sources: Any,
    model: str,
    is_current: bool = True,
) -> dict:
    """Insert a summary row. When is_current, demote the prior current row of the
    same (persona, scope, deal_id) group first. summary_id/generated_at are set
    in Python so the result is returned without RETURNING parsing.
    """
    sid = str(uuid.uuid4())
    generated_at = datetime.now(timezone.utc)
    cur = conn.cursor()
    if is_current:
        cur.execute(
            "UPDATE proc.bp_summary SET is_current = false "
            "WHERE persona = %s AND scope = %s "
            "AND deal_id IS NOT DISTINCT FROM %s AND is_current",
            (persona, scope, deal_id),
        )
    cur.execute(
        "INSERT INTO proc.bp_summary "
        "(summary_id, persona, persona_source, scope, deal_id, summary, "
        " data_snapshot, sources, model, is_current, generated_at) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
        (
            sid, persona, persona_source, scope, deal_id, summary,
            json.dumps(data_snapshot, default=str),
            json.dumps(sources, default=str) if sources is not None else None,
            model, is_current, generated_at,
        ),
    )
    conn.commit()
    return {
        "summary_id": sid,
        "persona": persona,
        "persona_source": persona_source,
        "scope": scope,
        "deal_id": deal_id,
        "summary": summary,
        "sources": sources,
        "generated_at": generated_at.isoformat(),
    }


def generate_summary(
    persona: str,
    deal_id: Optional[str] = None,
    as_of: Optional[str] = None,
    conn: Any = None,
) -> Optional[dict]:
    """Generate (and persist) a persona summary.

    deal_id present -> per-deal scope; absent -> portfolio. When as_of is set,
    regenerate over the nearest stored snapshot at/before that datetime (the
    result is stored as a historical, non-current row). Returns None when there
    is no underlying data; raises SnapshotNotFound / SummarizationError.
    """
    if conn is None:
        with get_conn() as own:
            return generate_summary(persona, deal_id, as_of, conn=own)

    scope = "deal" if deal_id else "portfolio"
    framing, persona_source = resolve_persona(persona, conn)

    if as_of is not None:
        cur = conn.cursor()
        cur.execute(
            "SELECT data_snapshot FROM proc.bp_summary "
            "WHERE scope = %s AND deal_id IS NOT DISTINCT FROM %s "
            "AND generated_at <= %s ORDER BY generated_at DESC LIMIT 1",
            (scope, deal_id, as_of),
        )
        row = cur.fetchone()
        if not row or row[0] is None:
            raise SnapshotNotFound(
                f"no snapshot at/before {as_of} for scope={scope} deal_id={deal_id}"
            )
        facts = row[0] if isinstance(row[0], dict) else json.loads(row[0])
        is_current = False
    else:
        facts = (
            gather_deal_context(deal_id, conn=conn)
            if deal_id
            else gather_portfolio_context(conn)
        )
        if facts is None:
            return None
        is_current = True

    text = ollama_cloud_generate(
        _build_persona_prompt(framing, facts),
        model=_SUMMARY_MODEL,
        temperature=0.0,
        num_predict=1024,
        timeout=120,
        retries=2,
    )
    if not text or not text.strip():
        raise SummarizationError(
            f"empty summary for persona={persona} deal_id={deal_id}"
        )

    sources = facts.get("sources") if isinstance(facts, dict) else None
    return _store_summary(
        conn,
        persona=persona,
        persona_source=persona_source,
        scope=scope,
        deal_id=deal_id,
        summary=text.strip(),
        data_snapshot=facts,
        sources=sources,
        model=_SUMMARY_MODEL,
        is_current=is_current,
    )
