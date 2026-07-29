"""Persona-driven summaries over the final (_trgt) procurement tables.

Mirrors ``deal_summary`` (direct SQL, cloud LLM, no local GPU). A persona is
resolved from the ``bp_prompt`` governance table (``prompt_type='summary_persona'``)
with a raw-string fallback. Results are cached and versioned in ``proc.bp_summary``.
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from src.services.db import get_conn
from src.services.ollama_client import ollama_generate
from src.services.deal_summary import (
    GROUNDING_RULE,
    NATIVE_CURRENCY_RULE,
    PORTFOLIO_CURRENCY_RULE,
    gather_deal_context,
    _summary_facts,
)

log = logging.getLogger(__name__)

# Summaries run on the LOCAL Ollama GPU (AgentNick:unified, the consolidated
# non-extraction brain). This was previously routed to the Ollama Cloud API to
# keep the small A10G free for extraction, but on the 96GB Blackwell there is
# ample room for both models — and the cloud host (api.ollama.com) does not host
# the custom AgentNick model, so cloud calls 404'd and retried. Override the
# model via PROCWISE_SUMMARY_MODEL.
_SUMMARY_MODEL = os.getenv("PROCWISE_SUMMARY_MODEL", "BeyondProcwise/AgentNick:unified")


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


_PERSONA_FORMAT = (
    "Structure your answer as:\n"
    "<one or two plain-English sentences covering what you are summarising>\n"
    "Key Points:\n"
    "• <label>: <finding>\n"
    "(3-6 bullets, reflecting the emphasis you were given above.)\n"
    "Next Steps:\n"
    "• <a specific action supported by the facts>\n"
    "(2-4 bullets. Omit this whole section if the facts support no action — "
    "never invent one to fill it.)\n"
    "Conclusion:\n"
    "<one short sentence>"
)


def _build_persona_prompt(framing: str, facts: dict) -> str:
    """Persona framing + shared grounding rules + a persona-shaped format.

    This used to delegate wholesale to ``deal_summary._build_prompt``, which
    orders a "summary of the deal" in EXACTLY three fixed sections (sentences,
    Key Outcomes, Conclusion). Two consequences: a portfolio summary was told it
    was summarising a deal, and a persona asking for leverage points, concession
    opportunities or next steps had nowhere to put them — the format instruction
    came after the persona text and won. The grounding rules are still shared
    verbatim with the deal path; only the format and the subject differ.
    """
    is_portfolio = str(facts.get("scope") or "").strip().lower() == "portfolio"
    subject = (
        "the procurement portfolio described below"
        if is_portfolio
        else "the deal described below"
    )
    currency_rule = PORTFOLIO_CURRENCY_RULE if is_portfolio else NATIVE_CURRENCY_RULE
    fact_json = json.dumps(_summary_facts(facts), indent=2, default=str)
    label = "Portfolio facts (JSON)" if is_portfolio else "Deal facts (JSON)"
    return (
        f"{framing.strip()}\n\n"
        f"Using ONLY the JSON facts below, write a precise summary of {subject}. "
        f"{GROUNDING_RULE} {currency_rule}\n\n"
        f"{_PERSONA_FORMAT}\n\n"
        f"{label}:\n{fact_json}\n"
    )


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
    # Normalize as_of: clients send "", "null", "undefined" (JS) or other junk
    # for "now". Only a PARSEABLE timestamp may drive the historical-snapshot
    # branch; anything else falls through to current generation rather than
    # crashing the query (`generated_at <= 'null'` -> InvalidDatetimeFormat).
    if isinstance(as_of, str):
        s = as_of.strip()
        if not s or s.lower() in ("null", "undefined", "none", "nan"):
            as_of = None
        else:
            try:
                datetime.fromisoformat(s.replace("Z", "+00:00"))
                as_of = s
            except ValueError:
                log.warning("summary: ignoring unparseable as_of=%r; generating current", as_of)
                as_of = None

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

    text = ollama_generate(
        _build_persona_prompt(framing, facts),
        model=_SUMMARY_MODEL,
        temperature=0.0,
        num_predict=1024,
        timeout=120,
        retries=2,
        think=False,  # AgentNick:unified is a reasoning model — keep answer in `response`
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


def _rows_as_dicts(cur) -> list[dict]:
    cols = [d[0] for d in (cur.description or [])]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def get_cached_summary(persona: str, deal_id: Optional[str] = None, conn: Any = None) -> Optional[dict]:
    """Return the current cached summary for (persona, deal_id), or None."""
    if conn is None:
        with get_conn() as own:
            return get_cached_summary(persona, deal_id, conn=own)
    scope = "deal" if deal_id else "portfolio"
    cur = conn.cursor()
    cur.execute(
        "SELECT summary_id, persona, persona_source, scope, deal_id, summary, "
        "sources, generated_at FROM proc.bp_summary "
        "WHERE persona = %s AND scope = %s AND deal_id IS NOT DISTINCT FROM %s "
        "AND is_current ORDER BY generated_at DESC LIMIT 1",
        (persona, scope, deal_id),
    )
    rows = _rows_as_dicts(cur)
    return rows[0] if rows else None


def list_summary_history(persona: str, deal_id: Optional[str] = None, conn: Any = None) -> list[dict]:
    """Return prior summaries for (persona, deal_id), newest first."""
    if conn is None:
        with get_conn() as own:
            return list_summary_history(persona, deal_id, conn=own)
    scope = "deal" if deal_id else "portfolio"
    cur = conn.cursor()
    cur.execute(
        "SELECT summary_id, scope, deal_id, generated_at, is_current, "
        "left(summary, 200) AS snippet FROM proc.bp_summary "
        "WHERE persona = %s AND scope = %s AND deal_id IS NOT DISTINCT FROM %s "
        "ORDER BY generated_at DESC",
        (persona, scope, deal_id),
    )
    return _rows_as_dicts(cur)


def get_summary_by_id(summary_id: str, conn: Any = None) -> Optional[dict]:
    """Return a single stored summary by id, or None."""
    if conn is None:
        with get_conn() as own:
            return get_summary_by_id(summary_id, conn=own)
    cur = conn.cursor()
    cur.execute(
        "SELECT summary_id, persona, persona_source, scope, deal_id, summary, "
        "sources, model, is_current, generated_at FROM proc.bp_summary "
        "WHERE summary_id = %s",
        (summary_id,),
    )
    rows = _rows_as_dicts(cur)
    return rows[0] if rows else None


def _distinct_deal_ids(conn: Any) -> list[str]:
    cur = conn.cursor()
    cur.execute(
        "SELECT DISTINCT deal_id FROM ("
        " SELECT deal_id FROM proc.bp_invoice_trgt "
        " UNION SELECT deal_id FROM proc.bp_purchase_order_trgt "
        " UNION SELECT deal_id FROM proc.bp_quote_trgt) t "
        "WHERE deal_id IS NOT NULL"
    )
    return [r[0] for r in cur.fetchall()]


def _summary_personas(conn: Any) -> list[str]:
    cur = conn.cursor()
    cur.execute(
        "SELECT prompt_name FROM proc.bp_prompt "
        "WHERE prompt_type = 'summary_persona' AND COALESCE(prompts_status,1)=1"
    )
    return [r[0] for r in cur.fetchall()]


_DEFAULT_MAX_DEALS = 50
_DEFAULT_BUDGET_SECONDS = 20 * 60
_DEFAULT_MAX_CONSECUTIVE_FAILURES = 3


def _recent_deal_ids(conn: Any) -> list[str]:
    """Every deal id, most recently touched first.

    Warming the cache is only worth anything for the deals somebody is likely to open,
    and the caller keeps just the head of this list.
    """
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT deal_id FROM ("
            "  SELECT deal_id, MAX(created_date) AS seen FROM proc.bp_invoice_trgt"
            "   WHERE deal_id IS NOT NULL GROUP BY deal_id"
            "  UNION ALL SELECT deal_id, MAX(created_date) FROM proc.bp_purchase_order_trgt"
            "   WHERE deal_id IS NOT NULL GROUP BY deal_id"
            "  UNION ALL SELECT deal_id, MAX(created_date) FROM proc.bp_quote_trgt"
            "   WHERE deal_id IS NOT NULL GROUP BY deal_id"
            ") t GROUP BY deal_id ORDER BY MAX(seen) DESC NULLS LAST"
        )
        rows = [r[0] for r in cur.fetchall()]
        if rows:
            return rows
    except Exception:  # pragma: no cover - schema drift; order is an optimisation only
        log.warning("summary precompute: recency ordering unavailable; using unordered ids")
    return _distinct_deal_ids(conn)


def precompute_summaries(
    personas: Optional[list[str]] = None,
    deal_ids: Optional[list[str]] = None,
    conn: Any = None,
    max_deals: Optional[int] = None,
    budget_seconds: Optional[float] = None,
    max_consecutive_failures: Optional[int] = None,
) -> dict:
    """Warm the summary cache for the portfolio and the most recent deals.

    This is an OPTIMISATION, not a source of truth: anything not warmed here is generated
    on demand by POST /summary, so leaving deals out costs a slower first open and nothing
    else.

    It is bounded three ways because it shares one Ollama model with every interactive
    request. Unbounded, it planned 3 personas x 5038 deals = 15114 sequential generations;
    live, that ran for six hours without finishing while each failing item burned ~150s in
    timeouts and retries, and every user-facing call queued behind it.

      max_deals                 - warm the newest N deals, not all of them
      budget_seconds            - give the whole run a wall clock, so it always ends
      max_consecutive_failures  - abort when the model is clearly unavailable; it will not
                                  recover by being asked another 15000 times

    An explicit deal_ids list is honoured in full — the caller asked for those.
    """
    if conn is None:
        with get_conn() as own:
            return precompute_summaries(
                personas, deal_ids, conn=own,
                max_deals=max_deals,
                budget_seconds=budget_seconds,
                max_consecutive_failures=max_consecutive_failures,
            )

    try:  # settings are optional so the function stays unit-testable in isolation
        from config.settings import settings as _settings
    except Exception:  # pragma: no cover
        _settings = None

    def _cfg(name, fallback):
        return getattr(_settings, name, fallback) if _settings is not None else fallback

    if max_deals is None:
        max_deals = int(_cfg("summary_precompute_max_deals", _DEFAULT_MAX_DEALS))
    if budget_seconds is None:
        budget_seconds = float(_cfg("summary_precompute_budget_minutes", 20)) * 60.0
    if max_consecutive_failures is None:
        max_consecutive_failures = int(
            _cfg("summary_precompute_max_consecutive_failures", _DEFAULT_MAX_CONSECUTIVE_FAILURES)
        )

    personas = personas or _summary_personas(conn)

    explicit = deal_ids is not None
    skipped_deals = 0
    if explicit:
        chosen = list(deal_ids)
    else:
        available = _recent_deal_ids(conn)
        chosen = available[:max_deals] if max_deals and max_deals > 0 else available
        skipped_deals = max(0, len(available) - len(chosen))

    scopes: list[Optional[str]] = [None] + chosen  # None = portfolio
    log.info(
        "summary precompute: %d personas x %d scopes (%d deals skipped, budget %.0fs)",
        len(personas), len(scopes), skipped_deals, budget_seconds or 0,
    )

    start = time.monotonic()
    generated = 0
    failed = 0
    consecutive = 0
    stopped_reason = None

    for persona in personas:
        for deal_id in scopes:
            if budget_seconds and (time.monotonic() - start) >= budget_seconds:
                stopped_reason = "budget"
                break
            try:
                generate_summary(persona, deal_id=deal_id, conn=conn)
                generated += 1
                consecutive = 0
            except Exception:  # logged, run continues unless the model is clearly down
                failed += 1
                consecutive += 1
                log.exception(
                    "precompute failed for persona=%s deal_id=%s", persona, deal_id
                )
                if max_consecutive_failures and consecutive >= max_consecutive_failures:
                    stopped_reason = "failures"
                    break
        if stopped_reason:
            break

    elapsed = time.monotonic() - start
    if stopped_reason:
        # Never let a truncated run read as a complete one.
        log.warning(
            "summary precompute stopped early (%s) after %.0fs: %d generated, %d failed",
            stopped_reason, elapsed, generated, failed,
        )
    if skipped_deals:
        log.info(
            "summary precompute warmed the %d most recent deals; %d not warmed "
            "(generated on demand at first open)", len(chosen), skipped_deals,
        )

    return {
        "generated": generated,
        "failed": failed,
        "personas": len(personas),
        "scopes": len(scopes),
        "skipped_deals": skipped_deals,
        "stopped_reason": stopped_reason,
        "elapsed_seconds": round(elapsed, 1),
    }
