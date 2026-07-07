"""AgentNick supplier research: agentic web tool-use loop → grounded enrichment.

AgentNick (local, via Ollama tool-calling) drives the research; the backend
executes the web_search/fetch_url tools. Every reported fact must cite a source
URL that the tools actually returned — uncited facts are DROPPED (anti-
hallucination). Only EMPTY, non-sensitive bp_supplier fields are auto-filled;
conflicts stay pending for review. Nothing is ever overwritten or fabricated.
"""
from __future__ import annotations

import json
import logging
import os
import re
from urllib.parse import urlparse

import requests

from src.services.supplier_enrichment.web_tools import fetch_url, web_search

log = logging.getLogger(__name__)

_OLLAMA_CHAT = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/") + "/api/chat"
_MODEL = os.getenv("SUPPLIER_RESEARCH_MODEL", "BeyondProcwise/AgentNick:unified")
_MAX_ROUNDS = int(os.getenv("SUPPLIER_RESEARCH_MAX_ROUNDS", "4"))
_APPLY_CONF = float(os.getenv("SUPPLIER_RESEARCH_APPLY_CONF", "0.75"))

# bp_supplier columns we will auto-fill (descriptive, low-harm). business_summary
# is researched but kept in the sidecar only (no column).
_APPLY_COLUMNS = ["website_url", "registered_country", "legal_structure",
                  "supplier_type", "city", "country"]
_RESEARCH_FIELDS = _APPLY_COLUMNS + ["business_summary"]
# Never researched or applied — fraud vectors / high harm if wrong.
_SENSITIVE = {"tax_id", "vat_number", "registration_number", "duns_number",
              "bank_name", "bank_account_number", "bank_swift", "bank_iban",
              "credit_limit_amount"}

_TOOLS = [
    {"type": "function", "function": {
        "name": "web_search",
        "description": "Search the web. Returns a list of {title, url, snippet}.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "fetch_url",
        "description": "Fetch the visible text of a web page by its URL.",
        "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}}},
]

_SYSTEM = (
    "You research a company using the web tools and report ONLY verified facts. "
    "First web_search, then fetch_url to read authoritative pages (the company's own "
    "website, official registries, reputable business directories). For EVERY fact you "
    "report you MUST include the exact source_url you read it from. If you cannot verify "
    "a fact from a fetched page, set its value to \"unknown\" — never guess. Do NOT report "
    "bank, tax, VAT, or registration numbers. When finished, output ONLY a JSON object:\n"
    '{"fields": {'
    '"website_url": {"value": "..", "source_url": "..", "confidence": 0.0}, '
    '"registered_country": {...}, "legal_structure": {...}, "supplier_type": {...}, '
    '"city": {...}, "country": {...}, "business_summary": {...}}}'
)


def _host(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().removeprefix("www.")
    except Exception:  # noqa: BLE001
        return ""


def _chat(messages: list[dict]) -> dict:
    r = requests.post(_OLLAMA_CHAT, json={
        "model": _MODEL, "messages": messages, "tools": _TOOLS,
        "stream": False, "think": False, "keep_alive": -1,
        "options": {"temperature": 0},
    }, timeout=180)
    r.raise_for_status()
    return r.json().get("message", {})


def _run_loop(supplier_name: str) -> tuple[str, set[str]]:
    """Drive AgentNick's tool-use loop. Returns (final_content, urls_seen)."""
    seen: set[str] = set()
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": f"Research this supplier and return the JSON: {supplier_name}"},
    ]
    for _ in range(_MAX_ROUNDS):
        msg = _chat(messages)
        messages.append(msg)
        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            return msg.get("content") or "", seen
        for tc in tool_calls:
            fn = tc.get("function", {})
            name = fn.get("name")
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:  # noqa: BLE001
                    args = {}
            if name == "web_search":
                results = web_search(str(args.get("query", "")))
                for r in results:
                    seen.add(r["url"])
                content = json.dumps(results)
            elif name == "fetch_url":
                url = str(args.get("url", ""))
                seen.add(url)
                content = fetch_url(url)[:4000]
            else:
                content = "unknown tool"
            messages.append({"role": "tool", "name": name, "content": content})
    # rounds exhausted — ask once more for the final JSON
    messages.append({"role": "user", "content": "Now output ONLY the final JSON object."})
    try:
        return _chat(messages).get("content") or "", seen
    except Exception:  # noqa: BLE001
        return "", seen


def _parse_fields(content: str) -> dict:
    if not content:
        return {}
    m = re.search(r"\{[\s\S]*\}", content)
    if not m:
        return {}
    try:
        data = json.loads(m.group(0))
    except Exception:  # noqa: BLE001
        return {}
    fields = data.get("fields", data) if isinstance(data, dict) else {}
    return fields if isinstance(fields, dict) else {}


def _ground(fields: dict, seen_urls: set[str]) -> dict:
    """Keep only fields whose citation host was actually visited. Drop the rest."""
    seen_hosts = {_host(u) for u in seen_urls if u}
    kept: dict = {}
    for name, f in fields.items():
        if name not in _RESEARCH_FIELDS or name in _SENSITIVE or not isinstance(f, dict):
            continue
        value = f.get("value")
        src = f.get("source_url") or ""
        if value in (None, "", "unknown", "n/a", "N/A"):
            continue
        if _host(src) and _host(src) in seen_hosts:
            kept[name] = {"value": value, "source_url": src,
                          "confidence": float(f.get("confidence") or 0.0)}
    return kept


def _apply(cur, supplier_id: str, fields: dict) -> dict:
    """Fill only EMPTY, non-sensitive columns. Returns applied {col:{value,source_url}}."""
    cur.execute(
        "SELECT " + ", ".join(_APPLY_COLUMNS) + " FROM proc.bp_supplier WHERE supplier_id = %s",
        (supplier_id,),
    )
    row = cur.fetchone()
    current = dict(zip(_APPLY_COLUMNS, row)) if row else {}
    applied: dict = {}
    for col in _APPLY_COLUMNS:
        if col in _SENSITIVE:
            continue
        f = fields.get(col)
        if not f or float(f.get("confidence") or 0.0) < _APPLY_CONF:
            continue
        cur_val = current.get(col)
        if cur_val is None or str(cur_val).strip() == "":
            cur.execute(
                f"UPDATE proc.bp_supplier SET {col} = %s, last_modified_by = 'agentnick_web', "
                "last_modified_date = now() WHERE supplier_id = %s",
                (str(f["value"])[:500], supplier_id),
            )
            applied[col] = {"value": f["value"], "source_url": f.get("source_url")}
        # else: differs from an existing value → leave pending for human review
    return applied


def research_and_enrich(supplier_id: str, conn) -> dict:
    """Research a supplier via AgentNick and store a grounded enrichment record."""
    with conn.cursor() as cur:
        cur.execute("SELECT supplier_name FROM proc.bp_supplier WHERE supplier_id = %s", (supplier_id,))
        row = cur.fetchone()
    if not row or not row[0]:
        return {"error": "supplier not found", "supplier_id": supplier_id}
    supplier_name = row[0]

    content, seen = _run_loop(supplier_name)
    fields = _ground(_parse_fields(content), seen)
    confs = [f["confidence"] for f in fields.values()]
    overall = round(sum(confs) / len(confs), 3) if confs else 0.0

    with conn.cursor() as cur:
        applied = _apply(cur, supplier_id, fields)
        cur.execute(
            "INSERT INTO proc.bp_supplier_enrichment "
            "(supplier_id, model, fields, citations, confidence, raw, apply_status, applied_fields) "
            "VALUES (%s,%s,%s::jsonb,%s::jsonb,%s,%s::jsonb,%s,%s::jsonb) RETURNING enrichment_id",
            (supplier_id, _MODEL, json.dumps(fields), json.dumps(sorted(seen)), overall,
             json.dumps({"content": content[:4000]}),
             "applied" if applied else "pending", json.dumps(applied)),
        )
        eid = cur.fetchone()[0]
    conn.commit()
    log.info("supplier research %s: %d cited fields, %d auto-applied (enrichment %d)",
             supplier_id, len(fields), len(applied), eid)
    return {"enrichment_id": eid, "supplier_id": supplier_id, "supplier_name": supplier_name,
            "fields": fields, "applied": applied, "confidence": overall,
            "citations": sorted(seen)}


def reject_enrichment(enrichment_id: int, reviewer: str, conn) -> dict:
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE proc.bp_supplier_enrichment SET apply_status='rejected', reviewed_by=%s, "
            "reviewed_date=now() WHERE enrichment_id=%s AND apply_status <> 'rejected'",
            (reviewer, enrichment_id),
        )
        if cur.rowcount == 0:
            raise ValueError(f"enrichment {enrichment_id} not found or already rejected")
    conn.commit()
    return {"enrichment_id": enrichment_id, "status": "rejected"}


def batch_research(conn, limit: int = 10) -> dict:
    """Research suppliers with an empty website_url (proxy for 'not yet enriched')."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT supplier_id FROM proc.bp_supplier "
            "WHERE (website_url IS NULL OR trim(website_url) = '') "
            "AND supplier_id NOT IN (SELECT supplier_id FROM proc.bp_supplier_enrichment) "
            "ORDER BY created_date DESC LIMIT %s",
            (limit,),
        )
        ids = [r[0] for r in cur.fetchall()]
    done = [research_and_enrich(sid, conn) for sid in ids]
    return {"researched": len(done), "results": done}
