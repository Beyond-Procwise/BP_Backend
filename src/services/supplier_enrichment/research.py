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
# Entity-match gate: fact-confidence is NOT entity-match. The official name the
# model found must fuzzy-match the supplier name before we auto-apply, so we
# never fill Supplier A's record with a same-industry different company's data.
# Below this, the enrichment stays PENDING for human review.
_APPLY_NAME_MATCH = float(os.getenv("SUPPLIER_RESEARCH_NAME_MATCH", "85"))

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
    "bank, tax, VAT, or registration numbers. Also report matched_name: the OFFICIAL company "
    "name exactly as it appears on the sources you read (so a human can verify you found the "
    "right company); if you are not confident you found the SAME company, set matched_name to "
    "\"unknown\". When finished, output ONLY a JSON object:\n"
    '{"matched_name": "..", "fields": {'
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
        "options": {"temperature": 0, "num_predict": 2048},
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


def _parse_result(content: str) -> tuple[str | None, dict]:
    """Return (matched_name, fields) from the model's final JSON."""
    if not content:
        return None, {}
    start = content.find("{")
    if start < 0:
        return None, {}
    # raw_decode parses the FIRST complete JSON object and ignores any trailing
    # junk (e.g. an extra closing brace the model sometimes appends, or a code
    # fence) — a plain json.loads on a greedy {...} match fails on that.
    try:
        data, _ = json.JSONDecoder().raw_decode(content[start:])
    except Exception:  # noqa: BLE001
        return None, {}
    if not isinstance(data, dict):
        return None, {}
    matched = data.get("matched_name")
    fields = data.get("fields", {})
    return (matched if isinstance(matched, str) else None), (fields if isinstance(fields, dict) else {})


def _ground(fields: dict, seen_urls: set[str]) -> dict:
    """Keep only fields whose citation host was actually visited. Drop the rest."""
    seen_hosts = {_host(u) for u in seen_urls if u}
    kept: dict = {}
    for name, f in fields.items():
        if name not in _RESEARCH_FIELDS or name in _SENSITIVE or not isinstance(f, dict):
            continue
        value = f.get("value")
        src = f.get("source_url") or ""
        # sentinel check is case-insensitive: the model emits "Unknown"/"UNKNOWN"
        # as often as "unknown", and those were being kept as if they were facts.
        if value is None or str(value).strip().lower() in ("", "unknown", "n/a", "na", "none", "not found"):
            continue
        if _host(src) and _host(src) in seen_hosts:
            kept[name] = {"value": value, "source_url": src,
                          "confidence": float(f.get("confidence") or 0.0)}
    return kept


def _col_limits(cur) -> dict:
    """Real varchar length caps for the apply columns (e.g. legal_structure is
    varchar(10)). Without this a long value raises StringDataRightTruncation,
    which aborts the transaction and loses the whole enrichment record."""
    cur.execute(
        "SELECT column_name, character_maximum_length FROM information_schema.columns "
        "WHERE table_schema = 'proc' AND table_name = 'bp_supplier' "
        "AND column_name = ANY(%s)",
        (_APPLY_COLUMNS,),
    )
    return {name: lim for name, lim in cur.fetchall() if lim}


def _apply(cur, supplier_id: str, fields: dict) -> dict:
    """Fill only EMPTY, non-sensitive columns. Returns applied {col:{value,source_url}}."""
    limits = _col_limits(cur)
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
            val = str(f["value"])[:500]
            lim = limits.get(col)
            if lim and len(val) > lim:
                # Don't silently store a mangled prefix ("Limited Liability
                # Company (LLC)" -> "Limited Li"); leave it for human review.
                log.info("skipping %s for %s: %d chars exceeds column limit %d",
                         col, supplier_id, len(val), lim)
                continue
            cur.execute(
                f"UPDATE proc.bp_supplier SET {col} = %s, last_modified_by = 'agentnick_web', "
                "last_modified_date = now() WHERE supplier_id = %s",
                (val, supplier_id),
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
    matched_name, raw_fields = _parse_result(content)
    fields = _ground(raw_fields, seen)
    confs = [f["confidence"] for f in fields.values()]
    overall = round(sum(confs) / len(confs), 3) if confs else 0.0

    # Entity-match gate: only auto-apply if the found company name matches the
    # supplier name. Otherwise keep everything pending for human review.
    from rapidfuzz import fuzz
    from src.services.extraction_v3.supplier_resolver import _strip_biz_suffix
    name_match = 0.0
    if matched_name and matched_name.strip().lower() != "unknown":
        name_match = float(fuzz.WRatio(_strip_biz_suffix(supplier_name) or supplier_name,
                                       _strip_biz_suffix(matched_name) or matched_name))
    entity_ok = name_match >= _APPLY_NAME_MATCH

    with conn.cursor() as cur:
        applied = _apply(cur, supplier_id, fields) if entity_ok else {}
        cur.execute(
            "INSERT INTO proc.bp_supplier_enrichment "
            "(supplier_id, model, fields, citations, confidence, raw, apply_status, applied_fields) "
            "VALUES (%s,%s,%s::jsonb,%s::jsonb,%s,%s::jsonb,%s,%s::jsonb) RETURNING enrichment_id",
            (supplier_id, _MODEL, json.dumps(fields), json.dumps(sorted(seen)), overall,
             json.dumps({"content": content[:4000], "matched_name": matched_name,
                         "name_match": name_match}),
             "applied" if applied else "pending", json.dumps(applied)),
        )
        eid = cur.fetchone()[0]
    conn.commit()
    log.info("supplier research %s: matched=%r name_match=%.0f, %d cited fields, %d auto-applied (enrichment %d)",
             supplier_id, matched_name, name_match, len(fields), len(applied), eid)
    return {"enrichment_id": eid, "supplier_id": supplier_id, "supplier_name": supplier_name,
            "matched_name": matched_name, "name_match": name_match, "entity_confirmed": entity_ok,
            "fields": fields, "applied": applied, "confidence": overall, "citations": sorted(seen)}


def apply_enrichment(enrichment_id: int, reviewer: str, conn) -> dict:
    """Human-approve a pending enrichment: apply its cited facts to EMPTY, non-
    sensitive supplier columns (the reviewer is the entity confirmation, so this
    bypasses the auto name-match gate — but still never overwrites and never
    touches sensitive fields)."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT supplier_id, fields, apply_status FROM proc.bp_supplier_enrichment "
            "WHERE enrichment_id = %s FOR UPDATE",
            (enrichment_id,),
        )
        row = cur.fetchone()
        if not row:
            raise ValueError(f"enrichment {enrichment_id} not found")
        supplier_id, fields, status = row
        if status == "rejected":
            raise ValueError("enrichment was rejected")
        if isinstance(fields, str):
            fields = json.loads(fields or "{}")
        applied = _apply(cur, supplier_id, fields or {})
        cur.execute(
            "UPDATE proc.bp_supplier_enrichment SET apply_status='applied', applied_fields=%s::jsonb, "
            "reviewed_by=%s, reviewed_date=now() WHERE enrichment_id=%s",
            (json.dumps(applied), reviewer, enrichment_id),
        )
    conn.commit()
    return {"enrichment_id": enrichment_id, "status": "applied", "applied": applied}


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
