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
import unicodedata
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


def _run_loop(supplier_name: str) -> tuple[str, dict[str, str]]:
    """Drive AgentNick's tool-use loop. Returns (final_content, evidence).

    `evidence` maps each URL to the exact text the model was shown for it, which is what the
    grounding guard checks claims against. Previously only the URLs were kept and the text was
    handed to the model and discarded, so nothing downstream could tell a page that was read
    from one that 403'd — both looked like "visited".

    Deliberately stores the SAME slice the model saw, not the full fetch: grounding against
    text the model never read would accept a lucky guess as though it had been sourced.
    """
    evidence: dict[str, str] = {}
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": f"Research this supplier and return the JSON: {supplier_name}"},
    ]
    for _ in range(_MAX_ROUNDS):
        msg = _chat(messages)
        messages.append(msg)
        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            return msg.get("content") or "", evidence
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
                    # Title+snippet IS text the model read, so it can legitimately ground a
                    # fact. Appended rather than assigned: the same URL can surface in more
                    # than one search, and a later fetch of it must not lose the snippet.
                    snippet = f"{r.get('title') or ''} {r.get('snippet') or ''}".strip()
                    evidence[r["url"]] = (evidence.get(r["url"], "") + " " + snippet).strip()
                content = json.dumps(results)
            elif name == "fetch_url":
                url = str(args.get("url", ""))
                content = fetch_url(url)[:4000]
                # A failed fetch returns "" and is recorded as such: the URL is known to have
                # been visited AND known to have yielded nothing, which is what lets the guard
                # refuse to ground on it instead of trusting the hostname.
                evidence[url] = (evidence.get(url, "") + " " + content).strip()
            else:
                content = "unknown tool"
            messages.append({"role": "tool", "name": name, "content": content})
    # rounds exhausted — ask once more for the final JSON
    messages.append({"role": "user", "content": "Now output ONLY the final JSON object."})
    try:
        return _chat(messages).get("content") or "", evidence
    except Exception:  # noqa: BLE001
        return "", evidence


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


_SENTINELS = ("", "unknown", "n/a", "na", "none", "not found", "not available", "null")

# Prose fields. A summary is a paraphrase by construction, so it can be neither confirmed by
# a verbatim search (which always fails) nor by token overlap (which is the hole that lets a
# fabricated clause through — see the grounding-guard digit-hole note). It is therefore never
# claimed to be verified. It has no bp_supplier column, so it reaches a human or nothing.
_PROSE_FIELDS = {"business_summary"}


def _norm_text(value: object) -> str:
    """Casefold, strip accents, and reduce every run of non-alphanumerics to one space.

    Space-padded so a plain `in` test lands on word boundaries: "us" must not match
    "industrial". This is FORMAT tolerance only — it makes "  Private   Limited Company  "
    match "private limited company" in the page. It deliberately does no alias or synonym
    expansion; see _value_supported.
    """
    if value is None:
        return ""
    s = unicodedata.normalize("NFKD", str(value))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^0-9A-Za-z]+", " ", s).casefold().strip()
    return f" {s} " if s else ""


def _norm_url(url: object) -> str:
    """Canonical key matching a citation to the page we actually showed the model.

    Folds the variants a model produces when echoing a URL back (scheme, host case, www,
    trailing slash, fragment) and NOTHING else. The query string is kept on purpose: on a
    company registry `?id=1` and `?id=2` are different companies, so dropping it would
    re-open host-level trust through the back door.
    """
    if not url:
        return ""
    try:
        p = urlparse(str(url).strip())
    except Exception:  # noqa: BLE001 - a malformed citation grounds nothing
        return ""
    if not p.netloc:
        return ""
    host = p.netloc.casefold()
    if host.startswith("www."):
        host = host[4:]
    key = f"{host}{(p.path or '').rstrip('/').casefold()}"
    return f"{key}?{p.query}" if p.query else key


def _value_supported(value: object, text: str) -> bool:
    """Does the claimed value actually appear in the text the model was shown?

    Substring-after-normalisation, not fuzzy similarity: "Brazil" must not be judged close
    enough to "United Kingdom". A value the page does not contain — including a correct
    alias the page happens not to use, like "UK" for "United Kingdom" — is dropped rather
    than guessed. Dropping costs a trip through human review, which loses nothing; guessing
    writes a wrong fact into a supplier record.
    """
    v = _norm_text(value)
    # A one-character value substring-matches nearly any page, so a match would carry no
    # information. Two is enough for real values ("US", "GB", "3M").
    if len(v.replace(" ", "")) < 2:
        return False
    return v in _norm_text(text)


def _value_host(value: object) -> str:
    """Host of a URL-valued field, tolerating a bare domain ("acme.example")."""
    raw = str(value or "").strip()
    if not raw:
        return ""
    if "://" not in raw:
        raw = "https://" + raw
    return _norm_url(raw).split("/")[0].split("?")[0]


def _ground(fields: dict, evidence: dict[str, str]) -> dict:
    """Keep only fields the cited page ACTUALLY SUPPORTS. Drop the rest.

    `evidence` maps every URL the model was shown to the exact text it was shown for that
    URL (search title+snippet, or fetched page body). Two things follow that the previous
    host-only check could not express:

      - A fetch that failed returns "" and is present-but-empty, so it grounds nothing. This
        is the case that mattered: a 403'd registry page was grounding three fields whose
        values came from the model's priors.
      - Grounding is per-URL, not per-host, so an invented path on a real host proves
        nothing.
    """
    by_url = {_norm_url(u): (t or "") for u, t in (evidence or {}).items()}
    kept: dict = {}
    for name, f in fields.items():
        if name not in _RESEARCH_FIELDS or name in _SENSITIVE or not isinstance(f, dict):
            continue
        value = f.get("value")
        src = f.get("source_url") or ""
        # Case-insensitive: the model emits "Unknown"/"UNKNOWN" as often as "unknown", and
        # those were being kept as if they were facts.
        if value is None or str(value).strip().lower() in _SENTINELS:
            continue

        key = _norm_url(src)
        if not key or key not in by_url:
            continue                      # uncited, or citing a page we never showed it
        text = by_url[key]
        if not text.strip():
            continue                      # visited, but yielded nothing to read

        if name in _PROSE_FIELDS:
            # Carried for the reviewer, explicitly NOT claimed as verified, and stripped of
            # the model's self-reported confidence so it can never be averaged into the
            # record's overall score as though it had been checked.
            kept[name] = {"value": value, "source_url": src, "confidence": 0.0,
                          "verified": False}
            continue

        if name == "website_url":
            # A URL is not page prose. It is supported either because the cited page IS on
            # that domain (you learned the company's site by standing on it), or because the
            # cited page names the domain in its text (a directory listing it).
            vhost = _value_host(value)
            page_host = key.split("/")[0].split("?")[0]
            ok = bool(vhost) and (vhost == page_host or _value_supported(vhost, text))
        else:
            ok = _value_supported(value, text)
        if ok:
            kept[name] = {"value": value, "source_url": src,
                          "confidence": float(f.get("confidence") or 0.0), "verified": True}
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
        # Only content-verified facts are ever written. Today the unverified entries are
        # prose, which has no column here — this makes that a rule rather than a coincidence
        # of _APPLY_COLUMNS, so adding a column later cannot open the gate.
        #
        # `is False` rather than falsy on purpose: rows stored before content verification
        # existed carry no `verified` key at all, and those were reviewed under the old
        # contract. Treating a missing key as a failure would silently make every pending
        # legacy enrichment unapprovable. Only an explicit False — checked and not supported
        # — blocks. _ground now always sets the key, so new records are fully governed.
        if f.get("verified") is False:
            log.info("skipping %s for %s: not content-verified against its cited page",
                     col, supplier_id)
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

    content, evidence = _run_loop(supplier_name)
    matched_name, raw_fields = _parse_result(content)
    fields = _ground(raw_fields, evidence)
    seen = set(evidence)
    # Only content-verified fields carry a confidence into the overall score. Unverified
    # prose is fixed at 0.0 by _ground, so including it would drag the average down as
    # though it were a weak fact rather than an unchecked one — average over the verified.
    confs = [f["confidence"] for f in fields.values() if f.get("verified")]
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
