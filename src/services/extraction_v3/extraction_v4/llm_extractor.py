"""LLM-based extraction fallback.

Used when regex/layout extraction leaves required fields empty. Sends the
document text to a local Ollama instance running BeyondProcwise/AgentNick:extract
(a Qwen2.5 7B fine-tuned for structured extraction; already preloaded with
24h keep_alive).

Hallucination guard: every value returned by the LLM is verified by substring
match against the source document text. Values that don't appear in the doc
are dropped (not used to overwrite missing fields).

Wiring point: dispatch._run_hybrid_v4 calls llm_fill_missing_required AFTER
the regex/layout extraction returns its raw dict. Only fields that are
missing from the regex/layout result get LLM values written into the dict.
The mapper then runs over the filled dict, so the rest of the pipeline
(adapter, persist) sees no special case.

Environment:
    LLM_EXTRACT_FALLBACK = 'on' (default) | 'off'
    LLM_EXTRACT_MODEL    = model name (default 'BeyondProcwise/AgentNick:extract')
    OLLAMA_BASE_URL      = inherited from engine
    LLM_EXTRACT_TIMEOUT  = seconds (default 60)
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

import requests

logger = logging.getLogger("extraction_v4.llm_extractor")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_EXTRACT_MODEL", "BeyondProcwise/AgentNick:extract")
LLM_TIMEOUT = int(os.getenv("LLM_EXTRACT_TIMEOUT", "60"))

# Cap text we send to the LLM — long contexts hurt latency more than accuracy
# for header-only extraction. The reference's flat_text is usually short.
MAX_TEXT_CHARS = 8000


# Required fields per doc-type (from YAML schemas). When any is missing after
# regex/layout extraction, we invoke the LLM. These match the field names
# used INSIDE the engine's raw dict (NOT the YAML field names — the mapper
# translates between them).
_REQUIRED_RAW_FIELDS = {
    "invoice": [
        "invoice_number", "invoice_date", "currency",
        # invoice_amount maps from "subtotal" or derived from total-tax
    ],
    "po": [
        "po_number", "supplier_name", "currency", "total_amount",
    ],
    "quote": [
        "quote_number", "quote_date", "currency", "total_amount",
    ],
}


_PROMPTS = {
    "invoice": """Extract these fields from the INVOICE text below. Return ONLY a JSON object.

INVOICE TEXT:
{text}

Return JSON with these keys (use null when value is genuinely absent — DO NOT GUESS):
- invoice_number: the invoice number/id (string, exactly as it appears)
- invoice_date: the invoice date in YYYY-MM-DD format
- due_date: due/payment date in YYYY-MM-DD format or null
- vendor_name: the supplier/vendor company name (NOT section headers like "Vendor Information")
- bill_to: the buyer/customer name or company
- currency: ISO code (USD, GBP, EUR, INR, etc.)
- subtotal: pre-tax amount (number, no currency symbol)
- tax: tax/VAT/GST amount (number, no currency symbol) or null
- total_amount: final total including tax (number)

CRITICAL RULES:
1. Use values EXACTLY as they appear in the document. Do not compute or infer.
2. Use null for fields you cannot find verbatim in the text.
3. Output ONLY the JSON object, no markdown, no commentary.""",

    "po": """Extract these fields from the PURCHASE ORDER text below. Return ONLY a JSON object.

PURCHASE ORDER TEXT:
{text}

Return JSON with these keys (use null when value is genuinely absent — DO NOT GUESS):
- po_number: the PO number/id (string, exactly as it appears)
- order_date: the PO/order date in YYYY-MM-DD format
- supplier_name: the vendor/supplier company name (NOT section headers like "Vendor Information")
- buyer_name: the buyer/client/customer name (organization or person)
- currency: ISO code (USD, GBP, EUR, INR, etc.)
- subtotal: pre-tax amount (number, no currency symbol) or null
- tax: tax/VAT/GST amount (number, no currency symbol) or null
- total_amount: final total amount payable (number)
- payment_terms: payment terms text or null

CRITICAL RULES:
1. Use values EXACTLY as they appear in the document. Do not compute or infer.
2. Use null for fields you cannot find verbatim in the text.
3. Output ONLY the JSON object, no markdown, no commentary.""",

    "quote": """Extract these fields from the QUOTE/QUOTATION text below. Return ONLY a JSON object.

QUOTE TEXT:
{text}

Return JSON with these keys (use null when value is genuinely absent — DO NOT GUESS):
- quote_number: the quote number/id (string, exactly as it appears)
- quote_date: the quote date in YYYY-MM-DD format
- validity_date: validity/expiry date in YYYY-MM-DD format or null
- supplier_name: the supplier/quoting company name (NOT section headers)
- buyer_name: the buyer/customer/client name (organization or person)
- currency: ISO code (USD, GBP, EUR, INR, etc.)
- subtotal: pre-tax amount (number) or null
- tax: tax/VAT/GST amount (number) or null
- total_amount: final total including tax (number)

CRITICAL RULES:
1. Use values EXACTLY as they appear in the document. Do not compute or infer.
2. Use null for fields you cannot find verbatim in the text.
3. Output ONLY the JSON object, no markdown, no commentary.""",
}


# Map LLM JSON keys → engine raw-dict keys (the engine uses different field
# names than the LLM prompt for clarity).
_LLM_TO_RAW_INVOICE = {
    "invoice_number": "invoice_number",
    "invoice_date": "invoice_date",
    "due_date": "due_date",
    "vendor_name": "vendor_name",
    "bill_to": "bill_to",
    "currency": "currency",
    "subtotal": "subtotal",
    "tax": "tax",
    "total_amount": "total_amount",
}
_LLM_TO_RAW_PO = {
    "po_number": "po_number",
    "order_date": "order_date",
    "supplier_name": "supplier_name",
    "buyer_name": "buyer_name",
    "currency": "currency",
    "subtotal": "subtotal",
    "tax": "tax",
    "total_amount": "total_amount",
    "payment_terms": "payment_terms",
}
_LLM_TO_RAW_QUOTE = {
    "quote_number": "quote_number",
    "quote_date": "quote_date",
    "validity_date": "validity_date",
    "supplier_name": "supplier_name",
    "buyer_name": "buyer_name",
    "currency": "currency",
    "subtotal": "subtotal",
    "tax": "tax",
    "total_amount": "total_amount",
}
_LLM_TO_RAW = {
    "invoice": _LLM_TO_RAW_INVOICE,
    "po": _LLM_TO_RAW_PO,
    "quote": _LLM_TO_RAW_QUOTE,
}


def _normalize_doc_type(dt: str) -> str:
    dt = (dt or "").lower().strip()
    if dt == "purchase_order":
        return "po"
    return dt


def _is_llm_enabled() -> bool:
    return (os.getenv("LLM_EXTRACT_FALLBACK", "on") or "on").lower() == "on"


def _has_required(raw: dict, doc_type: str) -> bool:
    """Return True iff all required engine raw-dict fields are non-empty."""
    needed = _REQUIRED_RAW_FIELDS.get(doc_type, [])
    for f in needed:
        v = raw.get(f)
        if v is None or (isinstance(v, str) and not v.strip()):
            return False
    return True


def _missing_fields(raw: dict, doc_type: str) -> list[str]:
    needed = _REQUIRED_RAW_FIELDS.get(doc_type, [])
    return [f for f in needed if not (raw.get(f) and str(raw.get(f)).strip())]


def _verify_substring(value: str, source_text_lower: str) -> bool:
    """True if the value (or a normalised form) appears in source text.

    Normalisations tried (in order):
      1. Exact lowercase substring
      2. Strip currency symbols + commas + spaces (for numbers)
      3. Strip non-alphanumeric (loose name matching)
    """
    if not value or not source_text_lower:
        return False
    v = str(value).strip().lower()
    if not v:
        return False
    if v in source_text_lower:
        return True
    # Try numeric: strip currency + commas + spaces, look for the digits
    v_digits = re.sub(r"[,$£€¥₹\s]", "", v)
    if v_digits and v_digits != v and v_digits in source_text_lower:
        return True
    # Loose: alphanumeric only (handles "ABC Corp." vs "ABC Corp")
    v_alpha = re.sub(r"[^a-z0-9]", "", v)
    src_alpha = re.sub(r"[^a-z0-9]", "", source_text_lower)
    if v_alpha and len(v_alpha) >= 3 and v_alpha in src_alpha:
        return True
    return False


def _call_ollama(prompt: str) -> dict:
    """POST to Ollama generate API with format=json. Returns parsed dict."""
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0,
                "num_predict": 1024,
            },
        },
        timeout=LLM_TIMEOUT,
    )
    response.raise_for_status()
    raw_output = (response.json().get("response") or "").strip()
    if not raw_output:
        return {}
    try:
        return json.loads(raw_output)
    except json.JSONDecodeError as exc:
        logger.warning(
            "LLM returned invalid JSON (%d chars): %s — first 300: %r",
            len(raw_output), exc, raw_output[:300],
        )
        # Try to salvage: find first {...} block
        m = re.search(r"\{.*?\}", raw_output, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        return {}


def llm_fill_missing_required(raw: dict, doc_type: str, source_text: str) -> dict:
    """Augment regex-extracted raw dict with LLM values for missing required fields.

    Args:
        raw: the engine's raw dict (mutable in-place; also returned).
        doc_type: 'invoice' | 'po' | 'quote' | 'purchase_order'.
        source_text: full document text (preferably the flat PDF text — what
            the regex extractor saw — so substring grounding is correct).

    Returns:
        Same `raw` dict, possibly with new keys filled in. Only required
        fields are touched. Values that don't appear in source_text are
        rejected (no hallucination).
    """
    if not _is_llm_enabled():
        return raw
    dt = _normalize_doc_type(doc_type)
    if dt not in _PROMPTS:
        return raw
    if _has_required(raw, dt):
        return raw  # nothing missing → skip LLM
    if not source_text or not source_text.strip():
        return raw

    missing = _missing_fields(raw, dt)
    logger.info(
        "LLM fallback: %s has missing required fields: %s",
        dt, missing,
    )
    text_for_prompt = source_text[:MAX_TEXT_CHARS]
    prompt = _PROMPTS[dt].format(text=text_for_prompt)
    try:
        llm_out = _call_ollama(prompt)
    except Exception as exc:
        logger.warning("LLM extraction failed: %s -- skipping fill", exc)
        return raw

    if not isinstance(llm_out, dict):
        logger.warning("LLM returned non-dict: %r", type(llm_out))
        return raw

    rename = _LLM_TO_RAW.get(dt, {})
    src_lower = source_text.lower()
    filled = []
    for llm_key, raw_key in rename.items():
        # Only fill fields that are currently empty
        existing = raw.get(raw_key)
        if existing and str(existing).strip():
            continue
        candidate = llm_out.get(llm_key)
        if candidate is None or (isinstance(candidate, str) and not candidate.strip()):
            continue
        candidate_str = str(candidate).strip()
        # Substring grounding — reject if not in source
        if not _verify_substring(candidate_str, src_lower):
            logger.info(
                "LLM-grounding REJECT %s=%r (not found in source text)",
                raw_key, candidate_str,
            )
            continue
        raw[raw_key] = candidate_str
        filled.append(f"{raw_key}={candidate_str!r}")

    if filled:
        logger.info("LLM fallback filled: %s", ", ".join(filled))
    else:
        logger.info("LLM fallback returned no usable (grounded) values")
    return raw
