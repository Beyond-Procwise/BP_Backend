"""LLM-based wholistic context-understanding layer.

Runs AFTER the regex / layout extraction pass, on every doc, to fill
any field the deterministic extractors couldn't capture. Uses a local
Ollama instance running BeyondProcwise/AgentNick:extract (Qwen2.5 7B
fine-tuned for structured extraction).

Design principles:

  1. Wholistic — the LLM is given the full document text and asked for
     the complete schema (all stg-mappable fields), not just a few
     "required" ones. This is the context-understanding layer the
     deterministic extractors don't have.

  2. Substring-grounded — every value the LLM returns must appear in
     the source text (after light normalisation: punctuation/whitespace
     stripping, currency-symbol stripping for numerics). Values that
     don't ground are dropped. No hallucination reaches the DB.

  3. Fill-only, never overwrite — if the regex / layout extractor
     already produced a value, the LLM does not replace it. The LLM
     is purely additive, filling the long tail that pattern matching
     misses (tax, address, expected-delivery, payment terms, etc.).

  4. Schema-driven — the prompt enumerates every field we want, with
     short descriptions, so the model knows what to look for.

Wiring: dispatch._run_hybrid_v4 calls llm_fill_all_extractable AFTER the
regex/layout pass returns its raw dict. Filled fields land in that dict
and the mapper sees no special case.

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

# Cap text we send to the LLM. Long contexts hurt latency more than they help
# accuracy for header-only extraction. The flat_text is usually short.
MAX_TEXT_CHARS = 12000


# All extractable fields per doc-type. The LLM is asked for every one of these.
# Keys are the LLM's JSON keys (also become engine raw-dict keys via _LLM_TO_RAW).
_FIELDS = {
    "invoice": [
        ("invoice_number", "invoice number / id, exactly as printed"),
        ("invoice_date", "invoice date in YYYY-MM-DD"),
        ("due_date", "due / payment date in YYYY-MM-DD"),
        ("vendor_name", "supplier / vendor company name (NOT a section header)"),
        ("vendor_address", "full vendor postal address as one string"),
        ("bill_to", "buyer / customer name (organisation or person)"),
        ("buyer_address", "full buyer / billing address"),
        ("po_number", "related PO number, if printed"),
        ("requested_by", "person who requested / authorised the order"),
        ("requested_date", "request date in YYYY-MM-DD"),
        ("payment_terms", "payment terms text (e.g. 'Net 30', '50% advance')"),
        ("currency", "ISO 4217 code (USD, GBP, EUR, INR, ...)"),
        ("subtotal", "pre-tax subtotal as a number, no currency symbol"),
        ("tax", "tax / VAT / GST amount as a number"),
        ("tax_percent", "tax rate as a percent number, e.g. 10 for 10%"),
        ("total_amount", "final total payable, including tax"),
    ],
    "po": [
        ("po_number", "purchase order number / id, exactly as printed"),
        ("order_date", "order date in YYYY-MM-DD"),
        ("expected_delivery_date", "expected / promised delivery date in YYYY-MM-DD"),
        ("supplier_name", "vendor / supplier company name (NOT a section header)"),
        ("supplier_address", "full supplier postal address as one string"),
        ("buyer_name", "buyer / client / customer name (organisation or person)"),
        ("buyer_address", "full buyer / billing address"),
        ("delivery_address", "ship-to / delivery address as one string"),
        ("requisition_id", "internal requisition / requisition reference number"),
        ("requested_by", "person who requested / authorised the order"),
        ("requested_date", "request date in YYYY-MM-DD"),
        ("payment_terms", "payment terms text"),
        ("currency", "ISO 4217 code"),
        ("subtotal", "pre-tax subtotal as a number"),
        ("tax", "tax / VAT / GST amount as a number"),
        ("tax_percent", "tax rate as a percent number"),
        ("total_amount", "grand total, including tax"),
        ("contract_id", "related contract / agreement reference"),
    ],
    "quote": [
        ("quote_number", "quote / quotation number, exactly as printed"),
        ("quote_date", "quote date in YYYY-MM-DD"),
        ("validity_date", "validity / expiry date in YYYY-MM-DD"),
        ("supplier_name", "supplier / quoting company name"),
        ("supplier_address", "full supplier address"),
        ("buyer_name", "buyer / customer / client name"),
        ("buyer_address", "full buyer address"),
        ("po_number", "related PO, if printed"),
        ("currency", "ISO 4217 code"),
        ("subtotal", "pre-tax subtotal as a number"),
        ("tax", "tax / VAT / GST amount as a number"),
        ("tax_percent", "tax rate as a percent number"),
        ("total_amount", "grand total, including tax"),
    ],
}


# Map LLM JSON keys -> engine raw-dict keys. The engine and the LLM use
# slightly different field names; this is the translation table.
_LLM_TO_RAW_INVOICE = {
    "invoice_number": "invoice_number",
    "invoice_date": "invoice_date",
    "due_date": "due_date",
    "vendor_name": "vendor_name",
    "vendor_address": "supplier_address",
    "bill_to": "bill_to",
    "buyer_address": "buyer_address",
    "po_number": "po_number",
    "requested_by": "requested_by",
    "requested_date": "requested_date",
    "payment_terms": "payment_terms",
    "currency": "currency",
    "subtotal": "subtotal",
    "tax": "tax",
    "tax_percent": "tax_percent",
    "total_amount": "total_amount",
}
_LLM_TO_RAW_PO = {
    "po_number": "po_number",
    "order_date": "order_date",
    "expected_delivery_date": "expected_delivery_date",
    "supplier_name": "supplier_name",
    "supplier_address": "supplier_address",
    "buyer_name": "buyer_name",
    "buyer_address": "buyer_address",
    "delivery_address": "delivery_address",
    "requisition_id": "requisition_id",
    "requested_by": "requested_by",
    "requested_date": "requested_date",
    "payment_terms": "payment_terms",
    "currency": "currency",
    "subtotal": "subtotal",
    "tax": "tax",
    "tax_percent": "tax_percent",
    "total_amount": "total_amount",
    "contract_id": "contract_id",
}
_LLM_TO_RAW_QUOTE = {
    "quote_number": "quote_number",
    "quote_date": "quote_date",
    "validity_date": "validity_date",
    "supplier_name": "supplier_name",
    "supplier_address": "supplier_address",
    "buyer_name": "buyer_name",
    "buyer_address": "buyer_address",
    "po_number": "po_number",
    "currency": "currency",
    "subtotal": "subtotal",
    "tax": "tax",
    "tax_percent": "tax_percent",
    "total_amount": "total_amount",
}
_LLM_TO_RAW = {
    "invoice": _LLM_TO_RAW_INVOICE,
    "po": _LLM_TO_RAW_PO,
    "quote": _LLM_TO_RAW_QUOTE,
}


_DOC_LABELS = {"invoice": "INVOICE", "po": "PURCHASE ORDER", "quote": "QUOTE / QUOTATION"}


def _build_prompt(doc_type: str, source_text: str) -> str:
    fields = _FIELDS[doc_type]
    label = _DOC_LABELS[doc_type]
    field_lines = "\n".join(f"- {k}: {desc}" for k, desc in fields)
    return f"""You are an expert at extracting structured data from business documents.
Read the {label} text below and return ONE JSON object containing every field
listed under SCHEMA. Use null when a field is genuinely absent — DO NOT GUESS.

CRITICAL RULES:
1. Use values EXACTLY as they appear in the document. Do not compute, infer, or rephrase.
2. Use null for any field you cannot find verbatim in the text.
3. Output ONLY the JSON object, no markdown, no commentary.

SCHEMA (return a JSON object with all of these keys):
{field_lines}

{label} TEXT:
{source_text}

JSON:"""


def _normalize_doc_type(dt: str) -> str:
    dt = (dt or "").lower().strip()
    if dt == "purchase_order":
        return "po"
    return dt


def _is_llm_enabled() -> bool:
    return (os.getenv("LLM_EXTRACT_FALLBACK", "on") or "on").lower() == "on"


def _empty_raw_fields(raw: dict, doc_type: str) -> list[str]:
    """Return the engine-raw keys whose value is currently empty."""
    rename = _LLM_TO_RAW.get(doc_type, {})
    out = []
    for raw_key in set(rename.values()):
        v = raw.get(raw_key)
        if v is None or (isinstance(v, str) and not v.strip()):
            out.append(raw_key)
    return out


_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_NUMERIC_RE = re.compile(r"^-?\d+(?:\.\d+)?$")


def _ground_date(value: str, source_text_lower: str) -> bool:
    """Date grounding: parse both sides and re-encode in multiple human formats.

    The LLM returns YYYY-MM-DD; the source typically prints 'August 5, 2025',
    '5 August 2025', '08/05/2025', etc. Try every common encoding.
    """
    if not _ISO_DATE_RE.match(value.strip()):
        return False
    try:
        from datetime import datetime
        dt = datetime.strptime(value.strip(), "%Y-%m-%d")
    except ValueError:
        return False
    # Try a battery of common formats. Source-text comparisons are case-insensitive.
    candidates = [
        dt.strftime("%B %-d, %Y").lower(),       # August 5, 2025
        dt.strftime("%B %d, %Y").lower(),        # August 05, 2025
        dt.strftime("%-d %B %Y").lower(),        # 5 August 2025
        dt.strftime("%d %B %Y").lower(),         # 05 August 2025
        dt.strftime("%-d %b %Y").lower(),        # 5 Aug 2025
        dt.strftime("%-d-%b-%Y").lower(),        # 5-Aug-2025
        dt.strftime("%-d-%b-%y").lower(),        # 5-Aug-25
        dt.strftime("%B %-d %Y").lower(),        # August 5 2025
        dt.strftime("%-m/%-d/%Y").lower(),       # 8/5/2025
        dt.strftime("%-d/%-m/%Y").lower(),       # 5/8/2025
        dt.strftime("%m/%d/%Y").lower(),         # 08/05/2025
        dt.strftime("%d/%m/%Y").lower(),         # 05/08/2025
        dt.strftime("%-d-%-m-%Y").lower(),       # 5-8-2025
        dt.strftime("%Y/%m/%d").lower(),         # 2025/08/05
        value.lower(),                           # 2025-08-05 (ISO itself)
    ]
    return any(c in source_text_lower for c in candidates)


def _ground_percent_or_number(value: str, source_text_lower: str) -> bool:
    """Numeric grounding: '10.0' grounds to '10%', '10', '10.00', '10.0', etc."""
    v = value.strip()
    if not _NUMERIC_RE.match(v):
        return False
    try:
        n = float(v)
    except ValueError:
        return False
    # Try integer form, decimal forms, and with/without trailing % sign
    int_form = str(int(n)) if n == int(n) else None
    forms = [v, f"{n:.0f}", f"{n:.1f}", f"{n:.2f}"]
    if int_form:
        forms += [int_form, int_form + "%", "(" + int_form + "%)"]
    forms += [v + "%", "(" + v + "%)"]
    return any(f.lower() in source_text_lower for f in forms)


def _field_specific_reject(field: str, value: str, source_text: str) -> bool:
    """Domain-specific guards: reject values that ground in source but under
    the wrong label (e.g. po_number captured from an 'Invoice Number' field).

    Returns True if the value should be rejected for this field.
    """
    if not value:
        return False
    src = source_text or ""
    val = value.strip()
    if field == "po_number":
        # Reject if the value appears immediately after an 'Invoice' label
        pat = re.compile(
            r"invoice\s*(?:number|no\.?|#)\s*[:\s]\s*" + re.escape(val) + r"\b",
            re.IGNORECASE,
        )
        if pat.search(src):
            return True
    elif field == "invoice_number":
        # Symmetric: reject if value appears under a 'PO Number' label
        pat = re.compile(
            r"(?:purchase\s*order|p\.?o\.?)\s*(?:number|no\.?|#)\s*[:\s]\s*"
            + re.escape(val) + r"\b",
            re.IGNORECASE,
        )
        if pat.search(src):
            return True
    elif field == "supplier_name":
        # Reject obvious section labels masquerading as a name
        bad = {
            "vendor information", "vendor name", "supplier information",
            "supplier name", "company name", "client information",
            "ordered by", "bill to", "ship to", "from", "to",
        }
        if val.lower().strip(":.,") in bad:
            return True
    return False


def _verify_substring(value: str, source_text_lower: str) -> bool:
    """True if value (or a normalised form) appears in source text.

    Tries: exact substring → numeric strip → alphanum strip → token coverage
    (for addresses) → date re-encoding → percent / number re-encoding.
    """
    if not value or not source_text_lower:
        return False
    v = str(value).strip().lower()
    if not v:
        return False
    if v in source_text_lower:
        return True
    # Numeric: strip currency + commas + spaces, look for digits
    v_digits = re.sub(r"[,$£€¥₹\s]", "", v)
    if v_digits and v_digits != v and v_digits in source_text_lower:
        return True
    # Alphanumeric-only (for names with punctuation differences)
    v_alpha = re.sub(r"[^a-z0-9]", "", v)
    src_alpha = re.sub(r"[^a-z0-9]", "", source_text_lower)
    if v_alpha and len(v_alpha) >= 3 and v_alpha in src_alpha:
        return True
    # Date re-encoding (LLM returns ISO; source uses human formats)
    if _ground_date(value, source_text_lower):
        return True
    # Numeric / percent re-encoding ('10.0' grounds to '10%')
    if _ground_percent_or_number(value, source_text_lower):
        return True
    # Sliding-window token coverage for addresses / multi-line values
    tokens = [t for t in re.findall(r"[a-z0-9]+", v) if len(t) >= 3]
    if len(tokens) >= 2:
        present = sum(1 for t in tokens if t in src_alpha)
        if present / len(tokens) >= 0.75:
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
                "num_predict": 2048,
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
        m = re.search(r"\{.*\}", raw_output, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        return {}


def llm_fill_all_extractable(raw: dict, doc_type: str, source_text: str) -> dict:
    """Wholistic context-understanding pass. Fill ANY empty extractable field.

    Args:
        raw: engine raw dict (mutated in place; also returned).
        doc_type: 'invoice' | 'po' | 'purchase_order' | 'quote'.
        source_text: full document text (for substring grounding).

    Returns:
        Same `raw` dict, possibly with newly filled keys. Existing values
        are never overwritten. Values that don't ground in source_text are
        rejected (no hallucination).
    """
    if not _is_llm_enabled():
        return raw
    dt = _normalize_doc_type(doc_type)
    if dt not in _FIELDS:
        return raw
    if not source_text or not source_text.strip():
        return raw

    empty_before = _empty_raw_fields(raw, dt)
    if not empty_before:
        return raw  # everything filled deterministically — skip LLM entirely

    logger.info(
        "LLM context layer: %s has %d empty extractable fields: %s",
        dt, len(empty_before), empty_before,
    )
    text_for_prompt = source_text[:MAX_TEXT_CHARS]
    prompt = _build_prompt(dt, text_for_prompt)
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
    rejected_grounding = []
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
            rejected_grounding.append(f"{raw_key}={candidate_str!r}")
            continue
        # Field-specific rejection (e.g. po_number captured from Invoice #)
        if _field_specific_reject(raw_key, candidate_str, source_text):
            logger.info(
                "LLM context layer REJECT %s=%r (wrong-label leak)",
                raw_key, candidate_str,
            )
            continue
        raw[raw_key] = candidate_str
        filled.append(f"{raw_key}={candidate_str!r}")

    if filled:
        logger.info("LLM context layer filled %d field(s): %s", len(filled), ", ".join(filled))
    if rejected_grounding:
        logger.info(
            "LLM context layer rejected %d ungrounded value(s): %s",
            len(rejected_grounding), ", ".join(rejected_grounding[:5]),
        )
    return raw


# Back-compat alias. The old name is kept so callers that import the prior
# function don't break. Behaviour matches the new wholistic implementation.
llm_fill_missing_required = llm_fill_all_extractable
