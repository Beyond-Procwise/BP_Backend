"""Context understanding layer — synthesize a clean stg row from the
raw extraction artifacts.

Sits between `_raw` and `_stg` in the renovation pipeline. The raw row
already has whatever the L1 regex / L2 NER / L3 grounded judge produced,
plus `parser_snapshot.full_text` (the entire document text). The problem
those layers can't solve is *context*: which company in the document is
the supplier vs the buyer? Is "QTY" a real org or a column header? Is
"Redkiln Way Horsham" a person or an address?

This module makes ONE Qwen2.5-VL call per document. The prompt:
  - hands Qwen the full text (capped at 16k chars)
  - lists the schema fields with their procurement meanings
  - includes the existing regex/NER candidates as hints
  - demands a single JSON object back

Anti-hallucination contract — same as L3 grounded_last_resort:
  - every STRING value Qwen emits must be a literal substring of
    full_text (after letter-spacing collapse + whitespace normalisation).
    Non-grounded strings are dropped to NULL.
  - DATE / MONEY / DECIMAL values are passed through the existing
    parsers (`parse_date`, `parse_amount`). Bind failures → NULL.
  - Derived fields (`exchange_rate_to_usd`, `converted_amount_usd`) are
    computed locally from `currency` × the relevant total. The FX table
    is intentionally inline — for production swap in a daily feed.

Failure modes:
  - Qwen unavailable / OOM / parse error → return the input row
    unchanged. The pipeline still completes; the row promotes with
    whatever L1/L2 produced (better than blocking the pipeline).
  - No `full_text` → return the input row unchanged.

Called from `promotion.promote()` BEFORE the INSERT INTO _stg.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

log = logging.getLogger(__name__)

# Cap on doc text included in the prompt. Real procurement docs come in
# well under this — 4 KB typical, 16 KB worst case.
MAX_DOC_TEXT_CHARS = 16000
# Cap on prompt response size. Sized for the BeyondProcwise/AgentNick
# thinking-capable model: the reasoning tokens consume the budget before
# the JSON emission, so we need headroom for both. 4096 is enough for
# the widest PO schema (~23 fields) with substantial thinking.
MAX_RESPONSE_TOKENS = 4096

# FX rates to USD — approximate mid-market as of 2026-05.
# Swap with a daily FX feed (Bloomberg / ECB / openexchangerates) when
# accounting needs daily precision. For procurement reporting these are
# good to ~2% which is acceptable.
_FX_TO_USD: dict[str, float] = {
    "USD": 1.00,
    "GBP": 1.27,
    "EUR": 1.08,
    "JPY": 0.0067,
    "AUD": 0.66,
    "CAD": 0.74,
    "INR": 0.012,
    "SGD": 0.74,
    "HKD": 0.13,
    "CHF": 1.15,
    "NZD": 0.61,
}


# (column_name, type, prompt-description) per doc_type. Types:
#   "str"     → must be a substring of full_text (grounded)
#   "date"    → parsed via parse_date → ISO YYYY-MM-DD
#   "money"   → parsed via parse_amount → float
#   "decimal" → parsed via parse_amount or float()
_INVOICE_FIELDS: list[tuple[str, str, str]] = [
    ("invoice_id", "str",
     "Invoice identifier as it appears in the document (e.g. 'INV132548', 'INV-2025-056', '039467'). Output the raw token — DO NOT add or remove a prefix."),
    ("po_id", "str",
     "Related Purchase Order number if the invoice mentions one (look for 'PO NO:', 'P.O. Number:', 'Purchase Order #')."),
    ("supplier_name", "str",
     "The COMPANY ISSUING the invoice (the supplier/seller). Typically appears in the document HEADING/MASTHEAD (e.g. '## INVOICE Duncan LLC' → 'Duncan LLC') or under a 'From:' / 'Vendor:' label. NOT the buyer."),
    ("buyer_id", "str",
     "The COMPANY RECEIVING the invoice. Found under 'Invoice To:', 'Bill To:', 'Customer:', or in a separate addressee block. If only an address appears with no company name, output null."),
    ("requested_by", "str",
     "PERSON NAME (not company) explicitly identified as the requester under 'Requested By:', 'Raised By:', 'Ordered By:'. If only a company or no requester label exists, output null."),
    ("requested_date", "date", "Date the request was raised — ISO YYYY-MM-DD."),
    ("invoice_date", "date",
     "Date the invoice was issued — ISO YYYY-MM-DD. Look for 'Invoice Date:', 'Date Issued:', or the date in the masthead."),
    ("due_date", "date", "Payment due date — ISO YYYY-MM-DD. Look for 'Due Date:', 'Payment Due:', 'Pay By:'."),
    ("payment_terms", "str",
     "Payment-terms text near 'Payment Terms:', 'Terms:', or 'Net X days' (e.g. 'Net 30 days', 'Net 14', '30 days from invoice date')."),
    ("currency", "str",
     "ISO 4217 code (GBP, USD, EUR, JPY, AUD, CAD, INR, SGD, HKD, CHF, NZD). Derive from currency symbols if no ISO code is present: £→GBP, $→USD, €→EUR, ¥→JPY."),
    ("invoice_amount", "money",
     "Pre-tax subtotal — the 'Subtotal' / 'Net Amount' / 'Sub-Total' value. Decimal number only (e.g. 2131.05)."),
    ("tax_percent", "decimal", "Tax rate percentage (e.g. 20 for 20%, 5 for 5%)."),
    ("tax_amount", "money", "Tax amount value next to 'Tax:', 'VAT:', 'GST:'. Decimal number."),
    ("invoice_total_incl_tax", "money",
     "Final payable amount AFTER tax — 'Grand Total', 'Total Amount Due', 'Total Payable', 'Total'. Decimal number."),
    ("country", "str",
     "Country of the supplier or buyer address. Derive from the address (e.g. 'RH13 5QH' → 'United Kingdom'). Output the full country name."),
    ("region", "str",
     "State / County / Province from the address. UK: county name (e.g. 'West Sussex'). US: state name."),
]

_PO_FIELDS: list[tuple[str, str, str]] = [
    ("po_id", "str", "PO identifier (e.g. '502001', 'PO-2025-001'). Output the raw token."),
    ("supplier_name", "str",
     "The COMPANY THIS PO IS ADDRESSED TO — the supplier. In POs the addressee is the supplier. Look under 'Recipient:', 'Send To:', 'Ship To:', 'Sold To:'. NOT the issuer."),
    ("buyer_id", "str", "The COMPANY ISSUING the PO (in the masthead / 'From:'). If absent, output null."),
    ("requested_by", "str", "Person who raised the PO under 'Requested By:' / 'Raised By:'. PERSON NAME only — not company or address."),
    ("requested_date", "date", "Date the requisition was raised — ISO YYYY-MM-DD."),
    ("order_date", "date", "PO issue date — ISO YYYY-MM-DD. Look for 'PO Date:', 'Order Date:', 'Date Issued:'."),
    ("expected_delivery_date", "date", "Expected/required delivery date — ISO YYYY-MM-DD."),
    ("payment_terms", "str", "Payment terms text."),
    ("currency", "str", "ISO 4217 code (GBP/USD/EUR/etc.)."),
    ("total_amount", "money", "Pre-tax subtotal."),
    ("tax_percent", "decimal", "Tax rate percentage."),
    ("tax_amount", "money", "Tax amount."),
    ("total_amount_incl_tax", "money", "Grand total / final payable amount including tax."),
    ("ship_to_country", "str", "Delivery country (full name)."),
    ("delivery_region", "str", "Delivery state/county."),
    ("delivery_address_line1", "str", "First line of the delivery address (street)."),
    ("delivery_address_line2", "str", "Second line of the delivery address if present."),
    ("delivery_city", "str", "Delivery city/town."),
    ("postal_code", "str", "Delivery postcode/ZIP."),
]

_QUOTE_FIELDS: list[tuple[str, str, str]] = [
    ("quote_id", "str", "Quote identifier (e.g. 'QUT30746', 'QUT-2025-051'). Output the raw token."),
    ("supplier_id", "str",
     "The COMPANY ISSUING the quote — the supplier. Usually in the masthead/heading. Output the company NAME — it will be resolved to an internal supplier ID downstream."),
    ("buyer_id", "str", "The COMPANY RECEIVING the quote (in 'Bill To:', 'Customer:', 'Quoted To:')."),
    ("supplier_address", "str", "Supplier's full postal address (single-line concatenation if multi-line)."),
    ("buyer_address", "str", "Buyer's full postal address."),
    ("quote_date", "date", "Quote issue date — ISO YYYY-MM-DD."),
    ("validity_date", "date", "Quote validity / expiry date — ISO YYYY-MM-DD."),
    ("currency", "str", "ISO 4217 code."),
    ("total_amount", "money", "Pre-tax subtotal."),
    ("tax_percent", "decimal", "Tax rate percentage."),
    ("tax_amount", "money", "Tax amount."),
    ("total_amount_incl_tax", "money", "Grand total including tax."),
    ("po_id", "str", "Related PO number if mentioned."),
    ("country", "str", "Country derived from address."),
    ("region", "str", "State/county derived from address."),
]

_FIELD_DEFS: dict[str, list[tuple[str, str, str]]] = {
    "invoice": _INVOICE_FIELDS,
    "purchase_order": _PO_FIELDS,
    "quote": _QUOTE_FIELDS,
}


# Filename pattern: most procurement-team uploads name the file as
# ``<SUPPLIER NAME> <PREFIX><DOC_ID> [for <REF_PREFIX><REF_ID>].<ext>``
# Examples:
#   "DUNCAN INV132666 for PO526702.pdf"
#   "DESIGN HOUSE AGENCY INV DHA-2025-145 for PO438295.pdf"
#   "DIXON, REYNOLDS ETC INV600820 for PO507222.pdf"
#   "WADE QUT30789.pdf"
# The supplier portion (everything before the first INV/PO/QUT prefix)
# is a very reliable signal because the uploader literally typed it.
# Supplier portion is non-greedy and stops at the first INV/PO/QUT prefix.
# The prefix may be followed by either digits directly ("INV132666"), or
# a hyphen+token ("INV-2025-058"), or a space+token ("INV DHA-2025-145").
_FILENAME_HEAD_RE = re.compile(
    r"""
    ^\s*(?P<supplier>[A-Z][A-Z0-9, '&\.\-]*?)
    \s+(?P<prefix>INV|PO|QUT|QTE|RFQ)
    (?P<sep>[\s\-_]?)
    (?P<doc_id>[A-Z0-9][A-Z0-9\-]*[A-Z0-9]|[A-Z0-9])
    """,
    re.IGNORECASE | re.VERBOSE,
)

_FILENAME_REF_RE = re.compile(
    r"\bfor\s+(?P<prefix>INV|PO|QUT|QTE|RFQ)[\s\-_]*(?P<id>[A-Z0-9][A-Z0-9\-]*[A-Z0-9])",
    re.IGNORECASE,
)


def parse_filename_hints(file_path: str | None) -> dict[str, str]:
    """Extract supplier + doc-ID + related-ID hints from the source filename.

    Returns at most: ``{supplier, invoice_id, po_id, quote_id}``. Empty dict
    when the filename doesn't match the standard pattern.

    The supplier value is the **uploader-typed** supplier label — usually
    a short form (e.g. "DUNCAN" for "Duncan LLC", "AQUARIUS" for
    "Aquarius Marketing Ltd"). Don't put it directly into supplier_name —
    use it as a HINT for Qwen to anchor the supplier identification.
    """
    if not file_path:
        return {}
    name = file_path.rsplit("/", 1)[-1] if "/" in file_path else file_path
    name = re.sub(r"\.(pdf|jpe?g|png|docx?|tiff?)$", "", name, flags=re.IGNORECASE).strip()
    m = _FILENAME_HEAD_RE.match(name)
    if not m:
        return {}
    supplier = m.group("supplier").strip(" .,-")
    # Strip trailing digits (e.g. "GOMEZ, GOOD ETC5") and common trailer words.
    supplier = re.sub(r"\s*\b(ETC\d*|LTD|INC|LLC|GROUP|CO|COMPANY)\b\s*$", "", supplier, flags=re.IGNORECASE).strip()
    supplier = re.sub(r"\s*\d+\s*$", "", supplier).strip()
    prefix = m.group("prefix").upper()
    sep = m.group("sep") or ""
    # Hyphen separator is part of the canonical token form (e.g.
    # "INV-2025-056"). Empty separator means the prefix and id are
    # naturally one token (e.g. "INV132666"). Whitespace/underscore
    # indicate the uploader typed "INV DHA-..." or "INV_2025-..." as
    # TWO separate tokens — the doc body in those cases usually has
    # just the doc_id without the prefix, so we'll surface the bare
    # form as the canonical hint to avoid steering the model toward
    # a mangled concatenation.
    sep_kept = sep if sep == "-" else ""
    prefer_bare_id = sep in (" ", "_")
    doc_id = m.group("doc_id")
    out: dict[str, str] = {"supplier": supplier}
    # For every ID, surface BOTH forms — with prefix ("QUT136700",
    # "INV-2025-056") preserving the original separator AND the
    # bare numeric/alphanumeric tail ("136700", "2025-056"). Document
    # layouts often include only one of the two; supplying both lets the
    # grounding check accept whichever the doc actually contains.
    bare_id = re.sub(r"^(?:INV|PO|QUT|QTE|RFQ)[\-_]?", "", doc_id, flags=re.IGNORECASE)
    def _prefixed(p: str, did: str) -> str:
        # Use the doc_id as-is if it already starts with the prefix;
        # otherwise prepend the prefix preserving the original separator
        # ("INV" + "-" + "2025-056" → "INV-2025-056").
        # When the filename separator was a space/underscore, the prefix
        # is a TYPE marker and the doc_id is the real ID — return doc_id
        # bare so we don't synthesise a token the doc body never contains.
        if prefer_bare_id:
            return did
        return did if did.upper().startswith(p) else f"{p}{sep_kept}{did}"

    if prefix == "INV":
        out["invoice_id"] = _prefixed("INV", doc_id)
        out["invoice_id_alt"] = bare_id
    elif prefix == "PO":
        out["po_id"] = _prefixed("PO", doc_id)
        out["po_id_alt"] = bare_id
    elif prefix in ("QUT", "QTE"):
        out["quote_id"] = _prefixed(prefix, doc_id)
        out["quote_id_alt"] = bare_id
    # Optional secondary reference after "for".
    m2 = _FILENAME_REF_RE.search(name[m.end():])
    if m2:
        ref_prefix = m2.group("prefix").upper()
        ref_id = m2.group("id")
        if ref_prefix == "PO" and "po_id" not in out:
            out["po_id"] = ref_id
        elif ref_prefix in ("QUT", "QTE") and "quote_id" not in out:
            out["quote_id"] = ref_id
        elif ref_prefix == "INV" and "invoice_id" not in out:
            out["invoice_id"] = ref_id
    return out


def synthesize(
    doc_type: str,
    full_text: str,
    raw_candidates: dict[str, Any],
    file_path: str | None = None,
) -> dict[str, Any]:
    """Produce a clean structured row from full_text + L1/L2/L3 candidates.

    Returns the synthesized dict. Caller (`promotion.promote`) overlays
    these values onto the `_raw` row before INSERT INTO `_stg`.

    Never raises — on any failure returns `raw_candidates` unchanged so
    the pipeline stays unblocked.

    ``file_path`` (when supplied) is parsed for supplier-name and ID hints.
    Those are passed to the Qwen prompt as authoritative signals because
    they were typed by the uploader and almost always reflect the correct
    supplier/document — far more reliable than parsing the document body
    when the layout is unusual.
    """
    fields = _FIELD_DEFS.get(doc_type)
    if not fields:
        log.warning("context_layer: no field definitions for doc_type=%r", doc_type)
        return raw_candidates
    if not (full_text and full_text.strip()):
        log.debug("context_layer: empty full_text for doc_type=%s — skipping", doc_type)
        return raw_candidates

    filename_hints = parse_filename_hints(file_path)
    prompt = _build_prompt(doc_type, full_text, fields, raw_candidates, filename_hints)
    raw = _call_llm(prompt)
    if not raw:
        log.warning("context_layer: LLM returned nothing for doc_type=%s", doc_type)
        return raw_candidates

    parsed = _parse_json(raw)
    if parsed is None:
        log.warning("context_layer: failed to parse JSON output (head=%r)", raw[:200])
        return raw_candidates

    cleaned = _validate_and_bind(parsed, full_text, fields)
    cleaned = _compute_derived(cleaned)
    log.info(
        "context_layer: doc_type=%s synthesized %d fields (non-null)",
        doc_type, sum(1 for v in cleaned.values() if v is not None),
    )
    return cleaned


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_PROMPT_HEADER = """You are a structured-data extractor for procurement documents.

Read the DOCUMENT TEXT below and output a single JSON object with the requested fields. The candidate hints come from regex/NER and are frequently WRONG — use the full document text as ground truth.

DOMAIN CONTEXT (very important — read carefully):
- The procurement system runs from "Assurity Ltd" (a UK company at 10 Redkiln Way, Horsham, West Sussex, RH13 5QH).
- Assurity Ltd is ALWAYS our BUYER (never the supplier). On invoices it is the BILL TO / Invoice To party; on purchase orders it is the issuer (buyer_id) and the addressee block contains the supplier; on quotes it is the recipient.
- The supplier is the OTHER party — the company whose name appears in the document masthead/heading on invoices and quotes, or under "Recipient:" / "Send To:" on purchase orders.
- If "Assurity Ltd" appears in the document, it is the buyer, NOT the supplier.
"""

_PROMPT_RULES = """OUTPUT RULES:
1. Output ONLY a single JSON object. No prose. No markdown code fences.
2. Every STRING value MUST be a literal substring of the DOCUMENT TEXT. If you cannot find a string value as a verbatim substring, output null.
3. For dates, output ISO format YYYY-MM-DD. You may CONVERT the format you see — output the date as ISO, not the raw string.
4. For money/decimal fields, output a JSON NUMBER without currency symbol or commas (e.g. 2131.05). Do NOT wrap numbers in quotes.
5. For currency, output a 3-letter ISO code (GBP, USD, EUR, JPY, AUD, CAD, INR, SGD, HKD, CHF, NZD). Derive from the symbol if no code is present: £→GBP $→USD €→EUR ¥→JPY.
6. If a field is NOT present or unclear, output null. DO NOT GUESS, DO NOT FABRICATE.
7. CRITICAL: distinguish SUPPLIER (issuer) from BUYER (recipient). The supplier appears in the masthead/heading of an invoice or quote and in the addressee block of a PO. The buyer is the other one. Never swap them.
8. Reject layout noise. "QTY", "TOTAL", "DESCRIPTION", "SUBTOTAL", street addresses, postcodes, and product/service names are NOT supplier names, buyer names, or person names.

TAX + TOTAL RECOVERY (very important — many invoices have these but in awkward layouts):
- A typical procurement money block has THREE values in order: Subtotal / Tax / Grand Total. They are NOT always labelled side by side — sometimes the labels are in one column and the numbers stack below in a second column.
- Pattern A — labelled rows:  "Subtotal: £8,333  Tax (20%): £1,666.60  Total: £9,999.60"
- Pattern B — stacked column: rows of just numbers under a header, where the FIRST number is the subtotal, the SECOND is the tax amount, the THIRD is the grand total. Verify with `subtotal + tax ≈ grand_total`.
- Pattern C — single "Total Amount Due" with one number: that's the GRAND TOTAL (incl. tax). If the doc shows ONLY this single number with no separate tax/subtotal line, set invoice_amount = grand_total - tax (if tax_amount is derivable from "Tax (X%)" anywhere), otherwise leave invoice_amount = grand_total and tax_amount = null.
- UK invoices very commonly use 20% VAT. If you can see "Tax (20%)" anywhere, even without an explicit amount, you can pair it with the subtotal in the SAME block. DO NOT compute the tax — only report the value as it appears in the document.
- Never put the GRAND TOTAL into invoice_amount/total_amount (those are pre-tax subtotals). The grand total goes into *_total_incl_tax.

COUNTRY + REGION DERIVATION (from any visible address):
- UK postcode like "RH13 5QH", "M17 1AB", "EC2A 3NW", "B7 4AX", "LS10 1QP",
  "SW4 7GH", "W1W 5QZ" → country = "United Kingdom".
- US ZIP like "10001" or "10001-1234" → country = "United States".
- For region: take the COUNTY/STATE token from the address. The county
  is sometimes CONCATENATED to the postcode in docling output
  (e.g. "SussexRH135QH") — still output it as the canonical name
  ("West Sussex"). Common UK county clues:
    "RH" postcode prefix → West Sussex
    "B" prefix (Birmingham) → West Midlands
    "M" prefix (Manchester) → Greater Manchester
    "LS" prefix (Leeds) → West Yorkshire
    "EC", "SW", "W1", etc. → Greater London
- For PURCHASE ORDERS specifically: ship_to_country is the country of
  the DELIVERY address (delivery_address_line1 + delivery_city +
  postal_code). If you populate ANY of the delivery_* fields with a UK
  postcode, ship_to_country MUST be 'United Kingdom'. Same for US.
- Only populate country/region/ship_to_country if an address IS visible.
  Don't invent these from filename.

"""


def _build_prompt(
    doc_type: str,
    full_text: str,
    fields: list[tuple[str, str, str]],
    candidates: dict[str, Any],
    filename_hints: dict[str, str] | None = None,
) -> str:
    capped = full_text[:MAX_DOC_TEXT_CHARS]
    field_lines = "\n".join(f'  "{n}" ({t}): {d}' for n, t, d in fields)
    hint_set = {f[0] for f in fields}
    # Money fields are EXCLUDED from the candidate hints. L1 regex picks
    # for these are unreliable on column-split layouts (a line_amount
    # gets misread as the subtotal, the model then computes a wrong tax
    # off it). Without the hint the model reads full_text fresh and
    # generally finds the right Subtotal/Tax/Total triple (verified
    # against WADE QUT30789 and TECHNOVA QUT103069 where hint-free
    # synthesis got the totals right, hint-augmented synthesis got them
    # catastrophically wrong).
    _money_field_names = {
        n for (n, t, _) in fields if t in ("money", "decimal")
    }
    hints = {
        k: v for k, v in (candidates or {}).items()
        if v is not None and k in hint_set and k not in _money_field_names
    }
    hint_lines = (
        "\n".join(f"  {k}: {v!r}" for k, v in hints.items())
        if hints else "  (no candidate hints)"
    )
    fn = filename_hints or {}
    if fn:
        fn_lines = (
            f"  supplier (uploader-typed): {fn.get('supplier', '(none)')!r}\n"
            + "".join(
                f"  {k}: {v!r}\n"
                for k, v in fn.items()
                if k != "supplier" and v
            )
        ).rstrip()
        fn_block = (
            "FILENAME HINTS (very reliable — typed by the uploader, almost always correct):\n"
            f"{fn_lines}\n"
            "USE THESE AGGRESSIVELY:\n"
            "1. If the filename supplier (e.g. 'DUNCAN', 'AQUARIUS', 'DIXON, REYNOLDS') appears\n"
            "   anywhere in the document text (as short form, ALL CAPS, title case, OR as the\n"
            "   first word of a longer registered name like 'Duncan LLC' or 'Aquarius Marketing\n"
            "   Ltd'), then the supplier_name / supplier_id IS that company. Output the form\n"
            "   that appears in the document (e.g. 'Duncan LLC' if the doc says 'Duncan LLC',\n"
            "   or 'DUNCAN' if only the all-caps form is present). DO NOT return null for\n"
            "   supplier when the filename hint is clearly anchored in the document.\n"
            "2. For invoice_id / po_id / quote_id: BOTH the prefixed form ('QUT136700',\n"
            "   'INV-2025-056') AND the bare numeric form ('136700', '2025-056') are provided\n"
            "   as hints. Output the EXACT verbatim token AS IT APPEARS in the document\n"
            "   body — preserve ALL hyphens, dots, and separators ('INV-2025-056' is NOT\n"
            "   the same as 'INV2025-056'). If the doc shows only the bare digits, output\n"
            "   the bare digits; if the doc shows the prefixed form, output that. Never\n"
            "   strip or add separators from what the document literally contains.\n"
            "3. Only override a filename hint if the document text explicitly and\n"
            "   unambiguously contradicts it (rare).\n\n"
        )
    else:
        fn_block = ""
    return (
        f"{_PROMPT_HEADER}\n"
        f"DOCUMENT TYPE: {doc_type}\n\n"
        f"{fn_block}"
        f"DOCUMENT TEXT:\n\"\"\"\n{capped}\n\"\"\"\n\n"
        f"FIELDS TO EXTRACT:\n{field_lines}\n\n"
        f"CANDIDATE HINTS (from regex/NER — may be wrong):\n{hint_lines}\n\n"
        f"{_PROMPT_RULES}"
        f"JSON OUTPUT:"
    )


# ---------------------------------------------------------------------------
# LLM call (Qwen2.5-VL inside procwise; Ollama fallback)
# ---------------------------------------------------------------------------


# BeyondProcwise/AgentNick — the procurement fine-tuned model. The
# :extract tag (7B Q8) is the live-extraction default — it fits in the
# ~12 GiB of GPU memory free alongside the procwise service. The 30B
# :latest tag is reserved for nightly fine-tuning (procwise downtime
# 1am-9am IST frees GPU for the larger model). Override with the
# PROCWISE_AGENTNICK_MODEL env var.
import os as _os
_LLM_MODEL = _os.getenv("PROCWISE_AGENTNICK_MODEL", "BeyondProcwise/AgentNick:extract")


def _call_llm(prompt: str) -> str | None:
    """Run the prompt through Ollama BeyondProcwise/AgentNick.

    Temperature 0 (deterministic), num_predict sized for the widest schema,
    short retry budget. A failed call returns None and the document blocks
    with a missing-required discrepancy — no fabrication, no silent partial.
    """
    try:
        from src.services.ollama_client import ollama_generate
        return ollama_generate(
            prompt,
            model=_LLM_MODEL,
            num_predict=MAX_RESPONSE_TOKENS,
            temperature=0.0,
            retries=2,
            timeout=120,
        )
    except Exception as exc:  # noqa: BLE001
        log.error("context_layer: ollama call failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Output parsing + validation
# ---------------------------------------------------------------------------


def _parse_json(raw: str) -> dict | None:
    """Locate and parse the best {…} block in the LLM response.

    Scans for ALL balanced `{...}` candidates (top-level only — not nested)
    and returns the one with the most fields that parses cleanly. This
    handles the case where the model echoes the prompt's literal placeholder
    `{"value": <string or null>, ...}` (invalid JSON) before emitting the
    real object, OR repeats the response twice. Falls back to a best-effort
    truncated-tail recovery if no complete block parses.
    """
    if not raw:
        return None
    text = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
    text = re.sub(r"```\s*$", "", text.strip())

    # Enumerate every balanced top-level {...} block.
    candidates: list[tuple[int, str]] = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        depth = 0
        in_str = False
        esc = False
        end = -1
        for j in range(i, n):
            c = text[j]
            if esc:
                esc = False
                continue
            if c == "\\":
                esc = True
                continue
            if c == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    end = j + 1
                    break
        if end < 0:
            break
        candidates.append((i, text[i:end]))
        i = end

    best: dict | None = None
    best_score = -1
    for _, snippet in candidates:
        try:
            obj = json.loads(snippet)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        # Score = number of non-null, non-placeholder values. Skip
        # echoed-prompt placeholders like {"value": "<string or null>"}.
        score = 0
        for v in obj.values():
            if v is None:
                continue
            if isinstance(v, str) and (
                v.startswith("<") or v.endswith(">") or v == "..."
            ):
                continue
            score += 1
        if score > best_score:
            best = obj
            best_score = score

    if best is not None and best_score >= 1:
        return best

    # No clean parse — try truncated-tail recovery on the LAST candidate
    # (most likely to be the real, partially-emitted response).
    if not candidates:
        # Maybe there's an open `{` with no close. Find the last `{`.
        last_open = text.rfind("{")
        if last_open < 0:
            return None
        snippet = text[last_open:]
    else:
        snippet = candidates[-1][1]
    # Drop a trailing incomplete `"...": <incomplete>` pair, close braces.
    snippet = re.sub(r",\s*\"[^\"]*\"\s*:\s*[^,\}]*$", "", snippet)
    snippet = re.sub(r"\"[^\"]*\"\s*:\s*[^,\}]*$", "", snippet)
    snippet = snippet.rstrip(",").rstrip()
    if not snippet.endswith("}"):
        snippet += "}"
    try:
        obj = json.loads(snippet)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError as exc:
        log.debug("context_layer: JSON decode failed: %s", exc)
        return None


# ID fields (invoice_id / po_id / quote_id) tolerate a prefix gap. The
# document text may say only "136700" while the masthead-style ID is
# "QUT136700" — or vice versa. We accept the candidate value if EITHER
# form (with or without the conventional prefix) is in the doc.
_ID_PREFIX_RE = re.compile(r"^(?:INV|PO|QUT|QTE|RFQ)[\-_]?", re.IGNORECASE)


def _id_field_grounded(value: str, full_text: str) -> bool:
    """Grounding for invoice/po/quote IDs — accept prefix-stripped form too."""
    if not value:
        return False
    if _is_grounded(value, full_text):
        return True
    bare = _ID_PREFIX_RE.sub("", value).strip()
    if bare and bare != value and _is_grounded(bare, full_text):
        return True
    return False


# Region names typically appear in addresses as a token at the tail of
# the line, sometimes concatenated to the postcode (docling output).
# "West Sussex" may show up as "Sussex" only (e.g. "SussexRH135QH"). We
# accept the region if the last token of the value appears as a
# case-insensitive substring of full_text. Conservative because we
# require the last token to be at least 4 chars long.
def _region_field_grounded(value: str, full_text: str) -> bool:
    if not value:
        return False
    if _is_grounded(value, full_text):
        return True
    parts = value.split()
    if not parts:
        return False
    last = parts[-1]
    if len(last) >= 4 and last.lower() in full_text.lower():
        return True
    # Try first token too (e.g. "Greater Manchester" → "Manchester" or
    # "Greater" — pick whichever is most distinctive). Require >=4 chars.
    first = parts[0]
    if len(first) >= 4 and first.lower() in full_text.lower():
        return True
    return False


_SUPPLIER_SUFFIX_RE = re.compile(
    r"\s*[,\.]?\s*\b(?:LLC|Ltd|Limited|Inc|Incorporated|Pvt|GmbH|Corp|Corporation|"
    r"Co\.?|Company|Studios|Agency|Group|Solutions|Services|Holdings|"
    r"Enterprises?|LLP|PLC|Marketing|Consulting|Trading)\b\.?\s*$",
    re.IGNORECASE,
)


def _supplier_name_grounded(value: str, full_text: str) -> bool:
    """Accept supplier_name if its distinctive stem appears in full_text.

    "Aquarius Marketing Ltd" → check "Aquarius" appears in the doc.
    Tolerates the LLM normalising a short masthead ("AQUARIUS") into the
    full registered form. Substring check is case-insensitive AND uses
    the letter-spacing-collapsed view of the doc — same grounding
    chain as L3.
    """
    if not value:
        return False
    # Cheapest: case-sensitive full-string substring.
    if _is_grounded(value, full_text):
        return True
    # Case-insensitive (supplier names appear in MIXED CASE / ALL CAPS /
    # title case across templates; the LLM normalises them).
    if value.lower() in full_text.lower():
        return True
    # Stem-stripped form: drop the business suffix and check the first
    # distinctive word case-insensitively. "Aquarius Marketing Ltd" →
    # "Aquarius" → matches the doc's "AQUARIUS".
    stem = _SUPPLIER_SUFFIX_RE.sub("", value).strip()
    if not stem:
        return False
    first_word = stem.split()[0]
    if len(first_word) < 4:
        return False
    if first_word.lower() in full_text.lower():
        return True
    # Final fallback: collapse letter-spacing in the doc and retry the
    # first-word check (handles docling's "A Q U A R I U S" runs).
    from src.services.extraction_v3.judge.grounded_last_resort import (
        _collapse_letter_spacing,
    )
    return first_word.lower() in _collapse_letter_spacing(full_text).lower()


def _money_grounded(amount: float, full_text: str) -> bool:
    """Anti-hallucination check for money values.

    The model sometimes invents totals on column-split layouts where the
    label/value pairing is ambiguous. Accept the amount only if some
    common rendering of the number appears literally in full_text. The
    candidates cover: integer / decimal, with/without thousand commas,
    with/without trailing ``.00``. Whitespace and currency symbols are
    NOT inserted into the candidate — we just check substring presence
    in the doc text. If none of the renderings ground, the value is
    dropped to NULL (no-fabrication rule).
    """
    if amount is None or not full_text:
        return False
    # Build candidates. We don't know what decimal precision the doc used,
    # so try both integer and 2-decimal forms.
    abs_amt = abs(amount)
    rounded = round(abs_amt, 2)
    candidates: set[str] = set()
    # Integer form (e.g. 7290) — only meaningful for amounts with .00.
    if abs_amt == int(abs_amt):
        i = int(abs_amt)
        candidates.add(str(i))
        # With thousand-commas: "7,290", "87,480"
        candidates.add(f"{i:,}")
    # 2-decimal form (e.g. "1240.06", "1,240.06")
    candidates.add(f"{rounded:.2f}")
    candidates.add(f"{rounded:,.2f}")
    # Drop trailing .00 variants in addition (e.g. "8965" vs "8965.00")
    for c in list(candidates):
        if c.endswith(".00"):
            candidates.add(c[:-3])
            candidates.add(c[:-3].replace(",", ""))
    # Substring presence in raw text.
    for c in candidates:
        if c in full_text:
            return True
    # Whitespace-normalised view (handles "£ 7,290" → "£7,290").
    ft_ws = _norm_ws_inline(full_text)
    for c in candidates:
        if c in ft_ws:
            return True
    # Numeric-tolerance fallback. The doc sometimes uses non-standard
    # precision (e.g. GOMEZ INV618706 prints TAX (20%) as "£233.916",
    # the model normalises to 233.92, and neither candidate substring
    # matches verbatim). Scan all numeric tokens in the doc and accept
    # when any matches the amount within ±0.01. Tolerance is tighter
    # than 1 cent so we don't accept incidental phone/account/postcode
    # digits — only true currency values get within rounding of each
    # other.
    import re as _re
    for tok in _re.finditer(r"[0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?", full_text):
        s = tok.group(0).replace(",", "")
        try:
            doc_num = float(s)
        except ValueError:
            continue
        if abs(doc_num - abs(amount)) < 0.01:
            return True
    return False


def _norm_ws_inline(s: str) -> str:
    """Like _norm_ws but in this module to avoid the L3 import on a hot path."""
    import re as _re
    return _re.sub(r"\s+", "", s)


def _is_grounded(value: str, full_text: str) -> bool:
    """Same progressive substring grounding as L3 grounded_last_resort —
    accepts: raw / whitespace-normalised / letter-spacing-collapsed."""
    from src.services.extraction_v3.judge.grounded_last_resort import (
        _collapse_letter_spacing, _norm_ws, _WS_RE,
    )
    if not value:
        return False
    if value in full_text:
        return True
    if _norm_ws(value) in _norm_ws(full_text):
        return True
    v_sq = _WS_RE.sub("", _collapse_letter_spacing(value))
    t_sq = _WS_RE.sub("", _collapse_letter_spacing(full_text))
    return bool(v_sq) and v_sq in t_sq


_COUNTRY_NAME_OK = re.compile(
    r"^(United Kingdom|United States|Canada|Australia|New Zealand|Singapore|"
    r"India|Hong Kong|Japan|Switzerland|Ireland|Germany|France|Spain|Italy|"
    r"Netherlands|Belgium|Sweden|Norway|Denmark|Finland|UK|USA|US|UAE|"
    r"United Arab Emirates)$",
    re.IGNORECASE,
)


def _validate_and_bind(
    parsed: dict,
    full_text: str,
    fields: list[tuple[str, str, str]],
) -> dict[str, Any]:
    """Validate Qwen output. STRING fields must ground; date/money/decimal
    are type-bound via the existing parsers. Country/region are exempt
    from grounding when they're derived from address (Qwen may output
    'United Kingdom' that's not literally in the doc — accept if it
    matches a known country name)."""
    from src.services.extraction_v2.parsers.amounts import parse_amount
    from src.services.extraction_v2.parsers.dates import parse_date

    out: dict[str, Any] = {}
    for name, ftype, _desc in fields:
        v = parsed.get(name)
        # Treat the literal string "null"/"None" (Qwen quirk) and empty
        # strings as missing.
        if v is None or (isinstance(v, str) and v.strip().lower() in ("", "null", "none", "n/a")):
            out[name] = None
            continue

        if ftype == "str":
            sval = str(v).strip()
            # Currency is an ISO code derived from the symbol in the doc
            # (£→GBP, $→USD, etc.). It will NOT appear as a literal
            # substring of full_text, so we validate it against the known
            # ISO list instead of grounding it.
            if name == "currency":
                code = sval.upper()
                if code in _FX_TO_USD:
                    out[name] = code
                else:
                    out[name] = None
                continue
            # supplier_name: when Qwen normalises the supplier (e.g. doc
            # says "AQUARIUS" but Qwen returns "Aquarius Marketing Ltd"
            # using its world knowledge), require only that ONE distinctive
            # token from the value appears in the document. This avoids
            # NULLing legitimate supplier names that the doc only mentions
            # in short form.
            if name == "supplier_name":
                if _supplier_name_grounded(sval, full_text):
                    out[name] = sval
                else:
                    log.debug("context_layer: dropped ungrounded supplier_name=%r", sval)
                    out[name] = None
                continue
            # ID fields tolerate prefix gap (QUT136700 / 136700).
            if name in ("invoice_id", "po_id", "quote_id"):
                if _id_field_grounded(sval, full_text):
                    out[name] = sval
                else:
                    log.debug("context_layer: dropped ungrounded id %s=%r", name, sval)
                    out[name] = None
                continue
            # Region/delivery_region tolerate concatenated/postcode-stuck
            # tokens (West Sussex inside SussexRH135QH).
            if name in ("region", "delivery_region"):
                if _region_field_grounded(sval, full_text):
                    out[name] = sval
                else:
                    log.debug("context_layer: dropped ungrounded region %s=%r", name, sval)
                    out[name] = None
                continue
            # ship_to_country/country accept the well-known country names
            # without requiring substring grounding (already handled below).
            # Country and region are commonly derived from postcode/address
            # rather than appearing verbatim — allow well-known country names
            # to pass even when not a literal substring.
            if name in ("country", "ship_to_country") and _COUNTRY_NAME_OK.match(sval):
                out[name] = sval
                continue
            if name == "region" and len(sval) <= 40:
                # Region names tend to appear verbatim in addresses; require
                # grounding except when it's a county/state alias common
                # in procurement (e.g. "West Sussex", "California"). Letting
                # the grounding check decide keeps the rule simple.
                if _is_grounded(sval, full_text):
                    out[name] = sval
                else:
                    out[name] = None
                continue
            if _is_grounded(sval, full_text):
                out[name] = sval
            else:
                log.debug("context_layer: dropped ungrounded %s=%r", name, sval)
                out[name] = None

        elif ftype == "date":
            # Accept ISO format directly, otherwise pass through parse_date.
            sval = str(v).strip()
            try:
                iso = parse_date(sval)
                out[name] = iso.isoformat() if iso else None
            except Exception:  # noqa: BLE001
                out[name] = None

        elif ftype == "money":
            try:
                if isinstance(v, (int, float)):
                    amount = float(v)
                else:
                    amt = parse_amount(str(v))
                    amount = float(amt) if amt is not None else None
                if amount is None:
                    out[name] = None
                    continue
                # Anti-hallucination: the model occasionally invents totals
                # that don't exist in the doc body (e.g. INV-005-30 had a
                # column-split layout where the model output 8965 but the
                # doc only contained 6750/7290/675/1215). Require the
                # number to be findable in full_text under common
                # currency/comma/decimal variants. If it's not grounded,
                # drop to NULL — per no-fabrication rule.
                if not _money_grounded(amount, full_text):
                    log.debug(
                        "context_layer: dropped ungrounded money %s=%s",
                        name, amount,
                    )
                    out[name] = None
                    continue
                out[name] = amount
            except Exception:  # noqa: BLE001
                out[name] = None

        elif ftype == "decimal":
            try:
                if isinstance(v, (int, float)):
                    out[name] = float(v)
                else:
                    s = str(v).rstrip("%").replace(",", "").strip()
                    out[name] = float(s)
            except (ValueError, TypeError):
                out[name] = None

        else:
            out[name] = v
    return out


# ---------------------------------------------------------------------------
# Derived fields: FX → USD
# ---------------------------------------------------------------------------


def _compute_derived(row: dict[str, Any]) -> dict[str, Any]:
    """Populate exchange_rate_to_usd + converted_amount_usd from currency
    and the most representative total available on the row.

    Also fills `*_total_incl_tax` when the model couldn't ground the
    printed total but both subtotal and tax_amount are grounded values.
    This is arithmetic over verified inputs — NOT fabrication — and
    closes the column-split-layout gap where the doc clearly states
    components but the total in the print appears in a position the model
    can't reliably reach (e.g. TECHWORLD QUT-005-022, WADE QUT30789).
    """
    # Tax_amount recovery from subtotal × tax_percent / 100 when the
    # model couldn't ground the printed value (e.g. doc prints
    # "£233.916" with 3dp and the substring check rejected the model's
    # 2dp "233.92" before the numeric-tolerance fallback existed). This
    # is arithmetic over two grounded values — not fabrication.
    for sub_f, pct_f, tax_f in (
        ("invoice_amount", "tax_percent", "tax_amount"),
        ("total_amount", "tax_percent", "tax_amount"),
    ):
        if (
            row.get(tax_f) is None
            and row.get(sub_f) is not None
            and row.get(pct_f) is not None
        ):
            try:
                sub = float(row[sub_f])
                pct = float(row[pct_f])
                if pct > 0:
                    row[tax_f] = round(sub * pct / 100.0, 2)
                    log.info(
                        "context_layer: derived %s=%s from %s × %s%%",
                        tax_f, row[tax_f], sub_f, pct_f,
                    )
                    break
            except (ValueError, TypeError):
                pass

    # Total recovery from subtotal + tax. Order: invoice variant first,
    # then PO/quote shared keys; whichever pair exists on the row wins.
    for sub_f, tax_f, tot_f in (
        ("invoice_amount", "tax_amount", "invoice_total_incl_tax"),
        ("total_amount", "tax_amount", "total_amount_incl_tax"),
    ):
        if row.get(tot_f) is None and row.get(sub_f) is not None and row.get(tax_f) is not None:
            try:
                row[tot_f] = round(float(row[sub_f]) + float(row[tax_f]), 2)
                log.info(
                    "context_layer: derived %s=%s from %s+%s",
                    tot_f, row[tot_f], sub_f, tax_f,
                )
            except (ValueError, TypeError):
                pass

    ccy_raw = row.get("currency")
    if not isinstance(ccy_raw, str):
        return row
    ccy = ccy_raw.upper().strip()
    rate = _FX_TO_USD.get(ccy)
    if rate is None:
        return row
    row["exchange_rate_to_usd"] = rate
    # Pick the most relevant total to convert. Prefer including-tax → including
    # tax of subordinate amounts → subtotal-only.
    for total_field in (
        "invoice_total_incl_tax", "total_amount_incl_tax",
        "invoice_amount", "total_amount",
    ):
        v = row.get(total_field)
        if v is not None:
            try:
                row["converted_amount_usd"] = round(float(v) * rate, 2)
                break
            except (ValueError, TypeError):
                pass
    return row
