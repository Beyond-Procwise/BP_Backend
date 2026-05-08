"""Deterministic post-extraction field recovery.

After the LLM, sanitizer, template, and existing derivation_rules have
all run, certain header fields still come back NULL even though the
data is present in the parsed document text. This module sweeps each
NULL field once and tries a focused regex / heuristic against the
parsed text.

**Hard constraint — no fabrication.** A field is only filled from
explicit evidence found in ``parsed_text``. If no evidence is present,
the value stays NULL. Region is derived from the document's own
postcode (the postcode IS in the address); never inferred from vendor
name, supplier_id, or anything outside the document. Currency
*conversion* (USD-equivalent amounts, rate lookup) is handled by
``extraction_validator._calculate_currency_conversion`` — this module
only recovers the *currency code itself* when it is missing.

Every recovery is logged with the field name, the source pattern, and
the matched substring so a reviewer can tell what came from where. The
module never overwrites non-NULL values.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field as dc_field
from datetime import date, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


__all__ = ["recover_fields", "RecoveryReport"]


# UK postcode (most common in this corpus): "SW1A 1AA", "RH13 5QH", "LS2 7JL"
_UK_POSTCODE = re.compile(
    r"\b([A-PR-UWYZ][A-HK-Y]?[0-9][0-9A-HJKPSTUW]?\s*[0-9][ABD-HJLNP-UW-Z]{2})\b",
)
# US ZIP (5 or 5+4): "90210", "10001-1234"
_US_ZIP = re.compile(r"\b(\d{5}(?:-\d{4})?)\b")
# EU postcodes (numeric, 4-5 digits) — fallback when no UK/US match
_EU_POSTCODE = re.compile(r"\b(\d{4,5})\b")

# Country lookup from postcode shape
_POSTCODE_TO_COUNTRY = {
    "UK": "United Kingdom",
    "US": "United States",
}

# Known UK county / region from common postcode prefixes (not exhaustive,
# but covers the common ones in the corpus).
_UK_REGION_BY_POSTCODE_PREFIX = {
    "SW": "Greater London", "SE": "Greater London", "NW": "Greater London",
    "NE": "Greater London", "EC": "Greater London", "WC": "Greater London",
    "E": "Greater London", "W": "Greater London", "N": "Greater London",
    "B": "West Midlands", "M": "Greater Manchester",
    "L": "Merseyside",   "G": "Glasgow",
    "EH": "Edinburgh",   "BS": "Bristol",
    "RH": "West Sussex", "BN": "East Sussex",
    "LS": "West Yorkshire", "BD": "West Yorkshire",
    "S": "South Yorkshire", "OX": "Oxfordshire",
    "CB": "Cambridgeshire", "PO": "Hampshire",
    "NP": "Newport",
}


@dataclass
class RecoveryReport:
    """What was recovered, from where, with which source pattern."""
    fields_recovered: list[tuple[str, str, str]] = dc_field(default_factory=list)

    def record(self, field: str, value: str, source: str) -> None:
        self.fields_recovered.append((field, str(value), source))


def _f(v) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def _is_empty(v) -> bool:
    return v is None or (isinstance(v, str) and not v.strip())


# -- payment_terms --------------------------------------------------------

# "Net 30", "Net 30 days", "30 days net", "Due on receipt", "Due in 14 days",
# "Payment due within 30 days", "Terms: Net 30", "Payable in 60 days"
_PAYMENT_TERMS_PATTERNS = [
    re.compile(r"\b[Nn]et\s+(\d{1,3})(?:\s*days?)?\b"),
    re.compile(r"\b(\d{1,3})\s*days?\s+net\b", re.IGNORECASE),
    re.compile(r"\bdue\s+(?:within|in)\s+(\d{1,3})\s*days?\b", re.IGNORECASE),
    re.compile(r"\bpayable\s+(?:within|in)\s+(\d{1,3})\s*days?\b", re.IGNORECASE),
    re.compile(r"\bpayment\s+(?:due|terms)\s*:?\s*(?:within|in)?\s*(\d{1,3})\s*days?\b", re.IGNORECASE),
    re.compile(r"\bterms?\s*:?\s*[Nn]et\s+(\d{1,3})\b"),
]

_PAYMENT_TERMS_LITERALS = [
    (re.compile(r"\b(?:due\s+on\s+receipt|payable\s+upon\s+receipt|cod|cash\s+on\s+delivery)\b", re.IGNORECASE),
     "Due on receipt"),
    (re.compile(r"\b(?:eom|end\s+of\s+month)\b", re.IGNORECASE), "EOM"),
    (re.compile(r"\b(?:cia|cash\s+in\s+advance|prepaid?)\b", re.IGNORECASE), "CIA"),
    (re.compile(r"\b(?:50\s*%\s*upfront|50/50)\b", re.IGNORECASE), "50% upfront"),
]


def _recover_payment_terms(text: str) -> Optional[str]:
    if not text:
        return None
    for pat in _PAYMENT_TERMS_PATTERNS:
        m = pat.search(text)
        if m:
            return f"Net {m.group(1)}"
    for pat, normalized in _PAYMENT_TERMS_LITERALS:
        if pat.search(text):
            return normalized
    return None


# -- validity_date / expected_delivery_date -------------------------------

_VALIDITY_DAYS_RE = re.compile(
    r"(?:valid\s+for|validity)\s*:?\s*(\d{1,3})\s*days?",
    re.IGNORECASE,
)
_VALIDITY_UNTIL_RE = re.compile(
    r"valid\s+(?:until|through|till|to)\s*:?\s*"
    r"(\d{1,2}[\s./-]+\w{3,9}[\s./-]+\d{2,4}|\d{4}-\d{2}-\d{2}|\w{3,9}\s+\d{1,2},?\s+\d{4})",
    re.IGNORECASE,
)
_EXPECTED_DELIVERY_RE = re.compile(
    r"(?:expected\s+delivery|delivery\s+date|deliver(?:y|ed)\s+by|"
    r"required\s+(?:by|date)|ship(?:ment|ping)\s+date)\s*:?\s*"
    r"(\d{1,2}[\s./-]+\w{3,9}[\s./-]+\d{2,4}|\d{4}-\d{2}-\d{2}|\w{3,9}\s+\d{1,2},?\s+\d{4})",
    re.IGNORECASE,
)


def _normalize_date(raw: str) -> Optional[str]:
    """Best-effort to convert a free-form date string to ISO-8601."""
    if not raw:
        return None
    raw = raw.strip().rstrip(",.")
    # Already ISO?
    if re.match(r"^\d{4}-\d{2}-\d{2}", raw):
        return raw[:10]
    # Try a few common formats
    from datetime import datetime
    for fmt in (
        "%d %b %Y", "%d %B %Y", "%d-%b-%Y", "%d-%B-%Y", "%d/%b/%Y", "%d/%B/%Y",
        "%d.%b.%Y", "%d.%B.%Y", "%b %d, %Y", "%B %d, %Y", "%b %d %Y", "%B %d %Y",
        "%d %b %y", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y",
    ):
        try:
            return datetime.strptime(raw, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def _recover_validity_date(text: str, quote_date: Optional[str]) -> Optional[str]:
    if not text:
        return None
    # "Valid until <date>"
    m = _VALIDITY_UNTIL_RE.search(text)
    if m:
        norm = _normalize_date(m.group(1))
        if norm:
            return norm
    # "Valid for N days" → quote_date + N
    m = _VALIDITY_DAYS_RE.search(text)
    if m and quote_date:
        try:
            base = date.fromisoformat(str(quote_date)[:10])
            n = int(m.group(1))
            return (base + timedelta(days=n)).isoformat()
        except (ValueError, TypeError):
            pass
    return None


def _recover_expected_delivery(text: str) -> Optional[str]:
    if not text:
        return None
    m = _EXPECTED_DELIVERY_RE.search(text)
    if m:
        norm = _normalize_date(m.group(1))
        if norm:
            return norm
    return None


# -- invoice_date / quote_date / order_date / due_date --------------------
#
# The schema has explicit date fields per doc type. The LLM sometimes
# misses them entirely. These regexes look for the document's own
# date label followed by a recognisable date format.

_INVOICE_DATE_RE = re.compile(
    r"(?:invoice\s+date|date\s+(?:of\s+)?(?:issue|issued)|issue\s+date|"
    r"^date\s*[:\-]|invoice\s+issued)\s*[:\-]?\s*"
    r"(\d{1,2}[\s./-]+\w{3,9}[\s./-]+\d{2,4}|\d{4}-\d{2}-\d{2}|"
    r"\w{3,9}\s+\d{1,2},?\s+\d{4}|\d{1,2}/\d{1,2}/\d{2,4})",
    re.IGNORECASE | re.MULTILINE,
)
_QUOTE_DATE_RE = re.compile(
    r"(?:quote\s+date|date\s+of\s+quote|quotation\s+date|"
    r"^date\s*[:\-]|quote\s+issued)\s*[:\-]?\s*"
    r"(\d{1,2}[\s./-]+\w{3,9}[\s./-]+\d{2,4}|\d{4}-\d{2}-\d{2}|"
    r"\w{3,9}\s+\d{1,2},?\s+\d{4}|\d{1,2}/\d{1,2}/\d{2,4})",
    re.IGNORECASE | re.MULTILINE,
)
_ORDER_DATE_RE = re.compile(
    r"(?:po\s+date|order\s+date|date\s+of\s+order|"
    r"date\s+raised|date\s+issued|^date\s*[:\-])\s*[:\-]?\s*"
    r"(\d{1,2}[\s./-]+\w{3,9}[\s./-]+\d{2,4}|\d{4}-\d{2}-\d{2}|"
    r"\w{3,9}\s+\d{1,2},?\s+\d{4}|\d{1,2}/\d{1,2}/\d{2,4})",
    re.IGNORECASE | re.MULTILINE,
)
_DUE_DATE_RE = re.compile(
    r"(?:due\s+date|payment\s+due|pay\s+by|payable\s+by)\s*[:\-]?\s*"
    r"(\d{1,2}[\s./-]+\w{3,9}[\s./-]+\d{2,4}|\d{4}-\d{2}-\d{2}|"
    r"\w{3,9}\s+\d{1,2},?\s+\d{4}|\d{1,2}/\d{1,2}/\d{2,4})",
    re.IGNORECASE | re.MULTILINE,
)


def _recover_date(text: str, regex) -> Optional[str]:
    if not text:
        return None
    m = regex.search(text)
    if m:
        return _normalize_date(m.group(1))
    return None


# -- incoterm / currency --------------------------------------------------

_INCOTERMS = {"EXW", "FCA", "CPT", "CIP", "DAP", "DPU", "DDP", "FAS", "FOB",
              "CFR", "CIF"}
_INCOTERM_RE = re.compile(
    r"\b(?:" + "|".join(_INCOTERMS) + r")\b",
)


def _recover_incoterm(text: str) -> Optional[str]:
    if not text:
        return None
    m = _INCOTERM_RE.search(text.upper())
    if m:
        return m.group(0)
    return None


_CURRENCY_FROM_HEADER_RE = re.compile(
    r"\b(USD|EUR|GBP|JPY|CNY|INR|AUD|CAD|CHF|SGD|HKD|NZD|SEK|NOK|DKK|PLN)\b"
)


def _recover_currency(text: str, header_currency: Optional[str]) -> Optional[str]:
    """Prefer header currency. Fall back to first ISO code in the text."""
    if header_currency and isinstance(header_currency, str):
        ccy = header_currency.strip().upper()
        if len(ccy) == 3:
            return ccy
    if not text:
        return None
    m = _CURRENCY_FROM_HEADER_RE.search(text)
    if m:
        return m.group(1)
    # Symbol fallback
    if "£" in text:
        return "GBP"
    if "€" in text:
        return "EUR"
    return None


# -- city / postcode / country / region -----------------------------------

# Anchors that begin a delivery address block in procurement docs.
# Only STRONG anchors that reliably introduce the delivery address are
# included. "Recipient:" / "Buyer:" are deliberately excluded because
# in real docs they often appear as a section label *after* the buyer
# address (which was already printed at the top), pointing at the
# supplier-contact block.
_DELIVERY_ANCHOR_RE = re.compile(
    r"(?:^|\n|\r)\s*"
    r"(?:Ship[ -]?[Tt]o|Deliver(?:y)?(?:\s*[Aa]ddress)?\s*[Tt]o|"
    r"Deliver\s*[Aa]ddress|Delivery\s*[Aa]ddress|"
    r"Ship\s+[Aa]ddress|Sent\s+[Tt]o|Send\s+[Tt]o)"
    r"\s*[:\-]?\s*",
    re.IGNORECASE,
)
# Anchors that END the delivery block (next section begins). When we see
# one of these inside the candidate window, we cut the block there so a
# downstream "Bill To" supplier postcode can't be picked up by mistake.
_DELIVERY_TERMINATOR_RE = re.compile(
    r"\b(?:Bill[ -]?[Tt]o|Invoice[ -]?[Tt]o|Sold[ -]?[Tt]o|"
    r"Supplier|Vendor|From\s*:|"
    r"Subtotal|Sub-?total|Total|Tax|VAT|"
    r"Item|Description|Quantity|Qty\b|"
    r"Payment|Terms|Notes)\b",
    re.IGNORECASE,
)
# How far past the anchor we'll accept as "the same address block" if no
# terminator fires. UK addresses with line wraps rarely exceed ~250 chars.
_DELIVERY_BLOCK_MAX = 300


def _delivery_block(text: str) -> Optional[str]:
    """Return the substring of `text` that contains the delivery address.

    Strategy: find the first delivery anchor ("Ship to:" / "Delivery
    address:" / etc.). Return the slice from there up to either (a) the
    next terminator anchor ("Bill to", "Subtotal", "Item", etc.) or (b)
    a fixed character cap, whichever fires first. Returns None when no
    anchor is present — callers should fall back to the global text.
    """
    if not text:
        return None
    m = _DELIVERY_ANCHOR_RE.search(text)
    if not m:
        return None
    start = m.end()
    # Cap by character count first so we don't scan the whole document
    end = min(start + _DELIVERY_BLOCK_MAX, len(text))
    candidate = text[start:end]
    # If a terminator appears inside, cut the candidate there
    term = _DELIVERY_TERMINATOR_RE.search(candidate)
    if term:
        candidate = candidate[:term.start()]
    return candidate


def _resolve_postcode_from_text(text: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Run the postcode regexes against `text`, return (postcode, country, region).

    Important: when ``text`` contains MULTIPLE distinct UK postcodes,
    we cannot tell which is the delivery postcode without an anchor.
    Returning a guess would be fabrication. We return ``(None, "United
    Kingdom", None)`` — country is safe (a UK postcode IS in the doc),
    but postcode/region are ambiguous → NULL. Callers can still derive
    region from a header.postal_code that the LLM extracted.
    """
    if not text:
        return None, None, None
    matches = _UK_POSTCODE.findall(text)
    if matches:
        # De-duplicate (case- and space-insensitive)
        seen = set()
        unique: list = []
        for raw in matches:
            norm = raw.upper().replace("  ", " ").strip()
            if norm not in seen:
                seen.add(norm)
                unique.append(norm)
        if len(unique) > 1:
            # Ambiguous — multiple UK postcodes, none can be claimed as
            # delivery without anchor evidence. Country is safe.
            return None, "United Kingdom", None
        postcode = unique[0]
        prefix_letters = re.match(r"^[A-Z]+", postcode).group(0)
        region = None
        for k in sorted(_UK_REGION_BY_POSTCODE_PREFIX, key=len, reverse=True):
            if prefix_letters.startswith(k):
                region = _UK_REGION_BY_POSTCODE_PREFIX[k]
                break
        return postcode, "United Kingdom", region
    m = _US_ZIP.search(text)
    if m and re.search(r"\b(USA|United States|U\.S\.A?)\b", text, re.IGNORECASE):
        return m.group(1), "United States", None
    return None, None, None


def _recover_postcode(text: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Return (postcode, country, region) for the DELIVERY address.

    Two-pass: first try the delivery-address block (anchored by "Ship
    To:" / "Delivery address:" / etc.). If no postcode is found there
    — or no anchor exists — fall back to the global first-match. The
    fallback is necessary for documents whose layout doesn't include
    an explicit anchor (one-address invoices, receipts, etc.).
    """
    if not text:
        return None, None, None
    block = _delivery_block(text)
    if block:
        result = _resolve_postcode_from_text(block)
        if result[0]:  # found a postcode inside the delivery block
            return result
    # No anchor, or anchor present but no postcode in its block
    return _resolve_postcode_from_text(text)


# UK city extraction: only attempt when an address-shaped line is
# present — i.e. one or more comma-separated address tokens directly
# precede the postcode on the same logical line. We pick the token
# IMMEDIATELY before the postcode, which is conventionally the post
# town. Anything more aggressive (multi-word grabs, line walks) risks
# pulling in company names. If the layout doesn't match, return None.
_CITY_TOKEN_BEFORE_POSTCODE_RE = re.compile(
    # The city sits on a line that's preceded by a comma OR a newline.
    # Both real layouts:
    #   "Building, Newport, NP20 4EG"      (comma before & after city)
    #   "Regent Street\nCambridge, CB2 1FD" (newline before city, comma after)
    #   "1 High St, Horsham RH13 5QH"      (comma before, space after)
    r"(?:[,\n]\s*)([A-Z][A-Za-z\-']{1,30}(?:\s+[A-Z][A-Za-z\-']{1,30})?)[\s,]+"
    r"([A-PR-UWYZ][A-HK-Y]?[0-9][0-9A-HJKPSTUW]?\s*[0-9][ABD-HJLNP-UW-Z]{2})\b",
)

# Words that look capitalised but are never post towns. Conservative —
# we'd rather miss a city than fabricate one.
_CITY_BLOCKLIST = {
    "Limited", "Ltd", "PLC", "LLP", "Inc", "Corp", "Corporation",
    "Industrial", "Estate", "Park", "Trading", "Business", "Centre",
    "Center", "House", "Office", "Building", "Unit", "Avenue", "Road",
    "Street", "Lane", "Drive", "Way", "Court", "Square", "Place",
    "England", "Wales", "Scotland", "United", "Kingdom",
    "Sussex", "Yorkshire", "Midlands", "London",
}


def _resolve_city_from_text(text: str) -> Optional[str]:
    """Apply the city regex to `text` with the conservative filters."""
    if not text:
        return None
    m = _CITY_TOKEN_BEFORE_POSTCODE_RE.search(text)
    if not m:
        return None
    candidate = m.group(1).strip()
    words = candidate.split()
    if not words:
        return None
    if any(w in _CITY_BLOCKLIST for w in words):
        return None
    if len(words) == 1 and 2 <= len(words[0]) <= 30:
        return words[0]
    if len(words) == 2 and all(w[0].isupper() and w[1:].isalpha() for w in words):
        return candidate
    return None


def _recover_city(text: str) -> Optional[str]:
    """Extract the delivery post town from the delivery-address block.

    Same two-pass strategy as ``_recover_postcode``: prefer the
    anchored delivery block, then fall back to the global text. This
    prevents picking up the supplier's city when the doc has both
    supplier and buyer addresses.
    """
    if not text:
        return None
    block = _delivery_block(text)
    if block:
        candidate = _resolve_city_from_text(block)
        if candidate:
            return candidate
    return _resolve_city_from_text(text)


# Explicit country mentions in the document. We accept ISO-3 codes,
# common ISO-2 codes when in clearly bounded context, and the most
# common written names. Anything else stays NULL.
_COUNTRY_NAME_RE = re.compile(
    r"\b(United Kingdom|Great Britain|United States of America|"
    r"United States|U\.?S\.?A\.?|Germany|France|Spain|Italy|Netherlands|"
    r"Belgium|Ireland|Switzerland|Sweden|Norway|Denmark|Finland|Poland|"
    r"Portugal|Austria|Greece|Czech Republic|Czechia|Hungary|Romania|"
    r"Bulgaria|Australia|New Zealand|Canada|Mexico|Brazil|Argentina|"
    r"India|China|Japan|Singapore|Hong Kong|South Korea|Taiwan|"
    r"United Arab Emirates|Saudi Arabia|South Africa)\b",
    re.IGNORECASE,
)
_COUNTRY_NAME_NORMALIZED = {
    "great britain": "United Kingdom",
    "united states of america": "United States",
    "u.s.a": "United States", "u.s.a.": "United States",
    "usa": "United States", "u.s": "United States", "u.s.": "United States",
    "czechia": "Czech Republic",
}


def _recover_country_explicit(text: str) -> Optional[str]:
    """Extract a country only when its name appears literally in the text."""
    if not text:
        return None
    m = _COUNTRY_NAME_RE.search(text)
    if not m:
        return None
    raw = m.group(1).strip()
    return _COUNTRY_NAME_NORMALIZED.get(raw.lower(), raw.title()
                                        if raw.lower() == raw.upper().lower()
                                        else raw)


# -- tax_amount / tax_percent --------------------------------------------
#
# Only fills from explicit document statements. Examples accepted:
#   "VAT @ 20%"             → tax_percent = 20
#   "Tax: £24.00"           → tax_amount = 24.00
#   "Sales Tax (8.5%):"     → tax_percent = 8.5
#   "VAT 20% £24.00"        → tax_percent = 20, tax_amount = 24.00
# We never compute tax_amount from subtotal*rate here; that's the job
# of the closure invariants (which flag inconsistencies, not fill them).

_TAX_PERCENT_RE = re.compile(
    r"(?:VAT|Sales\s+Tax|GST|Tax)\s*"
    r"(?:@|at|rate|\(|:)?\s*"
    r"(\d{1,2}(?:\.\d{1,2})?)\s*%",
    re.IGNORECASE,
)
_TAX_AMOUNT_LABELLED_RE = re.compile(
    r"(?:VAT|Sales\s+Tax|GST|Tax(?:\s+amount)?)\s*[:\-]\s*"
    r"[£€$]?\s*([0-9]{1,3}(?:[,\s][0-9]{3})*(?:\.[0-9]{2})?|[0-9]+\.[0-9]{2})",
    re.IGNORECASE,
)


def _recover_tax_percent(text: str) -> Optional[float]:
    if not text:
        return None
    m = _TAX_PERCENT_RE.search(text)
    if not m:
        return None
    try:
        v = float(m.group(1))
    except ValueError:
        return None
    if 0 < v <= 30:
        return v
    return None


def _recover_tax_amount(text: str) -> Optional[float]:
    if not text:
        return None
    m = _TAX_AMOUNT_LABELLED_RE.search(text)
    if not m:
        return None
    raw = m.group(1).replace(",", "").replace(" ", "")
    try:
        v = float(raw)
    except ValueError:
        return None
    return v if v >= 0 else None


# -- main entry -----------------------------------------------------------

# Per-doc-type recovery plan: which fields to attempt.
_DOC_TYPE_FIELDS = {
    "Invoice": (
        "payment_terms", "country", "region", "currency",
        "tax_amount", "tax_percent",
        "invoice_date", "due_date",
    ),
    "Purchase_Order": (
        "payment_terms", "expected_delivery_date", "incoterm",
        "ship_to_country", "delivery_region", "delivery_city", "postal_code",
        "currency", "tax_amount", "tax_percent",
        "order_date",
    ),
    "Quote": (
        "validity_date", "country", "region", "currency",
        "supplier_address", "tax_amount", "tax_percent",
        "quote_date",
    ),
    "Contract": (),
}


def recover_fields(
    header: dict,
    *,
    parsed_text: Optional[str],
    doc_type: str,
    file_path: Optional[str] = None,
) -> RecoveryReport:
    """Fill NULL header fields from `parsed_text` using deterministic patterns.

    Never overwrites non-NULL values. Returns a :class:`RecoveryReport`
    enumerating every recovery so callers can log + write provenance.
    """
    report = RecoveryReport()
    if not parsed_text:
        return report
    fields = _DOC_TYPE_FIELDS.get(doc_type, ())
    if not fields:
        return report

    text = parsed_text

    # payment_terms
    if "payment_terms" in fields and _is_empty(header.get("payment_terms")):
        v = _recover_payment_terms(text)
        if v:
            header["payment_terms"] = v
            report.record("payment_terms", v, "regex.payment_terms")

    # invoice_date / quote_date / order_date — primary doc-type date
    if "invoice_date" in fields and _is_empty(header.get("invoice_date")):
        v = _recover_date(text, _INVOICE_DATE_RE)
        if v:
            header["invoice_date"] = v
            report.record("invoice_date", v, "regex.invoice_date")
    if "quote_date" in fields and _is_empty(header.get("quote_date")):
        v = _recover_date(text, _QUOTE_DATE_RE)
        if v:
            header["quote_date"] = v
            report.record("quote_date", v, "regex.quote_date")
    if "order_date" in fields and _is_empty(header.get("order_date")):
        v = _recover_date(text, _ORDER_DATE_RE)
        if v:
            header["order_date"] = v
            report.record("order_date", v, "regex.order_date")
    if "due_date" in fields and _is_empty(header.get("due_date")):
        v = _recover_date(text, _DUE_DATE_RE)
        if v:
            header["due_date"] = v
            report.record("due_date", v, "regex.due_date")

    # validity_date (quote)
    if "validity_date" in fields and _is_empty(header.get("validity_date")):
        v = _recover_validity_date(text, header.get("quote_date"))
        if v:
            header["validity_date"] = v
            report.record("validity_date", v, "regex.validity")

    # expected_delivery_date (PO)
    if "expected_delivery_date" in fields and _is_empty(header.get("expected_delivery_date")):
        v = _recover_expected_delivery(text)
        if v:
            header["expected_delivery_date"] = v
            report.record("expected_delivery_date", v, "regex.delivery")

    # incoterm (PO)
    if "incoterm" in fields and _is_empty(header.get("incoterm")):
        v = _recover_incoterm(text)
        if v:
            header["incoterm"] = v
            report.record("incoterm", v, "regex.incoterm")

    # currency (any doc type, depending on schema column name)
    for ccy_key in ("currency", "default_currency"):
        if ccy_key in fields and _is_empty(header.get(ccy_key)):
            v = _recover_currency(text, header.get("currency"))
            if v:
                header[ccy_key] = v
                report.record(ccy_key, v, "regex.currency")
                break

    # tax_percent — only when the document explicitly states a rate
    if "tax_percent" in fields and _is_empty(header.get("tax_percent")):
        v = _recover_tax_percent(text)
        if v is not None:
            header["tax_percent"] = v
            report.record("tax_percent", str(v), "regex.tax_percent")

    # tax_amount — only when the document explicitly labels an amount
    if "tax_amount" in fields and _is_empty(header.get("tax_amount")):
        v = _recover_tax_amount(text)
        if v is not None:
            header["tax_amount"] = v
            report.record("tax_amount", f"{v:.2f}", "regex.tax_amount")

    # postcode → country/region/city, plus explicit country mention.
    #
    # Subtle but important: when deriving country/region, prefer the
    # postcode ALREADY persisted in the header over a fresh text scan.
    # The persisted postcode is the canonical delivery postcode (set by
    # LLM or earlier sanitizer); a fresh scan can pick up a different
    # postcode from the document and produce inconsistent header data
    # (e.g. postal_code='RH13 5QH' but delivery_region='West Midlands').
    if any(f in fields for f in ("postal_code", "country", "ship_to_country",
                                  "region", "delivery_region", "delivery_city")):
        # Step 1: text-derived candidate.
        text_pc, text_country, text_region = _recover_postcode(text)
        explicit_country = _recover_country_explicit(text)

        if text_pc and "postal_code" in fields and _is_empty(header.get("postal_code")):
            header["postal_code"] = text_pc
            report.record("postal_code", text_pc, "regex.uk_postcode")

        # NOTE: a previous version of this code overrode header.postal_code
        # with the FIRST postcode in the text on the assumption that
        # buyer/delivery addresses are printed first. That assumption
        # turned out to be backwards for many real procurement docs where
        # the SUPPLIER's letterhead is at the top (above the buyer's
        # delivery address). The override produced false data — exactly
        # what the user told us to avoid. We trust the LLM's postal_code
        # value when it's present; recovery only fills if it's empty.

        # Step 2: derive country/region from the POSTAL_CODE NOW IN THE
        # HEADER (not the fresh text scan). This guarantees consistency
        # between postal_code and the country/region we override.
        canonical_pc = header.get("postal_code")
        if canonical_pc:
            _, pc_country, pc_region = _resolve_postcode_from_text(str(canonical_pc))
        else:
            pc_country, pc_region = text_country, text_region

        # Country: prefer explicit name in document over postcode-derived.
        # Both are evidence from the document; explicit wins because it's
        # the document literally saying so.
        country = explicit_country or pc_country
        country_source = "regex.country_explicit" if explicit_country else "postcode.country"
        for country_key in ("country", "ship_to_country"):
            if country_key in fields and _is_empty(header.get(country_key)) and country:
                header[country_key] = country
                report.record(country_key, country, country_source)

        # Region: postcode-derived. Two cases —
        #   (1) field is empty → fill from postcode prefix lookup
        #   (2) field is set BUT contradicts the postcode-derived value
        #       (e.g. postcode 'M1 7HY' → 'Greater Manchester' but the LLM
        #       wrote 'West Sussex'). The LLM/few-shot bias is fabricating
        #       false data; correcting it from postcode evidence is NOT
        #       fabrication — it's removing a wrong value using a stronger
        #       signal that comes from the same address.
        for region_key in ("region", "delivery_region"):
            if region_key not in fields or not pc_region:
                continue
            existing = header.get(region_key)
            if _is_empty(existing):
                header[region_key] = pc_region
                report.record(region_key, pc_region, "postcode.region")
            elif str(existing).strip().lower() != pc_region.lower():
                # Contradicting evidence: postcode says X, header says Y.
                # Trust the postcode (paired with the same address) over
                # the LLM's free-text guess.
                header[region_key] = pc_region
                report.record(
                    region_key,
                    f"{pc_region} (overrode {existing!r})",
                    "postcode.region.contradicts",
                )

        if "delivery_city" in fields and _is_empty(header.get("delivery_city")):
            city = _recover_city(text)
            if city:
                header["delivery_city"] = city
                report.record("delivery_city", city, "regex.city_before_postcode")

    if report.fields_recovered:
        logger.info(
            "[FieldRecovery] %s %s: filled %d NULL field(s): %s",
            doc_type, file_path or "",
            len(report.fields_recovered),
            ", ".join(f"{f}={v[:40]!r} (via {s})"
                      for (f, v, s) in report.fields_recovered),
        )
    return report
