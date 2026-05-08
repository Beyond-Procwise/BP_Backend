"""Currency detection and normalization for extracted documents.

Two responsibilities:

  1. :func:`detect_currency` — given a document's parsed text + optional
     vendor country, return an ISO 4217 alpha-3 code (or ``None``) using
     a priority chain: explicit ISO code → unambiguous symbol →
     country-derived prior → number-formatting pattern.

  2. :func:`normalize_currency_in_place` — fill in ``header["currency"]``
     and any missing per-line ``item["currency"]`` from the detected
     currency, without overwriting values the extractor already supplied.

The ``CurrencyConsistency`` invariant in ``invariants.py`` then catches
documents where ``currency`` differs across header and lines.
"""
from __future__ import annotations

import re
from typing import Optional

__all__ = ["detect_currency", "normalize_currency_in_place"]


# -- ISO 4217 alpha-3 codes -----------------------------------------------

# Common ISO codes the system is likely to encounter. Not exhaustive —
# the regex-based detection captures any 3-letter all-caps token next
# to an amount, but this set is what we'll use for priors.
_KNOWN_ISO = {
    "USD", "EUR", "GBP", "JPY", "CNY", "INR", "AUD", "CAD",
    "CHF", "SGD", "HKD", "NZD", "SEK", "NOK", "DKK", "PLN",
    "MXN", "BRL", "ZAR", "TRY", "AED", "SAR", "ILS", "KRW",
    "RUB", "CZK", "HUF", "RON", "THB", "IDR", "PHP", "MYR", "VND",
}


# -- Symbol → set of plausible ISO codes ----------------------------------
# Symbols that map to exactly one currency are unambiguous; ambiguous
# symbols return a set, which the country prior must disambiguate.
_SYMBOL_MAP = {
    "£": {"GBP"},
    "€": {"EUR"},
    "₹": {"INR"},
    "₩": {"KRW"},
    "₪": {"ILS"},
    "฿": {"THB"},
    "₺": {"TRY"},
    "₽": {"RUB"},
    "$": {"USD", "CAD", "AUD", "SGD", "HKD", "MXN", "NZD"},
    "¥": {"JPY", "CNY"},
    "kr": {"SEK", "NOK", "DKK"},
    "Fr": {"CHF"},
    "zł": {"PLN"},
    "Ft": {"HUF"},
}


# Country-of-vendor → most-likely currency. Used to disambiguate $/¥/kr
# when the vendor's address gives a country.
_COUNTRY_TO_CCY = {
    "United States": "USD", "USA": "USD", "US": "USD",
    "United Kingdom": "GBP", "UK": "GBP", "GB": "GBP", "Britain": "GBP",
    "Germany": "EUR", "France": "EUR", "Italy": "EUR", "Spain": "EUR",
    "Netherlands": "EUR", "Belgium": "EUR", "Ireland": "EUR",
    "Austria": "EUR", "Portugal": "EUR", "Finland": "EUR",
    "Greece": "EUR", "Luxembourg": "EUR", "Estonia": "EUR",
    "Slovakia": "EUR", "Slovenia": "EUR", "Latvia": "EUR",
    "Lithuania": "EUR", "Cyprus": "EUR", "Malta": "EUR", "Croatia": "EUR",
    "Japan": "JPY",
    "China": "CNY", "PRC": "CNY",
    "India": "INR",
    "Australia": "AUD",
    "Canada": "CAD",
    "Switzerland": "CHF",
    "Singapore": "SGD",
    "Hong Kong": "HKD",
    "New Zealand": "NZD",
    "Sweden": "SEK", "Norway": "NOK", "Denmark": "DKK",
    "Poland": "PLN",
    "Mexico": "MXN",
    "Brazil": "BRL",
    "South Africa": "ZAR",
    "Turkey": "TRY",
    "Israel": "ILS",
    "South Korea": "KRW", "Korea": "KRW",
}


# Recognise an explicit ISO code adjacent to a number, e.g. "1234.56 EUR"
# or "USD 1,234.56".
_ISO_NEAR_AMOUNT = re.compile(
    r"(?:(?P<pre>[A-Z]{3})\s*[\d,.\s]+|"
    r"[\d,.\s]+\s*(?P<post>[A-Z]{3}))",
)


def _country_from_address_text(text: str) -> Optional[str]:
    """Return the longest-match country name found in `text`. Match is
    on whole-word boundaries so "us" inside "just" doesn't trigger.
    Two-letter abbreviations (US, UK, GB) only match at word boundaries.
    """
    if not text:
        return None
    best: Optional[str] = None
    best_len = 0
    for country in _COUNTRY_TO_CCY:
        pattern = r"\b" + re.escape(country) + r"\b"
        if re.search(pattern, text, re.IGNORECASE) and len(country) > best_len:
            best = country
            best_len = len(country)
    return best


def detect_currency(parsed_text: str,
                    *,
                    header_country: Optional[str] = None,
                    vendor_address: Optional[str] = None) -> Optional[str]:
    """Detect the document's currency from text + optional vendor country.

    Priority chain (first hit wins):
      1. Explicit ISO code (``EUR``, ``USD``, …) adjacent to an amount.
      2. Unambiguous currency symbol (``£``, ``€``, ``₹``, …).
      3. Country prior from ``header_country`` or ``vendor_address``.
      4. Disambiguated symbol (``$``/``¥``/``kr``) using the country prior.

    Returns the alpha-3 ISO code or ``None`` when nothing matches.
    """
    text = parsed_text or ""

    # Layer 1 — explicit ISO code
    for m in _ISO_NEAR_AMOUNT.finditer(text):
        code = m.group("pre") or m.group("post")
        if code and code.upper() in _KNOWN_ISO:
            return code.upper()

    # Country prior — used in layers 3+ but compute now
    country = (header_country or "").strip() or _country_from_address_text(
        vendor_address or text
    )
    country_ccy = _COUNTRY_TO_CCY.get(country) if country else None

    # Layer 2 — unambiguous single-currency symbol
    for sym, codes in _SYMBOL_MAP.items():
        if sym in text and len(codes) == 1:
            return next(iter(codes))

    # Layer 3 — country prior alone, when no symbol/code at all
    has_any_symbol = any(sym in text for sym in _SYMBOL_MAP)
    if not has_any_symbol and country_ccy:
        return country_ccy

    # Layer 4 — ambiguous symbol disambiguated by country prior
    for sym, codes in _SYMBOL_MAP.items():
        if sym in text and country_ccy in codes:
            return country_ccy

    # Fallback: the country prior even when symbol is ambiguous and no match
    return country_ccy


def normalize_currency_in_place(
    header: dict, line_items: list,
    *, parsed_text: Optional[str] = None,
    fallback_country: Optional[str] = None,
) -> Optional[str]:
    """Set ``header['currency']`` and any missing line ``currency`` to the
    detected ISO code. Never overwrites values already present.

    Returns the resolved currency, or ``None`` if detection fails.
    """
    existing = (header.get("currency") or "").strip().upper()
    if existing in _KNOWN_ISO:
        ccy = existing
    else:
        ccy = detect_currency(
            parsed_text or "",
            header_country=header.get("country") or fallback_country,
            vendor_address=header.get("supplier_address") or header.get("address"),
        )
        if ccy:
            header["currency"] = ccy
    if ccy:
        for item in line_items or []:
            if not (item.get("currency") or "").strip():
                item["currency"] = ccy
    return ccy
