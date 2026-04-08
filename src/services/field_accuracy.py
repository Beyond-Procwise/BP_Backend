"""Field accuracy utilities for procurement data extraction.

Provides locale-aware date parsing, European numeric format handling,
and improved currency detection.
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)

# --- Date Parsing ---

# Common date formats ordered by specificity (most specific first)
_DATE_FORMATS = [
    # ISO
    "%Y-%m-%d",
    "%Y/%m/%d",
    # EU day-first
    "%d/%m/%Y",
    "%d-%m-%Y",
    "%d.%m.%Y",
    "%d/%m/%y",
    "%d-%m-%y",
    "%d.%m.%y",
    # US month-first
    "%m/%d/%Y",
    "%m-%d-%Y",
    "%m/%d/%y",
    # Named months
    "%d %B %Y",
    "%d %b %Y",
    "%d %b, %Y",
    "%d %B, %Y",
    "%B %d, %Y",
    "%b %d, %Y",
    "%B %d %Y",
    "%b %d %Y",
    "%d-%b-%Y",
    "%d-%B-%Y",
    # Month-year only
    "%B %Y",
    "%b %Y",
]

_ORDINAL_RE = re.compile(r"(\d+)\s*(?:st|nd|rd|th)\b", re.IGNORECASE)
_MONTH_ABBREV_FIX = {"sept": "sep", "june": "jun", "july": "jul"}


def parse_date(
    raw: str,
    *,
    dayfirst: bool = True,
    vendor_hint: str = "",
) -> Optional[date]:
    """Parse a date string with locale awareness.

    Args:
        raw: Raw date string from document.
        dayfirst: If True (default), ambiguous dates like 01/02/2024 are
                  interpreted as DD/MM/YYYY (EU/UK). Set False for US vendors.
        vendor_hint: Optional vendor date_format_hint from profiles
                     (e.g., "DD/MM/YYYY", "MM/DD/YYYY").

    Returns:
        Parsed date or None if unparseable.
    """
    if not raw or not isinstance(raw, str):
        return None

    text = raw.strip()
    if not text:
        return None

    # Strip ordinals: "1st" -> "1", "23rd" -> "23"
    text = _ORDINAL_RE.sub(r"\1", text)

    # Normalize month abbreviations
    for wrong, right in _MONTH_ABBREV_FIX.items():
        text = re.sub(rf"\b{wrong}\b", right, text, flags=re.IGNORECASE)

    # If vendor hint specifies format, try it first
    if vendor_hint:
        hint_dayfirst = "DD" in vendor_hint.upper().split("/")[0] if "/" in vendor_hint else dayfirst
        if hint_dayfirst != dayfirst:
            dayfirst = hint_dayfirst

    # Try explicit formats first (avoids ambiguity of fuzzy parsing)
    formats = list(_DATE_FORMATS)
    if not dayfirst:
        # Prioritize US formats
        formats = [f for f in formats if f.startswith("%m")] + [
            f for f in formats if not f.startswith("%m")
        ]

    for fmt in formats:
        try:
            dt = datetime.strptime(text, fmt)
            return dt.date()
        except ValueError:
            continue

    # Fallback: dateutil with dayfirst control
    try:
        from dateutil import parser as dateutil_parser
        dt = dateutil_parser.parse(text, dayfirst=dayfirst, fuzzy=False)
        return dt.date()
    except Exception:
        pass

    # Last resort: fuzzy parsing (can extract date from surrounding text)
    try:
        from dateutil import parser as dateutil_parser
        dt = dateutil_parser.parse(text, dayfirst=dayfirst, fuzzy=True)
        return dt.date()
    except Exception:
        return None


# --- Numeric Cleaning ---

_CURRENCY_STRIP_RE = re.compile(r"[£$€¥₹\u00a3\u20ac\u00a5\u20b9A-Za-z,\s]")
_EUROPEAN_DECIMAL_RE = re.compile(
    r"^-?\d{1,3}(?:\.\d{3})+,\d{1,2}$"
)  # e.g., "1.234,56" or "12.345.678,90"
_PARENTHESIZED_NEGATIVE_RE = re.compile(r"^\((.+)\)$")


def clean_numeric(raw: Any) -> Optional[float]:
    """Parse a numeric value handling international formats.

    Handles:
    - Currency symbols: £1,234.56 -> 1234.56
    - European format: 1.234,56 -> 1234.56
    - Parenthesized negatives: (100.50) -> -100.50
    - Percentage: 20% -> 20.0
    - Thousands separators: 1,234,567.89 -> 1234567.89
    """
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)

    text = str(raw).strip()
    if not text:
        return None

    # Remove percentage sign
    text = text.replace("%", "").strip()

    # Check for parenthesized negative
    is_negative = False
    paren_match = _PARENTHESIZED_NEGATIVE_RE.match(text)
    if paren_match:
        text = paren_match.group(1).strip()
        is_negative = True

    if text.startswith("-"):
        is_negative = True
        text = text[1:].strip()

    # Detect European decimal format: 1.234,56
    if _EUROPEAN_DECIMAL_RE.match(text):
        # European: dots are thousands sep, comma is decimal
        text = text.replace(".", "").replace(",", ".")
    elif "," in text and "." in text:
        # Mixed: determine which is decimal separator
        last_comma = text.rfind(",")
        last_dot = text.rfind(".")
        if last_comma > last_dot:
            # Comma is decimal: 1.234,56
            text = text.replace(".", "").replace(",", ".")
        else:
            # Dot is decimal: 1,234.56
            text = text.replace(",", "")
    elif "," in text and "." not in text:
        # Could be thousands (1,234) or European decimal (1,5)
        parts = text.split(",")
        if len(parts) == 2 and len(parts[1]) <= 2:
            # Likely European decimal: 1,50 or 12,5
            text = text.replace(",", ".")
        else:
            # Likely thousands: 1,234 or 1,234,567
            text = text.replace(",", "")
    # else: just dots or no separators — leave as is

    # Strip remaining non-numeric characters
    text = re.sub(r"[^0-9.]", "", text)

    if not text:
        return None

    try:
        result = float(text)
        return -result if is_negative else result
    except ValueError:
        return None


# --- Currency Detection ---

_CURRENCY_SYMBOLS = {
    "£": "GBP",
    "€": "EUR",
    "¥": "JPY",
    "₹": "INR",
    "R$": "BRL",
    "kr": "SEK",
    "Fr": "CHF",
    "zł": "PLN",
}

# "$" is ambiguous — resolved by country/region context
_DOLLAR_CURRENCIES = {
    "US": "USD", "USA": "USD", "United States": "USD",
    "CA": "CAD", "Canada": "CAD",
    "AU": "AUD", "Australia": "AUD",
    "NZ": "NZD", "New Zealand": "NZD",
    "SG": "SGD", "Singapore": "SGD",
    "HK": "HKD", "Hong Kong": "HKD",
}

_CURRENCY_CODE_RE = re.compile(
    r"\b(USD|EUR|GBP|JPY|CAD|AUD|NZD|CHF|SEK|NOK|DKK|INR|BRL|SGD|HKD|ZAR|PLN|CZK|MXN|AED)\b",
    re.IGNORECASE,
)


def detect_currency(
    text: str,
    *,
    country: str = "",
    vendor_currency_hint: str = "",
) -> str:
    """Detect currency from text with disambiguation.

    Args:
        text: Document text to scan.
        country: Country hint for $ disambiguation.
        vendor_currency_hint: Currency hint from vendor profile.

    Returns:
        3-letter currency code (e.g., "USD", "GBP") or empty string.
    """
    if vendor_currency_hint:
        return vendor_currency_hint.upper()[:3]

    # Check for explicit currency codes first (most reliable)
    code_match = _CURRENCY_CODE_RE.search(text[:2000])
    if code_match:
        return code_match.group(1).upper()

    # Check for unambiguous symbols
    for symbol, code in _CURRENCY_SYMBOLS.items():
        if symbol in text[:2000]:
            return code

    # Handle ambiguous "$"
    if "$" in text[:2000]:
        country_upper = country.upper().strip()
        for key, code in _DOLLAR_CURRENCIES.items():
            if key.upper() in country_upper:
                return code
        return "USD"  # Default $ to USD

    return ""
