"""Validating types for the engineered extraction pipeline.

Every type in this module enforces its invariants at construction time.
A successfully-constructed value is guaranteed to be valid — downstream
code does not need to re-check format, range, or normalization. If input
is malformed, the constructor raises :class:`InvalidValue`. There is no
"None means invalid" convention here — that belongs in the parsers
layer (:mod:`extraction_v2.parsers`).

Design intent:
    - Types are load-bearing. ``buyer_id: SUP-AssurityLtd`` and
      ``buyer_id: 10 Redkiln Way`` are not the same kind of value;
      runtime types prevent the second from masquerading as the first.
    - Construction is total: ``Money(raw)`` either returns a Money or
      raises. Callers wanting a "try" semantics use the parsers module.
    - Normalization happens at the boundary: ``£1,234.56`` and
      ``1.234,56`` and ``1234.56`` all produce the same Money value.
"""
from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import Optional


__all__ = [
    "InvalidValue",
    "Money", "IsoDate", "Postcode",
    "InvoiceId", "PoId", "QuoteId",
    "Currency",
]


class InvalidValue(ValueError):
    """Raised when a typed constructor receives input that fails validation.

    The exception message identifies the type and the offending input so
    log lines and review-queue records carry actionable context.
    """


# ---------------------------------------------------------------------------
# Money
# ---------------------------------------------------------------------------

class Money(Decimal):
    """A monetary value normalized to two decimal places.

    Accepts: int, float, Decimal, or string with optional currency symbol
    (£/$/€/¥/₹), thousands separators (UK: ``,``; EU: ``.``), parens for
    negative ("(123.45)" → -123.45).

    Rejects: empty/None, non-numeric strings, values exceeding ``_MAX``.
    """

    _MAX = Decimal("9_999_999_999.99")
    _MIN = Decimal("-9_999_999_999.99")
    _CURRENCY_RE = re.compile(r"[£$€¥₹]")
    _SYMBOL_AMOUNT_RE = re.compile(r"^[A-Z]{3}\s+", re.I)  # "GBP 1234"

    def __new__(cls, raw):
        if raw is None or raw == "":
            raise InvalidValue("Money: input is None/empty")

        # Numeric inputs go through Decimal directly
        if isinstance(raw, (int, float)):
            try:
                d = Decimal(str(raw))
            except InvalidOperation as exc:
                raise InvalidValue(f"Money: {raw!r} not a valid number") from exc
        elif isinstance(raw, Decimal):
            d = raw
        elif isinstance(raw, str):
            d = cls._parse_string(raw)
        else:
            raise InvalidValue(f"Money: unsupported input type {type(raw).__name__}")

        # Range check — reject obvious outliers (likely extraction noise)
        if d > cls._MAX or d < cls._MIN:
            raise InvalidValue(f"Money: {d} out of plausible range")

        # Quantize to two decimal places
        normalized = d.quantize(Decimal("0.01"))
        return super().__new__(cls, normalized)

    @classmethod
    def _parse_string(cls, s: str) -> Decimal:
        original = s
        s = s.strip()
        if not s:
            raise InvalidValue("Money: empty string")

        # Reject "GBP 1234" — currency code embedded with amount confuses parsing
        if cls._SYMBOL_AMOUNT_RE.match(s):
            raise InvalidValue(f"Money: {original!r} contains currency code; pass amount only")

        # Strip currency symbols
        s = cls._CURRENCY_RE.sub("", s).strip()

        # Detect parentheses-negative
        is_negative = False
        m = re.match(r"^\((.+)\)$", s)
        if m:
            is_negative = True
            s = m.group(1).strip()

        # Detect leading minus
        if s.startswith("-"):
            is_negative = True
            s = s[1:].strip()

        # Strip leading + sign
        if s.startswith("+"):
            s = s[1:].strip()

        # Decimal-format heuristic:
        #   "1234.56"  -> en (decimal=., thousands=,)
        #   "1,234.56" -> en
        #   "1234,56"  -> eu (decimal=,)
        #   "1.234,56" -> eu (decimal=,, thousands=.)
        if "," in s and "." in s:
            # Whichever appears LAST is the decimal separator
            if s.rfind(",") > s.rfind("."):
                # EU: 1.234,56
                s = s.replace(".", "").replace(",", ".")
            else:
                # en: 1,234.56
                s = s.replace(",", "")
        elif "," in s and "." not in s:
            # Could be EU (1234,56) or en thousands-only (1,234)
            parts = s.split(",")
            if len(parts) == 2 and 1 <= len(parts[1]) <= 2:
                # 1234,56 — treat as EU decimal
                s = s.replace(",", ".")
            else:
                # 1,234 or 12,345,678 — thousands separator
                s = s.replace(",", "")
        # else: only "." or no separator — already in en form

        if not re.match(r"^\d+(?:\.\d+)?$", s):
            raise InvalidValue(f"Money: {original!r} is not a recognizable amount")

        try:
            d = Decimal(s)
        except InvalidOperation as exc:
            raise InvalidValue(f"Money: {original!r} failed Decimal parse") from exc

        if is_negative:
            d = -d
        return d


# ---------------------------------------------------------------------------
# IsoDate
# ---------------------------------------------------------------------------

class IsoDate(date):
    """A date in [2000-01-01, today + 5 years].

    Accepts: ISO-8601 strings, day-first formats ("10/10/2025" → Oct 10),
    "1st Oct, 2024", or a :class:`datetime.date` instance.

    Reject anything outside the plausible business-document window.
    """

    _MIN_YEAR = 2000

    @classmethod
    def _max_date(cls) -> date:
        return date.today() + timedelta(days=5 * 365)

    @classmethod
    def _min_date(cls) -> date:
        return date(cls._MIN_YEAR, 1, 1)

    def __new__(cls, raw):
        if raw is None:
            raise InvalidValue("IsoDate: input is None")

        if isinstance(raw, datetime):
            d = raw.date()
        elif isinstance(raw, date):
            d = raw
        elif isinstance(raw, str):
            d = cls._parse_string(raw)
        else:
            raise InvalidValue(f"IsoDate: unsupported input type {type(raw).__name__}")

        if not (cls._min_date() <= d <= cls._max_date()):
            raise InvalidValue(
                f"IsoDate: {d.isoformat()} outside plausible range "
                f"[{cls._min_date().isoformat()}, {cls._max_date().isoformat()}]"
            )

        return super().__new__(cls, d.year, d.month, d.day)

    @classmethod
    def _parse_string(cls, s: str) -> date:
        original = s
        s = s.strip()
        if not s:
            raise InvalidValue("IsoDate: empty string")

        # Fast path: ISO-8601
        m = re.match(r"^(\d{4})-(\d{2})-(\d{2})", s)
        if m:
            try:
                return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            except ValueError as exc:
                raise InvalidValue(f"IsoDate: {original!r} invalid ISO date") from exc

        # General path: dateparser handles dozens of formats
        try:
            import dateparser
        except ImportError:
            raise InvalidValue(
                "IsoDate: dateparser not installed; only ISO-8601 supported"
            )

        parsed = dateparser.parse(
            s,
            settings={
                "DATE_ORDER": "DMY",       # UK convention default
                "STRICT_PARSING": False,
                "PREFER_DAY_OF_MONTH": "first",
            },
        )
        if parsed is None:
            raise InvalidValue(f"IsoDate: {original!r} unparseable")
        return parsed.date()


# ---------------------------------------------------------------------------
# Postcode
# ---------------------------------------------------------------------------

class Postcode(str):
    """A normalized postcode with country inference.

    Recognizes UK (full or outcode-incode) and US ZIP (5 or 9-digit).
    Other countries can be added as needed.
    """

    _UK_RE = re.compile(r"^([A-Z]{1,2}\d[A-Z\d]?)\s*(\d[A-Z]{2})$", re.I)
    _US_RE = re.compile(r"^(\d{5})(?:-?(\d{4}))?$")

    def __new__(cls, raw):
        if not raw or not isinstance(raw, str):
            raise InvalidValue(f"Postcode: invalid input {raw!r}")
        s = raw.strip().upper()

        # Try UK first (more specific format)
        m = cls._UK_RE.match(s)
        if m:
            normalized = f"{m.group(1)} {m.group(2)}"
            obj = str.__new__(cls, normalized)
            obj._country = "United Kingdom"
            return obj

        # Try US ZIP
        m = cls._US_RE.match(s)
        if m:
            zip5, zip4 = m.group(1), m.group(2)
            normalized = f"{zip5}-{zip4}" if zip4 else zip5
            obj = str.__new__(cls, normalized)
            obj._country = "United States"
            return obj

        raise InvalidValue(f"Postcode: {raw!r} not a recognized format")

    @property
    def country(self) -> str:
        return self._country


# ---------------------------------------------------------------------------
# Document IDs (Invoice / PO / Quote)
# ---------------------------------------------------------------------------

class _DocId(str):
    """Common implementation for invoice / PO / quote IDs."""

    _PREFIXES: tuple[str, ...] = ()
    _DEFAULT_PREFIX: str = ""
    _BARE_NUMERIC_RE = re.compile(r"^\d{3,}(?:[-/]\d+)*$")
    _MIN_DIGITS = 3

    def __new__(cls, raw):
        if not raw or not isinstance(raw, str):
            raise InvalidValue(f"{cls.__name__}: invalid input {raw!r}")
        s = raw.strip()
        if not s:
            raise InvalidValue(f"{cls.__name__}: empty input")

        upper = s.upper()
        # Reject prefix-less or wrong-prefix garbage
        for forbidden in cls._forbidden_prefixes():
            if upper.startswith(forbidden):
                raise InvalidValue(
                    f"{cls.__name__}: {raw!r} has wrong prefix {forbidden!r}"
                )

        # Match a known prefix → uppercase the prefix part
        for prefix in cls._PREFIXES:
            if upper.startswith(prefix.upper()):
                # Reject naked prefix (e.g., "INV" with no digits)
                rest = s[len(prefix):]
                if not re.search(r"\d", rest):
                    raise InvalidValue(f"{cls.__name__}: {raw!r} prefix only, no digits")
                return str.__new__(cls, prefix.upper() + rest)

        # No prefix — check if bare-numeric (then add default prefix)
        if cls._BARE_NUMERIC_RE.match(s) and sum(c.isdigit() for c in s) >= cls._MIN_DIGITS:
            return str.__new__(cls, cls._DEFAULT_PREFIX + s)

        raise InvalidValue(f"{cls.__name__}: {raw!r} unrecognized format")

    @classmethod
    def _forbidden_prefixes(cls) -> tuple[str, ...]:
        # All known doc-id prefixes EXCEPT this class's own → reject
        all_prefixes = {"INV", "BILL", "PO", "PUR", "QUT", "QTE", "QUOTE"}
        own = {p.upper() for p in cls._PREFIXES}
        return tuple(all_prefixes - own)


class InvoiceId(_DocId):
    """An invoice ID — must start with INV/BILL or be normalized to INV+digits."""
    _PREFIXES = ("INV", "BILL")
    _DEFAULT_PREFIX = "INV"


class PoId(_DocId):
    """A purchase-order ID — must start with PO/PUR or normalize to PO+digits."""
    _PREFIXES = ("PO", "PUR")
    _DEFAULT_PREFIX = "PO"
    _MIN_DIGITS = 4  # PO numbers are usually ≥4 digits


class QuoteId(_DocId):
    """A quote ID — must start with QUT/QTE/QUOTE/Q+digits or normalize to QUT+digits."""
    _PREFIXES = ("QUT", "QTE", "QUOTE")
    _DEFAULT_PREFIX = "QUT"

    def __new__(cls, raw):
        # Accept "Q12345" style as well (no normalization needed)
        if isinstance(raw, str):
            s = raw.strip()
            if re.match(r"^Q\d{4,}$", s, re.I):
                return str.__new__(cls, s.upper())
        return super().__new__(cls, raw)


# ---------------------------------------------------------------------------
# Currency
# ---------------------------------------------------------------------------

class Currency(str):
    """ISO-4217 three-letter code, normalized from common symbols if present."""

    _ISO = frozenset({
        "GBP", "USD", "EUR", "JPY", "INR",
        "CAD", "AUD", "CHF", "CNY", "NZD", "ZAR", "SGD", "HKD", "AED",
        "SEK", "NOK", "DKK", "PLN", "CZK",
    })
    _SYMBOL_TO_ISO = {
        "£": "GBP",
        "$": "USD",
        "€": "EUR",
        "¥": "JPY",
        "₹": "INR",
    }

    def __new__(cls, raw):
        if not raw or not isinstance(raw, str):
            raise InvalidValue(f"Currency: invalid input {raw!r}")
        s = raw.strip()
        if not s:
            raise InvalidValue("Currency: empty input")

        # Symbol lookup
        if s in cls._SYMBOL_TO_ISO:
            return str.__new__(cls, cls._SYMBOL_TO_ISO[s])

        # ISO code (case-insensitive)
        upper = s.upper()
        if upper in cls._ISO:
            return str.__new__(cls, upper)

        raise InvalidValue(f"Currency: {raw!r} not a recognized ISO-4217 code or symbol")
