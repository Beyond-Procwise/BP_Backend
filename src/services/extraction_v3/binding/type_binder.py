"""Type-coerce a Candidate.value string into a typed Python value per the
schema's `type` field. Coercion failure → bind_error=True, never silent null."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from src.services.extraction_v3.schemas.candidate import Candidate

FieldType = Literal["string", "iso_date", "money", "decimal", "address", "postcode", "currency"]


@dataclass
class BoundCandidate:
    """Result of type coercion for a candidate value.

    Attributes:
        candidate: Original Candidate instance
        coerced_value: The typed value, or None if bind failed
        bind_error: True if coercion failed; False if successful
    """
    candidate: Candidate
    coerced_value: Any
    bind_error: bool


def bind_typed(candidate: Candidate, field_type: FieldType) -> BoundCandidate:
    """Coerce candidate.value to a typed Python value per field_type.

    Returns BoundCandidate with bind_error=True if coercion fails. The L3
    orchestrator routes bind_errors to the judge as a tiebreaker between
    conflicting candidates, and to the review queue if no candidate coerces.

    Args:
        candidate: The Candidate to coerce
        field_type: Target type (string, iso_date, money, decimal, address, postcode, currency)

    Returns:
        BoundCandidate with coerced_value and bind_error flag
    """
    raw = (candidate.value or "").strip()

    if field_type == "string":
        # No coercion needed — just validate non-empty
        if not raw:
            return BoundCandidate(candidate, None, bind_error=True)
        return BoundCandidate(candidate, raw, bind_error=False)

    if field_type == "money":
        from src.services.extraction_v2.parsers.amounts import parse_amount

        result = parse_amount(raw)
        if result is None:
            return BoundCandidate(candidate, None, bind_error=True)
        # Money is a Decimal subclass; convert to float for downstream consumption
        return BoundCandidate(candidate, float(result), bind_error=False)

    if field_type == "decimal":
        # Try parse_amount first (handles "1,234.56" etc.); fall back to float()
        from src.services.extraction_v2.parsers.amounts import parse_amount

        result = parse_amount(raw)
        if result is not None:
            return BoundCandidate(candidate, float(result), bind_error=False)
        try:
            return BoundCandidate(candidate, float(raw.replace(",", "")), bind_error=False)
        except (ValueError, TypeError, AttributeError):
            return BoundCandidate(candidate, None, bind_error=True)

    if field_type == "iso_date":
        from src.services.extraction_v2.parsers.dates import parse_date
        import re

        # Pre-filter: reject strings that don't contain recognisable date structure.
        # dateparser is too permissive — "$3" → "3rd of current month", "5" → today.
        # A legitimate date must have a 4-digit year, or a separator pattern
        # (e.g. 10/14), or a month name paired with at least one digit.
        # NOTE: dot is intentionally excluded from separators — "0.00" and "$3.95"
        # look like money and would incorrectly match \d{1,2}\.\d{1,2}.  Dates
        # using dot separators (European style "01.09") also contain a 4-digit
        # year in our dataset, so _has_year catches them.
        _has_year = re.search(r"20\d{2}", raw)
        _has_sep = re.search(r"\d{1,2}[/\-]\d{1,2}", raw)
        _has_month_name = re.search(
            r"(january|february|march|april|may|june|july|august|september|october|"
            r"november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)",
            raw, re.IGNORECASE,
        )
        if not (_has_year or _has_sep or (_has_month_name and re.search(r"\d", raw))):
            return BoundCandidate(candidate, None, bind_error=True)

        result = parse_date(raw)
        if result is None:
            return BoundCandidate(candidate, None, bind_error=True)
        # IsoDate is a date subclass; convert to ISO string
        return BoundCandidate(candidate, result.isoformat(), bind_error=False)

    if field_type == "address":
        from src.services.extraction_v2.parsers.addresses import parse_address

        # parse_address always returns ParsedAddress (never None); check if it has structured data
        # Require at least a postcode OR both city and a line address to be useful
        result = parse_address(raw)
        if result.postcode or (result.city and result.line1):
            return BoundCandidate(candidate, result, bind_error=False)
        return BoundCandidate(candidate, None, bind_error=True)

    if field_type == "postcode":
        from src.services.extraction_v2.parsers.postcodes import parse_postcode

        result = parse_postcode(raw)
        if result is None:
            return BoundCandidate(candidate, None, bind_error=True)
        # Postcode is a str subclass; return the string value
        return BoundCandidate(candidate, str(result), bind_error=False)

    if field_type == "currency":
        from src.services.extraction_v2.parsers.currency import parse_currency

        result = parse_currency(raw)
        if result is None:
            return BoundCandidate(candidate, None, bind_error=True)
        # Currency is a str subclass; return the string value
        return BoundCandidate(candidate, str(result), bind_error=False)

    raise ValueError(f"unsupported field_type: {field_type}")
