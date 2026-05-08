"""Tests for currency detection and normalization."""
from __future__ import annotations

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.services.extraction_v2.currency import (  # noqa: E402
    detect_currency, normalize_currency_in_place,
)


def test_detect_explicit_iso_code():
    assert detect_currency("Total 1,234.56 EUR") == "EUR"
    assert detect_currency("USD 999.00 due") == "USD"


def test_detect_unambiguous_symbol():
    assert detect_currency("£1,500 due 2026-05-04") == "GBP"
    assert detect_currency("€999.00") == "EUR"
    assert detect_currency("₹50,000") == "INR"


def test_detect_dollar_resolved_by_country():
    assert detect_currency("$1,000", header_country="Australia") == "AUD"
    assert detect_currency("$1,000", header_country="United States") == "USD"
    assert detect_currency("$1,000", header_country="Canada") == "CAD"


def test_detect_yen_resolved_by_country():
    assert detect_currency("¥10000", header_country="Japan") == "JPY"
    assert detect_currency("¥10000", header_country="China") == "CNY"


def test_detect_country_prior_alone_when_no_symbol():
    """When the document has no symbol at all, the country prior is used."""
    assert detect_currency("Total 1234.56", header_country="Germany") == "EUR"


def test_detect_returns_none_when_nothing_matches():
    assert detect_currency("just some prose") is None


def test_detect_country_from_vendor_address_text():
    text = "Acme Corp\n10 Main Street\nLondon\nUnited Kingdom\nTotal: $999"
    # $ is ambiguous — country in text resolves to GBP
    assert detect_currency(text) == "GBP"


def test_normalize_in_place_sets_header_currency():
    header = {"supplier_id": "X", "country": "Germany"}
    line_items = [{"line_amount": 100}, {"line_amount": 50}]
    out = normalize_currency_in_place(header, line_items, parsed_text="Total €150")
    assert out == "EUR"
    assert header["currency"] == "EUR"
    assert all(item["currency"] == "EUR" for item in line_items)


def test_normalize_in_place_does_not_overwrite_existing_currency():
    header = {"currency": "GBP"}
    line_items = [{"line_amount": 100}]
    out = normalize_currency_in_place(header, line_items, parsed_text="Total €150")
    assert out == "GBP"
    assert header["currency"] == "GBP"
    assert line_items[0]["currency"] == "GBP"


def test_normalize_in_place_does_not_overwrite_per_line_currency():
    header = {}
    line_items = [
        {"line_amount": 100, "currency": "USD"},
        {"line_amount": 50},  # missing — gets filled
    ]
    out = normalize_currency_in_place(
        header, line_items, parsed_text="Mixed €150 USD",
    )
    # Line 0 keeps its own currency; line 1 inherits the detected currency
    assert line_items[0]["currency"] == "USD"
    assert line_items[1]["currency"] in {"EUR", "USD"}


def test_normalize_in_place_handles_empty_lines():
    header = {}
    out = normalize_currency_in_place(header, [], parsed_text="Total £999")
    assert out == "GBP"
    assert header["currency"] == "GBP"
