"""Tests for parser wrappers: contract is 'returns typed value or None,
never raises'. These are thin wrappers around the typed constructors;
the heavy lifting lives in test_types.py and test_addresses.py."""
import pytest
from decimal import Decimal
from datetime import date

from services.extraction_v2.parsers import (
    parse_amount, parse_currency, parse_date, parse_postcode,
)


class TestParseAmount:
    def test_valid(self):
        assert parse_amount("£1,234.56") == Decimal("1234.56")

    @pytest.mark.parametrize("bad", [None, "", "abc", "1.2.3", "1e100", []])
    def test_returns_none_on_garbage(self, bad):
        assert parse_amount(bad) is None  # never raises


class TestParseDate:
    def test_valid(self):
        assert parse_date("2025-10-10") == date(2025, 10, 10)

    @pytest.mark.parametrize("bad", [None, "", "not a date", "2099-12-31", "1999-01-01"])
    def test_returns_none(self, bad):
        assert parse_date(bad) is None


class TestParsePostcode:
    def test_valid(self):
        pc = parse_postcode("rh13 5qh")
        assert str(pc) == "RH13 5QH"
        assert pc.country == "United Kingdom"

    @pytest.mark.parametrize("bad", [None, "", "abc", "ZZZ 999", 12345])
    def test_returns_none(self, bad):
        assert parse_postcode(bad) is None


class TestParseCurrency:
    def test_valid(self):
        assert parse_currency("£") == "GBP"
        assert parse_currency("usd") == "USD"

    @pytest.mark.parametrize("bad", [None, "", "XYZ", 12345])
    def test_returns_none(self, bad):
        assert parse_currency(bad) is None
