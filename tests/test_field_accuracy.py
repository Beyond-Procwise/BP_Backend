"""Tests for field accuracy utilities."""

from datetime import date

import pytest

from services.field_accuracy import parse_date, clean_numeric, detect_currency


class TestParseDate:
    def test_iso_format(self):
        assert parse_date("2024-03-15") == date(2024, 3, 15)

    def test_eu_day_first(self):
        assert parse_date("15/03/2024") == date(2024, 3, 15)

    def test_eu_day_first_dots(self):
        assert parse_date("15.03.2024") == date(2024, 3, 15)

    def test_us_format_with_dayfirst_false(self):
        assert parse_date("03/15/2024", dayfirst=False) == date(2024, 3, 15)

    def test_named_month(self):
        assert parse_date("15 March 2024") == date(2024, 3, 15)

    def test_named_month_us(self):
        assert parse_date("March 15, 2024") == date(2024, 3, 15)

    def test_ordinal(self):
        assert parse_date("1st January 2024") == date(2024, 1, 1)
        assert parse_date("23rd March 2024") == date(2024, 3, 23)

    def test_abbreviated_month(self):
        assert parse_date("15 Mar 2024") == date(2024, 3, 15)

    def test_sept_abbreviation(self):
        assert parse_date("15 Sept 2024") == date(2024, 9, 15)

    def test_empty_returns_none(self):
        assert parse_date("") is None
        assert parse_date(None) is None

    def test_vendor_hint_mm_dd(self):
        # Ambiguous date, vendor hint says MM/DD
        result = parse_date("01/02/2024", vendor_hint="MM/DD/YYYY")
        assert result == date(2024, 1, 2)

    def test_vendor_hint_dd_mm(self):
        result = parse_date("01/02/2024", vendor_hint="DD/MM/YYYY")
        assert result == date(2024, 2, 1)


class TestCleanNumeric:
    def test_simple_number(self):
        assert clean_numeric("1234.56") == 1234.56

    def test_with_currency_symbol(self):
        assert clean_numeric("£1,234.56") == 1234.56

    def test_euro_symbol(self):
        assert clean_numeric("€500.00") == 500.0

    def test_european_format(self):
        assert clean_numeric("1.234,56") == 1234.56

    def test_european_large(self):
        assert clean_numeric("12.345.678,90") == 12345678.90

    def test_european_small(self):
        assert clean_numeric("1,50") == 1.50

    def test_parenthesized_negative(self):
        assert clean_numeric("(100.50)") == -100.50

    def test_negative(self):
        assert clean_numeric("-500") == -500.0

    def test_percentage(self):
        assert clean_numeric("20%") == 20.0

    def test_thousands_separator(self):
        assert clean_numeric("1,234,567.89") == 1234567.89

    def test_integer(self):
        assert clean_numeric(42) == 42.0

    def test_float_passthrough(self):
        assert clean_numeric(3.14) == 3.14

    def test_none_returns_none(self):
        assert clean_numeric(None) is None

    def test_empty_string(self):
        assert clean_numeric("") is None

    def test_non_numeric(self):
        assert clean_numeric("N/A") is None


class TestDetectCurrency:
    def test_explicit_code(self):
        assert detect_currency("Total: 1,234.56 GBP") == "GBP"

    def test_pound_symbol(self):
        assert detect_currency("Total: £1,234.56") == "GBP"

    def test_euro_symbol(self):
        assert detect_currency("Total: €500.00") == "EUR"

    def test_dollar_default_usd(self):
        assert detect_currency("Total: $1,000.00") == "USD"

    def test_dollar_with_canada_country(self):
        assert detect_currency("Total: $1,000.00", country="Canada") == "CAD"

    def test_dollar_with_australia(self):
        assert detect_currency("Total: $1,000.00", country="Australia") == "AUD"

    def test_vendor_hint_overrides(self):
        assert detect_currency("Total: $1,000.00", vendor_currency_hint="EUR") == "EUR"

    def test_no_currency_found(self):
        assert detect_currency("No currency here") == ""

    def test_code_takes_priority_over_symbol(self):
        assert detect_currency("Amount: $500 CAD") == "CAD"
