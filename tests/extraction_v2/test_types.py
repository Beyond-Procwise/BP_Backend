"""Tests for the validating type system. Every type's constructor must
either return a valid normalized value or raise InvalidValue. Code paths
that produce a typed value have already passed validation."""
from datetime import date, timedelta
from decimal import Decimal

import pytest

from services.extraction_v2.types import (
    Currency,
    InvalidValue,
    InvoiceId,
    IsoDate,
    Money,
    Postcode,
    PoId,
    QuoteId,
)


# ---------------------------------------------------------------------------
# Money
# ---------------------------------------------------------------------------

class TestMoney:
    @pytest.mark.parametrize("raw,expected", [
        ("1234.56",   Decimal("1234.56")),
        ("1234",      Decimal("1234.00")),
        ("£1,234.56", Decimal("1234.56")),
        ("$1,234.56", Decimal("1234.56")),
        ("€1,234.56", Decimal("1234.56")),
        ("1.234,56",  Decimal("1234.56")),     # EU format
        ("1234,56",   Decimal("1234.56")),     # bare EU
        ("(123.45)",  Decimal("-123.45")),     # parens-negative (credit)
        (1234,        Decimal("1234.00")),
        (1234.56,     Decimal("1234.56")),
        (Decimal("1234.56"), Decimal("1234.56")),
    ])
    def test_constructs_clean(self, raw, expected):
        assert Money(raw) == expected

    @pytest.mark.parametrize("bad", [
        "abc", "", "  ", "1.2.3", "1e100", None,
        "GBP 1234",  # currency code mixed with amount
        "twelve",
    ])
    def test_rejects_garbage(self, bad):
        with pytest.raises(InvalidValue):
            Money(bad)

    def test_rejects_above_max(self):
        with pytest.raises(InvalidValue):
            Money("99999999999.99")  # > 9_999_999_999.99

    def test_two_decimal_normalization(self):
        assert Money("1234.5") == Decimal("1234.50")
        assert Money("1234")   == Decimal("1234.00")
        assert Money("1234.567").quantize(Decimal("0.01")) == Decimal("1234.57")

    def test_addition_works(self):
        a = Money("100.00")
        b = Money("50.50")
        assert a + b == Decimal("150.50")


# ---------------------------------------------------------------------------
# IsoDate
# ---------------------------------------------------------------------------

class TestIsoDate:
    @pytest.mark.parametrize("raw,expected", [
        ("2025-10-10",     date(2025, 10, 10)),
        ("10/10/2025",     date(2025, 10, 10)),  # day-first (UK)
        ("1st Oct, 2024",  date(2024, 10, 1)),
        ("Oct 1, 2024",    date(2024, 10, 1)),
        ("01-10-2024",     date(2024, 10, 1)),
        ("2024-01-10",     date(2024, 1, 10)),
    ])
    def test_parses_common_formats(self, raw, expected):
        assert IsoDate(raw) == expected

    @pytest.mark.parametrize("bad", [
        "2099-12-31",         # too far future
        "1999-01-01",         # before 2000
        "not a date",
        "",
        None,
        "2024-13-01",         # invalid month
        "2024-01-32",         # invalid day
    ])
    def test_rejects_out_of_range_or_garbage(self, bad):
        with pytest.raises(InvalidValue):
            IsoDate(bad)

    def test_accepts_date_object(self):
        d = date(2025, 1, 15)
        assert IsoDate(d) == d

    def test_today_within_range(self):
        IsoDate(date.today())  # must not raise


# ---------------------------------------------------------------------------
# Postcode
# ---------------------------------------------------------------------------

class TestPostcode:
    @pytest.mark.parametrize("raw,country", [
        ("RH13 5QH",   "United Kingdom"),
        ("RH135QH",    "United Kingdom"),
        ("B3 1AA",     "United Kingdom"),
        ("B31AA",      "United Kingdom"),
        ("EC2A 3NW",   "United Kingdom"),
        ("M1 1AE",     "United Kingdom"),
        ("SW16 7JD",   "United Kingdom"),
        ("LE19 1WX",   "United Kingdom"),
        ("10001",      "United States"),
        ("90210",      "United States"),
        ("90210-1234", "United States"),
    ])
    def test_detects_country(self, raw, country):
        pc = Postcode(raw)
        assert pc.country == country

    @pytest.mark.parametrize("normalized_pair", [
        ("rh13 5qh", "RH13 5QH"),
        ("b3 1aa",   "B3 1AA"),
        ("RH135QH",  "RH13 5QH"),
        ("B31AA",    "B3 1AA"),
    ])
    def test_normalizes(self, normalized_pair):
        raw, expected = normalized_pair
        assert str(Postcode(raw)) == expected

    @pytest.mark.parametrize("bad", [
        "", "abc", "ZZZ 999", "12345 67890", "RH13", None,
    ])
    def test_rejects_garbage(self, bad):
        with pytest.raises(InvalidValue):
            Postcode(bad)


# ---------------------------------------------------------------------------
# InvoiceId / PoId / QuoteId
# ---------------------------------------------------------------------------

class TestInvoiceId:
    @pytest.mark.parametrize("raw,expected", [
        ("INV600820",      "INV600820"),
        ("INV-2026-01602", "INV-2026-01602"),
        ("inv600820",      "INV600820"),       # uppercase normalize
        ("132548",         "INV132548"),       # bare-numeric → prefix added
        ("BILL-001",       "BILL-001"),        # alt prefix accepted
    ])
    def test_normalizes_prefix(self, raw, expected):
        assert str(InvoiceId(raw)) == expected

    @pytest.mark.parametrize("bad", [
        "", None, "abc", "PO12345", "INV", "QUT123",
    ])
    def test_rejects_non_invoice(self, bad):
        with pytest.raises(InvalidValue):
            InvoiceId(bad)


class TestPoId:
    @pytest.mark.parametrize("raw,expected", [
        ("PO507222", "PO507222"),
        ("po-12345", "PO-12345"),
        ("507222",   "PO507222"),       # bare-numeric → prefix added
        ("PUR-001",  "PUR-001"),
    ])
    def test_normalizes(self, raw, expected):
        assert str(PoId(raw)) == expected

    @pytest.mark.parametrize("bad", [
        "", None, "abc", "INV12345", "PO", "1",  # too short
    ])
    def test_rejects(self, bad):
        with pytest.raises(InvalidValue):
            PoId(bad)


class TestQuoteId:
    @pytest.mark.parametrize("raw,expected", [
        ("QUT103107",      "QUT103107"),
        ("QTE-2026-00487", "QTE-2026-00487"),
        ("103107",         "QUT103107"),    # bare-numeric → prefix added
        ("Q10483",         "Q10483"),
    ])
    def test_normalizes(self, raw, expected):
        assert str(QuoteId(raw)) == expected

    @pytest.mark.parametrize("bad", [
        "", None, "abc", "INV12345", "PO12345", "QUT",
    ])
    def test_rejects(self, bad):
        with pytest.raises(InvalidValue):
            QuoteId(bad)


# ---------------------------------------------------------------------------
# Currency
# ---------------------------------------------------------------------------

class TestCurrency:
    @pytest.mark.parametrize("raw,expected", [
        ("GBP", "GBP"),
        ("USD", "USD"),
        ("gbp", "GBP"),
        ("£",   "GBP"),
        ("$",   "USD"),
        ("€",   "EUR"),
        ("¥",   "JPY"),
        ("₹",   "INR"),
    ])
    def test_iso_or_symbol(self, raw, expected):
        assert Currency(raw) == expected

    @pytest.mark.parametrize("bad", [
        "", None, "XYZ", "DOLLARS", "12345",
    ])
    def test_rejects_unknown(self, bad):
        with pytest.raises(InvalidValue):
            Currency(bad)
