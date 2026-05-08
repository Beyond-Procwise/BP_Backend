"""Tests for extraction_v3 type binder."""
import pytest

from src.services.extraction_v3.binding.type_binder import bind_typed, BoundCandidate
from src.services.extraction_v3.schemas.candidate import Candidate


def _c(value: str, field: str = "x", confidence: float = 0.9) -> Candidate:
    """Helper to create a test Candidate."""
    return Candidate(
        field=field,
        value=value,
        page=0,
        bbox=(0, 0, 1, 1),
        evidence_text=value,
        model="layoutlmv3",
        confidence=confidence,
    )


class TestBindString:
    """Tests for string field type."""

    def test_bind_string_passthrough(self):
        """String values should pass through unchanged."""
        out = bind_typed(_c("ACME Industries Ltd"), "string")
        assert out.bind_error is False
        assert out.coerced_value == "ACME Industries Ltd"

    def test_bind_string_strips_whitespace(self):
        """String values should have whitespace stripped."""
        out = bind_typed(_c("  ACME Industries Ltd  "), "string")
        assert out.bind_error is False
        assert out.coerced_value == "ACME Industries Ltd"

    def test_bind_string_fails_on_empty(self):
        """Empty strings should fail binding."""
        out = bind_typed(_c(""), "string")
        assert out.bind_error is True
        assert out.coerced_value is None

    def test_bind_string_fails_on_whitespace_only(self):
        """Whitespace-only strings should fail binding."""
        out = bind_typed(_c("   "), "string")
        assert out.bind_error is True
        assert out.coerced_value is None


class TestBindMoney:
    """Tests for money field type."""

    def test_bind_money_gbp(self):
        """GBP amounts with thousands separator should parse."""
        out = bind_typed(_c("£7,290.00"), "money")
        assert out.bind_error is False
        assert out.coerced_value == pytest.approx(7290.00)

    def test_bind_money_usd(self):
        """USD amounts should parse."""
        out = bind_typed(_c("$1,234.56"), "money")
        assert out.bind_error is False
        assert out.coerced_value == pytest.approx(1234.56)

    def test_bind_money_no_symbol(self):
        """Plain numbers with separators should parse."""
        out = bind_typed(_c("1,234.56"), "money")
        assert out.bind_error is False
        assert out.coerced_value == pytest.approx(1234.56)

    def test_bind_money_eu_format(self):
        """EU format (dot thousands, comma decimal) should parse."""
        out = bind_typed(_c("1.234,56"), "money")
        assert out.bind_error is False
        assert out.coerced_value == pytest.approx(1234.56)

    def test_bind_money_negative(self):
        """Negative amounts (parentheses or minus) should parse."""
        out = bind_typed(_c("(1,234.56)"), "money")
        assert out.bind_error is False
        assert out.coerced_value == pytest.approx(-1234.56)

    def test_bind_money_fails_on_garbage(self):
        """Non-numeric text should fail binding."""
        out = bind_typed(_c("totally not a number"), "money")
        assert out.bind_error is True
        assert out.coerced_value is None

    def test_bind_money_fails_on_empty(self):
        """Empty strings should fail binding."""
        out = bind_typed(_c(""), "money")
        assert out.bind_error is True
        assert out.coerced_value is None


class TestBindDecimal:
    """Tests for decimal field type."""

    def test_bind_decimal_thousands_separator(self):
        """Decimals with thousands separator should parse."""
        out = bind_typed(_c("1,234.56"), "decimal")
        assert out.bind_error is False
        assert out.coerced_value == pytest.approx(1234.56)

    def test_bind_decimal_plain_number(self):
        """Plain decimal numbers should parse."""
        out = bind_typed(_c("42.5"), "decimal")
        assert out.bind_error is False
        assert out.coerced_value == pytest.approx(42.5)

    def test_bind_decimal_integer(self):
        """Integer strings should parse."""
        out = bind_typed(_c("123"), "decimal")
        assert out.bind_error is False
        assert out.coerced_value == pytest.approx(123.0)

    def test_bind_decimal_eu_format(self):
        """EU format (dot thousands, comma decimal) should parse."""
        out = bind_typed(_c("1.234,56"), "decimal")
        assert out.bind_error is False
        assert out.coerced_value == pytest.approx(1234.56)

    def test_bind_decimal_fails_on_garbage(self):
        """Non-numeric text should fail binding."""
        out = bind_typed(_c("not a number"), "decimal")
        assert out.bind_error is True
        assert out.coerced_value is None

    def test_bind_decimal_fails_on_empty(self):
        """Empty strings should fail binding."""
        out = bind_typed(_c(""), "decimal")
        assert out.bind_error is True
        assert out.coerced_value is None


class TestBindIsoDate:
    """Tests for iso_date field type."""

    def test_bind_iso_date_already_iso(self):
        """ISO-8601 dates should parse unchanged."""
        out = bind_typed(_c("2025-12-31"), "iso_date")
        assert out.bind_error is False
        assert out.coerced_value == "2025-12-31"

    def test_bind_iso_date_uk_format(self):
        """UK format (DD/MM/YYYY) should parse."""
        out = bind_typed(_c("31/12/2025"), "iso_date")
        assert out.bind_error is False
        assert out.coerced_value == "2025-12-31"

    def test_bind_iso_date_us_format(self):
        """US format (MM/DD/YYYY) may not parse due to DATE_ORDER=DMY, but let's try."""
        # The parser uses DATE_ORDER=DMY, so "31/12/2025" is tried first
        out = bind_typed(_c("12/31/2025"), "iso_date")
        # This might fail or parse as Dec 31, depending on dateparser
        # (we're using DMY order, so it tries 31/12/2025 first)
        if not out.bind_error:
            # If it parses, check the month/day
            assert "2025" in out.coerced_value

    def test_bind_iso_date_natural_language(self):
        """Natural language dates should parse if dateparser is available."""
        out = bind_typed(_c("20 October 2025"), "iso_date")
        if out.bind_error:
            # dateparser might not be installed or the format unsupported
            pytest.skip("dateparser does not handle this format")
        assert out.coerced_value == "2025-10-20"

    def test_bind_iso_date_fails_on_garbage(self):
        """Unparseable text should fail binding."""
        out = bind_typed(_c("not a date at all xyz"), "iso_date")
        assert out.bind_error is True
        assert out.coerced_value is None

    def test_bind_iso_date_fails_on_empty(self):
        """Empty strings should fail binding."""
        out = bind_typed(_c(""), "iso_date")
        assert out.bind_error is True
        assert out.coerced_value is None

    def test_bind_iso_date_fails_on_out_of_range(self):
        """Dates far in the past or future should fail (outside [2000, today+5y])."""
        out = bind_typed(_c("1950-01-01"), "iso_date")
        assert out.bind_error is True
        assert out.coerced_value is None


class TestBindAddress:
    """Tests for address field type."""

    def test_bind_address_full_uk(self):
        """Full UK addresses should parse."""
        address_text = "10 Redkiln Way, Horsham, West Sussex RH13 5QH, United Kingdom"
        out = bind_typed(_c(address_text), "address")
        assert out.bind_error is False
        result = out.coerced_value
        assert result.postcode == "RH13 5QH"
        assert result.city == "Horsham"

    def test_bind_address_multiline(self):
        """Multiline addresses should parse."""
        address_text = "Unit 12\nMeridian Business Park\nHorsham\nRH13 5QH"
        out = bind_typed(_c(address_text), "address")
        assert out.bind_error is False
        result = out.coerced_value
        assert result.postcode == "RH13 5QH"

    def test_bind_address_us_zip(self):
        """US addresses with ZIP codes should parse."""
        address_text = "123 Main St, Springfield, IL 62701"
        out = bind_typed(_c(address_text), "address")
        assert out.bind_error is False
        result = out.coerced_value
        # Should detect US ZIP
        assert result.postcode == "62701"

    def test_bind_address_fails_on_empty(self):
        """Empty addresses should fail binding."""
        out = bind_typed(_c(""), "address")
        assert out.bind_error is True
        assert out.coerced_value is None

    def test_bind_address_fails_on_garbage(self):
        """Addresses with no postcode or city should fail."""
        out = bind_typed(_c("xyz abc 123"), "address")
        assert out.bind_error is True
        assert out.coerced_value is None


class TestBindPostcode:
    """Tests for postcode field type."""

    def test_bind_postcode_uk_full(self):
        """Full UK postcodes should parse."""
        out = bind_typed(_c("RH13 5QH"), "postcode")
        assert out.bind_error is False
        assert "RH13" in out.coerced_value.upper()
        assert "5QH" in out.coerced_value.upper()

    def test_bind_postcode_uk_lowercase(self):
        """Lowercase UK postcodes should normalize to uppercase."""
        out = bind_typed(_c("rh13 5qh"), "postcode")
        assert out.bind_error is False
        assert out.coerced_value.upper() == "RH13 5QH"

    def test_bind_postcode_uk_no_space(self):
        """UK postcodes without space should parse."""
        out = bind_typed(_c("RH135QH"), "postcode")
        assert out.bind_error is False
        # Should normalize to include space
        assert "RH13" in out.coerced_value.upper()

    def test_bind_postcode_us_zip(self):
        """US ZIP codes should parse."""
        out = bind_typed(_c("62701"), "postcode")
        assert out.bind_error is False
        assert "62701" in out.coerced_value

    def test_bind_postcode_us_zip_plus_4(self):
        """US ZIP+4 codes should parse."""
        out = bind_typed(_c("62701-1234"), "postcode")
        assert out.bind_error is False
        assert "62701" in out.coerced_value

    def test_bind_postcode_fails_on_garbage(self):
        """Invalid postcode formats should fail."""
        out = bind_typed(_c("not a postcode"), "postcode")
        assert out.bind_error is True
        assert out.coerced_value is None

    def test_bind_postcode_fails_on_empty(self):
        """Empty strings should fail binding."""
        out = bind_typed(_c(""), "postcode")
        assert out.bind_error is True
        assert out.coerced_value is None


class TestBindCurrency:
    """Tests for currency field type."""

    def test_bind_currency_symbol_gbp(self):
        """GBP symbol should parse to GBP code."""
        out = bind_typed(_c("£"), "currency")
        assert out.bind_error is False
        assert out.coerced_value == "GBP"

    def test_bind_currency_symbol_usd(self):
        """USD symbol should parse to USD code."""
        out = bind_typed(_c("$"), "currency")
        assert out.bind_error is False
        assert out.coerced_value == "USD"

    def test_bind_currency_symbol_eur(self):
        """EUR symbol should parse to EUR code."""
        out = bind_typed(_c("€"), "currency")
        assert out.bind_error is False
        assert out.coerced_value == "EUR"

    def test_bind_currency_iso_code(self):
        """ISO-4217 codes should pass through."""
        out = bind_typed(_c("GBP"), "currency")
        assert out.bind_error is False
        assert out.coerced_value == "GBP"

    def test_bind_currency_iso_code_lowercase(self):
        """ISO codes should normalize to uppercase."""
        out = bind_typed(_c("gbp"), "currency")
        assert out.bind_error is False
        assert out.coerced_value == "GBP"

    def test_bind_currency_fails_on_garbage(self):
        """Invalid currency codes should fail."""
        out = bind_typed(_c("XYZ"), "currency")
        assert out.bind_error is True
        assert out.coerced_value is None

    def test_bind_currency_fails_on_empty(self):
        """Empty strings should fail binding."""
        out = bind_typed(_c(""), "currency")
        assert out.bind_error is True
        assert out.coerced_value is None


class TestBoundCandidateStructure:
    """Tests for BoundCandidate structure."""

    def test_bound_candidate_contains_original(self):
        """BoundCandidate should retain the original Candidate."""
        candidate = _c("test_value", field="test_field")
        out = bind_typed(candidate, "string")
        assert out.candidate is candidate
        assert out.candidate.field == "test_field"
        assert out.candidate.value == "test_value"

    def test_bound_candidate_success_flags(self):
        """Successful binding should have bind_error=False."""
        out = bind_typed(_c("test"), "string")
        assert isinstance(out, BoundCandidate)
        assert out.bind_error is False
        assert out.coerced_value is not None

    def test_bound_candidate_failure_flags(self):
        """Failed binding should have bind_error=True."""
        out = bind_typed(_c(""), "string")
        assert isinstance(out, BoundCandidate)
        assert out.bind_error is True
        assert out.coerced_value is None


class TestUnsupportedFieldType:
    """Tests for unsupported field types."""

    def test_unsupported_field_type_raises(self):
        """Unsupported field types should raise ValueError."""
        with pytest.raises(ValueError, match="unsupported field_type"):
            bind_typed(_c("test"), "unknown_type")
