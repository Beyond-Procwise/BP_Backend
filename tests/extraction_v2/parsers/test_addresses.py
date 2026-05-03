"""Tests for the address parser. The parser splits a multi-line address
block into structured parts: line1 / line2 / city / postcode / country.

Contract: ``parse_address(raw)`` returns a ``ParsedAddress`` (possibly
with None fields). Never raises. Abstention is encoded as None in the
respective field.
"""
import pytest

from services.extraction_v2.parsers.addresses import ParsedAddress, parse_address


class TestParseAddressUK:
    """UK address blocks observed in production data."""

    def test_assurity_full(self):
        a = parse_address("10 Redkiln Way, Horsham, West Sussex RH13 5QH, United Kingdom")
        assert a.line1 == "10 Redkiln Way"
        assert a.city == "Horsham"
        assert a.postcode == "RH13 5QH"
        assert a.country == "United Kingdom"

    def test_assurity_multiline(self):
        a = parse_address(
            "10 Redkiln Way\nHorsham\nWest Sussex RH13 5QH\nUnited Kingdom"
        )
        assert a.line1 == "10 Redkiln Way"
        assert a.city == "Horsham"
        assert a.postcode == "RH13 5QH"
        assert a.country == "United Kingdom"

    def test_birmingham_with_floor(self):
        a = parse_address(
            "3rd Floor, Regent House\n45 Market Street\nBirmingham B3 1AA"
        )
        assert "Regent House" in (a.line1 or "")
        assert "Market Street" in (a.line2 or "") or "Market Street" in (a.line1 or "")
        assert a.city == "Birmingham"
        assert a.postcode == "B3 1AA"
        assert a.country == "United Kingdom"

    def test_unit_at_business_park(self):
        a = parse_address(
            "Unit 12, Meridian Business Park\nPhoenix Way\nLeicester LE19 1WX"
        )
        assert a.line1 == "Unit 12, Meridian Business Park"
        assert a.line2 == "Phoenix Way"
        assert a.city == "Leicester"
        assert a.postcode == "LE19 1WX"

    def test_london_short_form(self):
        a = parse_address("478 Branding Lane, SW16 7JD, London, UK")
        assert a.line1 == "478 Branding Lane"
        assert a.city == "London"
        assert a.postcode == "SW16 7JD"
        assert a.country == "United Kingdom"

    def test_postcode_no_space(self):
        a = parse_address("10 Redkiln Way Horsham West Sussex RH135QH")
        assert a.postcode == "RH13 5QH"   # parser normalizes

    def test_just_postcode(self):
        a = parse_address("RH13 5QH")
        assert a.line1 is None
        assert a.city is None
        assert a.postcode == "RH13 5QH"
        assert a.country == "United Kingdom"


class TestParseAddressUS:
    def test_us_simple(self):
        a = parse_address("123 Main Street, Springfield, IL 62701, USA")
        assert a.line1 == "123 Main Street"
        assert a.city == "Springfield"
        assert a.postcode == "62701"
        assert a.country == "United States"


class TestParseAddressEdgeCases:
    @pytest.mark.parametrize("bad", ["", "   ", None])
    def test_empty_returns_blank(self, bad):
        a = parse_address(bad)
        assert a == ParsedAddress(None, None, None, None, None)

    def test_no_postcode(self):
        a = parse_address("Some Street, Some City, Some Country")
        # No postcode → can't reliably split; everything stays in line1
        assert a.postcode is None
        # We don't assert specific city/line1 here — best-effort only
        # The important contract: parser doesn't raise

    def test_garbage_input(self):
        a = parse_address("foo bar baz")
        # No identifiable structure — line1 may have content but postcode/country None
        assert a.postcode is None
        assert a.country is None

    def test_handles_trailing_punctuation(self):
        a = parse_address("10 Redkiln Way, Horsham, West Sussex RH13 5QH.")
        assert a.postcode == "RH13 5QH"

    def test_two_addresses_picks_one(self):
        # Sometimes the source mashes two addresses; we pick the first
        # complete one (with postcode)
        a = parse_address(
            "10 Redkiln Way, Horsham, West Sussex RH13 5QH, "
            "and also 478 Branding Lane, SW16 7JD"
        )
        # Should anchor on the first postcode found
        assert a.postcode == "RH13 5QH"


class TestParsedAddressDataclass:
    def test_is_immutable(self):
        a = ParsedAddress("a", "b", "c", None, "d")
        with pytest.raises(Exception):
            a.line1 = "x"  # type: ignore[misc]

    def test_equality(self):
        assert (
            ParsedAddress("a", "b", "c", None, "d")
            == ParsedAddress("a", "b", "c", None, "d")
        )
