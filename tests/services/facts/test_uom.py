"""The UoM normaliser must map the corpus's real units and refuse its junk.

The 28 values below are every distinct unit_of_measure across
bp_quote_line_items_trgt, bp_po_line_items_trgt and bp_invoice_line_items_trgt.
Fourteen are units. Fourteen are payment terms, scope descriptions or prices
that landed in the UoM column, and coercing any of them into a unit would be
fabrication.
"""
from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.services.facts.uom import UOM_UNMAPPED, normalise_uom  # noqa: E402

MAPPED = {
    "each": ("each", "count"),
    "case": ("case", "count"),
    "pack": ("pack", "count"),
    "box": ("box", "count"),
    "seat": ("seat", "count"),
    "licence": ("licence", "count"),
    "shipment": ("shipment", "count"),
    "hour": ("hour", "time"),
    "day": ("day", "time"),
    "week": ("week", "time"),
    "month": ("month", "time"),
    "year": ("year", "time"),
    "tonne": ("tonne", "mass"),
    "metre": ("metre", "length"),
}

UNMAPPED = [
    "included",
    "annual in advance",
    "30 days from quote date",
    "21 days from quote date",
    "45 days from quote date",
    "30 days from invoice",
    "45 days from invoice",
    "transition 5 weeks",
    "transition 6 weeks",
    "transition 7 weeks",
    "onboarding 7 weeks",
    "onboarding 10 weeks",
    "implementation (one-off, fixed) — £72,000.00",
    "implementation (one-off, fixed) — £58,000.00",
]


@pytest.mark.parametrize("raw,expected", sorted(MAPPED.items()))
def test_real_units_normalise(raw, expected):
    r = normalise_uom(raw)
    assert (r.canonical, r.dimension) == expected
    assert UOM_UNMAPPED not in r.reason_codes


@pytest.mark.parametrize("raw", UNMAPPED)
def test_junk_is_refused_not_coerced(raw):
    r = normalise_uom(raw)
    assert r.canonical is None, f"{raw!r} was coerced to {r.canonical!r}"
    assert r.dimension is None
    assert UOM_UNMAPPED in r.reason_codes


@pytest.mark.parametrize("raw", ["EACH", " Each ", "eaches", "EA", "hrs", "Hours"])
def test_case_whitespace_and_common_abbreviations(raw):
    """Real documents do not spell units the way the seeder did."""
    assert normalise_uom(raw).canonical is not None


@pytest.mark.parametrize("raw", [None, "", "   "])
def test_absent_uom_is_unmapped_not_defaulted(raw):
    """A missing unit must never silently become 'each' — benchmark_live already
    does that at its own layer, and doing it here would bake the guess into the
    fact base."""
    r = normalise_uom(raw)
    assert r.canonical is None
    assert UOM_UNMAPPED in r.reason_codes


def test_result_is_hashable_and_frozen():
    r = normalise_uom("each")
    with pytest.raises(Exception):
        r.canonical = "box"  # type: ignore[misc]


def test_time_units_carry_a_factor_to_a_common_basis():
    """Cross-document comparison needs hour/day/week/month/year on one basis.
    Months and years are calendar-ambiguous, so the factor is stated in days
    with the convention recorded in reason_codes, not silently assumed."""
    assert normalise_uom("day").factor == Decimal("1")
    assert normalise_uom("week").factor == Decimal("7")
    assert normalise_uom("hour").factor is not None
    r = normalise_uom("month")
    assert r.factor is not None
    assert any("CALENDAR_CONVENTION" in c for c in r.reason_codes)
