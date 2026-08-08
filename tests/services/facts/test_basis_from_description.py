"""Recovering a billing period from the item description.

Some documents state the unit in the description rather than the unit column:
'HR Advisory & Employment Law Retainer (Quarterly)' is billed per quarter, and
the UoM column holds the deliverable noun instead.

This is inference over free text, which is where fabrication starts, so the
rule is deliberately narrow: a parenthetical whose ENTIRE content is a period
adverb. Every other shape in this corpus is a trap, and the negative cases
below are all real strings from proc.bp_product_master and the _trgt line
tables:

    'Enterprise Licence - 12 months'   a term length, billed per licence
    'Advanced Package (3 months)'      a duration, not a rate
    'Tier 3 Marketing (Months 1-10)'   a range
    '(4 visits per month)'             visit frequency, not a billing basis
    'Monthly Design Package'           an adjective in the product's name

A derived basis must also never be mistaken for one the document stated in the
unit column, so it carries BASIS_FROM_DESCRIPTION.
"""
from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.services.facts.uom import (  # noqa: E402
    BASIS_FROM_DESCRIPTION,
    basis_from_description,
)


@pytest.mark.parametrize("text,expected", [
    ("HR Advisory & Employment Law Retainer (Quarterly)", "quarter"),
    ("Payroll Managed Service (Quarterly)", "quarter"),
    ("Support Package (Monthly)", "month"),
    ("Licence (Annual)", "year"),
    ("Cleaning (Weekly)", "week"),
    ("Cover (Daily)", "day"),
    ("retainer (quarterly)", "quarter"),
    ("Service  ( Monthly )", "month"),
])
def test_a_bare_period_parenthetical_is_recovered(text, expected):
    r = basis_from_description(text)
    assert r is not None, f"{text!r} should yield a basis"
    assert r.canonical == expected
    assert BASIS_FROM_DESCRIPTION in r.reason_codes


@pytest.mark.parametrize("text", [
    # durations and term lengths -- NOT billing periods
    "Digital Learning Platform (Enterprise Licence - 12 months)",
    "Digital Learning Platform (Enterprise Licence – 12 months)",
    "Advanced Package (3 months)",
    "Bespoke Marketing Services (1 Month)",
    "Social Media Management - 3 Months",
    # ranges
    "Tier 3 Marketing Services (Months 1-10) 3-5 Posts Per Week",
    "Description Tier 3 Marketing Services (Months 1–10)",
    # a frequency of something other than billing
    "Enterprise IT Consulting Package (4 visits per month)",
    "Social Media Builder - 12 posts/month, engagement monitoring",
    # the period word is part of the product's name
    "Monthly Design & Marketing Package",
    "Monthly Design & Marketing Package April Payment",
    "Monthly Fee for Marketing Services",
    "Monthly management fee",
    "Quarterly Business Review Service",
    "MONTHLY COST: GENERAL IT CONSULTANT",
    # nothing at all
    "HP LaserJet Pro M404dn Mono Laser Printer",
    "",
])
def test_everything_else_is_refused(text):
    """A wrong basis is worse than none: it makes two incomparable numbers look
    comparable, which is the failure this whole model exists to prevent."""
    assert basis_from_description(text) is None, f"{text!r} was parsed"


def test_none_and_non_strings_are_safe():
    assert basis_from_description(None) is None
    assert basis_from_description(12345) is None  # type: ignore[arg-type]


def test_conflicting_periods_are_refused_rather_than_guessed():
    """Two different periods in one description means the document is not
    telling us one thing. Picking the first would be arbitrary."""
    assert basis_from_description("Package (Monthly) add-on (Quarterly)") is None


def test_the_same_period_twice_is_not_a_conflict():
    r = basis_from_description("Retainer (Quarterly) renewal (Quarterly)")
    assert r is not None and r.canonical == "quarter"


def test_a_recovered_period_carries_its_calendar_convention():
    r = basis_from_description("Retainer (Quarterly)")
    assert r.factor == Decimal("90")
    assert any("CALENDAR_CONVENTION" in c for c in r.reason_codes)


def test_a_long_parenthetical_is_refused_even_if_it_contains_a_period_word():
    """Bounded on purpose. A long parenthetical is prose, and prose mentioning
    a month is not a statement that the price is per month."""
    assert basis_from_description(
        "Service (billed in arrears on a monthly basis by agreement)") is None
