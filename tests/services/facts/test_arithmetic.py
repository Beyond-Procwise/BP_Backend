"""quantity × unit_rate = extended_line is what TESTS a role assignment
instead of trusting a label. Measured on bp_sqldb it holds for 78-93% of real
lines, and is structurally blind on the ~19% where quantity is 1."""
from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.services.facts.arithmetic import check_line_arithmetic  # noqa: E402
from src.services.facts.models import ArithmeticState  # noqa: E402

D = Decimal


def test_consistent_line():
    assert check_line_arithmetic(D("2"), D("86.94"), D("173.88")) is ArithmeticState.CONSISTENT


def test_a_total_booked_as_a_unit_rate_is_caught():
    """The historical bug: the line total landed in unit_price. With quantity
    40 the arithmetic is off by exactly the quantity factor."""
    assert check_line_arithmetic(D("40"), D("4586.65"), D("4586.65")) is ArithmeticState.INCONSISTENT


def test_quantity_one_is_untestable_not_consistent():
    """With quantity 1 a unit rate and a total are numerically identical, so
    the check CANNOT pass — it must abstain. Returning CONSISTENT here would
    manufacture confidence for a fifth of the corpus."""
    assert check_line_arithmetic(D("1"), D("500"), D("500")) is ArithmeticState.UNTESTABLE_QUANTITY_ONE


@pytest.mark.parametrize("q,u,e", [
    (None, D("10"), D("20")), (D("2"), None, D("20")), (D("2"), D("10"), None),
])
def test_missing_input_is_untestable(q, u, e):
    assert check_line_arithmetic(q, u, e) is ArithmeticState.UNTESTABLE_MISSING_INPUT


def test_rounding_tolerance_is_absolute_not_proportional():
    """A proportional tolerance scales the blind spot with the value — the
    larger the line, the more error it hides. completeness.py already learned
    this the hard way."""
    assert check_line_arithmetic(D("3"), D("10.00"), D("30.01")) is ArithmeticState.CONSISTENT
    assert check_line_arithmetic(D("3"), D("10000.00"), D("30001.00")) is ArithmeticState.INCONSISTENT


def test_zero_quantity_is_untestable_not_a_division():
    assert check_line_arithmetic(D("0"), D("10"), D("0")) is ArithmeticState.UNTESTABLE_MISSING_INPUT


def test_quantity_one_abstains_even_when_the_numbers_disagree():
    """The abstention is about what the check CAN establish, not about whether
    the two numbers happen to match. At quantity 1 the arithmetic carries no
    information about the role either way."""
    assert check_line_arithmetic(D("1"), D("500"), D("650")) is ArithmeticState.UNTESTABLE_QUANTITY_ONE


def test_negative_quantity_does_not_masquerade_as_quantity_one():
    """A credit note line is a real case; -1 is not 1 and must not borrow the
    quantity-one abstention."""
    assert check_line_arithmetic(D("-1"), D("10"), D("-10")) is ArithmeticState.CONSISTENT


def test_no_float_creeps_into_the_comparison():
    """Accepting a float here would reintroduce the drift the Decimal pipeline
    exists to prevent; it must be coerced, not compared as a float."""
    assert check_line_arithmetic(2, 86.94, 173.88) is ArithmeticState.CONSISTENT
