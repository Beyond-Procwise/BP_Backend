"""The window an analytic answer covers.

Scope is mandatory in the contract, and the period is the half of it a reader
checks first: a spend figure with no period attached is the question, not the
answer. It is computed here from a clock and carried through the answer, so
nothing downstream has to describe a window it was never told.

The fiscal year opens in April by default — the convention this organisation
reports on — and the month is settable, because that is a business fact rather
than a law.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

DEFAULT_FY_START_MONTH = 4


@dataclass(frozen=True)
class Period:
    start: date
    end: date
    label: str


def _fy_label(start: date) -> str:
    """FY26 is the year that opened in April 2026, as the reports are titled."""
    return f"FY{start.year % 100:02d} YTD"


def fiscal_year_to_date(today: date, fy_start_month: int = DEFAULT_FY_START_MONTH) -> Period:
    """From the opening of the current fiscal year up to and including ``today``."""
    year = today.year if today.month >= fy_start_month else today.year - 1
    start = date(year, fy_start_month, 1)
    return Period(start=start, end=today, label=_fy_label(start))


def _same_day_last_year(value: date) -> date:
    """The same date a year earlier — 29 February resolves to the 28th."""
    try:
        return value.replace(year=value.year - 1)
    except ValueError:
        return value.replace(year=value.year - 1, day=28)


def prior_year_equivalent(period: Period) -> Period:
    """The same span one year back, which is what a year-on-year delta compares to."""
    start = _same_day_last_year(period.start)
    end = _same_day_last_year(period.end)
    return Period(start=start, end=end, label=_fy_label(start))
