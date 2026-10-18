"""Date maths for /ask, done in code and handed to the model as a fact (test F2).

A model asked "has a contract signed on 15 June 2024 with a 21 month term expired?" has to add 21 months to a date and
compare the result with today. It does that unreliably (the live answer wrote the dates in a format the output filter then
withheld), and it does not know today's date. Here the arithmetic is plain date code:

  * a start date and a term found in the question  ->  expiry = start + term, and whether it is past or future;
  * a contract record (start and end dates, or start and term_months)  ->  the same, from the record.

The result is a short block of text for the prompt, marked as computed. Only unambiguous dates are read: a month written
as a word, or an ISO date. "03/04/2026" is day-first in the UK and month-first in the US, so it is not guessed at.
"""
from __future__ import annotations

import calendar
import datetime as dt
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

_MONTHS = {name.lower(): i for i, name in enumerate(calendar.month_name) if name}
_MONTHS.update({name.lower(): i for i, name in enumerate(calendar.month_abbr) if name})
_MONTH_RX = "|".join(sorted(_MONTHS, key=len, reverse=True))
_DAY_FIRST = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({_MONTH_RX})\.?,?\s+(\d{{4}})\b", re.I)
_MONTH_FIRST = re.compile(rf"\b({_MONTH_RX})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b", re.I)
_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_TERM = re.compile(r"\b(\d{1,3})[\s-]*(year|yr|month|mo|week|day)s?\b", re.I)
_START_CUE = re.compile(r"\b(?:signed|sign|effective|commenc\w*|start\w*|began|begin|from|dated|executed|entered\s+into)\b", re.I)


def today() -> dt.date:
    return dt.date.today()


def add_months(d: dt.date, months: int) -> dt.date:
    """``d`` plus whole months, clamping to the end of a shorter month (31 Jan + 1 month = 28/29 Feb)."""
    index = d.year * 12 + (d.month - 1) + months
    year, month = divmod(index, 12)
    return dt.date(year, month + 1, min(d.day, calendar.monthrange(year, month + 1)[1]))


def _dates(text: str) -> List[dt.date]:
    found: List[tuple] = []
    for m in _DAY_FIRST.finditer(text):
        found.append((m.start(), _safe(int(m.group(3)), _MONTHS[m.group(2).lower()], int(m.group(1)))))
    for m in _MONTH_FIRST.finditer(text):
        found.append((m.start(), _safe(int(m.group(3)), _MONTHS[m.group(1).lower()], int(m.group(2)))))
    for m in _ISO.finditer(text):
        found.append((m.start(), _safe(int(m.group(1)), int(m.group(2)), int(m.group(3)))))
    return [d for _, d in sorted(found, key=lambda t: t[0]) if d is not None]


def _safe(y: int, m: int, d: int) -> Optional[dt.date]:
    try:
        return dt.date(y, m, d)
    except ValueError:
        return None


@dataclass(frozen=True)
class DateFacts:
    start: Optional[dt.date]
    term: Optional[str]            # e.g. "21 months"
    expiry: dt.date
    today: dt.date
    source: str                    # "the question" | "the contract record"

    @property
    def days_from_today(self) -> int:
        return (self.expiry - self.today).days

    @property
    def expired(self) -> bool:
        return self.expiry <= self.today


def from_question(text: str, now: Optional[dt.date] = None) -> Optional[DateFacts]:
    """Start date + term written in the question -> the expiry. None unless there is exactly one date and one term."""
    if not isinstance(text, str):
        return None
    dates = _dates(text)
    terms = [m for m in _TERM.finditer(text)]
    if len(dates) != 1 or len(terms) != 1 or not _START_CUE.search(text):
        return None
    n, unit = int(terms[0].group(1)), terms[0].group(2).lower()
    start = dates[0]
    try:
        if unit in ("year", "yr"):
            expiry, label = add_months(start, 12 * n), f"{n} year{'s' if n != 1 else ''}"
        elif unit in ("month", "mo"):
            expiry, label = add_months(start, n), f"{n} month{'s' if n != 1 else ''}"
        elif unit == "week":
            expiry, label = start + dt.timedelta(weeks=n), f"{n} week{'s' if n != 1 else ''}"
        else:
            expiry, label = start + dt.timedelta(days=n), f"{n} day{'s' if n != 1 else ''}"
    except (OverflowError, ValueError):
        return None
    return DateFacts(start, label, expiry, now or today(), "the question")


def from_contract_row(row: Dict[str, Any], now: Optional[dt.date] = None) -> Optional[DateFacts]:
    """A contract record -> its expiry: the stored end date, else start + term_months. None if neither is there."""
    start = row.get("contract_start_date")
    end = row.get("contract_end_date")
    months = row.get("term_months")
    start = start if isinstance(start, dt.date) else None
    if isinstance(end, dt.date):
        return DateFacts(start, f"{int(months)} months" if months else None, end, now or today(), "the contract record")
    if start is not None and months not in (None, ""):
        try:
            return DateFacts(start, f"{int(months)} months", add_months(start, int(months)), now or today(), "the contract record")
        except (ValueError, OverflowError):
            return None
    return None


def render(facts: Optional[DateFacts]) -> str:
    """The computed facts as prompt text. Empty when there is nothing to say."""
    if facts is None:
        return ""
    lines = [f"Date calculation (done in code from {facts.source}; today is {facts.today.isoformat()}):"]
    if facts.start is not None:
        lines.append(f"- Start date: {facts.start.isoformat()}")
    if facts.term:
        lines.append(f"- Term: {facts.term}")
    lines.append(f"- Expiry date: {facts.expiry.isoformat()}")
    days = abs(facts.days_from_today)
    if facts.expired:
        lines.append(f"- Status: EXPIRED. The expiry date was {days} day{'s' if days != 1 else ''} before today.")
    else:
        lines.append(f"- Status: NOT YET EXPIRED. It expires in {days} day{'s' if days != 1 else ''}.")
    lines.append("State the expiry date and this status exactly as given; do not recompute them.")
    return "\n".join(lines)
