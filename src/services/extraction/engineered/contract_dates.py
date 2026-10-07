"""When a contract starts and ends, read from its own words.

WHY THIS EXISTS. On 2026-10-04 the real Marketing Agreement would not promote on
bp_sqldb: `contract_start_date` was NULL and blocking, on a document that states
its start date three times over --

    ... is entered into on June 12, 2025 ( ' the Effective Date') by and between ...
    ... (referred to as the "Effective Date"). It will end on December 12, 2025
    Name: John Smith  Signature: ______  Date: June 12, 2025

-- and nothing read any of them. `contract_start_date` and `contract_end_date`
declare sixteen `canonical_labels` between them in
`extraction_schemas/contract.yaml` and have **no patterns at all**: the same
structural hole `supplier_id` had. The only path was the context layer, which
returned nothing, and the field carried no provenance entry because no candidate
was ever produced.

NOT a date-parsing problem. The runtime binder reads every one of these shapes
(`parse_date("June 12, 2025") -> 2025-06-12`, dateparser 1.4.0 in `.venv`). The
four `TestIsoDate` failures in the suite are the two-venv trap -- `venv`, which
pytest uses, has no dateparser -- and not a defect in the date code. So this
module's whole job is to PRODUCE the candidate; parsing already worked.

HOW IT READS. A labelled field first (`Effective Date: 5 January 2026`), then
prose (`entered into on 5 January 2026`, `It will end on 4 January 2029`). In both
cases the date must be the FIRST thing after the label or connector, inside a short
window -- which is what stops `shall be effective on the date of signing this
Agreement` from reaching forward to some later date, and what stops
`Effective Date: 5 January 2026 End Date: 4 January 2029` (one line, as the parser
renders it) from reading the end date as the start.

WHAT IT REFUSES, each one a test:
  * a label with no date after it -- the real document names "Effective Date"
    twice and states no date in either place;
  * a bare `Date:` label, which in a signature block sits beside every field and
    means the day somebody signed. Reading that as the start is how a renewal
    gets the wrong anniversary;
  * a date in prose with no start/end wording ("The Parties met on 3 February");
  * a value dateparser cannot parse ("the first Tuesday after Michaelmas");
  * a term whose start falls after its end -- that is a misread, not two facts,
    and both are dropped.

A TERM STATED AS A DURATION. Some contracts never print an end date: "shall
commence on 5 January 2026 and shall continue for a period of thirtysix (36)
months". When -- and only when -- no end date is printed, the end is worked out as
start + N months - 1 day and the candidate carries `DERIVED_END_PATTERN` as its
pattern_name, so provenance says "derived", never "read". Narrow on purpose: the
duration must follow the start date in the same sentence ("continue for", "remain
in force for"), so a payment term, a notice period, a price-fix window or a renewal
period is never mistaken for the term; words and digits must agree; and a start on
a day the end month lacks (31 January + 1 month) is refused, because there is no
single answer.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from ..types import Candidate, Span
from .contract_parties import CONFIDENCE, _squeeze

START_FIELD = "contract_start_date"
END_FIELD = "contract_end_date"

#: pattern_name recorded in a field's provenance when the end was worked out from
#: a stated duration rather than read as a date. The marker IS the provenance.
DERIVED_END_PATTERN = "contract_term_duration_derived"

_MONTH = (r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
          r"jul(?:y)?|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|"
          r"dec(?:ember)?)")

#: A date, in the shapes contracts actually print. Anchored by the caller at the
#: start of the window, so it is never found loose in a sentence.
_DATE = re.compile(
    r"(?i)^\s*("
    r"\d{4}-\d{2}-\d{2}"                                        # 2026-01-05
    r"|\d{1,2}(?:st|nd|rd|th)?\s+" + _MONTH + r"\.?,?\s+\d{4}"  # 5 January 2026
    r"|" + _MONTH + r"\.?\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}"  # June 12, 2025
    r"|\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}"                         # 12/06/2025
    r")"
)

#: `Effective Date:`. A bare `Date:` is deliberately absent -- see the docstring.
_START_LABEL = re.compile(
    r"(?:^|[\s\n])(?i:effective\s+date|start\s+date|commencement\s+date|"
    r"contract\s+start\s+date|date\s+of\s+agreement|agreement\s+date|"
    r"contract\s+date|signed\s+date)\s*[:–]\s*"
)
_END_LABEL = re.compile(
    r"(?:^|[\s\n])(?i:expiry\s+date|end\s+date|termination\s+date|"
    r"contract\s+end\s+date|expiration\s+date|valid\s+until|expires\s+on|"
    r"term\s+end)\s*[:–]\s*"
)

#: Prose that introduces the start of a term, immediately followed by the date.
#: STRONG: wording that can only be about this document's own term.
_START_PROSE_STRONG = re.compile(
    r"(?i)(?:"
    r"\b(?:shall\s+|will\s+)?commenc(?:e|es|ing)\s+(?:on|from)\s+"
    r"|\beffective\s+(?:on|from|as\s+of)\s+"
    r"|\bwith\s+effect\s+from\s+"
    r"|\b(?:shall\s+)?begin(?:s|ning)?\s+on\s+"
    r")"
)

#: WEAK: "dated 5 January 2026" is about whatever was named just before it, which
#: may be ANOTHER agreement -- "governed by Framework Agreement No. FA-2026-0042
#: dated 5 January 2026" gave an order form its framework's start date, a month
#: early and entirely plausible. Tried only when nothing strong is stated, and
#: refused when a cited agreement sits in front of it (_CITED_AGREEMENT).
_START_PROSE_WEAK = re.compile(
    r"(?i)\b(?:entered\s+into|made|executed|signed|dated)\s+(?:on|as\s+of)?\s*"
)

#: Another document's name, immediately before a date. The date is that
#: document's, not this one's.
_CITED_AGREEMENT = re.compile(
    r"(?i)(?:framework|master|parent|principal|head|umbrella)\s+"
    r"(?:services?\s+)?(?:agreement|contract)"
    r"(?:\s*(?:number|no|ref|reference)\.?\s*:?\s*[A-Z0-9][A-Z0-9\-/\.]*)?\s*$"
)
_END_PROSE = re.compile(
    r"(?i)(?:"
    r"\b(?:shall\s+|will\s+)?(?:end|expire|terminate)s?\s+on\s+"
    r"|\b(?:up\s+to\s+and\s+including|through\s+to|through|until)\s+"
    r")"
)


@dataclass(frozen=True)
class ContractDates:
    """ISO values for the column, and the literal text they were read from."""

    start: Optional[str] = None
    end: Optional[str] = None
    start_text: Optional[str] = None
    end_text: Optional[str] = None
    end_derived: bool = False


#: Every spelling accepted, in full. Keyed on the WHOLE word, not a three-letter
#: prefix: prefix matching read "Februbry 5, 2026" as February and would read
#: "Octopus 5, 2026" as October. _DATE's own alternation would reject both in
#: context, but _iso has to be right on its own or the guard is one layer thick.
_MONTH_NUMBER = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

_ISO_SHAPE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
_DAY_MONTH_YEAR = re.compile(r"(?i)^(\d{1,2})(?:st|nd|rd|th)?\s+([a-z]{3,9})\.?,?\s+(\d{4})$")
_MONTH_DAY_YEAR = re.compile(r"(?i)^([a-z]{3,9})\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})$")
_NUMERIC = re.compile(r"^(\d{1,2})[/.-](\d{1,2})[/.-](\d{2,4})$")


def _iso(raw: str) -> Optional[str]:
    """`raw` as YYYY-MM-DD, or None if it is not one of the shapes _DATE accepts.

    DELIBERATELY DEPENDENCY-FREE. The first version called the pipeline's
    `parse_date`, which is `dateparser` underneath -- installed in `.venv`, which
    the server runs, and NOT in `venv`, which pytest runs. So the reader produced
    nothing at all under test while working in production: the exact two-venv trap
    that hid the supplier bug for months, this time pointed at my own tests. A
    module that only matches four date shapes can convert those four itself, and
    then it behaves the same everywhere.

    The numeric form is read DAY-FIRST (12/06/2025 -> 2025-06-12). That is a
    choice, not a guess: it matches what `dateparser` already answers for this
    corpus, which is UK (GBP, "the laws of England and Wales"), so the pipeline
    does not change its mind about a date depending on which reader saw it.
    """
    text = _squeeze(raw)

    m = _ISO_SHAPE.match(text)
    if m:
        y, mo, d = (int(g) for g in m.groups())
        return _build(y, mo, d)

    m = _DAY_MONTH_YEAR.match(text)
    if m:
        d, mon, y = m.group(1), m.group(2).lower(), m.group(3)
        if mon in _MONTH_NUMBER:
            return _build(int(y), _MONTH_NUMBER[mon], int(d))
        return None

    m = _MONTH_DAY_YEAR.match(text)
    if m:
        mon, d, y = m.group(1).lower(), m.group(2), m.group(3)
        if mon in _MONTH_NUMBER:
            return _build(int(y), _MONTH_NUMBER[mon], int(d))
        return None

    m = _NUMERIC.match(text)
    if m:
        d, mo, y = (int(g) for g in m.groups())
        if y < 100:
            y += 2000
        return _build(y, mo, d)

    return None


def _build(year: int, month: int, day: int) -> Optional[str]:
    """A real calendar date, or None. 31 February is not a date."""
    from datetime import date
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None


def _first_date_after(text: str, pos: int, window: int = 40) -> Optional[tuple[str, str]]:
    """``(iso, literal)`` for a date that starts within `window` chars of `pos`.

    Anchored: the date must be the first thing there. A loose search would read
    "effective on the date of signing this Agreement ... It will end on
    December 12, 2025" as a start date of December 12th.
    """
    m = _DATE.match(text[pos:pos + window])
    if not m:
        return None
    literal = m.group(1)
    iso = _iso(literal)
    return (iso, literal) if iso else None


def _cites_another_agreement(text: str, pos: int) -> bool:
    """Does another agreement's name sit immediately before `pos`?

    Checked on the 90 characters in front of the connector, whitespace collapsed,
    so "...governed by Framework Agreement No. FA-2026-0042 dated" is recognised
    however the parser wrapped it.
    """
    return bool(_CITED_AGREEMENT.search(_squeeze(text[max(0, pos - 90):pos])))


def _from(text: str, label: re.Pattern, *prose: re.Pattern) -> tuple[Optional[str], Optional[str]]:
    """A labelled date if the document has one, else prose, strongest tier first.

    The tiers matter: an order form that cites its framework's date BEFORE stating
    its own term was given the framework's date, because the first prose match won
    and the weak connector came first in the text.
    """
    for m in label.finditer(text):
        got = _first_date_after(text, m.end())
        if got:
            return got
    for pattern in prose:
        for m in pattern.finditer(text):
            if pattern is _START_PROSE_WEAK and _cites_another_agreement(text, m.start()):
                continue          # that date belongs to the agreement named here
            got = _first_date_after(text, m.end())
            if got:
                return got
    return (None, None)

_ONES = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
         "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
         "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
         "eighteen": 18, "nineteen": 19}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60}

#: A duration immediately after the start date: "and shall continue for a period of
#: thirtysix (36) months". Anchored to the end of the start date, so a duration
#: elsewhere in the document is never in reach.
_DURATION_AFTER_START = re.compile(
    r"(?i)^[\s,]*(?:and\s+)?(?:shall\s+|will\s+)?"
    r"(?:continue|remain\s+in\s+(?:force|effect)|run)\s+for\s+"
    r"(?:an?\s+)?(?:(?:initial\s+)?(?:period|term)\s+of\s+)?"
    r"(?P<num>\d{1,4}|[a-z][a-z\- ]{0,28}?)\s*(?:\((?P<digits>\d{1,4})\)\s*)?"
    r"(?P<unit>months?|years?)\b"
)

#: Longest term worth believing, in months. 9999 years is a misread.
_MAX_TERM_MONTHS = 600


def _number(text: str) -> Optional[int]:
    """`36`, `thirty-six`, `thirty six`, `thirtysix`, `three` -> int; else None."""
    t = _squeeze(text).lower().replace("-", " ").strip()
    if t.isdigit():
        return int(t)
    if t in _ONES:
        return _ONES[t]
    if t in _TENS:
        return _TENS[t]
    parts = t.split()
    if len(parts) == 2 and parts[0] in _TENS and parts[1] in _ONES and _ONES[parts[1]] < 10:
        return _TENS[parts[0]] + _ONES[parts[1]]
    if len(parts) == 1:                               # "thirtysix": the hyphen lost
        for tens, value in _TENS.items():
            rest = t[len(tens):]
            if t.startswith(tens) and rest in _ONES and _ONES[rest] < 10:
                return value + _ONES[rest]
    return None


def _end_from_duration(start_iso: str, months: int) -> Optional[str]:
    """start + `months` - 1 day, or None when the end month lacks the start's day."""
    from calendar import monthrange
    from datetime import date, timedelta
    s = date.fromisoformat(start_iso)
    year, month0 = divmod(s.month - 1 + months, 12)
    year, month = s.year + year, month0 + 1
    if s.day > monthrange(year, month)[1]:
        return None                                   # 31 Jan + 1 month: no single answer
    return (date(year, month, s.day) - timedelta(days=1)).isoformat()


def _derive_end(text: str, start_iso: str) -> tuple[Optional[str], Optional[str]]:
    """``(iso, literal)``: the end implied by a duration stated with the start."""
    for pattern in (_START_PROSE_STRONG, _START_PROSE_WEAK):
        for m in pattern.finditer(text):
            if pattern is _START_PROSE_WEAK and _cites_another_agreement(text, m.start()):
                continue
            dm = _DATE.match(text[m.end():m.end() + 40])
            if not dm or _iso(dm.group(1)) != start_iso:
                continue                              # a different date: not THIS start
            tail_at = m.end() + dm.end()
            d = _DURATION_AFTER_START.match(text[tail_at:tail_at + 200])
            if not d:
                continue
            n = _number(d.group("num"))
            if n is None:
                continue
            if d.group("digits") is not None and int(d.group("digits")) != n:
                continue                              # "thirty (36)": refuse, do not pick
            months = n * 12 if d.group("unit").lower().startswith("year") else n
            if not 1 <= months <= _MAX_TERM_MONTHS:
                continue
            iso = _end_from_duration(start_iso, months)
            if iso:
                return iso, text[m.start():tail_at + d.end()].strip()
    return (None, None)


def read_dates(full_text: str) -> ContractDates:
    """The contract's term, or None for either end it does not state."""
    if not full_text:
        return ContractDates()

    start, start_text = _from(full_text, _START_LABEL,
                              _START_PROSE_STRONG, _START_PROSE_WEAK)
    end, end_text = _from(full_text, _END_LABEL, _END_PROSE)
    end_derived = False
    if not end and start:
        end, end_text = _derive_end(full_text, start)
        end_derived = bool(end)

    # A term that ends before it begins has been misread. Two wrong dates are
    # worse than two empty ones, and a start date is what blocks promotion -- so
    # it must not be filled with a value this module cannot stand behind.
    if start and end and start > end:
        return ContractDates()

    return ContractDates(start=start, end=end,
                         start_text=start_text, end_text=end_text,
                         end_derived=end_derived)


def date_candidates(full_text: str) -> list[Candidate]:
    """`contract_start_date` / `contract_end_date`, ISO-valued.

    The value is ISO because that is what the column takes; the Span keeps the
    document's own rendering, so the grounding gate is never handed a string the
    page does not contain.
    """
    d = read_dates(full_text)
    out: list[Candidate] = []
    for field, value, literal in ((START_FIELD, d.start, d.start_text),
                                  (END_FIELD, d.end, d.end_text)):
        if not value:
            continue
        out.append(Candidate(
            field=field,
            value=value,
            span=Span(page=0, bbox=(0.0, 0.0, 0.0, 0.0), text=literal or value),
            source="date",
            pattern_name=(DERIVED_END_PATTERN if field == END_FIELD and d.end_derived
                          else "contract_term_clause"),
            confidence=CONFIDENCE,
        ))
    return out


__all__ = ["ContractDates", "read_dates", "date_candidates",
           "START_FIELD", "END_FIELD", "DERIVED_END_PATTERN"]


# ---------------------------------------------------------------------------
# Correcting rows stored before anything read a contract's term.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DateCorrection:
    start: Optional[str]
    end: Optional[str]
    changed: bool
    reason: str
    end_derived: bool = False


def decide_date_correction(
    *,
    full_text: str,
    stored_start: Optional[str],
    stored_end: Optional[str],
    provenance_source: Optional[str],
) -> DateCorrection:
    """Should this row's term change, and to what?

    Three rules, not the parties' four. The difference is deliberate: the entity
    sweep never produced a contract date -- these fields have no patterns and no
    NER type, which is why they were NULL -- so there is no wrong value of the
    sweep's to clear, only an absent one to fill. And where this reader finds
    nothing, a stored date STAYS: the context layer may have grounded a shape
    these four do not cover, and absence of a read is not evidence of a wrong
    value.

    1. a human-confirmed value is never touched;
    2. a row with no stored text cannot be re-read;
    3. what the document states wins over what is stored -- including over NULL,
       which is the case every row stored before 2026-10-04 is in.
    """
    from .contract_parties import HUMAN_SOURCE

    if (provenance_source or "").lower() == HUMAN_SOURCE:
        return DateCorrection(stored_start, stored_end, False,
                              "a human confirmed this value; nothing overrules that")
    if not (full_text or "").strip():
        return DateCorrection(stored_start, stored_end, False,
                              "no stored text to re-read; left as found")

    d = read_dates(full_text)
    start = d.start or stored_start
    # A derived end never replaces a stored one: arithmetic on a duration is weaker
    # evidence than a value the context layer grounded or a person confirmed.
    derived = d.end_derived and not stored_end
    end = (d.end if (d.end and (not d.end_derived or derived)) else None) or stored_end
    changed = (start != stored_start) or (end != stored_end)
    if not d.start and not d.end:
        return DateCorrection(stored_start, stored_end, False,
                              "the document states no term this reader recognises; "
                              "left as found rather than cleared")
    return DateCorrection(
        start, end, changed,
        ("read from the document's own term wording"
         + ("" if changed else " and already stored correctly")
         + ("; the end is DERIVED from the stated duration" if derived else "")),
        end_derived=derived,
    )


__all__ += ["DateCorrection", "decide_date_correction"]
