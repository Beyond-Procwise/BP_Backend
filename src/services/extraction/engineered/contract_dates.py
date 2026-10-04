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
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from ..types import Candidate, Span
from .contract_parties import CONFIDENCE, _squeeze

START_FIELD = "contract_start_date"
END_FIELD = "contract_end_date"

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


def read_dates(full_text: str) -> ContractDates:
    """The contract's term, or None for either end it does not state."""
    if not full_text:
        return ContractDates()

    start, start_text = _from(full_text, _START_LABEL,
                              _START_PROSE_STRONG, _START_PROSE_WEAK)
    end, end_text = _from(full_text, _END_LABEL, _END_PROSE)

    # A term that ends before it begins has been misread. Two wrong dates are
    # worse than two empty ones, and a start date is what blocks promotion -- so
    # it must not be filled with a value this module cannot stand behind.
    if start and end and start > end:
        return ContractDates()

    return ContractDates(start=start, end=end,
                         start_text=start_text, end_text=end_text)


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
            pattern_name="contract_term_clause",
            confidence=CONFIDENCE,
        ))
    return out


__all__ = ["ContractDates", "read_dates", "date_candidates",
           "START_FIELD", "END_FIELD"]


# ---------------------------------------------------------------------------
# Correcting rows stored before anything read a contract's term.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DateCorrection:
    start: Optional[str]
    end: Optional[str]
    changed: bool
    reason: str


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
    end = d.end or stored_end
    changed = (start != stored_start) or (end != stored_end)
    if not d.start and not d.end:
        return DateCorrection(stored_start, stored_end, False,
                              "the document states no term this reader recognises; "
                              "left as found rather than cleared")
    return DateCorrection(
        start, end, changed,
        "read from the document's own term wording"
        + ("" if changed else " and already stored correctly"),
    )


__all__ += ["DateCorrection", "decide_date_correction"]
