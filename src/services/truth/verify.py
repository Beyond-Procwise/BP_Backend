"""Does this extracted value actually appear in the document it came from?

Three outcomes, not two. ``src/services/extraction_v3/grounding.py`` answers the
same question with a bool and returns True when the document is unavailable --
right for a runtime guard that must not block a user over a missing PDF, fatal
here, where "could not check" would silently become "correct" and inflate every
score built on it.

It also treats the digit-signature of a value (three or more digits appearing
anywhere) as grounding. This does not: an invoice total of 1234.56 is not
verified by a postcode containing 123.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

VERIFIED = "verified"
UNSUPPORTED = "unsupported"
UNVERIFIABLE = "unverifiable"


@dataclass(frozen=True)
class Verdict:
    outcome: str
    rule: str
    span: Optional[str] = None


def _norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def _candidate_numbers(text: str) -> set[str]:
    """Every number in the text, normalised to a plain decimal string.

    Handles 1,234.56 and the European 1.234,56 alike: whichever separator
    appears LAST is the decimal point.
    """
    out: set[str] = set()
    for raw in re.findall(r"\d[\d.,]*\d|\d", text):
        last_comma, last_dot = raw.rfind(","), raw.rfind(".")
        if last_comma > last_dot:
            plain = raw.replace(".", "").replace(",", ".")
        else:
            plain = raw.replace(",", "")
        try:
            out.add(f"{float(plain):.4f}")
        except ValueError:
            continue
    return out


def _date_renderings(value: str) -> set[str]:
    try:
        d = datetime.strptime(value, "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return set()
    return {
        d.strftime(fmt)
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%d.%m.%Y",
                    "%d %B %Y", "%d %b %Y", "%B %d, %Y", "%b %d, %Y")
    }


def verify_field(field: str, value: Any, source_text: Optional[str]) -> Verdict:
    """One field, one source text, one verdict. Pure: no I/O, no model."""
    if value is None:
        return Verdict(UNVERIFIABLE, "value-absent")
    if not source_text or not source_text.strip():
        return Verdict(UNVERIFIABLE, "no-source-text")

    haystack = _norm_text(source_text)

    # Numbers compare as numbers. A textual match would miss 2,400.00 == 2400.0
    # and would accept 1234.56 inside 91234.567.
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        target = f"{float(value):.4f}"
        if target in _candidate_numbers(source_text):
            return Verdict(VERIFIED, "number-match", str(value))
        return Verdict(UNSUPPORTED, "number-absent")

    text = str(value).strip()
    if not text:
        return Verdict(UNVERIFIABLE, "value-empty")

    # A bare number arriving as a string is still a number.
    if re.fullmatch(r"-?\d[\d.,]*", text):
        try:
            target = f"{float(text.replace(',', '')):.4f}"
            if target in _candidate_numbers(source_text):
                return Verdict(VERIFIED, "number-match", text)
            return Verdict(UNSUPPORTED, "number-absent")
        except ValueError:
            pass

    for rendering in _date_renderings(text):
        if _norm_text(rendering) in haystack:
            return Verdict(VERIFIED, "date-rendering", rendering)

    needle = _norm_text(text)
    # Anchored at a token boundary: "Ltd" must not verify inside "Ultditch".
    if re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", haystack):
        return Verdict(VERIFIED, "text-match", text)

    return Verdict(UNSUPPORTED, "text-absent")
