"""Does this extracted value actually appear in the document it came from?

Three outcomes, not two. ``src/services/extraction_v3/grounding.py`` answers the
same question with a bool and returns True when the document is unavailable --
right for a runtime guard that must not block a user over a missing PDF, fatal
here, where "could not check" would silently become "correct".

The same principle is turned on this checker. A review ran a negative control:
substitute an arbitrary WRONG quantity into a real document and ask whether it
verified. A fabricated quantity of 5 verified against 95.5% of documents,
because every page contains a 5 somewhere -- in a date, a page number, a line
number. A match that carries almost no evidence is not a verification, so a bare
small integer is now ``unverifiable`` rather than ``verified``. That lowers
coverage and is the honest trade: an accuracy built on coincidence is worth less
than a smaller number that means something.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Optional

VERIFIED = "verified"
UNSUPPORTED = "unsupported"
UNVERIFIABLE = "unverifiable"

# Below this, a whole number is common enough on an arbitrary page that finding
# it says almost nothing. Above it, a bare integer is distinctive.
_DISCRIMINATING_MAGNITUDE = 1000

# A currency code is verified by the symbol the document actually prints.
_CURRENCY_SYMBOLS = {
    "GBP": ("£",), "EUR": ("€",), "USD": ("$", "US$"), "JPY": ("¥",),
    "INR": ("₹",), "AUD": ("A$", "$"), "CAD": ("C$", "$"), "CHF": ("CHF",),
    "AED": ("د.إ",), "CNY": ("¥",), "SEK": ("kr",), "NOK": ("kr",),
}

_DATE_INPUT_FORMATS = (
    "%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S",
    "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%d.%m.%Y", "%Y/%m/%d",
)


@dataclass(frozen=True)
class Verdict:
    outcome: str
    rule: str
    span: Optional[str] = None


def _norm_text(s: str) -> str:
    """Fold to a comparable form.

    NFKC collapses the typographic apostrophe in "O’Brien" onto the straight one
    the extraction holds, and the ligatures docling emits onto their letters.
    """
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("’", "'").replace("‘", "'")
    s = s.replace("“", '"').replace("”", '"')
    s = s.replace("–", "-").replace("—", "-")
    # Zero-width characters are not text; a page of them has nothing to check.
    s = re.sub(r"[​‌‍﻿]", "", s)
    return re.sub(r"\s+", " ", s).strip().lower()


@dataclass(frozen=True)
class _Number:
    value: float
    written_with_punctuation: bool


def _candidate_numbers(text: str) -> list[_Number]:
    """Every number in the text, with whether it was written with a separator.

    The separator rule used to read "whichever appears LAST is the decimal
    point", which fires wrongly when there is no dot at all: rfind(".") returns
    -1, so "1,234" parsed as 1.234 and 158 real money values across the corpus
    were scored unsupported while printed plainly on the page. "1,234,567" threw
    and was swallowed entirely.
    """
    out: list[_Number] = []
    for sign, raw in re.findall(r"(-?)(\d[\d.,]*\d|\d)", text):
        has_comma, has_dot = "," in raw, "." in raw
        if has_comma and has_dot:
            # Whichever comes last is the decimal separator.
            plain = (raw.replace(".", "").replace(",", ".")
                     if raw.rfind(",") > raw.rfind(".") else raw.replace(",", ""))
        elif has_comma:
            # Grouping (1,234) unless it looks like a European decimal (1,50).
            groups = raw.split(",")
            plain = (raw.replace(",", ".")
                     if len(groups) == 2 and len(groups[1]) != 3
                     else raw.replace(",", ""))
        elif has_dot:
            groups = raw.split(".")
            # 1.234.567 is grouping, not three decimal points.
            plain = raw.replace(".", "") if len(groups) > 2 else raw
        else:
            plain = raw
        try:
            out.append(_Number(float(f"{sign}{plain}"), has_comma or has_dot))
        except ValueError:
            continue
    return out


def _is_discriminating(value: float, matched: _Number) -> bool:
    """Would finding this number be evidence, or a coincidence?"""
    if value != int(value):
        return True                       # a fractional part is distinctive
    if abs(value) >= _DISCRIMINATING_MAGNITUDE:
        return True
    return matched.written_with_punctuation   # "490.00" is a figure; "5" is not


def _date_renderings(value: str) -> set[str]:
    parsed: Optional[date] = None
    text = str(value).strip()
    for fmt in _DATE_INPUT_FORMATS:
        try:
            parsed = datetime.strptime(text, fmt).date()
            break
        except (TypeError, ValueError):
            continue
    if parsed is None:
        return set()

    out: set[str] = set()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%d.%m.%Y",
                "%d %B %Y", "%d %b %Y", "%B %d, %Y", "%b %d, %Y", "%Y/%m/%d"):
        out.add(parsed.strftime(fmt))
    # Unpadded days and months: a document says "5 March 2024", not "05".
    out.add(f"{parsed.day} {parsed.strftime('%B')} {parsed.year}")
    out.add(f"{parsed.day} {parsed.strftime('%b')} {parsed.year}")
    out.add(f"{parsed.strftime('%B')} {parsed.day}, {parsed.year}")
    out.add(f"{parsed.day}/{parsed.month}/{parsed.year}")
    return out


def _match_number(value: float, source_text: str) -> Verdict:
    """Every matching candidate is considered, and the strongest wins.

    Returning on the first match let a weak one shadow a strong one: a tax
    amount of 0 against "VAT (0%): 0.00" matched the bare 0 inside "(0%)" and
    was called undiscriminating, while the figure 0.00 sat two characters away.
    """
    matched = [c for c in _candidate_numbers(source_text) if c.value == value]
    if not matched:
        return Verdict(UNSUPPORTED, "number-absent")
    if any(_is_discriminating(value, c) for c in matched):
        return Verdict(VERIFIED, "number-match", str(value))
    return Verdict(UNVERIFIABLE, "low-discrimination", str(value))


def verify_field(field: str, value: Any, source_text: Optional[str]) -> Verdict:
    """One field, one source text, one verdict. Pure: no I/O, no model."""
    if value is None:
        return Verdict(UNVERIFIABLE, "value-absent")
    if isinstance(value, bool):
        # "Paid: yes" does not contain "True". Grounding cannot judge a flag.
        return Verdict(UNVERIFIABLE, "not-groundable-type")
    if isinstance(value, (list, dict)):
        return Verdict(UNVERIFIABLE, "not-groundable-type")
    if not source_text or not source_text.strip():
        return Verdict(UNVERIFIABLE, "no-source-text")

    haystack = _norm_text(source_text)
    if not haystack:
        # Non-whitespace but meaningless: a zero-width space, an image-only page.
        return Verdict(UNVERIFIABLE, "no-source-text")

    if isinstance(value, (int, float)):
        return _match_number(float(value), source_text)

    text = str(value).strip()
    if not text:
        return Verdict(UNVERIFIABLE, "value-empty")

    # A currency code is verified by the symbol the page actually prints.
    if field.endswith("currency") and text.upper() in _CURRENCY_SYMBOLS:
        code = text.upper()
        if re.search(rf"(?<!\w){re.escape(code.lower())}(?!\w)", haystack):
            return Verdict(VERIFIED, "currency-code", code)
        for symbol in _CURRENCY_SYMBOLS[code]:
            if _norm_text(symbol) and _norm_text(symbol) in haystack:
                return Verdict(VERIFIED, "currency-symbol", symbol)
        return Verdict(UNSUPPORTED, "currency-absent")

    for rendering in _date_renderings(text):
        if _norm_text(rendering) in haystack:
            return Verdict(VERIFIED, "date-rendering", rendering)

    # A bare number arriving as a string is still a number, but keep the written
    # form: "007" is not 7.
    if re.fullmatch(r"-?\d[\d.,]*", text):
        needle = _norm_text(text)
        if re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", haystack):
            return Verdict(VERIFIED, "text-match", text)
        try:
            return _match_number(float(text.replace(",", "")), source_text)
        except ValueError:
            pass

    needle = _norm_text(text)
    # Anchored at a token boundary: "Ltd" must not verify inside "Ultditch".
    if re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", haystack):
        return Verdict(VERIFIED, "text-match", text)

    return Verdict(UNSUPPORTED, "text-absent")
