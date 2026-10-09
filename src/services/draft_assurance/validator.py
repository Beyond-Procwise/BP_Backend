"""Deterministic checks on a drafted email. Code, not an LLM.

Every function here is pure: text and allowed values in, violations out. A figure
in the draft is acceptable only if it equals a value the assurance layer put in
the allowed set -- a Postgres fact, a reasoned value with a basis, or a value the
payload carried that no row confirmed (accepted, but reported as unverified).
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from dateutil import parser as _dateparser

_REF = re.compile(
    r"\b(?:PO|INV|RFQ|RFP|RFI|QUT|QT|Q|SO|CN|DN|REF)[-/ ]?\d{2,}[A-Z\d/-]*\b", re.IGNORECASE)
_MONTHS = ("january|february|march|april|may|june|july|august|september|october|"
           "november|december")
_DATE = re.compile(
    rf"\b(?:\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{_MONTHS})(?:\s+\d{{4}})?"
    rf"|(?:{_MONTHS})\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s+\d{{4}})?"
    rf"|\d{{4}}-\d{{2}}-\d{{2}}|\d{{1,2}}/\d{{1,2}}(?:/\d{{2,4}})?)\b", re.IGNORECASE)
_NUMBER = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?(?![\w])(?!\s?(?:st|nd|rd|th)\b)")
_PLACEHOLDER = re.compile(r"\[[^\]\n]{2,}\]")
# A deadline is a target STATED AS A DEADLINE: a cue ("by", "no later than", "deadline is"...) then a date,
# a day, or a time. A bare date is not one: "our contract started 1 March 2025" or "PO 12/34" used to pass.
# Kept apart from _DATE on purpose: _DATE also decides which figures are dates, and widening it moves that check.
_MON = _MONTHS + "|jan|feb|mar|apr|jun|jul|aug|sept?|oct|nov|dec"
_DAY = "monday|tuesday|wednesday|thursday|friday|saturday|sunday"
_TARGET = (
    rf"(?:\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{_MON})\b\.?(?:\s+\d{{4}})?"
    rf"|(?:{_MON})\b\.?\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s+\d{{4}})?"
    rf"|\d{{4}}-\d{{2}}-\d{{2}}|\d{{1,2}}/\d{{1,2}}(?:/\d{{2,4}})?"
    rf"|(?:(?:this|next)\s+)?(?:{_DAY})\b"
    rf"|\d{{1,2}}(?:st|nd|rd|th)"
    rf"|today|tonight|tomorrow"
    rf"|(?:the\s+)?end\s+of\s+(?:the\s+|this\s+|next\s+)?(?:business\s+|working\s+)?(?:day|week|month)"
    rf"|close\s+of\s+(?:business|play)|cob|eod|eow|noon|midday|\d{{1,2}}(?::\d{{2}})?\s*(?:am|pm))\b")
_DEADLINE = re.compile(
    rf"\b(?:by|before|until|till|no\s+later\s+than|not\s+later\s+than|on\s+or\s+before|deadline(?:\s+\w+){{0,4}}\s+(?:is|of|:)|due)"
    rf"\s*:?\s+(?:the\s+)?{_TARGET}"
    rf"|\bwithin\s+(?:the\s+next\s+)?\d+\s+(?:business\s+|working\s+)?(?:hours?|days?|weeks?)\b",
    re.IGNORECASE)


def to_decimal(token: Any) -> Optional[Decimal]:
    try:
        return Decimal(str(token).replace(",", "").replace("%", "").strip())
    except (InvalidOperation, ValueError):
        return None


def numbers_in(obj: Any) -> Set[Decimal]:
    """Every numeric value inside a string, number, list or dict."""

    found: Set[Decimal] = set()
    if obj is None or isinstance(obj, bool):
        return found
    if isinstance(obj, (int, float, Decimal)):
        d = to_decimal(obj)
        return {d} if d is not None else found
    if isinstance(obj, dict):
        for v in obj.values():
            found |= numbers_in(v)
        return found
    if isinstance(obj, (list, tuple, set)):
        for v in obj:
            found |= numbers_in(v)
        return found
    for m in _NUMBER.finditer(str(obj)):
        d = to_decimal(m.group(0))
        if d is not None:
            found.add(d)
    return found


def _date_key(text: str) -> Optional[Tuple[Optional[int], int, int]]:
    try:
        probe = _dateparser.parse(text, default=_dateparser.parse("1900-01-01"), dayfirst=True)
    except (ValueError, OverflowError):
        return None
    has_year = bool(re.search(r"\d{4}", text)) or bool(re.search(r"/\d{2}$", text))
    return (probe.year if has_year else None, probe.month, probe.day)


def dates_in(obj: Any) -> Set[Tuple[Optional[int], int, int]]:
    out = set()
    items: Iterable[Any] = obj if isinstance(obj, (list, tuple, set)) else [obj]
    for item in items:
        for m in _DATE.finditer(str(item or "")):
            k = _date_key(m.group(0))
            if k:
                out.add(k)
    return out


def _same_date(a, b) -> bool:
    return a[1:] == b[1:] and (a[0] is None or b[0] is None or a[0] == b[0])


def _v(kind: str, detail: str, severity: str = "fail") -> Dict[str, str]:
    return {"kind": kind, "detail": detail, "severity": severity}


def check_figures(text: str, allowed_numbers: Set[Decimal], allowed_dates: Set,
                  allowed_refs: Set[str]) -> List[Dict[str, str]]:
    violations: List[Dict[str, str]] = []
    scrubbed = text
    allowed_ref_norm = {re.sub(r"[-/ ]", "", r).upper() for r in allowed_refs}
    for m in _REF.finditer(text):
        if re.sub(r"[-/ ]", "", m.group(0)).upper() not in allowed_ref_norm:
            violations.append(_v("ungrounded_reference", m.group(0)))
    scrubbed = _REF.sub(" ", scrubbed)
    for m in _DATE.finditer(scrubbed):
        k = _date_key(m.group(0))
        if k is None or not any(_same_date(k, a) for a in allowed_dates):
            violations.append(_v("ungrounded_date", m.group(0)))
    scrubbed = _DATE.sub(" ", scrubbed)
    for m in _NUMBER.finditer(scrubbed):
        d = to_decimal(m.group(0))
        if d is not None and d not in allowed_numbers:
            violations.append(_v("ungrounded_figure", m.group(0)))
    return violations


def check_leaks(text: str, never_state: Dict[str, Set[Decimal]]) -> List[Dict[str, str]]:
    present = {to_decimal(m.group(0)) for m in _NUMBER.finditer(text)}
    return [_v("internal_figure_leaked", key)
            for key, values in never_state.items() if present & values]


def check_patterns(text: str, patterns: Dict[str, str]) -> List[Dict[str, str]]:
    out = []
    for name, rx in patterns.items():
        try:
            hit = re.search(rx, text, re.IGNORECASE)
        except re.error:
            out.append(_v("bad_pattern", name))
            continue
        if hit:
            out.append(_v("forbidden_content", f"{name}: {hit.group(0)[:60]}"))
    if _PLACEHOLDER.search(text):
        out.append(_v("unresolved_placeholder", _PLACEHOLDER.search(text).group(0)))
    return out


_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")
# A phone number: an optional +, then digits with spaces, dots, dashes or brackets, at least 9 digits in all. A date has 8 at
# most and a reference keeps its letter prefix (the lookbehind refuses a run that follows a letter or a dash).
_PHONE = re.compile(r"(?<![\w-])\+?\(?\d[\d\s().-]{7,}\d(?![\w-])")


def _digits(value: Any) -> str:
    return re.sub(r"\D", "", str(value))


def check_contact_details(text: str, allowed: Iterable[Any]) -> List[Dict[str, str]]:
    """Email addresses and phone numbers that are not on record. Found live 2026-10-09: model drafts invent both."""

    pool = " ".join(str(a) for a in allowed if a is not None).lower()
    pool_digits = {_digits(a) for a in re.split(r"[\s,;]+", " ".join(str(a) for a in allowed if a is not None)) if _digits(a)}
    pool_digits |= {_digits(a) for a in allowed if a is not None and _digits(a)}
    out = []
    for m in _EMAIL.finditer(text or ""):
        if m.group(0).lower() not in pool:
            out.append(_v("ungrounded_contact_detail", m.group(0)))
    scrubbed = _EMAIL.sub(" ", text or "")
    for m in _PHONE.finditer(scrubbed):
        raw = m.group(0).strip(" .-")
        digits = _digits(raw)
        if len(digits) < 9 or _DATE.fullmatch(raw):
            continue
        if digits not in pool_digits:
            out.append(_v("ungrounded_contact_detail", raw))
    return out


def check_required(text: str, elements: List[str], asks: Iterable[str]) -> List[Dict[str, str]]:
    out = []
    for name in elements:
        if name == "explicit_ask":
            ok = "?" in text or re.search(r"\bplease\b", text, re.IGNORECASE) or any(
                str(a).strip() and str(a).strip().lower() in text.lower() for a in asks)
        elif name == "deadline":
            ok = bool(_DEADLINE.search(text))
        else:
            out.append(_v("unknown_required_element", name))
            continue
        if not ok:
            out.append(_v("missing_required_element", name))
    return out


def check_length(text: str, target: int) -> List[Dict[str, str]]:
    words = len(re.findall(r"\b\w+\b", text))
    if target and words > target:
        return [_v("over_length", f"{words} words, target {target}", "warn")]
    return []


def figures_in(text: str) -> Set[Decimal]:
    """The standalone numbers in a text, once references and dates are set aside."""

    scrubbed = _DATE.sub(" ", _REF.sub(" ", text or ""))
    return {d for d in (to_decimal(m.group(0)) for m in _NUMBER.finditer(scrubbed)) if d is not None}
