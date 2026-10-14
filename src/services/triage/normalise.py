"""Normalisation helpers (spec §4, step 1). Pure."""
from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from difflib import SequenceMatcher
from typing import Any, Optional

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_DAYS = re.compile(r"(\d+)\s*days?\b")
_NET = re.compile(r"\bnet\s*(\d+)\b")


def to_decimal(value: Any) -> Optional[Decimal]:
    if value is None:
        return None
    text = str(value).replace(",", "").strip()
    if not text:
        return None
    try:
        return Decimal(text)
    except (InvalidOperation, ValueError):
        return None


def to_confidence(value: Any) -> Optional[float]:
    """Extraction confidence as 0-1. The _trgt tables store 0-100."""
    d = to_decimal(value)
    if d is None:
        return None
    f = float(d)
    return round(f / 100.0, 6) if f > 1.0 else f


def norm_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(_NON_ALNUM.sub(" ", str(value).lower()).split())


def similarity(a: Any, b: Any) -> float:
    na, nb = norm_text(a), norm_text(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    return SequenceMatcher(None, na, nb).ratio()


def terms_days(value: Any) -> Optional[int]:
    """'30 days — due 30 Jul 2025' and 'Net 30' are both 30."""
    text = norm_text(value)
    m = _DAYS.search(text) or _NET.search(text)
    return int(m.group(1)) if m else None
