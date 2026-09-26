"""Turning whatever arrived into a number, a date, or a string.

Quotes and supplier replies carry money as "£12,300.00", quantities as "500 ea",
and dates in half a dozen shapes. These read them, and answer None when a value
cannot be read rather than guessing — an unparseable price is not zero.

Every function here was a method on NegotiationAgent that never touched `self`.
They are unchanged apart from losing that argument.
"""
from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Any, List, Optional


def parse_money(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        cleaned = re.sub(r"[\s,]", "", text)
        match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
        if match:
            try:
                return float(match.group())
            except ValueError:
                return None
    return None


def parse_quantity(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        match = re.search(r"\d+(?:\.\d+)?", text.replace(",", ""))
        if match:
            try:
                return float(match.group())
            except ValueError:
                return None
    return None


def parse_term_days(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        candidate = int(round(float(value)))
        return candidate if candidate > 0 else None
    if isinstance(value, str):
        text = value.strip().lower()
        if not text:
            return None
        numbers = re.findall(r"\d+(?:\.\d+)?", text)
        if not numbers:
            return None
        try:
            numeric = float(numbers[0])
        except ValueError:
            return None
        if "week" in text and numeric > 0:
            return int(round(numeric * 7))
        return int(round(numeric)) if numeric > 0 else None
    return None


def parse_date(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        for fmt in (
            "%Y-%m-%d",
            "%d/%m/%Y",
            "%m/%d/%Y",
            "%d-%m-%Y",
            "%d %b %Y",
            "%b %d, %Y",
        ):
            try:
                dt = datetime.strptime(text, fmt)
                return dt.replace(tzinfo=timezone.utc).isoformat()
            except ValueError:
                continue
        try:
            dt = datetime.fromisoformat(text)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).isoformat()
        except ValueError:
            return None
    return None


def format_currency(value: Optional[float], currency: Optional[str]) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return ""
    code = (currency or "GBP").upper()
    symbol = (
        "£"
        if code == "GBP"
        else "$"
        if code == "USD"
        else "€"
        if code == "EUR"
        else "₹"
        if code == "INR"
        else ""
    )
    formatted = f"{amount:,.2f}"
    return f"{symbol}{formatted}" if symbol else f"{formatted} {code}"


def normalise_currency(value: Any) -> Optional[str]:
    if not value:
        return None
    if isinstance(value, str):
        trimmed = value.strip().upper()
        if len(trimmed) == 3:
            return trimmed
    return None


def parse_lead_weeks(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
        return number if number <= 12 else round(number / 7.0, 2)
    except ValueError:
        pass
    lowered = text.lower()
    digits = "".join(ch for ch in lowered if (ch.isdigit() or ch == "."))
    try:
        numeric = float(digits)
    except ValueError:
        return None
    if "week" in lowered:
        return numeric
    if "day" in lowered or "business" in lowered:
        return round(numeric / 7.0, 2)
    return None


def coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def positive_int(value: Any, *, fallback: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        return fallback
    return parsed if parsed > 0 else fallback


def coerce_text(value: Any) -> Optional[str]:
    if isinstance(value, str):
        text = value.strip()
        if text:
            return text
    return None


def ensure_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]
