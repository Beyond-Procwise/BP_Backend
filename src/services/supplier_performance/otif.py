"""OTIF: the share of deliveries that were on time AND in full.

One row per delivery obligation (a PO line, with the quantity received against it and the date it
was complete or last received; partial receipts are totalled by the caller).

  on time  : received no later than promised + ``late_tolerance_days``, and, only if
             ``early_tolerance_days`` is given, no earlier than promised - that many days.
  in full  : received >= ordered x (1 - ``in_full_tolerance``). Over-delivery is in full, and counted.

Each rate is taken over the deliveries that can be assessed for it. A delivery with no promised
date, no receipt, no quantity or an unreadable date is counted as unassessed for that measure, never
as a failure: our missing data is not charged to the supplier. With nothing assessable the answer is
``None`` (the registered formula turns that into UNASSESSED), never a rate of zero.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, Mapping, Optional, Sequence


def _date(v: Any) -> Optional[date]:
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    if isinstance(v, str) and v.strip():
        try:
            return date.fromisoformat(v.strip()[:10])
        except ValueError:
            return None
    return None


def _num(v: Any) -> Optional[float]:
    try:
        return None if v is None or isinstance(v, bool) else float(v)
    except (TypeError, ValueError):
        return None


def otif(
    *, deliveries: Sequence[Mapping[str, Any]], late_tolerance_days: float,
    in_full_tolerance: float, early_tolerance_days: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    if late_tolerance_days < 0 or (early_tolerance_days is not None and early_tolerance_days < 0):
        raise ValueError("tolerances in days cannot be negative")
    if not 0.0 <= in_full_tolerance <= 1.0:
        raise ValueError("in_full_tolerance is a share between 0 and 1")

    on_time_n = on_time_ok = in_full_n = in_full_ok = both_n = both_ok = over = 0
    for row in deliveries:
        promised, received = _date(row.get("promised_date")), _date(row.get("received_date"))
        ordered, got = _num(row.get("ordered_qty")), _num(row.get("received_qty"))
        on_time = in_full = None
        if promised is not None and received is not None:
            delta = (received - promised).days
            on_time = delta <= late_tolerance_days and (
                early_tolerance_days is None or delta >= -early_tolerance_days)
        if ordered is not None and got is not None and ordered > 0:
            in_full = got >= ordered * (1.0 - in_full_tolerance)
            over += 1 if got > ordered else 0
        if on_time is not None:
            on_time_n += 1
            on_time_ok += 1 if on_time else 0
        if in_full is not None:
            in_full_n += 1
            in_full_ok += 1 if in_full else 0
        if on_time is not None and in_full is not None:
            both_n += 1
            both_ok += 1 if (on_time and in_full) else 0
    if not on_time_n and not in_full_n:
        return None

    rate = lambda ok, n: (ok / n) if n else None  # noqa: E731
    otif_rate = rate(both_ok, both_n)
    return {
        "deliveries": len(deliveries),
        "on_time_rate": rate(on_time_ok, on_time_n), "on_time_assessed": on_time_n,
        "in_full_rate": rate(in_full_ok, in_full_n), "in_full_assessed": in_full_n,
        "otif_rate": otif_rate, "otif_assessed": both_n,
        "otif_score": None if otif_rate is None else otif_rate * 100.0,
        "over_delivered": over,
        "unassessed": len(deliveries) - both_n,
    }
