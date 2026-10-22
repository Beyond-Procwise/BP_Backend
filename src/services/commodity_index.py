"""The commodity index register: published series a person loaded, and the move between two points.

``load_series`` appends the points of one series (a corrected value is loaded again and supersedes the
earlier row for that period). ``lookup`` is what the agent tool commodity_index_lookup calls: it returns
the series as recorded (publisher, unit, source), the points in a window and the move between the first
and last recorded points in it. It estimates nothing: no series means "index history not available", a
window with one point has no move. Table: ``proc.bp_commodity_index``
(deploy/sql/2026-10-22_bp_commodity_index.sql).
"""
from __future__ import annotations

import logging
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Callable, Dict, List, Optional, Sequence

from src.services.contract_tools import store

log = logging.getLogger(__name__)

MAX_POINTS_PER_LOAD = 2000
MAX_POINTS_RETURNED = 120
MAX_SERIES_LISTED = 50

_INSERT_SQL = """
    INSERT INTO proc.bp_commodity_index
        (series_code, publisher, description, currency, unit, period_date, value, source_reference, recorded_by)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
"""
# Series whose code matches exactly, or whose code, description or publisher contains the words.
_SERIES_SQL = """
    SELECT series_code, MAX(publisher), MAX(description), MAX(currency), MAX(unit),
           MIN(period_date), MAX(period_date), COUNT(DISTINCT period_date)::int
      FROM proc.bp_commodity_index
     WHERE series_code = %s OR series_code ILIKE %s OR description ILIKE %s OR publisher ILIKE %s
     GROUP BY series_code
     ORDER BY series_code
     LIMIT 6
"""
_LIST_SQL = """
    SELECT series_code, MAX(publisher), MAX(description), MAX(currency), MAX(unit),
           MIN(period_date), MAX(period_date), COUNT(DISTINCT period_date)::int
      FROM proc.bp_commodity_index
     GROUP BY series_code
     ORDER BY series_code
     LIMIT %s
"""
# The latest loaded value per period: a reload supersedes by created_date, nothing is edited.
_POINTS_SQL = """
    SELECT DISTINCT ON (period_date) period_date, value, source_reference, recorded_by, created_date
      FROM proc.bp_commodity_index
     WHERE series_code = %s AND period_date >= %s AND period_date <= %s
     ORDER BY period_date, created_date DESC, index_id DESC
"""


class InvalidSeries(ValueError):
    """The series cannot be loaded as given; the message says what is wrong."""


def _plain(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    return value


def _date(value: Any, name: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value or "").strip()[:10])
    except ValueError:
        raise InvalidSeries(f"{name} must be a date like 2026-01-31") from None


def validate_series(series_code: Any, publisher: Any, points: Any, *, currency: Any = None, unit: Any = None,
                    description: Any = None, source_reference: Any = None) -> Dict[str, Any]:
    code = str(series_code or "").strip()
    if not code or len(code) > 80:
        raise InvalidSeries("series_code is required: the publisher's code for the series, up to 80 characters")
    pub = str(publisher or "").strip()
    if not pub:
        raise InvalidSeries("publisher is required: who publishes the series")
    if not isinstance(points, (list, tuple)) or not points:
        raise InvalidSeries("points is required: a list of {period_date, value}")
    if len(points) > MAX_POINTS_PER_LOAD:
        raise InvalidSeries(f"at most {MAX_POINTS_PER_LOAD} points per load")
    clean: Dict[date, Decimal] = {}
    for i, p in enumerate(points):
        if not isinstance(p, dict):
            raise InvalidSeries(f"point {i + 1} must be an object with period_date and value")
        day = _date(p.get("period_date"), f"point {i + 1} period_date")
        try:
            value = Decimal(str(p.get("value")).strip())
        except (InvalidOperation, ValueError, TypeError):
            raise InvalidSeries(f"point {i + 1} value must be a number") from None
        if not value.is_finite():
            raise InvalidSeries(f"point {i + 1} value must be a finite number")
        if day in clean:
            raise InvalidSeries(f"point {i + 1} repeats the period {day.isoformat()} within one load")
        clean[day] = value
    text = lambda v: (str(v).strip() or None) if v not in (None, "") else None  # noqa: E731
    return {"series_code": code, "publisher": pub, "description": text(description), "currency": text(currency),
            "unit": text(unit), "source_reference": text(source_reference),
            "points": [{"period_date": d, "value": v} for d, v in sorted(clean.items())]}


def load_series(series_code: Any, publisher: Any, points: Any, by: str, *, currency: Any = None, unit: Any = None,
                description: Any = None, source_reference: Any = None,
                execute: Optional[Callable[[str, Sequence[Any]], Any]] = None) -> Dict[str, Any]:
    """Append the points of one series. ``by`` is the signed-in person, never taken from the body."""
    who = str(by or "").strip()
    if not who:
        raise InvalidSeries("the loading person is unknown; sign in to load a series")
    clean = validate_series(series_code, publisher, points, currency=currency, unit=unit,
                            description=description, source_reference=source_reference)
    run = execute or store.rows
    for p in clean["points"]:
        run(_INSERT_SQL, [clean["series_code"], clean["publisher"], clean["description"], clean["currency"],
                          clean["unit"], p["period_date"], p["value"], clean["source_reference"], who])
    return {"series_code": clean["series_code"], "publisher": clean["publisher"], "points_loaded": len(clean["points"]),
            "first_period": clean["points"][0]["period_date"].isoformat(),
            "last_period": clean["points"][-1]["period_date"].isoformat(), "recorded_by": who}


def _series(row: tuple) -> Dict[str, Any]:
    code, pub, desc, ccy, unit, first, last, n = row
    return {"series_code": code, "publisher": pub, "description": desc, "currency": ccy, "unit": unit,
            "first_period": _plain(first), "last_period": _plain(last), "periods": int(n or 0)}


def list_series(*, rows: Optional[Callable[[str, Sequence[Any]], List[tuple]]] = None) -> Dict[str, Any]:
    try:
        found = (rows or store.rows)(_LIST_SQL, [MAX_SERIES_LISTED])
    except store.StoreUnavailable:
        return {"found": False, "available": False,
                "note": "The index register could not be read just now. This does not mean no series is loaded."}
    series = [_series(r) for r in found]
    return {"found": bool(series), "series": series, "count": len(series),
            "note": ("Only series a person loaded are listed; an index not listed is not available here. "
                     if series else "No index series has been loaded. Index history is not available; ask the user "
                                    "to load the series (code, publisher, points) before any index is used.")}


def _move(points: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if len(points) < 2:
        return None
    first, last = points[0], points[-1]
    fv, lv = Decimal(str(first["value"])), Decimal(str(last["value"]))
    if fv == 0:
        return {"from_period": first["period_date"], "to_period": last["period_date"], "from_value": _plain(fv),
                "to_value": _plain(lv), "change_pct": None,
                "note": "the first recorded value is zero, so a percentage move is undefined"}
    pct = (lv - fv) / fv * 100
    return {"from_period": first["period_date"], "to_period": last["period_date"], "from_value": _plain(fv),
            "to_value": _plain(lv), "change_pct": float(round(pct, 4)),
            "note": "between the first and last recorded points in the window; not an average, not a forecast"}


def lookup(series: Any = None, start: Any = None, end: Any = None, *,
           rows: Optional[Callable[[str, Sequence[Any]], List[tuple]]] = None) -> Dict[str, Any]:
    """One series as recorded, its points in a window and the move across them; or the list of series."""
    read = rows or store.rows
    code = str(series or "").strip()
    if not code:
        out = list_series(rows=read)
        out["note"] = "Give the series code to read its values. " + out["note"]
        return out
    try:
        lo = _date(start, "start") if start not in (None, "") else date(1900, 1, 1)
        hi = _date(end, "end") if end not in (None, "") else date(2999, 12, 31)
    except InvalidSeries as exc:
        return {"found": False, "note": str(exc) + "."}
    if lo > hi:
        return {"found": False, "note": "start is after end."}
    try:
        matches = read(_SERIES_SQL, [code, f"%{code}%", f"%{code}%", f"%{code}%"])
    except store.StoreUnavailable:
        return {"found": False, "available": False,
                "note": "The index register could not be read just now. This does not mean the series is not loaded, "
                        "and index history is not available until it can be read."}
    exact = [m for m in matches if m[0] == code]
    if not matches:
        return {"found": False, "loaded": False,
                "note": f"No loaded index series matches '{code}'. Index history is not available: ask the user for "
                        "the code, publisher and values and have them loaded; do not estimate values."}
    if len(matches) > 1 and not exact:
        return {"found": False, "ambiguous": True, "candidates": [_series(m) for m in matches],
                "note": "Several loaded series match; ask the user which one, then pass its series_code."}
    meta = _series(exact[0] if exact else matches[0])
    try:
        found = read(_POINTS_SQL, [meta["series_code"], lo, hi])
    except store.StoreUnavailable:
        return {"found": False, "available": False,
                "note": "The index register could not be read just now. Index history is not available until it can be read."}
    points = [{"period_date": _plain(d), "value": _plain(v), "source_reference": src, "recorded_by": by,
               "loaded_at": _plain(at)} for d, v, src, by, at in found]
    shown = points if len(points) <= MAX_POINTS_RETURNED else points[:MAX_POINTS_RETURNED // 2] + points[-MAX_POINTS_RETURNED // 2:]
    out = {"found": True, "series": meta, "window": {"start": lo.isoformat() if start else None, "end": hi.isoformat() if end else None},
           "points_in_window": len(points), "points": shown, "move": _move(points),
           "note": "Values as loaded (a reloaded period shows its latest value), with who loaded each and from where. "
                   "The move is between the first and last recorded points in the window only; nothing is "
                   "interpolated, averaged or forecast. The series' publisher and code are as recorded; check "
                   "that the supplier does not influence it."}
    if not points:
        out["note"] = "The series is loaded but has no points in this window; index history is not available for it. " + out["note"]
    elif len(points) == 1:
        out["note"] = "Only one point is in this window, so no move can be given. " + out["note"]
    if len(points) > MAX_POINTS_RETURNED:
        out["note"] += f" Showing the first and last {MAX_POINTS_RETURNED // 2} of {len(points)} points."
    return out
