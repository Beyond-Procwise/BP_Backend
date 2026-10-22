"""The compliance screening register: checks people or providers performed on a supplier, as recorded.

``record`` writes one check (append-only; a later check supersedes an earlier one by date). ``lookup``
is what the agent tools screen_sanctions, financial_health_score, esg_rating_lookup,
forced_labour_screen, conflict_minerals_check and monitor_risk_events call: it returns the latest
recorded check per screened subject, with provider, reference and dates, and performs no check
itself. No record is "not performed", which is never a clear result. Table:
``proc.bp_supplier_screening`` (deploy/sql/2026-10-22_bp_supplier_screening.sql).
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional, Sequence

from src.services import supplier_resolve
from src.services.contract_tools import store

log = logging.getLogger(__name__)

#: check_type -> what the tool is called, and what the check is, in plain words
CHECK_TYPES: Dict[str, Dict[str, str]] = {
    "sanctions": {"tool": "screen_sanctions", "label": "sanctions screening"},
    "financial_health": {"tool": "financial_health_score", "label": "financial health assessment"},
    "esg_rating": {"tool": "esg_rating_lookup", "label": "ESG rating"},
    "forced_labour": {"tool": "forced_labour_screen", "label": "forced-labour screening"},
    "conflict_minerals": {"tool": "conflict_minerals_check", "label": "conflict-minerals check"},
    "risk_event": {"tool": "monitor_risk_events", "label": "risk-event review"},
}
RESULTS = ("clear", "flagged", "rated", "not_performed")
MAX_SUBJECTS = 40
MAX_SIGNALS = 20

_INSERT_SQL = """
    INSERT INTO proc.bp_supplier_screening
        (supplier_id, check_type, subject, result, rating, score, details, provider, reference,
         checked_on, valid_until, recorded_by)
    VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s, %s)
    RETURNING screening_id, created_date
"""
_LATEST_SQL = """
    SELECT DISTINCT ON (subject) subject, result, rating, score, details, provider, reference,
           checked_on, valid_until, recorded_by, created_date
      FROM proc.bp_supplier_screening
     WHERE supplier_id = %s AND check_type = %s
     ORDER BY subject, checked_on DESC, screening_id DESC
"""
_HISTORY_SQL = """
    SELECT screening_id, check_type, subject, result, rating, score, details, provider, reference,
           checked_on, valid_until, recorded_by, created_date
      FROM proc.bp_supplier_screening
     WHERE supplier_id = %s
     ORDER BY checked_on DESC, screening_id DESC
     LIMIT %s
"""
# The product's own risk-signal feed (src/repositories/supplier_risk_repo.py), read beside the
# recorded risk-event reviews so a monitor call shows both what was reviewed and what was recorded.
_SIGNALS_SQL = """
    SELECT signal_type, severity, source, payload, occurred_at
      FROM proc.supplier_risk_signals
     WHERE supplier_id = %s
     ORDER BY occurred_at DESC
     LIMIT %s
"""


class InvalidEntry(ValueError):
    """The entry cannot be recorded as given; the message says what is wrong."""


def _plain(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    return value


def _as_date(value: Any, name: str, *, required: bool) -> Optional[date]:
    if value is None or value == "":
        if required:
            raise InvalidEntry(f"{name} is required (a date like 2026-01-31)")
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value).strip()[:10])
    except ValueError:
        raise InvalidEntry(f"{name} must be a date like 2026-01-31") from None


def validate(entry: Dict[str, Any], *, today: Optional[date] = None) -> Dict[str, Any]:
    """The entry as it will be stored, or raise InvalidEntry naming the first problem."""
    today = today or date.today()
    sid = str(entry.get("supplier_id") or "").strip()
    if not sid:
        raise InvalidEntry("supplier_id is required")
    check_type = str(entry.get("check_type") or "").strip()
    if check_type not in CHECK_TYPES:
        raise InvalidEntry(f"check_type must be one of: {', '.join(CHECK_TYPES)}")
    result = str(entry.get("result") or "").strip()
    if result not in RESULTS:
        raise InvalidEntry(f"result must be one of: {', '.join(RESULTS)}")
    provider = str(entry.get("provider") or "").strip()
    if not provider:
        raise InvalidEntry("provider is required: who performed the check (a provider or a person)")
    checked_on = _as_date(entry.get("checked_on"), "checked_on", required=True)
    if checked_on > today:
        raise InvalidEntry("checked_on cannot be in the future")
    valid_until = _as_date(entry.get("valid_until"), "valid_until", required=False)
    if valid_until is not None and valid_until < checked_on:
        raise InvalidEntry("valid_until cannot be before checked_on")
    score = entry.get("score")
    if score is not None:
        try:
            score = float(score)
        except (TypeError, ValueError):
            raise InvalidEntry("score must be a number") from None
    details = entry.get("details")
    if details is None:
        details = {}
    if not isinstance(details, dict):
        raise InvalidEntry("details must be an object")
    subject = str(entry.get("subject") or "supplier").strip() or "supplier"
    rating = entry.get("rating")
    rating = str(rating).strip() if rating not in (None, "") else None
    reference = entry.get("reference")
    reference = str(reference).strip() if reference not in (None, "") else None
    return {"supplier_id": sid, "check_type": check_type, "subject": subject, "result": result,
            "rating": rating, "score": score, "details": details, "provider": provider,
            "reference": reference, "checked_on": checked_on, "valid_until": valid_until}


def record(entry: Dict[str, Any], by: str, *, execute: Optional[Callable[[str, Sequence[Any]], List[tuple]]] = None,
           today: Optional[date] = None) -> Dict[str, Any]:
    """Append one check. ``by`` is the signed-in person; it is never taken from the entry."""
    who = str(by or "").strip()
    if not who:
        raise InvalidEntry("the recording person is unknown; sign in to record a check")
    clean = validate(entry, today=today)
    params = [clean["supplier_id"], clean["check_type"], clean["subject"], clean["result"], clean["rating"],
              clean["score"], json.dumps(clean["details"]), clean["provider"], clean["reference"],
              clean["checked_on"], clean["valid_until"], who]
    rows = (execute or store.rows)(_INSERT_SQL, params)
    sid, created = rows[0] if rows else (None, None)
    return {**{k: _plain(v) for k, v in clean.items()}, "screening_id": sid, "recorded_by": who,
            "created_date": _plain(created)}


def _check_row(row: tuple, today: date) -> Dict[str, Any]:
    subject, result, rating, score, details, provider, reference, checked_on, valid_until, by, created = row
    valid_until = _as_date(valid_until, "valid_until", required=False) if valid_until is not None else None
    if isinstance(details, str):
        try:
            details = json.loads(details)
        except ValueError:
            details = {"as_stored": details}
    return {"subject": subject, "result": result, "rating": rating, "score": _plain(score),
            "details": details or {}, "provider": provider, "reference": reference,
            "checked_on": _plain(checked_on), "valid_until": _plain(valid_until),
            "expired": bool(valid_until is not None and valid_until < today),
            "recorded_by": by, "recorded_at": _plain(created)}


def lookup(supplier: Any = None, check_type: str = "", *,
           rows: Optional[Callable[[str, Sequence[Any]], List[tuple]]] = None,
           today: Optional[date] = None) -> Dict[str, Any]:
    """The latest recorded check of one type for one supplier, per screened subject."""
    if check_type not in CHECK_TYPES:
        return {"found": False, "note": f"check_type must be one of: {', '.join(CHECK_TYPES)}."}
    label = CHECK_TYPES[check_type]["label"]
    today = today or date.today()
    read = rows or store.rows
    resolved, reply = supplier_resolve.resolve(supplier, rows=read)
    if reply is not None:
        if reply.get("available") is False:
            reply["note"] += f" No {label} could be looked up, and that is not a clear result."
        return reply
    sid, sname = resolved
    try:
        latest = read(_LATEST_SQL, [sid, check_type])
        signals = read(_SIGNALS_SQL, [sid, MAX_SIGNALS]) if check_type == "risk_event" else []
    except store.StoreUnavailable:
        return {"found": False, "available": False,
                "note": f"The screening register could not be read just now. This does not mean no {label} "
                        "was recorded, and it is not a clear result."}
    checks = [_check_row(r, today) for r in latest[:MAX_SUBJECTS]]
    performed = [c for c in checks if c["result"] != "not_performed"]
    out: Dict[str, Any] = {"found": True, "supplier": {"supplier_id": sid, "name": sname},
                           "check_type": check_type, "check": label, "performed": bool(performed),
                           "checks": checks, "flagged": any(c["result"] == "flagged" for c in checks),
                           "expired": any(c["expired"] for c in performed),
                           "performed_by_this_tool": False}
    if check_type == "risk_event":
        out["signals"] = [{"signal_type": t, "severity": _plain(sev), "source": src,
                           "payload": (json.loads(p) if isinstance(p, str) else p) or {},
                           "occurred_at": _plain(at)} for t, sev, src, p, at in signals]
    if not performed:
        out["note"] = (f"No {label} is recorded for this supplier. This is NOT a clear result: ask compliance to "
                       "perform the check and record it in the screening register, and treat the check as "
                       "'not performed' until then.")
    else:
        out["note"] = (f"The latest recorded {label} per screened subject, as the provider stated it, with the date "
                       "it was performed. This tool performed no check itself. Quote the result, provider and "
                       "date; a check marked expired needs refreshing before it is relied on.")
    if check_type == "risk_event" and not signals:
        out["note"] += " No risk signals are recorded in the platform's own feed for this supplier."
    return out


_LATEST_ALL_SQL = """
    SELECT DISTINCT ON (check_type, subject) check_type, subject, result, rating, score, details, provider,
           reference, checked_on, valid_until, recorded_by, created_date
      FROM proc.bp_supplier_screening
     WHERE supplier_id = %s
     ORDER BY check_type, subject, checked_on DESC, screening_id DESC
"""


def summary(supplier: Any = None, *, rows: Optional[Callable[[str, Sequence[Any]], List[tuple]]] = None,
            today: Optional[date] = None) -> Dict[str, Any]:
    """Every check type for one supplier in one read: the latest record per subject, or 'not performed'.
    Built because a skill that called the six single-check tools in turn spent its whole round budget
    before it could answer (live, 2026-10-22). Performs no check; no record is never clear."""
    today = today or date.today()
    read = rows or store.rows
    resolved, reply = supplier_resolve.resolve(supplier, rows=read)
    if reply is not None:
        if reply.get("available") is False:
            reply["note"] += " No screening could be looked up, and that is not a clear result."
        return reply
    sid, sname = resolved
    try:
        latest = read(_LATEST_ALL_SQL, [sid])
        signals = read(_SIGNALS_SQL, [sid, MAX_SIGNALS])
    except store.StoreUnavailable:
        return {"found": False, "available": False,
                "note": "The screening register could not be read just now. This does not mean no check was "
                        "recorded, and it is not a clear result."}
    by_type: Dict[str, List[Dict[str, Any]]] = {}
    for r in latest:
        by_type.setdefault(r[0], []).append(_check_row(r[1:], today))
    checks: Dict[str, Dict[str, Any]] = {}
    for ct, meta in CHECK_TYPES.items():
        entries = by_type.get(ct, [])[:MAX_SUBJECTS]
        performed = [c for c in entries if c["result"] != "not_performed"]
        checks[ct] = {"check": meta["label"], "tool": meta["tool"], "performed": bool(performed),
                      "flagged": any(c["result"] == "flagged" for c in entries),
                      "expired": any(c["expired"] for c in performed),
                      "subjects": [c["subject"] for c in performed], "checks": entries}
    not_performed = [ct for ct, c in checks.items() if not c["performed"]]
    flagged = [ct for ct, c in checks.items() if c["flagged"]]
    expired = [ct for ct, c in checks.items() if c["expired"]]
    out: Dict[str, Any] = {"found": True, "supplier": {"supplier_id": sid, "name": sname}, "checks": checks,
                           "not_performed": not_performed, "flagged": flagged, "expired": expired,
                           "signals": [{"signal_type": t, "severity": _plain(sev), "source": src,
                                        "payload": (json.loads(p) if isinstance(p, str) else p) or {},
                                        "occurred_at": _plain(at)} for t, sev, src, p, at in signals],
                           "performed_by_this_tool": False}
    if len(not_performed) == len(CHECK_TYPES):
        out["note"] = ("No check of any kind is recorded for this supplier. This is NOT a clear result: every check is "
                       "'not performed' until compliance performs it and records it in the screening register.")
    else:
        out["note"] = ("Latest recorded result per check and screened subject, as the provider stated it, with dates. "
                       "This tool performed no check. A check listed under not_performed is not clear; one under "
                       "expired needs refreshing; one under flagged blocks until compliance resolves it.")
    if not signals:
        out["note"] += " No risk signals are recorded in the platform's own feed for this supplier."
    return out


def history(supplier_id: str, *, rows: Optional[Callable[[str, Sequence[Any]], List[tuple]]] = None,
            limit: int = 100, today: Optional[date] = None) -> List[Dict[str, Any]]:
    """Every recorded check for one supplier_id, newest first (for the screen; no resolution)."""
    today = today or date.today()
    found = (rows or store.rows)(_HISTORY_SQL, [supplier_id, max(1, min(int(limit), 500))])
    out = []
    for r in found:
        screening_id, check_type = r[0], r[1]
        item = _check_row(r[2:], today)
        item.update({"screening_id": screening_id, "check_type": check_type})
        out.append(item)
    return out
