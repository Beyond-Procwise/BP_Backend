"""contract_pipeline: the contracts falling due within a horizon, each with its stored calendar and
notice clauses, in one call. Read-only.

Built because the portfolio skill, reading the register and then one tracker per contract, spent its
six rounds before it could answer (live, 2026-10-22). This runs the register search and the
obligation tracker for the soonest contracts inside one tool call, so the agent gets the whole
pipeline at once: end date, auto-renewal, value, the clauses that mention notice, termination,
renewal or review quoted word for word, and whether the clauses were ever read. Nothing is
scheduled or changed, and no notice deadline is computed from a period the records do not state.
"""
from __future__ import annotations

from datetime import date
from typing import Any, Callable, Dict, List, Optional, Sequence

from . import search, store, tracker

DEFAULT_HORIZON_DAYS = 180
MAX_HORIZON_DAYS = 730
DEFAULT_DETAIL = 10
MAX_DETAIL = 15
LIST_LIMIT = search.MAX_LIMIT


def _int(value: Any, default: int, lo: int, hi: int) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, n))


def build(expires_within_days: Any = None, supplier: Any = None, status: Any = None, detail: Any = None, *,
          rows: Optional[Callable[[str, Sequence[Any]], List[tuple]]] = None,
          today: Optional[date] = None) -> Dict[str, Any]:
    horizon = _int(expires_within_days, DEFAULT_HORIZON_DAYS, 1, MAX_HORIZON_DAYS)
    n_detail = _int(detail, DEFAULT_DETAIL, 1, MAX_DETAIL)
    today = today or date.today()
    found = search.search(supplier=supplier, status=status, expires_within_days=horizon, limit=LIST_LIMIT, rows=rows)
    if found.get("available") is False:
        return {"found": False, "available": False, "horizon_days": horizon,
                "note": "The contract register could not be read just now. This does not mean nothing falls due."}
    # The register stores the flag as text ("Yes"/"No"); the search returns it as stored, so it is read here.
    contracts = [{**c, "auto_renew": tracker._yes(c.get("auto_renew"))} for c in (found.get("contracts") or [])]
    contracts.sort(key=lambda c: (c.get("end_date") is None, c.get("end_date") or ""))
    items: List[Dict[str, Any]] = []
    for c in contracts[:n_detail]:
        t = tracker.track(c["contract_id"], rows=rows, today=today)
        item = {**c, "days_to_end": None, "calendar": [], "notice_clauses": [], "clauses_read": None,
                "administration_clauses": 0}
        if c.get("end_date"):
            try:
                item["days_to_end"] = (date.fromisoformat(c["end_date"]) - today).days
            except ValueError:
                pass
        if t.get("available") is False:
            item["tracker"] = "could not be read"
        elif t.get("found"):
            item["calendar"] = t.get("calendar") or []
            item["clauses_read"] = bool(t.get("clauses_read"))
            item["administration_clauses"] = len(t.get("administration_clauses") or [])
            item["notice_clauses"] = [cl for cl in (t.get("administration_clauses") or []) if cl.get("mentions_notice")]
            item["decision_basis"] = ("notice clause quoted: read the period from it" if item["notice_clauses"]
                                      else "no notice clause stored" if item["clauses_read"]
                                      else "clauses never read for this contract")
        else:
            item["tracker"] = t.get("note")
        items.append(item)
    listed_only = [{k: c.get(k) for k in ("contract_id", "title", "supplier", "end_date", "auto_renew", "total_value", "currency")}
                   for c in contracts[n_detail:]]
    auto = [c["contract_id"] for c in contracts if c.get("auto_renew")]
    out: Dict[str, Any] = {"found": bool(contracts), "as_of": today.isoformat(), "horizon_days": horizon,
                           "contracts_due": len(contracts), "detailed": items, "listed_only": listed_only,
                           "auto_renewing": auto,
                           "not_read_for_clauses": [i["contract_id"] for i in items if i.get("clauses_read") is False],
                           "no_notice_clause_stored": [i["contract_id"] for i in items if i.get("clauses_read") and not i["notice_clauses"]],
                           "registered": False, "scheduled": False}
    notes = [f"Contracts on the register ending within {horizon} days of {today.isoformat()}, soonest first; the first "
             f"{min(n_detail, len(contracts))} carry their stored calendar and the clauses that mention notice, "
             "termination, renewal or review, quoted word for word.",
             "The notice period is not a stored field: read it from a quoted notice clause; where none is quoted the "
             "decision date is unknown, and a contract never read for clauses may still contain one. Nothing is "
             "scheduled or changed by this call. The register covers only uploaded contracts."]
    if not contracts:
        notes.insert(0, "No contract on the register ends within the horizon.")
    if found.get("note") and "More than" in str(found.get("note")):
        notes.append(found["note"])
    out["note"] = " ".join(notes)
    return out
