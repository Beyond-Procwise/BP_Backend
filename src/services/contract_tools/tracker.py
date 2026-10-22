"""obligation_tracker: a contract's administration calendar, from what is stored. Read-only.

The register (``bp_contract_master``) gives the dates, status, auto-renewal flag and renewal term as
uploaded. The clauses the reader found (``bp_contract_obligation``, every quote checked against the
document) give the obligations that have a date, a trigger or a notice attached. Nothing here is
registered, scheduled or reminded: the tool tells the agent what the records say is due, and says what
it cannot tell. In particular the notice period is NOT a stored field, so a notice deadline is never
computed here: the clause is quoted and the agent must read the period from it with the user.
"""
from __future__ import annotations

import re
from datetime import date
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional, Sequence

from . import clauses, store

DEFAULT_HORIZON_DAYS = 180
MAX_HORIZON_DAYS = 730
MAX_CLAUSES = 40

_MASTER_SQL = """
    SELECT c.contract_id, c.contract_title, c.contract_type, COALESCE(s.supplier_name, c.supplier_id::text),
           c.contract_start_date, c.contract_end_date, c.contract_lifecycle_status, c.currency,
           c.total_contract_value, c.auto_renew_flag, c.renewal_term, c.payment_terms
      FROM proc.bp_contract_master c
      LEFT JOIN proc.bp_supplier s ON s.supplier_id = c.supplier_id
     WHERE (c.contract_id = %s OR c.contract_title ILIKE %s){supplier}
     ORDER BY c.contract_end_date NULLS LAST
     LIMIT 6
"""
_SUPPLIER_CLAUSE = " AND COALESCE(s.supplier_name, '') ILIKE %s"
_COLUMNS = ("contract_id", "title", "type", "supplier", "start_date", "end_date", "status", "currency",
            "total_value", "auto_renew", "renewal_term", "payment_terms")

# Words that mark a clause as part of running the contract: something falls due, must be given, or
# changes on a date or an event. Matched against the clause name and its quote, case-insensitively.
_ADMIN_WORDS = re.compile(
    r"notice|terminat|renew|expir|review|adjust|index|rebate|audit|report|deadline|escalat|"
    r"within\s+\d+\s+(?:calendar\s+|business\s+|working\s+)?(?:days?|months?|weeks?)|"
    r"annual|quarter|monthly|each year|every year|anniversary|price (?:increase|change|revision)",
    re.IGNORECASE)
_NOTICE_WORDS = re.compile(r"notice|terminat|non-renew|not to renew", re.IGNORECASE)


def _yes(flag: Any) -> bool:
    return flag is True or str(flag or "").strip().lower() in ("yes", "true", "y", "1")


def _plain(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    return value


def _horizon(value: Any) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return DEFAULT_HORIZON_DAYS
    return max(1, min(n, MAX_HORIZON_DAYS))


def _calendar(header: Dict[str, Any], today: date, horizon_days: int) -> List[Dict[str, Any]]:
    """Dated events that follow from stored fields alone. Each names the field it came from."""
    items: List[Dict[str, Any]] = []
    end = header.get("end_date")
    if isinstance(end, str):
        try:
            end = date.fromisoformat(end)
        except ValueError:
            end = None
    if end is None:
        items.append({"event": "expiry", "date": None, "source": "contract_end_date",
                      "note": "No end date is stored; ask the user for it and label it unverified."})
        return items
    days = (end - today).days
    items.append({"event": "expiry", "date": end.isoformat(), "days_from_today": days, "source": "contract_end_date",
                  "passed": days < 0, "within_horizon": 0 <= days <= horizon_days})
    if header.get("auto_renew"):
        items.append({"event": "auto-renewal", "date": end.isoformat(), "days_from_today": days,
                      "source": "auto_renew_flag", "renewal_term": header.get("renewal_term"),
                      "note": ("The contract renews on expiry unless notice is given. The notice period is not a "
                               "stored field, so the notice deadline is not computed here: read it from the "
                               "quoted notice clause with the user, and treat an unquoted period as unknown.")})
    return items


def track(contract: Any = None, supplier: Any = None, horizon_days: Any = None, *,
          rows: Optional[Callable[[str, Sequence[Any]], List[tuple]]] = None,
          today: Optional[date] = None) -> Dict[str, Any]:
    ident = str(contract).strip() if contract is not None else ""
    if not ident:
        return {"found": False, "note": "Give the contract: its contract_id, document_id, or words from its title."}
    read = rows or store.rows
    today = today or date.today()
    horizon = _horizon(horizon_days)
    sup = supplier.strip() if isinstance(supplier, str) and supplier.strip() else None
    sql = _MASTER_SQL.format(supplier=_SUPPLIER_CLAUSE if sup else "")
    params: List[Any] = [ident, f"%{ident}%"] + ([f"%{sup}%"] if sup else [])
    try:
        masters = read(sql, params)
    except store.StoreUnavailable:
        return {"found": False, "available": False,
                "note": "The contract register could not be read just now. This does not mean the contract is unknown."}
    if len(masters) > 1:
        return {"found": False, "ambiguous": True,
                "candidates": [{k: _plain(v) for k, v in zip(_COLUMNS, r)
                                if k in ("contract_id", "title", "supplier", "end_date", "status")} for r in masters],
                "note": "Several contracts match; ask the user which one, then pass its contract_id."}

    header: Optional[Dict[str, Any]] = None
    if masters:
        header = {k: _plain(v) for k, v in zip(_COLUMNS, masters[0])}
        header["auto_renew"] = _yes(masters[0][9])
    lookup = header["contract_id"] if header else ident
    read_clauses = clauses.extract(lookup, rows=read)
    if read_clauses.get("available") is False:
        return {"found": False, "available": False,
                "note": "The stored clauses could not be read just now. This does not mean the contract has none."}
    if header is None and not read_clauses.get("read"):
        return {"found": False,
                "note": ("No contract in the register and no stored clauses match. The contract may not have been "
                         "uploaded; ask the user for the expiry, notice period and clause, and label them unverified.")}

    admin, notice = [], 0
    for c in read_clauses.get("clauses") or []:
        text = f"{c.get('name') or ''} {c.get('quote') or ''}"
        if _ADMIN_WORDS.search(text):
            item = dict(c)
            item["mentions_notice"] = bool(_NOTICE_WORDS.search(text))
            notice += 1 if item["mentions_notice"] else 0
            admin.append(item)

    out: Dict[str, Any] = {
        "found": True, "as_of": today.isoformat(), "horizon_days": horizon,
        "contract": header, "in_register": header is not None,
        "calendar": _calendar(header, today, horizon) if header else [],
        "clauses_read": bool(read_clauses.get("read")),
        "clauses_stored": read_clauses.get("clauses_stored", 0),
        "administration_clauses": admin[:MAX_CLAUSES],
        "notice_clauses": notice,
        "registered": False, "scheduled": False,
    }
    notes = ["Dates and flags are as stored in the register; clauses are quoted word for word. Nothing is "
             "registered or scheduled by this tool.",
             "The notice period is not a stored field: no notice deadline is computed. Read it from a quoted "
             "notice clause; if none is quoted, ask the user and label it unverified."]
    if header is None:
        notes.append("This contract is not in the register (no dates stored); only its clauses are known.")
    elif not read_clauses.get("read"):
        notes.append("The contract has not been read for clauses, so no obligations are listed; a notice or "
                     "termination clause may still exist. Ask the user to quote it.")
    elif not admin:
        notes.append("Clauses are stored but none mention a date, notice, renewal or review; the reader may "
                     "have missed some, so say 'not found in the stored clauses', not 'none'.")
    out["note"] = " ".join(notes)
    return out
