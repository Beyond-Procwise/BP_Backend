"""search_contracts: find contracts by words, supplier, status or expiry window."""
from __future__ import annotations

from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional, Sequence

from . import store

DEFAULT_LIMIT = 10
MAX_LIMIT = 25

_SELECT = """
    SELECT c.contract_id, c.contract_title, c.contract_type, COALESCE(s.supplier_name, c.supplier_id::text),
           c.contract_start_date, c.contract_end_date, c.contract_lifecycle_status, c.currency,
           c.total_contract_value, c.auto_renew_flag, c.renewal_term, c.governing_law, c.payment_terms
      FROM proc.bp_contract_master c
      LEFT JOIN proc.bp_supplier s ON s.supplier_id = c.supplier_id
"""
_COLUMNS = ("contract_id", "title", "type", "supplier", "start_date", "end_date", "status", "currency",
            "total_value", "auto_renew", "renewal_term", "governing_law", "payment_terms")


def _limit(value: Any) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return DEFAULT_LIMIT
    return max(1, min(n, MAX_LIMIT))


def search(query: Any = None, supplier: Any = None, status: Any = None, expires_within_days: Any = None,
           limit: Any = None, *, rows: Optional[Callable[[str, Sequence[Any]], List[tuple]]] = None) -> Dict[str, Any]:
    read = rows or store.rows
    where: List[str] = []
    params: List[Any] = []
    if isinstance(query, str) and query.strip():
        where.append("(c.contract_title ILIKE %s OR c.contract_type ILIKE %s OR c.spend_category ILIKE %s)")
        params += [f"%{query.strip()}%"] * 3
    if isinstance(supplier, str) and supplier.strip():
        where.append("COALESCE(s.supplier_name, '') ILIKE %s")
        params.append(f"%{supplier.strip()}%")
    if isinstance(status, str) and status.strip():
        where.append("c.contract_lifecycle_status ILIKE %s")
        params.append(status.strip())
    if expires_within_days is not None:
        try:
            days = int(expires_within_days)
        except (TypeError, ValueError):
            return {"found": False, "note": "expires_within_days must be a whole number of days."}
        where.append("c.contract_end_date BETWEEN CURRENT_DATE AND CURRENT_DATE + %s * INTERVAL '1 day'")
        params.append(days)
    if not where:
        return {"found": False, "note": "Give at least one of: query, supplier, status, expires_within_days."}
    n = _limit(limit)
    sql = _SELECT + " WHERE " + " AND ".join(where) + " ORDER BY c.contract_end_date NULLS LAST LIMIT %s"
    try:
        found = read(sql, params + [n + 1])
    except store.StoreUnavailable:
        return {"found": False, "available": False,
                "note": "The contract register could not be read just now. This does not mean there are no contracts."}
    contracts = [{k: (str(v) if hasattr(v, "isoformat") else float(v) if isinstance(v, Decimal) else v)
                  for k, v in zip(_COLUMNS, r)} for r in found[:n]]
    out: Dict[str, Any] = {"found": bool(contracts), "contracts": contracts, "count": len(contracts)}
    if len(found) > n:
        out["note"] = f"More than {n} match; narrow the search to see the rest."
    elif not contracts:
        out["note"] = "No contract in the register matches. Contracts not uploaded are not covered."
    return out
