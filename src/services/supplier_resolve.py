"""Turn what a person or an agent calls a supplier into the one supplier_id on record, or say why not.

Shared by the tools that take a supplier (scorecard, screening register). Matching is by exact id or
by words in the recorded name or trading name; several matches are returned for the caller to ask
about, never picked from.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from src.services.contract_tools import store

_SQL = """
    SELECT s.supplier_id::text, s.supplier_name
      FROM proc.bp_supplier s
     WHERE s.supplier_id::text = %s OR s.supplier_name ILIKE %s OR s.trading_name ILIKE %s
     ORDER BY s.supplier_name
     LIMIT 6
"""

OUTAGE = {"found": False, "available": False,
          "note": "The supplier records could not be read just now. This does not mean the supplier is unknown."}


def resolve(supplier: Any, *, rows: Optional[Callable[[str, Sequence[Any]], List[tuple]]] = None
            ) -> Tuple[Optional[Tuple[str, str]], Optional[Dict[str, Any]]]:
    """``((supplier_id, name), None)`` when exactly one supplier matches, else ``(None, reply)`` where
    ``reply`` is the tool result to return as is (missing name, none found, several found, outage)."""
    name = str(supplier).strip() if supplier is not None else ""
    if not name:
        return None, {"found": False, "note": "Give the supplier's name or supplier_id."}
    try:
        matches = (rows or store.rows)(_SQL, [name, f"%{name}%", f"%{name}%"])
    except store.StoreUnavailable:
        return None, dict(OUTAGE)
    if not matches:
        return None, {"found": False,
                      "note": f"No supplier on record matches '{name}'. Suppliers not on record cannot be looked up."}
    if len(matches) > 1:
        return None, {"found": False, "ambiguous": True,
                      "candidates": [{"supplier_id": sid, "name": nm} for sid, nm in matches],
                      "note": "Several suppliers match; ask the user which one, then pass its supplier_id."}
    sid, sname = matches[0]
    return (sid, sname), None
