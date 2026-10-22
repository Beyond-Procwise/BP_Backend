"""extract_clauses: the clauses already read from a contract, quoted word for word.

Reads the grounded obligations stored when the contract was uploaded; every quote was checked
against the document then. Nothing is read by the model here. A contract that has not been read
yet says so; it does not return an empty list.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

from . import store

TYPES = ("must_perform", "entitled_to", "triggered_by", "penalised_by", "risk_transfer")
MAX_CLAUSES = 60

_SQL = """
    SELECT document_id, contract_id, name, obligation_type, clause_ref, source_quote
      FROM proc.bp_contract_obligation
     WHERE grounded AND (document_id = %s OR contract_id = %s)
"""
_PARTIES_SQL = """
    SELECT p.obligation_id, p.entity_name FROM proc.bp_contract_obligation_party p
      JOIN proc.bp_contract_obligation o USING (obligation_id)
     WHERE o.grounded AND (o.document_id = %s OR o.contract_id = %s)
"""
_ID_SQL = _SQL.replace("SELECT document_id,", "SELECT obligation_id, document_id,")


def extract(document_id: Any = None, clause_type: Any = None, containing: Any = None, *,
            rows: Optional[Callable[[str, Sequence[Any]], List[tuple]]] = None) -> Dict[str, Any]:
    ident = str(document_id).strip() if document_id is not None else ""
    if not ident:
        return {"found": False, "note": "Give the document_id (or contract_id) of the contract to read."}
    if clause_type is not None and clause_type not in TYPES:
        return {"found": False, "note": f"clause_type must be one of: {', '.join(TYPES)}."}
    read = rows or store.rows
    try:
        found = read(_ID_SQL, [ident, ident])
        parties = read(_PARTIES_SQL, [ident, ident])
    except store.StoreUnavailable:
        return {"found": False, "available": False,
                "note": "The stored clauses could not be read just now. This does not mean the contract has none."}
    if not found:
        return {"found": False, "read": False,
                "note": ("No clauses are stored for this id. The contract may not have been uploaded or read yet. "
                         "Ask the user to quote the clause; label it unchecked until they do.")}
    by_ob: Dict[Any, List[str]] = {}
    for oid, name in parties:
        by_ob.setdefault(oid, []).append(name)
    needle = containing.strip().lower() if isinstance(containing, str) and containing.strip() else None
    clauses = []
    for oid, doc, contract, name, otype, ref, quote in found:
        if clause_type and otype != clause_type:
            continue
        if needle and needle not in f"{name} {quote}".lower():
            continue
        clauses.append({"clause_ref": ref, "type": otype, "name": name, "quote": quote,
                        "parties": by_ob.get(oid, [])})
    clauses.sort(key=lambda c: [int(p) if p.isdigit() else 0 for p in str(c["clause_ref"]).split(".")])
    out: Dict[str, Any] = {"found": bool(clauses), "read": True, "document_id": ident,
                           "clauses_stored": len(found), "clauses": clauses[:MAX_CLAUSES],
                           "note": ("Quotes are word for word from the document. Only obligations the reader found "
                                    "are listed; a clause not listed may still be in the contract.")}
    if not clauses:
        out["note"] = "Clauses are stored for this contract, but none match that filter. " + out["note"]
    return out
