"""Contract obligations API — read the grounded obligations extracted from contract prose.

Read-only. Extraction runs in the background (it takes minutes per contract), so nothing
here calls a model; it serves what the obligation service already grounded and persisted.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Query

from src.services.db import get_conn

log = logging.getLogger(__name__)

router = APIRouter(prefix="/obligations", tags=["Contract Obligations"])


def _rows(sql: str, params: tuple) -> list[dict]:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(sql, params)
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]


def _with_parties(obligations: list[dict]) -> list[dict]:
    """Attach each obligation's parties — the hyperedge is meaningless without them."""
    if not obligations:
        return []
    ids = tuple(o["obligation_id"] for o in obligations)
    parties = _rows(
        """
        SELECT obligation_id, entity_name, entity_type
          FROM proc.bp_contract_obligation_party
         WHERE obligation_id IN %s
        """,
        (ids,),
    )
    by_id: dict[int, list[dict]] = {}
    for p in parties:
        by_id.setdefault(p["obligation_id"], []).append(
            {"name": p["entity_name"], "type": p["entity_type"]}
        )
    for o in obligations:
        o["parties"] = by_id.get(o["obligation_id"], [])
    return obligations


@router.get("/contract/{document_id}")
def obligations_for_contract(document_id: str):
    """Every grounded obligation read from one contract, plus the run status.

    The status matters as much as the list. Without it, a contract we failed to read and a
    contract that genuinely has no obligations both return `[]`, and the caller cannot tell
    the difference — a green zero nobody earned.
    """
    rows = _rows(
        """
        SELECT obligation_id, document_id, contract_id, name, obligation_type,
               clause_ref, source_quote
          FROM proc.bp_contract_obligation
         WHERE document_id = %s
         ORDER BY clause_ref
        """,
        (document_id,),
    )
    run = _rows(
        """
        SELECT status, n_grounded, n_dropped, error
          FROM proc.bp_contract_obligation_run
         WHERE document_id = %s
        """,
        (document_id,),
    )
    return {
        "document_id": document_id,
        # "not_extracted" — we have never attempted to read this contract.
        "run": run[0] if run else {"status": "not_extracted"},
        "obligations": _with_parties(rows),
    }


@router.get("")
def search_obligations(
    party: str | None = Query(None, description="Entity bound by the obligation, e.g. a supplier"),
    obligation_type: str | None = Query(None, description="must_perform, penalised_by, ..."),
):
    """Cross-contract query. Filtering by party is a join over the hyperedge's members."""
    sql = """
        SELECT DISTINCT o.obligation_id, o.document_id, o.contract_id, o.name,
               o.obligation_type, o.clause_ref, o.source_quote
          FROM proc.bp_contract_obligation o
          JOIN proc.bp_contract_obligation_party p USING (obligation_id)
         WHERE (%s IS NULL OR p.entity_name ILIKE %s)
           AND (%s IS NULL OR o.obligation_type = %s)
         ORDER BY o.document_id, o.clause_ref
    """
    like = f"%{party}%" if party else None
    rows = _rows(sql, (party, like, obligation_type, obligation_type))
    return {"count": len(rows), "obligations": _with_parties(rows)}
