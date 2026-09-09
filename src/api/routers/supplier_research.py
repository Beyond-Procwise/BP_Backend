"""Supplier web-research API — AgentNick researches a supplier and enriches it.

Grounded (cited) facts only; auto-fills empty non-sensitive fields; provenance in
proc.bp_supplier_enrichment. Never overwrites, never fabricates.
"""
from __future__ import annotations

import logging
import os

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from src.services.db import get_conn
from src.services.supplier_enrichment import research as R

log = logging.getLogger(__name__)

from api.auth import require_user
from api.endpoint_gate import require as gate
router = APIRouter(prefix="/suppliers", tags=["Supplier Research"])


def _enabled() -> bool:
    return os.getenv("SUPPLIER_RESEARCH_ENABLED", "1") not in ("0", "false", "False")


class RejectBody(BaseModel):
    reviewer: str = "api"


@router.post("/research/batch")
def batch(limit: int = 10, principal=Depends(require_user)):
    gate("research.web", principal, agent="SupplierResearchRouter")
    if not _enabled():
        raise HTTPException(status_code=403, detail="supplier research disabled")
    with get_conn() as c:
        return R.batch_research(c, limit=max(1, min(limit, 25)))


@router.get("/enrichment/reviews")
def enrichment_reviews(status: str = "pending", limit: int = 50):
    """One-place review payload: each pending enrichment with matched_name,
    name_match, citations, the researched fields, the supplier's CURRENT values,
    and which empty fields would be filled on approve."""
    with get_conn() as c, c.cursor() as cur:
        cur.execute(
            "SELECT e.enrichment_id, e.supplier_id, s.supplier_name, "
            "e.raw->>'matched_name' AS matched_name, "
            "NULLIF(e.raw->>'name_match','')::float AS name_match, "
            "e.confidence, e.fields, e.citations, e.apply_status, e.created_date "
            "FROM proc.bp_supplier_enrichment e "
            "LEFT JOIN proc.bp_supplier s ON s.supplier_id = e.supplier_id "
            "WHERE (%s = 'all' OR e.apply_status = %s) "
            "ORDER BY e.created_date DESC LIMIT %s",
            (status, status, max(1, min(limit, 200))),
        )
        cols = [d[0] for d in cur.description]
        reviews = [dict(zip(cols, r)) for r in cur.fetchall()]
        # attach the supplier's current values + a would-fill preview per review
        for rv in reviews:
            cur.execute(
                "SELECT " + ", ".join(R._APPLY_COLUMNS) + " FROM proc.bp_supplier WHERE supplier_id = %s",
                (rv["supplier_id"],),
            )
            sv = cur.fetchone()
            current = dict(zip(R._APPLY_COLUMNS, sv)) if sv else {}
            fields = rv.get("fields") or {}
            rv["current"] = current
            rv["would_fill"] = [
                col for col in R._APPLY_COLUMNS
                if fields.get(col) and (current.get(col) is None or str(current.get(col)).strip() == "")
            ]
    return {"count": len(reviews), "reviews": reviews}


@router.post("/enrichment/{enrichment_id}/apply")
def apply(enrichment_id: int, body: RejectBody, principal=Depends(require_user)):
    gate("supplier.write", principal, agent="SupplierResearchRouter",
         context={"enrichment_id": enrichment_id})
    """Human-approve a pending enrichment (fills empty non-sensitive fields)."""
    with get_conn() as c:
        try:
            return R.apply_enrichment(enrichment_id, body.reviewer, c)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc))


@router.post("/enrichment/{enrichment_id}/reject")
def reject(enrichment_id: int, body: RejectBody):
    with get_conn() as c:
        try:
            return R.reject_enrichment(enrichment_id, body.reviewer, c)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc))


@router.post("/{supplier_id}/research")
def research(supplier_id: str, principal=Depends(require_user)):
    gate("research.web", principal, agent="SupplierResearchRouter",
         context={"supplier_id": supplier_id})
    if not _enabled():
        raise HTTPException(status_code=403, detail="supplier research disabled")
    with get_conn() as c:
        result = R.research_and_enrich(supplier_id, c)
    if result.get("error"):
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@router.get("/{supplier_id}/enrichment")
def get_enrichment(supplier_id: str):
    with get_conn() as c, c.cursor() as cur:
        cur.execute(
            "SELECT enrichment_id, created_date, model, fields, citations, confidence, "
            "apply_status, applied_fields, reviewed_by, reviewed_date "
            "FROM proc.bp_supplier_enrichment WHERE supplier_id = %s "
            "ORDER BY created_date DESC LIMIT 1",
            (supplier_id,),
        )
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="no enrichment for supplier")
        cols = [d[0] for d in cur.description]
    return dict(zip(cols, row))
