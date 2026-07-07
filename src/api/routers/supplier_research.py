"""Supplier web-research API — AgentNick researches a supplier and enriches it.

Grounded (cited) facts only; auto-fills empty non-sensitive fields; provenance in
proc.bp_supplier_enrichment. Never overwrites, never fabricates.
"""
from __future__ import annotations

import logging
import os

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.services.db import get_conn
from src.services.supplier_enrichment import research as R

log = logging.getLogger(__name__)

router = APIRouter(prefix="/suppliers", tags=["Supplier Research"])


def _enabled() -> bool:
    return os.getenv("SUPPLIER_RESEARCH_ENABLED", "1") not in ("0", "false", "False")


class RejectBody(BaseModel):
    reviewer: str = "api"


@router.post("/research/batch")
def batch(limit: int = 10):
    if not _enabled():
        raise HTTPException(status_code=403, detail="supplier research disabled")
    with get_conn() as c:
        return R.batch_research(c, limit=max(1, min(limit, 25)))


@router.post("/enrichment/{enrichment_id}/reject")
def reject(enrichment_id: int, body: RejectBody):
    with get_conn() as c:
        try:
            return R.reject_enrichment(enrichment_id, body.reviewer, c)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc))


@router.post("/{supplier_id}/research")
def research(supplier_id: str):
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
