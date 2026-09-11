"""Supplier web-research API — AgentNick researches a supplier and PROPOSES.

Grounded (cited) facts only; provenance in proc.bp_supplier_enrichment. Research
writes nothing: it leaves a proposal pending and /enrichment/{id}/apply — a
person — is the only path to the supplier master (A19.39). Never overwrites,
never fabricates, never researches a bank or tax identifier.
"""
from __future__ import annotations

import json
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
    """AutonomousOperationPolicy (P9): whether queries about a supplier may
    leave the tenant at all."""
    from src.services.governed_limits import limit as _governed_limit

    return bool(_governed_limit("autonomous_operation", "supplier_research_enabled",
                                env="SUPPLIER_RESEARCH_ENABLED", cast=bool))


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
            if isinstance(fields, str):
                fields = json.loads(fields or "{}")
            rv["current"] = current
            # Asked of the same function the approve button runs, rather than
            # re-derived here. The local version only checked "researched and
            # currently empty", so it could offer a field that verification or
            # the column width would then refuse — the queue promising a fill
            # that silently did not happen.
            rv["would_fill"] = sorted(R.fillable(cur, rv["supplier_id"], fields))
    return {"count": len(reviews), "reviews": reviews}


@router.post("/enrichment/{enrichment_id}/apply")
def apply(enrichment_id: int, body: RejectBody, principal=Depends(require_user)):
    """Human-approve a pending enrichment (fills empty non-sensitive fields).

    Since research stopped writing on its own, this is the only path into
    ``proc.bp_supplier`` — so the row has to say which person took it. The
    authenticated principal is used in preference to the ``reviewer`` the caller
    typed, which is a label anyone can set. With ASK_AUTH_MODE=off there is no
    principal and the label is all there is; P8 is where that stops being
    acceptable across the API generally.
    """
    gate("supplier.write", principal, agent="SupplierResearchRouter",
         context={"enrichment_id": enrichment_id})
    reviewer = getattr(principal, "subject", None) or body.reviewer
    with get_conn() as c:
        try:
            return R.apply_enrichment(enrichment_id, reviewer, c)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc))


@router.post("/enrichment/{enrichment_id}/reject")
def reject(enrichment_id: int, body: RejectBody, principal=Depends(require_user)):
    """Reject a proposed enrichment. Signed by the token, not by `body.reviewer`."""
    with get_conn() as c:
        try:
            reviewer = getattr(principal, "subject", None) or None
            return R.reject_enrichment(enrichment_id, reviewer, c)
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
