"""AI summary + reconciliation of a procurement deal.

GET  /deals/{deal_id}/summary   — consolidates the deal's final (_trgt) records,
line items, action trail and discrepancies, then returns a clear-text summary.
POST /deals/{deal_id}/reconcile — compares the deal's documents (amount/currency/
supplier/tax) and records consolidation actions; returns the verdicts.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException

from src.services.deal_summary import summarize_deal, SummarizationError
from src.services.reconciliation import reconcile_deal

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/deals", tags=["Deals"])


@router.get("/orphans", summary="PO/invoice documents awaiting an anchoring quote")
def get_deal_orphans() -> dict[str, Any]:
    """Quote-anchored model: list PO/invoice docs whose deal has no quote yet."""
    from src.services.db import get_conn
    try:
        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                "select deal_id, deal_name, doc_type, doc_pk, supplier_name, "
                "amount, currency, doc_date, status from proc.bp_deal_orphans "
                "order by amount desc nulls last")
            cols = [d[0] for d in cur.description]
            items = [dict(zip(cols, r)) for r in cur.fetchall()]
    except Exception as exc:
        logger.exception("deal orphans read failed")
        raise HTTPException(status_code=500, detail=str(exc))
    return {"orphans": items, "count": len(items),
            "generated_at": datetime.now(timezone.utc).isoformat()}


@router.get("/{deal_id}/summary", summary="AI summary of a procurement deal")
def get_deal_summary(deal_id: str) -> dict[str, Any]:
    try:
        result = summarize_deal(deal_id)
    except SummarizationError as exc:
        raise HTTPException(status_code=502, detail=f"Summarization failed: {exc}")
    except Exception as exc:  # DB or unexpected error
        logger.exception("deal summary failed for %s", deal_id)
        raise HTTPException(status_code=500, detail=str(exc))
    if result is None:
        raise HTTPException(status_code=404, detail=f"No deal found for deal_id={deal_id}")
    result["generated_at"] = datetime.now(timezone.utc).isoformat()
    return result


@router.post("/{deal_id}/reconcile", summary="Reconcile a deal's documents")
def post_deal_reconcile(deal_id: str) -> dict[str, Any]:
    try:
        result = reconcile_deal(deal_id)
    except Exception as exc:  # DB or unexpected error
        logger.exception("deal reconcile failed for %s", deal_id)
        raise HTTPException(status_code=500, detail=str(exc))
    if result is None:
        raise HTTPException(status_code=404, detail=f"No deal found for deal_id={deal_id}")
    result["generated_at"] = datetime.now(timezone.utc).isoformat()
    return result
