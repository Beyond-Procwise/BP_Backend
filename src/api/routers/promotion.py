"""Confidence-gated _stg -> _trgt promotion driven by the linking engine.

POST /promotion/run — score each not-yet-promoted staged invoice/quote against
its parent PO and promote the ones whose extraction confidence and document
link score both pass. deal_id is left for the SQL trigger.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query

from src.services.linking_engine import promote_ready

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/promotion", tags=["Promotion"])


@router.post("/run", summary="Promote eligible _stg rows to _trgt via the linking engine")
def run_promotion(
    doc_type: Optional[str] = Query(None, description="invoice | quote (default: both)"),
    limit: Optional[int] = Query(None, description="max rows per doc type"),
) -> dict[str, Any]:
    doc_types = (doc_type,) if doc_type else ("invoice", "quote")
    if any(d not in ("invoice", "quote") for d in doc_types):
        raise HTTPException(status_code=400, detail="doc_type must be 'invoice' or 'quote'")
    try:
        return promote_ready(doc_types=doc_types, limit=limit)
    except Exception as exc:
        logger.exception("promotion run failed")
        raise HTTPException(status_code=500, detail=str(exc))
