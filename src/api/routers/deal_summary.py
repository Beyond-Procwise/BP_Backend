"""AI summary of a procurement deal.

GET /deals/{deal_id}/summary — consolidates the deal's final (_trgt) records,
line items, action trail and discrepancies, then returns a clear-text summary.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException

from src.services.deal_summary import summarize_deal, SummarizationError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/deals", tags=["Deals"])


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
