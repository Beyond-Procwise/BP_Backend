"""Negotiate-page dashboard for a procurement deal.

GET /deals/{deal_id}/negotiate — returns the full negotiate-page payload
(summary, proposal snapshot, offer version history, negotiation KPIs + strategy,
cost-over-time, volume trend, baseline-vs-current proposal summary, demand-vs-
volume), computed from the deal's final (_trgt) records with safe fallbacks.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException

from src.services.negotiate_dashboard import build_negotiate_dashboard

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/deals", tags=["Negotiate"])


@router.get("/{deal_id}/negotiate", summary="Negotiate-page dashboard for a deal")
def get_negotiate_dashboard(deal_id: str) -> dict[str, Any]:
    try:
        result = build_negotiate_dashboard(deal_id)
    except Exception as exc:  # DB or unexpected error
        logger.exception("negotiate dashboard failed for %s", deal_id)
        raise HTTPException(status_code=500, detail=str(exc))
    if result is None:
        raise HTTPException(status_code=404, detail=f"No deal found for deal_id={deal_id}")
    result["generated_at"] = datetime.now(timezone.utc).isoformat()
    return result
