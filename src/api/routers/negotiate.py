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

from fastapi import APIRouter, Depends, HTTPException

from api.auth import require_user
from src.services.negotiate_dashboard import build_negotiate_dashboard
from src.services.negotiation_advice import apply_turn, build_advice

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


# Advice routes sit on this deal-scoped router (prefix "/deals"), beside the
# dashboard they belong to: /deals/{deal_id}/advice. The plan's interface list
# writes them as "/negotiate/...", but that prefix does not exist — the live
# dashboard route is /deals/{deal_id}/negotiate — and Task 7 does not touch
# main.py, so a second top-level router was never intended.
@router.get("/{deal_id}/advice", summary="Grounded negotiation advice for a deal")
def get_negotiate_advice(deal_id: str) -> dict[str, Any]:
    try:
        result = build_advice(deal_id)
    except Exception as exc:
        logger.exception("negotiation advice failed for %s", deal_id)
        raise HTTPException(status_code=500, detail=str(exc))
    if result is None:
        raise HTTPException(status_code=404, detail=f"No deal {deal_id}")
    return result


@router.post("/{deal_id}/advice/message", summary="One advice conversation turn")
def post_negotiate_advice_message(deal_id: str,
                                  body: dict[str, Any],
                                  principal=Depends(require_user)) -> dict[str, Any]:
    # A stated fact changes the advice, and its stated_by was the literal
    # "buyer". Who stated it is the token, or nobody.
    try:
        result = apply_turn(deal_id, body or {},
                            created_by=getattr(principal, "subject", None) or None)
    except Exception as exc:
        logger.exception("advice turn failed for %s", deal_id)
        raise HTTPException(status_code=500, detail=str(exc))
    if result is None:
        raise HTTPException(status_code=404, detail=f"No deal {deal_id}")
    return result


@router.delete("/{deal_id}/advice/fact/{fact_key}",
               summary="Withdraw a buyer-stated fact")
def delete_negotiate_advice_fact(deal_id: str, fact_key: str,
                                 principal=Depends(require_user)) -> dict[str, Any]:
    # Withdrawing can seed the advice row first, and that row has a created_by.
    try:
        result = apply_turn(deal_id, {"action": "withdraw_fact",
                                      "fact_key": fact_key},
                            created_by=getattr(principal, "subject", None) or None)
    except Exception as exc:
        logger.exception("fact withdrawal failed for %s", deal_id)
        raise HTTPException(status_code=500, detail=str(exc))
    if result is None:
        raise HTTPException(status_code=404, detail=f"No deal {deal_id}")
    return result
