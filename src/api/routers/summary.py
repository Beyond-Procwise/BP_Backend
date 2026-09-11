"""Persona-driven summaries over the procurement target tables.

POST /summary            — regenerate + store a summary (on-demand refresh).
GET  /summary            — return the latest cached summary (fast).
GET  /summary/history    — list prior summaries for a persona/scope.
GET  /summary/{id}       — fetch one stored summary.
POST /summary/precompute — warm the cache for personas x scopes.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.auth import require_user
import src.services.summary_agent as summary_agent

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/summary", tags=["Summary"])


class SummaryRequest(BaseModel):
    persona: str
    deal_id: Optional[str] = None
    as_of: Optional[str] = None


class PrecomputeRequest(BaseModel):
    personas: Optional[list[str]] = None
    deal_ids: Optional[list[str]] = None


@router.post("", summary="Generate (refresh) a persona summary")
def post_summary(req: SummaryRequest, principal=Depends(require_user)) -> dict[str, Any]:
    try:
        result = summary_agent.generate_summary(
            req.persona, deal_id=req.deal_id, as_of=req.as_of
        )
    except summary_agent.SnapshotNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except summary_agent.SummarizationError as exc:
        raise HTTPException(status_code=502, detail=f"Summarization failed: {exc}")
    except Exception as exc:  # DB or unexpected
        logger.exception("summary generation failed")
        raise HTTPException(status_code=500, detail=str(exc))
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"No data for persona={req.persona} deal_id={req.deal_id}",
        )
    return result


@router.get("", summary="Latest cached persona summary")
def get_summary(persona: str, deal_id: Optional[str] = None) -> dict[str, Any]:
    try:
        result = summary_agent.get_cached_summary(persona, deal_id)
    except Exception as exc:
        logger.exception("cached summary read failed")
        raise HTTPException(status_code=500, detail=str(exc))
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"No cached summary for persona={persona} deal_id={deal_id}; POST to generate.",
        )
    return result


@router.get("/history", summary="Historical summaries for a persona/scope")
def get_history(persona: str, deal_id: Optional[str] = None) -> dict[str, Any]:
    try:
        return {"history": summary_agent.list_summary_history(persona, deal_id)}
    except Exception as exc:
        logger.exception("summary history read failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/precompute", summary="Warm the summary cache")
def post_precompute(req: PrecomputeRequest, principal=Depends(require_user)) -> dict[str, Any]:
    try:
        return summary_agent.precompute_summaries(
            personas=req.personas, deal_ids=req.deal_ids
        )
    except Exception as exc:
        logger.exception("summary precompute failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/{summary_id}", summary="Fetch one stored summary by id")
def get_one(summary_id: str) -> dict[str, Any]:
    try:
        result = summary_agent.get_summary_by_id(summary_id)
    except Exception as exc:
        logger.exception("summary fetch failed")
        raise HTTPException(status_code=500, detail=str(exc))
    if result is None:
        raise HTTPException(status_code=404, detail=f"No summary {summary_id}")
    return result
