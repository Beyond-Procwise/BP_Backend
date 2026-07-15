"""Opportunities-page dashboard + lifecycle.

GET  /opportunities/dashboard        — KPIs, savings pipeline, monthly trends,
                                        identified-vs-completed, detailed list.
GET  /opportunities                  — detailed opportunities list (paged).
POST /opportunities/{id}/stage       — advance an opportunity's lifecycle stage.
POST /opportunities/sync             — upsert miner JSON findings into the table.
POST /opportunities/link-deals       — backfill deal_id from each opportunity's anchoring quote.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from src.services.opportunity_dashboard import build_opportunities_dashboard, detailed_opportunities
from src.services.opportunity_linkage import link_opportunities_to_deals
from src.services.opportunity_store import set_stage, sync_findings_from_json
from src.services.db import get_conn

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/opportunities", tags=["Opportunities"])


class StageUpdate(BaseModel):
    stage: str
    realised_savings: Optional[float] = None


@router.get("/dashboard", summary="Opportunities-page dashboard")
def get_dashboard(limit: int = Query(100, ge=1, le=1000)) -> dict[str, Any]:
    try:
        result = build_opportunities_dashboard(limit=limit)
    except Exception as exc:
        logger.exception("opportunities dashboard failed")
        raise HTTPException(status_code=500, detail=str(exc))
    result["generated_at"] = datetime.now(timezone.utc).isoformat()
    return result


@router.get("", summary="Detailed opportunities list")
def list_opportunities(limit: int = Query(100, ge=1, le=1000)) -> dict[str, Any]:
    try:
        with get_conn() as conn:
            items = detailed_opportunities(conn.cursor(), limit)
    except Exception as exc:
        logger.exception("opportunities list failed")
        raise HTTPException(status_code=500, detail=str(exc))
    return {"opportunities": items, "count": len(items)}


@router.post("/link-deals", summary="Link opportunities to their deal via anchoring quote")
def post_link_deals() -> dict[str, Any]:
    try:
        n = link_opportunities_to_deals()
    except Exception as exc:  # noqa: BLE001
        logger.exception("opportunity->deal linkage failed")
        raise HTTPException(status_code=500, detail=str(exc))
    return {"status": "ok", "linked": n}


@router.post("/{opportunity_id}/stage", summary="Advance an opportunity's lifecycle stage")
def post_stage(opportunity_id: str, body: StageUpdate) -> dict[str, Any]:
    try:
        set_stage(opportunity_id, body.stage, body.realised_savings)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("stage update failed for %s", opportunity_id)
        raise HTTPException(status_code=500, detail=str(exc))
    return {"status": "ok", "opportunity_id": opportunity_id, "stage": body.stage}


@router.post("/sync", summary="Sync miner JSON findings into proc.bp_opportunity")
def post_sync(path: Optional[str] = None) -> dict[str, Any]:
    findings_path = path or os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
        "opportunity_findings.json")
    try:
        result = sync_findings_from_json(findings_path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"findings file not found: {findings_path}")
    except Exception as exc:
        logger.exception("opportunity sync failed")
        raise HTTPException(status_code=500, detail=str(exc))
    return {"status": "ok", **result}
