"""Opportunities-page dashboard + lifecycle.

GET  /opportunities/dashboard        — KPIs, savings pipeline, monthly trends,
                                        identified-vs-completed, detailed list.
GET  /opportunities                  — detailed opportunities list (paged).
GET  /opportunities/buckets          — supplier+item buckets, each with its levers.
POST /opportunities/{id}/stage       — advance an opportunity's lifecycle stage.
POST /opportunities/{id}/review      — re-open or keep a rejection after new information.
POST /opportunities/sync             — upsert miner JSON findings into the table.
POST /opportunities/link-deals       — backfill deal_id from each opportunity's anchoring quote.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from api.auth import require_user
from api.endpoint_gate import require as gate

from src.services.opportunity_dashboard import build_opportunities_dashboard, detailed_opportunities
from src.services.opportunity_linkage import link_opportunities_to_deals
from src.services.lifecycle import IllegalTransition
from src.services.opportunity_buckets import (
    _mapping as as_mapping, group as group_buckets, list_buckets)
from src.services.opportunity_store import resolve_review, set_stage, sync_findings_from_json
from src.services.db import get_conn

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/opportunities", tags=["Opportunities"])


class StageUpdate(BaseModel):
    stage: str
    realised_savings: Optional[float] = None


class ReviewDecision(BaseModel):
    decision: str   # "reopen" | "keep_rejected"


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


@router.get("/buckets", summary="Opportunity buckets, each with its levers")
def get_buckets() -> dict[str, Any]:
    """One bucket per supplier and item that carries a unit price and a volume, with
    every lever found on it; opportunities with no such pairing are listed alone."""
    try:
        result = list_buckets()
    except Exception as exc:
        logger.exception("opportunity buckets failed")
        raise HTTPException(status_code=500, detail=str(exc))
    result["bucketCount"] = len(result["buckets"])
    result["individualCount"] = len(result["individual"])
    return result


@router.get("/by-deal/{deal_id}", summary="Opportunities for one deal (report-scoped)")
def get_opportunities_by_deal(deal_id: str) -> dict:
    from src.services.db import get_conn
    cols = ["opportunity_id", "detector_type", "category_id", "supplier_name",
            "item_description", "financial_impact_gbp", "stage",
            "ml_priority_score", "quote_id"]
    # Read for grouping only -- which rows are levers on one supplier and item, and
    # whether new information is waiting on a rejection. Not returned as they are.
    grouping = ["supplier_id", "item_id", "unit_price", "quantity", "calculation_details",
                "source_records", "detected_on", "review_requested_at", "review_detail"]
    try:
        with get_conn() as conn:
            cur = conn.cursor()
            cur.execute(
                f"select {', '.join(cols + grouping)} from proc.bp_opportunity "
                "where deal_id = %s order by financial_impact_gbp desc nulls last",
                (deal_id,))
            rows = cur.fetchall() or []
    except Exception as exc:  # noqa: BLE001
        logger.exception("by-deal opportunities failed for %s", deal_id)
        raise HTTPException(status_code=500, detail=str(exc))
    full = [dict(zip(cols + grouping, r)) for r in rows]
    buckets = group_buckets(full)["buckets"]
    in_bucket = {str(lever["opportunityId"]): b["bucketId"]
                 for b in buckets for lever in b["levers"]}
    opportunities = []
    for row in full:
        asked = row.get("review_requested_at") is not None
        opportunities.append({
            **{c: row.get(c) for c in cols},
            "bucket_id": in_bucket.get(str(row.get("opportunity_id"))),
            "review_requested": asked,
            "what_changed": as_mapping(row.get("review_detail")) if asked else None,
        })
    return {"deal_id": deal_id, "opportunities": opportunities, "buckets": buckets}


@router.post("/link-deals", summary="Link opportunities to their deal via anchoring quote")
def post_link_deals(principal=Depends(require_user)) -> dict[str, Any]:
    gate("opportunity.mine", principal, agent="opportunities_api")
    try:
        n = link_opportunities_to_deals()
    except Exception as exc:  # noqa: BLE001
        logger.exception("opportunity->deal linkage failed")
        raise HTTPException(status_code=500, detail=str(exc))
    return {"status": "ok", "linked": n}


@router.post("/{opportunity_id}/stage", summary="Advance an opportunity's lifecycle stage")
def post_stage(opportunity_id: str, body: StageUpdate,
               principal=Depends(require_user)) -> dict[str, Any]:
    try:
        set_stage(opportunity_id, body.stage, body.realised_savings)
    except IllegalTransition as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("stage update failed for %s", opportunity_id)
        raise HTTPException(status_code=500, detail=str(exc))
    return {"status": "ok", "opportunity_id": opportunity_id, "stage": body.stage}


@router.post("/{opportunity_id}/review",
             summary="Re-open or keep a rejected opportunity after new information")
def post_review(opportunity_id: str, body: ReviewDecision,
                principal=Depends(require_user)) -> dict[str, Any]:
    gate("opportunity.review", principal, agent="opportunities_api",
         context={"opportunity_id": opportunity_id, "decision": body.decision})
    try:
        stage = resolve_review(opportunity_id, body.decision,
                               decided_by=getattr(principal, "subject", None) or None)
    except IllegalTransition as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("review decision failed for %s", opportunity_id)
        raise HTTPException(status_code=500, detail=str(exc))
    return {"status": "ok", "opportunity_id": opportunity_id,
            "decision": body.decision, "stage": stage}


@router.post("/sync", summary="Sync miner JSON findings into proc.bp_opportunity")
def post_sync(path: Optional[str] = None, principal=Depends(require_user)) -> dict[str, Any]:
    gate("opportunity.mine", principal, agent="opportunities_api")
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
