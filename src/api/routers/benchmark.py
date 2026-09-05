"""Benchmark pricing endpoints (deterministic core over HTTP).

GET  /benchmark/by-deal/{deal_id} — run the engine over a deal's quote lines
     against pooled PO/invoice price history from bp_sqldb (_trgt), read-only.
POST /benchmark/preview           — run the pure engine on a caller-supplied
     payload (quote + points + lookup tables); no DB access.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from src.services.formulas import ensure_registered, evaluate
from services.benchmark.models import BenchmarkPoint, BenchmarkSettings, QuoteLine
from services.benchmark_live import benchmark_deal
from src.services.db import get_conn

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/benchmark", tags=["Benchmark"])


@router.get("/by-deal/{deal_id}", summary="Benchmark a deal's quote lines against live price history")
def benchmark_by_deal(
    deal_id: str,
    min_points: int = Query(3, ge=1, le=50, description="Evidence gate (default 3 = fail closed per spec)"),
    method: str = Query("weighted", description="weighted | simple | median"),
) -> dict[str, Any]:
    try:
        settings = BenchmarkSettings(min_data_points=min_points, method=method)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    try:
        with get_conn() as conn:
            return benchmark_deal(conn.cursor(), deal_id, settings)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("benchmark by-deal failed for %s", deal_id)
        raise HTTPException(status_code=500, detail=str(exc))


class BenchmarkPreviewRequest(BaseModel):
    quote: QuoteLine
    points: list[BenchmarkPoint]
    location_index_table: dict[str, float] = Field(default_factory=dict)
    index_table: dict[str, float] = Field(default_factory=dict)
    settings: Optional[BenchmarkSettings] = None


@router.post("/preview", summary="Run the deterministic benchmark engine on a supplied payload")
def benchmark_preview(body: BenchmarkPreviewRequest) -> dict[str, Any]:
    ensure_registered()
    outcome = evaluate("benchmark.adjusted_price", {
        "quote": body.quote,
        "points": body.points,
        "location_index_table": body.location_index_table,
        "index_table": body.index_table,
        "settings": body.settings or BenchmarkSettings(),
    })
    if outcome.unassessed:
        # The contract refused the payload. Say so with the reason rather than
        # returning an empty result that reads like "no benchmark exists".
        raise HTTPException(status_code=422, detail=outcome.why())
    return outcome.value.model_dump()
