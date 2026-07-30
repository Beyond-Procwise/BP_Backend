"""GET /spendiq/value-summary — the Value Found headline (W1).
One read-model over discrepancies + opportunities + benchmark deltas.
Spec: docs/superpowers/specs/2026-07-30-value-found-design.md"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from src.services import value_summary_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/spendiq", tags=["Value Found"])


@router.get("/value-summary", summary="Evidence-backed value found / recovered / potential")
def get_value_summary() -> dict:
    try:
        result = value_summary_service.build_value_summary()
    except Exception as exc:                     # the service isolates per-source failures;
        logger.exception("value-summary failed")  # reaching here means something structural
        raise HTTPException(status_code=500, detail=str(exc))
    result["generated_at"] = datetime.now(timezone.utc).isoformat()
    return result
