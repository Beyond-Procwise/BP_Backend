"""Discrepancy triage for one deal.

GET  /triage/deals/{deal_id}      the deal's verdict now, with its S1/S2 findings
POST /triage/deals/{deal_id}/run  re-check the deal and update the Action Centre

Spec: docs/superpowers/specs/2026-09-24-discrepancy-triage-p2p-design.md §9.1
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from api.auth import require_user
from src.services.governed_limits import LimitUnavailable
from src.services.triage import engine

router = APIRouter(prefix="/triage", tags=["Triage"])


def _unavailable(exc: Exception) -> HTTPException:
    return HTTPException(status_code=503, detail=f"triage tolerances unavailable: {exc}")


@router.get("/deals/{deal_id}", summary="A deal's triage verdict and the findings that need action")
def get_deal_triage(deal_id: str) -> dict[str, Any]:
    try:
        view = engine.triage_deal_view(deal_id)
    except LimitUnavailable as exc:
        raise _unavailable(exc) from exc
    if view is None:
        raise HTTPException(status_code=404, detail=f"no documents for deal {deal_id}")
    return view


@router.post("/deals/{deal_id}/run", summary="Re-triage one deal and update the Action Centre")
def run_deal_triage(deal_id: str, principal=Depends(require_user)) -> dict[str, Any]:
    try:
        report = engine.run_triage([deal_id], "single")
    except LimitUnavailable as exc:
        raise _unavailable(exc) from exc
    if report.failed:
        raise HTTPException(status_code=500, detail=report.failed.get(deal_id, "triage failed"))
    if report.deals_done == 0:
        raise HTTPException(status_code=404, detail=f"no documents for deal {deal_id}")
    return {"run_id": report.run_id, "writes": dict(report.write_counts),
            **(engine.triage_deal_view(deal_id) or {})}
