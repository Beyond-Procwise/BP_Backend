"""The commodity index register: load a published series; read what was loaded.

Loading is a write by a named person, gated twice: the onboarding permission setting
``index_series_record`` (Admin unless an administrator lowers it; unreadable = Admin) and the action gate
``index_series.record`` for the audit record. Reading asks ``contract.read`` (index history serves the
contract pricing and savings work). Agents never load here; their tool only reads.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from api.auth import require_user
from api.endpoint_gate import require as gate
from src.services import commodity_index as CI
from src.services import permission_settings, rbac
from src.services.contract_tools import store

log = logging.getLogger(__name__)
router = APIRouter(prefix="/commodity-index", tags=["Commodity index"])

_RANK = {"Viewer": 1, "Buyer": 2, "Approver": 3, "Admin": 4}
SETTING = "index_series_record"


class Point(BaseModel):
    period_date: str
    value: float


class SeriesBody(BaseModel):
    publisher: str = Field(..., min_length=1)
    points: List[Point] = Field(..., min_length=1)
    description: Optional[str] = None
    currency: Optional[str] = None
    unit: Optional[str] = None
    source_reference: Optional[str] = None


def _minimum_role() -> str:
    """Who may load a series. An unknown or unreadable setting means Admin, never a lower role."""
    try:
        role = permission_settings.minimum_role(SETTING, fallback="Admin")
    except Exception as exc:  # noqa: BLE001 - unreadable is not permission
        log.warning("commodity index: permission setting unreadable, requiring Admin: %s", exc)
        return "Admin"
    return role if role in _RANK else "Admin"


def _role_allows(principal) -> None:
    minimum = _minimum_role()
    role = rbac.effective_role(principal)
    if _RANK.get(role, 0) < _RANK[minimum]:
        raise HTTPException(status_code=403, detail=f"loading an index series needs the {minimum} role or higher")


@router.post("/{series_code}")
def load_series(series_code: str, body: SeriesBody, principal=Depends(require_user)) -> Dict[str, Any]:
    _role_allows(principal)
    gate("index_series.record", principal, agent="CommodityIndexRouter",
         context={"series_code": series_code, "publisher": body.publisher, "points": len(body.points)})
    who = getattr(principal, "subject", None) or ""
    try:
        return CI.load_series(series_code, body.publisher, [p.model_dump() for p in body.points], who,
                              currency=body.currency, unit=body.unit, description=body.description,
                              source_reference=body.source_reference)
    except CI.InvalidSeries as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    except store.StoreUnavailable as exc:
        raise HTTPException(status_code=503, detail="the index register could not be written just now") from exc


@router.get("")
def list_series(principal=Depends(require_user)) -> Dict[str, Any]:
    gate("contract.read", principal, agent="CommodityIndexRouter")
    return CI.list_series()


@router.get("/{series_code}")
def read_series(series_code: str, start: Optional[str] = None, end: Optional[str] = None,
                principal=Depends(require_user)) -> Dict[str, Any]:
    gate("contract.read", principal, agent="CommodityIndexRouter", context={"series_code": series_code})
    return CI.lookup(series_code, start, end)
