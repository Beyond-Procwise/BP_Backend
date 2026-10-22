"""The compliance screening register: record what a check found; read what was recorded.

Recording is a write by a named person, gated twice: the onboarding permission setting
``screening_record`` (Admin unless an administrator lowers it; unreadable = Admin) and the action gate
``supplier.screening.record`` for the audit record. Reading asks ``supplier.read``. Agents never
record here; their tools only read.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from api.auth import require_user
from api.endpoint_gate import require as gate
from src.services import permission_settings, rbac
from src.services import screening_register as SR
from src.services.contract_tools import store

log = logging.getLogger(__name__)
router = APIRouter(prefix="/screening", tags=["Supplier screening"])

_RANK = {"Viewer": 1, "Buyer": 2, "Approver": 3, "Admin": 4}
SETTING = "screening_record"


class ScreeningBody(BaseModel):
    supplier_id: str = Field(..., min_length=1)
    check_type: str
    result: str
    provider: str = Field(..., min_length=1)
    checked_on: str
    subject: Optional[str] = None
    rating: Optional[str] = None
    score: Optional[float] = None
    reference: Optional[str] = None
    valid_until: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


def _minimum_role() -> str:
    """Who may record a check. An unknown or unreadable setting means Admin, never a lower role."""
    try:
        role = permission_settings.minimum_role(SETTING, fallback="Admin")
    except Exception as exc:  # noqa: BLE001 - unreadable is not permission
        log.warning("screening: permission setting unreadable, requiring Admin: %s", exc)
        return "Admin"
    return role if role in _RANK else "Admin"


def _role_allows(principal) -> None:
    minimum = _minimum_role()
    role = rbac.effective_role(principal)
    if _RANK.get(role, 0) < _RANK[minimum]:
        raise HTTPException(status_code=403, detail=f"recording a screening result needs the {minimum} role or higher")


@router.post("")
def record_check(body: ScreeningBody, principal=Depends(require_user)) -> Dict[str, Any]:
    _role_allows(principal)
    gate("supplier.screening.record", principal, agent="ScreeningRouter",
         context={"supplier_id": body.supplier_id, "check_type": body.check_type, "result": body.result})
    who = getattr(principal, "subject", None) or ""
    try:
        return SR.record(body.model_dump(), who)
    except SR.InvalidEntry as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    except store.StoreUnavailable as exc:
        raise HTTPException(status_code=503, detail="the screening register could not be written just now") from exc


@router.get("/types")
def check_types() -> Dict[str, Any]:
    return {"check_types": {k: v["label"] for k, v in SR.CHECK_TYPES.items()}, "results": list(SR.RESULTS)}


@router.get("/{supplier_id}")
def supplier_history(supplier_id: str, principal=Depends(require_user)) -> Dict[str, Any]:
    gate("supplier.read", principal, agent="ScreeningRouter", context={"supplier_id": supplier_id})
    try:
        checks = SR.history(supplier_id)
    except store.StoreUnavailable as exc:
        raise HTTPException(status_code=503, detail="the screening register could not be read just now") from exc
    return {"supplier_id": supplier_id, "count": len(checks), "checks": checks}
