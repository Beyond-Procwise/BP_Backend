"""Onboarding permission settings -- the administrator's choice of who may do each sensitive action.

GET  /admin/permissions            -- every setting with the role now in force.
PUT  /admin/permissions/{key}      -- make a role the minimum for one setting.
POST /admin/permissions/seed       -- write the safe starting value for any setting that has none.

Saving writes the policy row the gate already reads (services/permission_settings); it takes effect
when the policy cache reloads.
"""
from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.auth import require_user
from api.endpoint_gate import require as gate
from src.services import permission_settings as ps

router = APIRouter(prefix="/admin/permissions", tags=["permissions"])


class RoleBody(BaseModel):
    role: Optional[str] = None      # for a role setting
    value: Optional[str] = None     # for a choice setting (e.g. workflow_question_answer)


def _who(principal: Any) -> str:
    return str(getattr(principal, "subject", None) or "unknown")


@router.get("", summary="Permission settings and the role now in force for each")
def list_settings(principal=Depends(require_user)) -> dict[str, Any]:
    gate("policy.read", principal, agent="permission_settings_api")
    return {"settings": ps.current()}


@router.put("/{key}", summary="Set the minimum role for one permission setting")
def change_setting(key: str, body: RoleBody, principal=Depends(require_user)) -> dict[str, Any]:
    gate("policy.write", principal, agent="permission_settings_api", context={"setting": key})
    try:
        if key in {c.key for c in ps.CHOICES}:
            if body.value is None:
                raise HTTPException(status_code=422, detail=f"{key} needs a value")
            return ps.set_choice(key, body.value, _who(principal))
        if body.role is None:
            raise HTTPException(status_code=422, detail=f"{key} needs a role")
        return ps.set_role(key, body.role, _who(principal))
    except ps.InvalidChoice as exc:
        raise HTTPException(status_code=422, detail=f"{exc}")
    except ps.UnknownSetting:
        raise HTTPException(status_code=404, detail=f"{key} is not a permission setting")
    except ps.InvalidRole as exc:
        raise HTTPException(status_code=422, detail=f"{exc}")
    except ps.SettingConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@router.post("/seed", summary="Write the starting value for every setting that has none")
def seed(principal=Depends(require_user)) -> dict[str, Any]:
    gate("policy.write", principal, agent="permission_settings_api", context={"op": "seed"})
    return ps.seed_defaults(_who(principal))
