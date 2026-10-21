"""Onboarding deal-visibility setting -- which deals each role may see and change.

GET  /admin/visibility           -- every role with its read and write scope.
PUT  /admin/visibility/{role}    -- change one role's read and/or write scope.
POST /admin/visibility/seed      -- write the narrow starting value if no setting exists yet.

The default is "only deals the person created or approved". This stores the choice
(services/deal_visibility); the screens that list deals apply it.
"""
from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.auth import require_user
from api.endpoint_gate import require as gate
from src.services import deal_visibility as dv
from src.services import permission_settings as ps

router = APIRouter(prefix="/admin/visibility", tags=["permissions"])


class ScopeBody(BaseModel):
    read: Optional[str] = None
    write: Optional[str] = None


def _who(principal: Any) -> str:
    return str(getattr(principal, "subject", None) or "unknown")


@router.get("", summary="Deal visibility by role")
def list_visibility(principal=Depends(require_user)) -> dict[str, Any]:
    gate("policy.read", principal, agent="deal_visibility_api")
    return {"roles": dv.current(), "readScopes": list(dv.READ_SCOPES), "writeScopes": list(dv.WRITE_SCOPES)}


@router.put("/{role}", summary="Set what one role may see and change")
def change_visibility(role: str, body: ScopeBody, principal=Depends(require_user)) -> dict[str, Any]:
    gate("policy.write", principal, agent="deal_visibility_api", context={"role": role})
    try:
        return dv.set_scope(role, _who(principal), read=body.read, write=body.write)
    except ps.InvalidRole:
        raise HTTPException(status_code=404, detail=f"{role} is not a role")
    except (dv.InvalidScope, dv.AdminCannotBeNarrowed) as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except ps.SettingConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@router.post("/seed", summary="Write the starting visibility if none exists")
def seed(principal=Depends(require_user)) -> dict[str, Any]:
    gate("policy.write", principal, agent="deal_visibility_api", context={"op": "seed"})
    return dv.seed_defaults(_who(principal))
