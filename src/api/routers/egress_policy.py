"""The outside-research / egress policy -- what may leave the system, as the administrator's rules.

GET  /admin/egress-policy        -- the rules in force (empty until seeded; the seed is shown alongside).
PUT  /admin/egress-policy        -- replace the rules (version-checked, audited, history kept).
POST /admin/egress-policy/seed   -- write the starting rules if there are none.

Shadow mode: the egress gate records what these rules WOULD decide on every outbound call and blocks
nothing. Enforcing is a later, separate switch.
"""
from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.auth import require_user
from api.endpoint_gate import require as gate
from src.services import egress_policy as ep

router = APIRouter(prefix="/admin/egress-policy", tags=["permissions"])


class RulesBody(BaseModel):
    rules: Dict[str, Any]
    version: int            # the version the administrator was looking at


def _who(principal: Any) -> str:
    return str(getattr(principal, "subject", None) or "unknown")


@router.get("", summary="The egress rules in force")
def read(principal=Depends(require_user)) -> dict[str, Any]:
    gate("policy.read", principal, agent="egress_policy_api")
    return ep.current()


@router.put("", summary="Replace the egress rules")
def replace(body: RulesBody, principal=Depends(require_user)) -> dict[str, Any]:
    gate("policy.write", principal, agent="egress_policy_api", context={"op": "replace"})
    try:
        return ep.replace_rules(body.rules, by=_who(principal), expected_version=body.version)
    except ep.InvalidRules as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except ep.RulesConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@router.post("/seed", summary="Write the starting egress rules if there are none")
def seed(principal=Depends(require_user)) -> dict[str, Any]:
    gate("policy.write", principal, agent="egress_policy_api", context={"op": "seed"})
    return ep.seed(by=_who(principal))
