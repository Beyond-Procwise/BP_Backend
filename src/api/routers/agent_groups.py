"""Agent groups — the sets of agents a user keeps putting on the canvas together.

A group is STORED and stamped back onto a canvas. It is never compiled and never run,
so nothing here imports the workflow compiler: see agent_group_repo's module docstring
for why a selection cannot go through validate_saved_graph.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from orchestration.workflow_compiler import _known_agents
from repositories import agent_group_repo as repo

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent-groups", tags=["Agent Groups"])


class Member(BaseModel):
    agent_slug: str
    display_name: str = ""
    dx: float = 0
    dy: float = 0


class Link(BaseModel):
    from_index: int
    to_index: int


class GroupBody(BaseModel):
    name: str
    members: List[Member] = Field(default_factory=list)
    links: List[Link] = Field(default_factory=list)


class GroupPatch(BaseModel):
    name: Optional[str] = None
    members: Optional[List[Member]] = None
    links: Optional[List[Link]] = None


def _validate(members: List[Member], links: List[Link]) -> None:
    """Fail here, at save time, with a reason. Note what is NOT checked: entry nodes,
    cycles and connectivity. A group is not a flow — a selection of two agents from
    different branches is a perfectly good group and would be a rejected workflow."""
    if not members:
        raise HTTPException(status_code=400, detail="A group needs at least one agent.")
    known = _known_agents()
    for m in members:
        if m.agent_slug not in known:
            raise HTTPException(status_code=400, detail=f"unknown agent: {m.agent_slug!r}")
    n = len(members)
    for e in links:
        if not (0 <= e.from_index < n and 0 <= e.to_index < n):
            raise HTTPException(
                status_code=400,
                detail=f"Link {e.from_index} -> {e.to_index} points outside the group.",
            )


@router.get("")
def list_groups() -> Dict[str, Any]:
    return {"groups": repo.list_active()}


@router.post("")
def create_group(body: GroupBody) -> Dict[str, Any]:
    _validate(body.members, body.links)
    gid = repo.create(
        name=body.name,
        members=[m.model_dump() for m in body.members],
        links=[e.model_dump() for e in body.links],
    )
    return {"group_id": gid}


@router.put("/{group_id}")
def update_group(group_id: int, body: GroupPatch) -> Dict[str, Any]:
    if repo.get(group_id) is None:
        raise HTTPException(status_code=404, detail="No such group.")
    if body.members is not None or body.links is not None:
        _validate(body.members or [], body.links or [])
    repo.update(
        group_id,
        name=body.name,
        members=None if body.members is None else [m.model_dump() for m in body.members],
        links=None if body.links is None else [e.model_dump() for e in body.links],
    )
    return {"ok": True}


@router.delete("/{group_id}")
def delete_group(group_id: int) -> Dict[str, Any]:
    repo.soft_delete(group_id)
    return {"ok": True}
