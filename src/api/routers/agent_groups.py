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

# The public catalogue loader, not orchestration.workflow_compiler._known_agents.
# That underscore-prefixed helper is workflow_compiler's own private convenience
# wrapper around this same function; importing it from here would make this
# router's slug validation depend on an implementation detail of a module it
# otherwise has nothing to do with (a group is never compiled — see this
# module's docstring). A rename of _known_agents would fail loudly at import,
# but a return-type change (e.g. dict-keyed-by-slug -> list) would silently
# break _validate_slugs here instead of there. Going straight to the public
# agents.definitions.load_agent_definitions() removes that coupling entirely.
from agents.definitions import load_agent_definitions
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


def _validate_slugs(members: List[Member]) -> None:
    """Every member's agent_slug must exist in the live agent catalogue."""
    known = {defn["slug"] for defn in load_agent_definitions()}
    for m in members:
        if m.agent_slug not in known:
            raise HTTPException(status_code=400, detail=f"unknown agent: {m.agent_slug!r}")


def _validate_links(links: List[Link], member_count: int) -> None:
    """Every link must reference two members that are actually IN the group.

    ``member_count`` is the count links are being checked against -- on a
    links-only PUT that is the group's CURRENTLY STORED member count, not an
    empty list, since no new members were supplied to replace them."""
    for e in links:
        if not (0 <= e.from_index < member_count and 0 <= e.to_index < member_count):
            raise HTTPException(
                status_code=400,
                detail=f"Link {e.from_index} -> {e.to_index} points outside the group.",
            )


def _validate(members: List[Member], links: List[Link]) -> None:
    """Fail here, at save time, with a reason. Note what is NOT checked: entry nodes,
    cycles and connectivity. A group is not a flow — a selection of two agents from
    different branches is a perfectly good group and would be a rejected workflow."""
    if not members:
        raise HTTPException(status_code=400, detail="A group needs at least one agent.")
    _validate_slugs(members)
    _validate_links(links, len(members))


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
    """A partial update validates only what was actually supplied.

    A PUT touching only ``links`` must still work -- it does not re-supply
    ``members``, so there is nothing wrong to reject there, and its links are
    bounds-checked against the group's CURRENTLY STORED members (fetched
    below), not against an empty list, which every links-only PUT would
    otherwise fail against unconditionally.
    """
    current = repo.get(group_id)
    if current is None:
        raise HTTPException(status_code=404, detail="No such group.")

    if body.members is not None:
        if not body.members:
            raise HTTPException(status_code=400, detail="A group needs at least one agent.")
        _validate_slugs(body.members)
        member_count = len(body.members)
    else:
        member_count = len(current["members"])

    if body.links is not None:
        _validate_links(body.links, member_count)

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
