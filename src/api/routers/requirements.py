"""Procurement requirements gathering API.

POST /requirements/message      — one elicitation turn (next question or completed requirement)
GET  /requirements/{id}         — fetch a persisted requirement
GET  /requirements              — list requirements
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from src.services import requirement_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/requirements", tags=["Requirements"])


class RequirementMessage(BaseModel):
    session_id: Optional[str] = None
    message: Optional[str] = None
    brief: Optional[str] = None
    created_by: Optional[str] = None
    category: Optional[str] = None


def _run_requirements_turn(app_state: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Run one RequirementsAgent turn via the app's orchestrator/agent_nick.

    Isolated for testability — patched in unit tests so the route can be
    exercised without the full agent stack. Mirrors run.py's pattern of
    reaching the orchestrator off ``request.app.state``.
    """
    import uuid
    from agents.agent_factory import AgentFactory
    from agents.base_agent import AgentContext

    orchestrator = getattr(app_state, "orchestrator", None)
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator unavailable")
    agent = AgentFactory(orchestrator.agent_nick).create("requirements")
    ctx = AgentContext(
        workflow_id=payload.get("session_id") or uuid.uuid4().hex,
        agent_id="requirements",
        user_id=payload.get("created_by") or "api",
        input_data=dict(payload),
    )
    output = agent.run(ctx)
    return dict(output.data or {})


def _run_requirements_workflow(app_state: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Run the declarative requirements→sourcing workflow via the orchestrator.

    gather_requirement → (complete) → rank_suppliers → (ranking) → draft_emails.
    Isolated for testability — patched in unit tests."""
    orchestrator = getattr(app_state, "orchestrator", None)
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator unavailable")
    return orchestrator.execute_workflow("requirements_to_ranking", dict(payload))


def _events_for(result: Dict[str, Any]) -> List[Dict[str, str]]:
    """Build SSE-style progress events describing the turn for a live chat UI."""
    events: List[Dict[str, str]] = [{"event": "thinking", "message": "Reviewing requirement"}]
    if result.get("complete"):
        events.append({"event": "complete",
                       "message": result.get("summary", "Requirement captured.")})
    else:
        events.append({"event": "question",
                       "message": result.get("next_question", "")})
    return events


@router.post("/message", summary="Run one requirements elicitation turn")
def post_message(body: RequirementMessage, request: Request) -> Dict[str, Any]:
    try:
        result = _run_requirements_turn(request.app.state, body.model_dump(exclude_none=True))
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("requirements turn failed")
        raise HTTPException(status_code=500, detail=str(exc))
    return {
        "result": result,
        "events": _events_for(result),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


@router.post("/run-workflow", summary="Run the requirements→sourcing workflow")
def post_run_workflow(body: RequirementMessage, request: Request) -> Dict[str, Any]:
    try:
        result = _run_requirements_workflow(request.app.state, body.model_dump(exclude_none=True))
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("requirements workflow failed")
        raise HTTPException(status_code=500, detail=str(exc))
    return {
        "workflow": "requirements_to_ranking",
        "result": result,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/{requirement_id}", summary="Fetch a procurement requirement")
def get_requirement(requirement_id: str) -> Dict[str, Any]:
    try:
        row = requirement_service.get_requirement(requirement_id)
    except Exception as exc:
        logger.exception("requirement fetch failed for %s", requirement_id)
        raise HTTPException(status_code=500, detail=str(exc))
    if row is None:
        raise HTTPException(status_code=404, detail=f"No requirement {requirement_id}")
    return row


@router.get("", summary="List procurement requirements")
def list_requirements(limit: int = 50, offset: int = 0) -> Dict[str, Any]:
    try:
        items = requirement_service.list_requirements(limit=limit, offset=offset)
    except Exception as exc:
        logger.exception("requirement list failed")
        raise HTTPException(status_code=500, detail=str(exc))
    return {"requirements": items, "count": len(items)}
