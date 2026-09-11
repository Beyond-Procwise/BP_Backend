"""Procurement requirements gathering API.

POST /requirements/message            — one scoping turn: a proposed scope when the buyer
                                        asks what the requirements should be, otherwise the
                                        next elicitation question (or the completed requirement)
POST /requirements/run-workflow       — start the requirements->sourcing workflow (async, returns job_id)
GET  /requirements/workflow/{job_id}  — poll an async workflow job's status/result
GET  /requirements/{id}               — fetch a persisted requirement
GET  /requirements                    — list requirements
"""
from __future__ import annotations

import logging
import threading
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from api.auth import require_user
from src.services import requirement_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/requirements", tags=["Requirements"])

# In-process registry of async workflow jobs (job_id -> {status, result, error,
# started_at}). The sourcing chain (supplier_ranking + email drafting) runs for
# minutes, so /run-workflow returns immediately and the client polls
# /workflow/{job_id}. Bounded so a long-lived server doesn't grow unboundedly.
_WORKFLOW_JOBS: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
_WORKFLOW_JOBS_MAX = 256


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
    from agents.base_agent import AgentContext

    orchestrator = getattr(app_state, "orchestrator", None)
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator unavailable")

    # Take the agent from the live registry, not from a second factory. This route
    # used to build its own RequirementsAgent via AgentFactory, whose AGENT_CONTRACTS
    # duplicated agent_definitions.json — so the agent serving this endpoint was a
    # different object, built from a different catalogue, than the one the
    # orchestrator dispatched. One registry, one instance.
    agent = orchestrator.agent_nick.agents.get("requirements")
    if agent is None:
        raise HTTPException(status_code=503, detail="Requirements agent not registered")
    ctx = AgentContext(
        workflow_id=payload.get("session_id") or uuid.uuid4().hex,
        agent_id="requirements",
        # Whoever the route resolved from the token, or nobody -- not "api",
        # which the agent then wrote to bp_requirement.created_by as a person.
        user_id=payload.get("created_by") or "",
        input_data=dict(payload),
    )
    output = agent.run(ctx)
    return dict(output.data or {})


def _set_job(job_id: str, **fields: Any) -> None:
    """Update a job record, keeping the registry bounded."""
    job = _WORKFLOW_JOBS.get(job_id, {})
    job.update(fields)
    _WORKFLOW_JOBS[job_id] = job
    _WORKFLOW_JOBS.move_to_end(job_id)
    while len(_WORKFLOW_JOBS) > _WORKFLOW_JOBS_MAX:
        _WORKFLOW_JOBS.popitem(last=False)


def _launch_workflow(app_state: Any, payload: Dict[str, Any], job_id: str) -> None:
    """Start the requirements→sourcing workflow in the background and track it.

    gather_requirement → (complete) → rank_suppliers → (ranking) → draft_emails.
    Returns immediately; the job result is recorded in ``_WORKFLOW_JOBS`` when the
    (minutes-long) chain finishes. Isolated for testability."""
    _set_job(job_id, status="running", result=None, error=None,
             started_at=datetime.now(timezone.utc).isoformat())
    orchestrator = getattr(app_state, "orchestrator", None)
    if orchestrator is None:
        _set_job(job_id, status="failed", error="Orchestrator unavailable")
        return

    def _run() -> None:
        try:
            result = orchestrator.execute_workflow("requirements_to_ranking", dict(payload))
            _set_job(job_id, status="completed", result=result)
        except Exception as exc:  # pragma: no cover - background failure path
            logger.exception("async requirements workflow failed")
            _set_job(job_id, status="failed", error=str(exc))

    executor = getattr(orchestrator, "executor", None)
    if executor is not None:
        executor.submit(_run)
    else:  # pragma: no cover - fallback when no shared executor
        threading.Thread(target=_run, daemon=True).start()


def _events_for(result: Dict[str, Any]) -> List[Dict[str, str]]:
    """Build SSE-style progress events describing the turn for a live chat UI."""
    mode = result.get("mode") or "elicitation"
    opening = ("Drafting a scope" if mode in ("proposed_scope", "scope_accepted")
               else "Reviewing requirement")
    events: List[Dict[str, str]] = [{"event": "thinking", "message": opening}]
    # A proposed scope is the substance of the turn, so it gets its own event
    # ahead of the question — a client that renders only `question` would show
    # the confirm prompt and hide the scope it refers to.
    scope = result.get("scope") or {}
    if scope.get("areas"):
        events.append({
            "event": "scope",
            "message": f"Proposed {len(scope['areas'])} requirement areas "
                       f"({scope.get('family_label', 'general')})",
        })
    if result.get("complete"):
        events.append({"event": "complete",
                       "message": result.get("summary", "Requirement captured.")})
    # Not `elif`: a turn can both complete the required fields AND propose a scope
    # that still needs confirming. Dropping the question there left the buyer
    # looking at a scope with nothing asked of them.
    if result.get("next_question"):
        events.append({"event": "question", "message": result["next_question"]})
    return events


def _payload(body: RequirementMessage, principal: Any) -> Dict[str, Any]:
    """The turn's input, with the requirement's author taken from the token.

    `body.created_by` stays on the model because clients send it, but it is not
    read: it became bp_requirement.created_by, so a caller could open a
    requirement in someone else's name. With no principal the key is absent and
    the requirement names nobody.
    """
    payload = body.model_dump(exclude_none=True)
    payload.pop("created_by", None)
    subject = getattr(principal, "subject", None) or None
    if subject:
        payload["created_by"] = subject
    return payload


@router.post("/message", summary="Run one requirements elicitation turn")
def post_message(body: RequirementMessage, request: Request,
                 principal=Depends(require_user)) -> Dict[str, Any]:
    try:
        result = _run_requirements_turn(request.app.state, _payload(body, principal))
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


@router.post("/run-workflow", summary="Start the requirements→sourcing workflow (async)")
def post_run_workflow(body: RequirementMessage, request: Request,
                      principal=Depends(require_user)) -> Dict[str, Any]:
    job_id = uuid.uuid4().hex
    _launch_workflow(request.app.state, _payload(body, principal), job_id)
    return {
        "workflow": "requirements_to_ranking",
        "job_id": job_id,
        "status": "running",
        "poll": f"/requirements/workflow/{job_id}",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/workflow/{job_id}", summary="Poll an async workflow job")
def get_workflow_job(job_id: str) -> Dict[str, Any]:
    job = _WORKFLOW_JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"No workflow job {job_id}")
    return {"job_id": job_id, **job}


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
