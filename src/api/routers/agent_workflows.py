"""Agent workflows — the canvas is the procurement process.

The DAG the user draws is saved, compiled and executed. Before it runs, the
workflow works out what it cannot know and asks the human for it. It never guesses.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from orchestration.elicitation import pending_requests
from orchestration.node_governance import governance_for
from orchestration.workflow_compiler import (
    GraphValidationError, compile_graph, validate_saved_graph,
)
from repositories import agent_workflow_repo as repo
from repositories import workflow_input_request_repo as reqrepo

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent-workflows", tags=["Agent Workflows"])

# Schema is ensured ONCE (app startup, see api/main.py's lifespan) rather than
# on every request — a DDL round-trip (CREATE TABLE/ALTER TABLE ADD COLUMN IF
# NOT EXISTS) on every one of these 8 handlers was needless work on the hot
# path; no other router in this codebase does that.


class WorkflowBody(BaseModel):
    name: str
    graph: Dict[str, Any]
    description: str = ""


class RunBody(BaseModel):
    payload: Dict[str, Any] = Field(default_factory=dict)
    user_id: str = "system"


class AnswerBody(BaseModel):
    request_id: int
    answer: Any
    answered_by: str = "human"


def _entry_of(graph: Dict[str, Any]) -> str:
    targets = {e["target"] for e in (graph.get("edges") or [])}
    return next(n["id"] for n in graph["nodes"] if n["id"] not in targets)


def _describe_nodes(graph: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        {"node_id": n["id"], "agent_slug": n["agent_slug"],
         "governance": governance_for(n["agent_slug"])}
        for n in graph.get("nodes", [])
    ]


@router.get("")
def list_workflows() -> Dict[str, Any]:
    return {"workflows": repo.list_active()}


@router.post("")
def create_workflow(body: WorkflowBody) -> Dict[str, Any]:
    try:
        validate_saved_graph(body.graph)
    except GraphValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    wid = repo.create(name=body.name, graph=body.graph, entry_node=_entry_of(body.graph),
                      description=body.description)
    return {"workflow_id": wid}


@router.get("/{workflow_id}")
def get_workflow(workflow_id: int) -> Dict[str, Any]:
    wf = repo.get(workflow_id)
    if not wf:
        raise HTTPException(status_code=404, detail="No such workflow")
    return wf


@router.put("/{workflow_id}")
def update_workflow(workflow_id: int, body: WorkflowBody) -> Dict[str, Any]:
    try:
        validate_saved_graph(body.graph)
    except GraphValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    repo.update(workflow_id, name=body.name, graph=body.graph,
                entry_node=_entry_of(body.graph), description=body.description)
    return {"ok": True}


@router.delete("/{workflow_id}")
def delete_workflow(workflow_id: int) -> Dict[str, Any]:
    repo.soft_delete(workflow_id)
    return {"ok": True}


@router.post("/{workflow_id}/run")
def run_workflow(workflow_id: int, body: RunBody, request: Request) -> Dict[str, Any]:
    wf = repo.get(workflow_id)
    if not wf:
        raise HTTPException(status_code=404, detail="No such workflow")

    run_id = f"awf-{workflow_id}-{uuid.uuid4().hex[:8]}"
    answers = reqrepo.answers_for(run_id)          # empty on a fresh run

    missing = pending_requests(wf["graph"], body.payload, answers)
    if missing:
        # The workflow does not guess. It stops and asks. The ORIGINAL payload
        # is persisted here — BEFORE returning awaiting_input — so it survives
        # a process restart and is not silently dropped by the time the human
        # answers and the run resumes. agent_workflow_id is recorded too, so
        # submit_input can resolve this run back to its saved workflow later
        # without parsing the run_id string.
        reqrepo.create_run(run_id, agent_workflow_id=workflow_id, payload=body.payload,
                            status="awaiting_input")
        reqrepo.raise_requests(run_id, missing, agent_workflow_id=workflow_id)
        return {
            "run_id": run_id, "status": "awaiting_input",
            "pending": reqrepo.open_requests(run_id),
            "nodes": _describe_nodes(wf["graph"]),
        }

    # Nothing outstanding — but this run must still be claimed atomically
    # before it executes, exactly like the resume path in submit_input, so a
    # replay of this same request can never execute the workflow twice.
    reqrepo.create_run(run_id, agent_workflow_id=workflow_id, payload=body.payload,
                        status="pending")
    return _claim_and_execute(request, run_id, wf, {**body.payload, **answers}, body.user_id)


@router.get("/runs/{run_id}")
def get_run(run_id: str) -> Dict[str, Any]:
    """Surface the run's real persisted status alongside its pending questions,
    so an operator/UI can tell 'executing' / 'completed' / 'failed' /
    'awaiting_input' apart — an unknown run and a completed run were
    previously indistinguishable (both returned an empty ``pending`` list).
    """
    run_row = reqrepo.get_run(run_id)
    if run_row is None:
        raise HTTPException(status_code=404, detail=f"No such run {run_id}")
    return {
        "run_id": run_id,
        "status": run_row["status"],
        "pending": reqrepo.open_requests(run_id),
    }


@router.post("/runs/{run_id}/input")
def submit_input(run_id: str, body: AnswerBody, request: Request) -> Dict[str, Any]:
    """The human answers. If nothing else is outstanding, the run proceeds."""
    reqrepo.answer(body.request_id, body.answer, body.answered_by)

    still_open = reqrepo.open_requests(run_id)
    if still_open:
        return {"run_id": run_id, "status": "awaiting_input", "pending": still_open}

    # Resolve the run back to its saved workflow from persisted state (the
    # agent_workflow_id recorded on the run's request rows), not by parsing
    # the run_id string — the run_id format is an implementation detail and
    # parsing it is brittle: it breaks the moment that format changes and can
    # silently mis-parse.
    workflow_id = reqrepo.workflow_id_for(run_id)
    if workflow_id is None:
        raise HTTPException(status_code=404, detail=f"No workflow found for run {run_id}")
    wf = repo.get(workflow_id)
    if not wf:
        raise HTTPException(status_code=404, detail="No such workflow")

    # The engine must see what the human supplied here AND what was supplied
    # up front on the original run call — the answers win on conflict, but
    # nothing the human already gave up front is ever dropped.
    original_payload = reqrepo.payload_for(run_id)
    answers = reqrepo.answers_for(run_id)
    return _claim_and_execute(request, run_id, wf, {**original_payload, **answers}, "human")


def _claim_and_execute(request: Request, run_id: str, wf: Dict[str, Any],
                        input_data: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    """Atomically claim this run for execution, then execute it exactly once.

    The claim is a single conditional UPDATE (see
    ``workflow_input_request_repo.claim_for_execution``) so a replayed final
    answer, or two answers landing concurrently on the last two outstanding
    questions, cannot both win — only one caller ever executes the workflow.
    A caller that does not win gets the run's current persisted state back
    instead of running it again (the agents this can trigger include
    email_dispatch, negotiation and supplier_interaction — running twice
    means sending real emails twice).
    """
    if not reqrepo.claim_for_execution(run_id):
        run_row = reqrepo.get_run(run_id)
        status = run_row["status"] if run_row else "completed"
        # Same shape as the winning path below (node_statuses, errors included)
        # so a client never has to special-case the claim-lost response.
        return {
            "run_id": run_id, "status": status, "pending": [],
            "nodes": _describe_nodes(wf["graph"]),
            "node_statuses": {}, "errors": [],
        }

    try:
        result = _execute(request, run_id, wf, input_data, user_id)
    except Exception:
        reqrepo.finish_run(run_id, "failed")
        raise
    reqrepo.finish_run(run_id, "failed" if result.get("errors") else "completed")
    return result


def _execute(request: Request, run_id: str, wf: Dict[str, Any],
             input_data: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    orchestrator = getattr(request.app.state, "orchestrator", None)
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not available")
    engine = getattr(orchestrator, "_workflow_engine", None)
    if engine is None:
        raise HTTPException(status_code=503, detail="Workflow engine not available")

    graph = compile_graph(wf["name"], wf["graph"])
    state = engine.execute(graph, input_data=input_data, user_id=user_id, workflow_id=run_id)

    return {
        "run_id": run_id,
        "status": getattr(state, "status", "completed"),
        "node_statuses": {k: getattr(v, "value", v) for k, v in state.node_statuses.items()},
        "errors": state.errors,
        "nodes": _describe_nodes(wf["graph"]),
        "pending": [],
    }
