"""Agent workflows — the canvas is the procurement process.

The DAG the user draws is saved, compiled and executed. Before it runs, the
workflow works out what it cannot know and asks the human for it. It never guesses.
"""

from __future__ import annotations

import logging
import re
import threading
import uuid
from collections import OrderedDict
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from orchestration.elicitation import pending_requests
from orchestration.node_governance import governance_for
from orchestration.workflow_compiler import (
    GraphValidationError, compile_graph, validate_saved_graph,
)
from repositories import agent_workflow_repo as repo
from repositories import workflow_input_request_repo as reqrepo

from api.auth import require_user
from api.endpoint_gate import require as gate
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
    # Accepted because clients send it; never read. The run is started by the
    # token. It defaulted to "system" and was passed straight in as the user.
    user_id: Optional[str] = None


class AnswerBody(BaseModel):
    request_id: int
    answer: Any
    answered_by: str = "human"


def _is_blank_answer(value: Any) -> bool:
    """None, an empty/whitespace-only string, or an empty list/dict — the same
    "presence isn't enough, the value must be real" test elicitation.py applies
    to a group before treating it as satisfied (see CRITICAL 2). Without this,
    ``POST /runs/{id}/input {"answer": ""}`` marked the question answered and,
    once it was the last open one, executed the workflow on a blank."""
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    if isinstance(value, (list, dict)) and len(value) == 0:
        return True
    return False


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
def create_workflow(
    body: WorkflowBody, principal=Depends(require_user)
) -> Dict[str, Any]:
    gate("workflow.save", principal, agent="AgentWorkflowsRouter")
    try:
        validate_saved_graph(body.graph)
    except GraphValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    wid = repo.create(name=body.name, graph=body.graph, entry_node=_entry_of(body.graph),
                      description=body.description,
                      # Who saved it: the token, or nobody. It was never passed,
                      # so the repo's "system" default named every author.
                      created_by=getattr(principal, "subject", None) or None)
    return {"workflow_id": wid}


@router.get("/{workflow_id}")
def get_workflow(workflow_id: int) -> Dict[str, Any]:
    wf = repo.get(workflow_id)
    if not wf:
        raise HTTPException(status_code=404, detail="No such workflow")
    return wf


@router.put("/{workflow_id}")
def update_workflow(
    workflow_id: int, body: WorkflowBody, principal=Depends(require_user)
) -> Dict[str, Any]:
    gate("workflow.save", principal, agent="AgentWorkflowsRouter",
         context={"workflow_id": workflow_id})
    try:
        validate_saved_graph(body.graph)
    except GraphValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    repo.update(workflow_id, name=body.name, graph=body.graph,
                entry_node=_entry_of(body.graph), description=body.description)
    return {"ok": True}


@router.delete("/{workflow_id}")
def delete_workflow(
    workflow_id: int, principal=Depends(require_user)
) -> Dict[str, Any]:
    gate("workflow.save", principal, agent="AgentWorkflowsRouter",
         context={"workflow_id": workflow_id, "deleting": True})
    repo.soft_delete(workflow_id)
    return {"ok": True}


@router.post("/{workflow_id}/run")
def run_workflow(
    workflow_id: int, body: RunBody, request: Request,
    principal=Depends(require_user),
) -> Dict[str, Any]:
    gate("workflow.run", principal, agent="AgentWorkflowsRouter",
         context={"workflow_id": workflow_id})
    wf = repo.get(workflow_id)
    if not wf:
        raise HTTPException(status_code=404, detail="No such workflow")

    run_id = f"awf-{workflow_id}-{uuid.uuid4().hex[:8]}"
    answers = reqrepo.answers_for(run_id)          # empty on a fresh run
    # Recorded on the run row, because a run that stops to ask a question is
    # resumed by whoever ANSWERS -- who is not necessarily who started it.
    started_by = getattr(principal, "subject", None) or None

    missing = pending_requests(wf["graph"], body.payload, answers)
    if missing:
        # The workflow does not guess. It stops and asks. The ORIGINAL payload
        # is persisted here — BEFORE returning awaiting_input — so it survives
        # a process restart and is not silently dropped by the time the human
        # answers and the run resumes. agent_workflow_id is recorded too, so
        # submit_input can resolve this run back to its saved workflow later
        # without parsing the run_id string.
        reqrepo.create_run(run_id, agent_workflow_id=workflow_id, payload=body.payload,
                            status="awaiting_input", initiated_by=started_by)
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
                        status="pending", initiated_by=started_by)
    return _claim_and_execute(request, run_id, wf, {**body.payload, **answers}, started_by)


@router.get("/runs/{run_id}")
def get_run(run_id: str) -> Dict[str, Any]:
    """The run's persisted status, its pending questions, AND its live
    per-node progress — the poll target that lets the canvas light nodes up
    while a run executes in the background instead of freezing on a held
    request (programme item A3).
    """
    run_row = reqrepo.get_run(run_id)
    if run_row is None:
        raise HTTPException(status_code=404, detail=f"No such run {run_id}")

    status = run_row["status"]
    state = _LIVE_RUNS.get(run_id)

    # A row saying "executing" with no live state in this process means the
    # server restarted mid-run: the thread is gone and no amount of polling
    # will finish it. Heal it to failed on first sight rather than reporting
    # "executing" forever.
    if status == "executing" and state is None:
        reqrepo.finish_run(run_id, "failed")
        return {
            "run_id": run_id, "status": "failed",
            "pending": [],
            "node_statuses": {}, "node_results": {},
            "errors": ["The run was interrupted by a server restart — run it again."],
            "nodes": [],
        }

    # The run ROW carries the workflow id (create_run stores it); the
    # request-rows resolver is only a fallback for legacy rows. A run that
    # asked no questions has NO request rows, so resolving through them
    # alone returned None and progress rendered empty (live bug, 2026-08-22).
    graph = None
    workflow_id = run_row.get("agent_workflow_id") or reqrepo.workflow_id_for(run_id)
    if workflow_id is not None:
        wf = repo.get(workflow_id)
        if wf:
            graph = wf["graph"]

    out: Dict[str, Any] = {
        "run_id": run_id,
        "status": status,
        "pending": reqrepo.open_requests(run_id),
        "node_statuses": {}, "node_results": {}, "errors": [],
        "nodes": _describe_nodes(graph) if graph else [],
    }
    if state is not None and graph is not None:
        out["node_statuses"] = {
            k: getattr(v, "value", v) for k, v in state.node_statuses.items()
        }
        out["node_results"] = _summarise_node_results(state.node_results, graph)
        out["errors"] = _readable_errors(state.errors, graph)
    return out


@router.post("/runs/{run_id}/input")
def submit_input(run_id: str, body: AnswerBody, request: Request,
                 principal=Depends(require_user)) -> Dict[str, Any]:
    """The human answers. If nothing else is outstanding, the run proceeds.

    request_id is scoped to run_id: a request_id that belongs to a
    different run (or does not exist at all) is rejected outright rather
    than silently no-op'd or, worse, applied to the wrong run's row. See
    workflow_input_request_repo.answer for why the audit trail depends on
    this.
    """
    if _is_blank_answer(body.answer):
        raise HTTPException(status_code=400, detail="answer must not be blank")

    owner_run_id = reqrepo.request_run_id(body.request_id)
    if owner_run_id is None:
        raise HTTPException(status_code=404, detail=f"No such request {body.request_id}")
    if owner_run_id != run_id:
        raise HTTPException(
            status_code=404,
            detail=f"Request {body.request_id} does not belong to run {run_id}",
        )

    # A no-op if this request was already answered (e.g. a replayed final
    # answer) -- idempotent, not an error, so a client retry still gets a
    # 200 with the run's current state instead of failing.
    #
    # Who answered is the token. This row is the HITL audit trail -- what a
    # person was asked and what they said -- and `body.answered_by` (default
    # "human") is a name the caller types. It stays on the model because
    # clients send it; it is not read as identity, and with no principal the
    # answer is recorded against nobody.
    reqrepo.answer(run_id, body.request_id, body.answer,
                   getattr(principal, "subject", None) or None)

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
    # Started by whoever started it -- not "human", and not whoever answered
    # last. The answerer is on the answer row; a self-approval check needs the
    # two kept apart.
    return _claim_and_execute(request, run_id, wf, {**original_payload, **answers},
                              reqrepo.initiator_for(run_id))


# A run gets this long to finish in-request before the caller is answered
# "executing" and left to poll GET /runs/{run_id}. Short runs still feel
# instant; a GPU-bound run no longer holds the connection for minutes.
_SYNC_GRACE_SECONDS = 1.0

# Live WorkflowState per run, readable by GET /runs/{run_id} while the
# background thread mutates it (the engine mutates the resume_state object in
# place, so a snapshot read here is always current). Completed runs stay until
# the cap evicts them — the canvas polls once more AFTER completion to fetch
# the final results. In-process on purpose: this service runs one worker, and
# the durable run row (proc, via reqrepo) still owns the status of record.
_LIVE_RUNS: "OrderedDict[str, Any]" = OrderedDict()
_LIVE_RUNS_CAP = 100


def _remember_live_run(run_id: str, state: Any) -> None:
    _LIVE_RUNS[run_id] = state
    _LIVE_RUNS.move_to_end(run_id)
    while len(_LIVE_RUNS) > _LIVE_RUNS_CAP:
        _LIVE_RUNS.popitem(last=False)


def _claim_and_execute(request: Request, run_id: str, wf: Dict[str, Any],
                        input_data: Dict[str, Any], user_id: Optional[str]) -> Dict[str, Any]:
    """Atomically claim this run, then execute it exactly once — in the
    background.

    The claim is a single conditional UPDATE (see
    ``workflow_input_request_repo.claim_for_execution``) so a replayed final
    answer, or two answers landing concurrently on the last two outstanding
    questions, cannot both win — only one caller ever executes the workflow.
    A caller that does not win gets the run's current persisted state back
    instead of running it again (the agents this can trigger include
    email_dispatch, negotiation and supplier_interaction — running twice
    means sending real emails twice).

    Execution happens on a daemon thread; the request waits at most
    ``_SYNC_GRACE_SECONDS`` and then answers with whatever state the run has
    reached — "executing" if it is still going, final state if it finished
    inside the grace. The canvas polls GET /runs/{run_id} for the rest.
    """
    if not reqrepo.claim_for_execution(run_id):
        run_row = reqrepo.get_run(run_id)
        status = run_row["status"] if run_row else "completed"
        # Same shape as the winning path below (node_statuses, errors included)
        # so a client never has to special-case the claim-lost response.
        return {
            "run_id": run_id, "status": status, "pending": [],
            "nodes": _describe_nodes(wf["graph"]),
            "node_statuses": {}, "node_results": {}, "errors": [],
        }

    # Anything that must fail loudly — no orchestrator, an uncompilable graph —
    # fails HERE, before a thread exists, and marks the claimed run failed so
    # it is not left "executing" forever.
    try:
        orchestrator = getattr(request.app.state, "orchestrator", None)
        if orchestrator is None:
            raise HTTPException(status_code=503, detail="Orchestrator not available")
        engine = getattr(orchestrator, "_workflow_engine", None)
        if engine is None:
            raise HTTPException(status_code=503, detail="Workflow engine not available")
        graph = compile_graph(wf["name"], wf["graph"])
    except Exception:
        reqrepo.finish_run(run_id, "failed")
        raise

    from orchestration.workflow_engine import WorkflowState

    state = WorkflowState(
        workflow_id=run_id,
        workflow_name=wf["name"],
        user_id=user_id,
        started_at=datetime.utcnow().isoformat(),
        shared_data=dict(input_data or {}),
    )
    _remember_live_run(run_id, state)

    worker = threading.Thread(
        target=_run_to_completion,
        args=(engine, graph, state, run_id, user_id),
        name=f"agent-workflow-{run_id}",
        daemon=True,
    )
    worker.start()
    worker.join(timeout=_SYNC_GRACE_SECONDS)

    run_row = reqrepo.get_run(run_id)
    return {
        "run_id": run_id,
        "status": run_row["status"] if run_row else "executing",
        "node_statuses": {
            k: getattr(v, "value", v) for k, v in state.node_statuses.items()
        },
        "node_results": _summarise_node_results(state.node_results, wf["graph"]),
        "errors": _readable_errors(state.errors, wf["graph"]),
        "nodes": _describe_nodes(wf["graph"]),
        "pending": [],
    }


def _run_to_completion(engine: Any, graph: Any, state: Any, run_id: str,
                       user_id: Optional[str]) -> None:
    """The background body of a run. Owns the terminal status of the run row:
    whatever happens — a clean finish, agent-level errors, or the engine
    itself raising — the row leaves "executing"."""
    try:
        engine.execute(graph, input_data=dict(state.shared_data),
                       user_id=user_id, workflow_id=run_id, resume_state=state)
        final = "failed" if (state.errors or state.status == "failed") else "completed"
    except Exception as exc:  # noqa: BLE001 — a thread that dies silently strands the run
        logger.exception("agent workflow run %s crashed", run_id)
        state.status = "failed"
        state.errors.append({"error": str(exc)})
        final = "failed"
    reqrepo.finish_run(run_id, final)


# Keys an agent writes for the process, not the person: run bookkeeping, context
# snapshots, plans. A result card that shipped these would put the system's
# internals on the wire — the exact thing _readable_errors stopped doing for
# failures, done here for successes.
_INTERNAL_RESULT_KEYS = {
    "action_id", "context", "plan", "routing_history", "task_profile",
    "policy_context", "knowledge_base", "input_data", "shared_data", "raw",
    "raw_response", "trace", "messages", "prompt",
}

# Prose keys, in the order a human would want them as the card's one-line answer.
_HEADLINE_KEYS = ("summary", "message", "answer", "recommendation", "status_message")

_MAX_FACTS = 6
_MAX_VALUE_CHARS = 120
# A reasoning node's answer is prose — a recommendation and its reasons — so the
# card carries it in full (bounded), where the headline is only its first line.
_MAX_ANSWER_CHARS = 2000


def _fact_label(key: str) -> str:
    return key.replace("_", " ").strip().capitalize()


_MARKDOWN_MARKS = re.compile(r"\*\*|__|`")


def _plain(text: str) -> str:
    """The card renders plain text; a model's **bold** and `code` marks would
    show as literal symbols."""
    return _MARKDOWN_MARKS.sub("", text)


def _summarise_node_results(
    node_results: Dict[str, Any], graph: Dict[str, Any]
) -> Dict[str, Dict[str, Any]]:
    """One renderable card per node: a headline and a few flat facts.

    Summarised, never forwarded: scalars are kept (truncated), lists become
    counts, and nested dicts, underscored keys and known-internal keys do not
    leave the process at all. Everything that does leave goes through the
    output-safety gate — an agent's summary sentence can name a table or a
    stack frame just as easily as an error string can.
    """
    from services import output_safety as osafe

    labels = {n.get("id"): n.get("agent_slug") for n in (graph.get("nodes") or [])}

    cards: Dict[str, Dict[str, Any]] = {}
    for node_id, data in (node_results or {}).items():
        data = data if isinstance(data, dict) else {}

        headline = ""
        for key in _HEADLINE_KEYS:
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                headline = _plain(value.strip())[:200]
                break

        facts: List[Dict[str, str]] = []
        for key, value in data.items():
            if len(facts) >= _MAX_FACTS:
                break
            if not isinstance(key, str) or key.startswith("_"):
                continue
            if key in _INTERNAL_RESULT_KEYS or key in _HEADLINE_KEYS:
                continue
            if isinstance(value, bool):
                shown = "yes" if value else "no"
            elif isinstance(value, int):
                shown = str(value)
            elif isinstance(value, float):
                # 44154.802707999974 is a computation artefact, not a figure
                # a person reads.
                shown = f"{value:.2f}"
            elif isinstance(value, str):
                if not value.strip() or len(value) > _MAX_VALUE_CHARS:
                    continue
                shown = value.strip()
            elif isinstance(value, list):
                if not value:
                    continue
                shown = f"{len(value)} item" + ("" if len(value) == 1 else "s")
            else:
                continue  # dicts and anything exotic stay in the process
            facts.append({"label": _fact_label(key), "value": shown})

        if not headline:
            headline = (
                f"Produced {facts[0]['value']} — {facts[0]['label'].lower()}"
                if facts and facts[0]["value"].endswith("items")
                else "Completed"
            )

        card: Dict[str, Any] = {
            "agent_slug": labels.get(node_id) or "",
            "headline": osafe.enforce(headline, where="workflow result headline"),
            "facts": osafe.scrub_payload(facts, where="workflow result facts"),
        }
        answer = data.get("answer")
        if isinstance(answer, str) and answer.strip():
            card["answer"] = osafe.enforce(
                _plain(answer.strip())[:_MAX_ANSWER_CHARS], where="workflow result answer"
            )
        cards[node_id] = card
    return cards


def _readable_errors(errors: Any, graph: Dict[str, Any]) -> List[str]:
    """Turn the engine's structured failures into sentences a human can act on.

    The engine records a failure as ``{node, agent, error, data}`` and we were handing that
    straight to the browser. The client renders errors as text — it does
    ``errors.join('; ')`` and ``errors.map(escH)`` — so every failure arrived as the literal
    string "[object Object]". The run had genuinely executed and genuinely failed, and the
    only thing the user could see was that it had done *something* unspeakable. That is why
    the Submit button looked dead: it worked perfectly and then told them nothing.

    Two things are fixed by returning strings. The user gets the actual reason ("the document
    reference you supplied matched no documents"), and ``data`` — which carried the agent's
    whole payload, including its context snapshot, its action id and its internal plan — stops
    being shipped to the browser at all. It was never renderable and never should have left
    the process.

    Every message goes through the output-safety gate, because ``error`` is often ``str(exc)``
    and a psycopg2 exception names the table and the column it choked on.
    """
    from services import output_safety as osafe

    labels = {n.get("id"): n.get("agent_slug") for n in (graph.get("nodes") or [])}

    out: List[str] = []
    for e in errors or []:
        if not isinstance(e, dict):
            out.append(osafe.enforce(str(e), where="workflow error"))
            continue
        node = e.get("node") or ""
        agent = e.get("agent") or labels.get(node) or "agent"
        reason = osafe.enforce(str(e.get("error") or "Unknown error"), where="workflow error")
        pretty = str(agent).replace("_", " ").title()
        out.append(f"{pretty} — {reason}")
    return out
