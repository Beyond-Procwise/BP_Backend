from fastapi import APIRouter, Depends, HTTPException, Request
from typing import Dict, Any, Optional
import logging
import os
from pydantic import BaseModel

from orchestration.orchestrator import Orchestrator

# Ensure GPU-related environment variables are set for agent operations
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("OLLAMA_USE_GPU", "1")
os.environ.setdefault("OLLAMA_NUM_PARALLEL", "4")
os.environ.setdefault("OMP_NUM_THREADS", "8")

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agents", tags=["Agents"])


def get_agent_nick(request: Request):
    """Get AgentNick from app state"""
    if not hasattr(request.app.state, 'agent_nick') or not request.app.state.agent_nick:
        raise HTTPException(status_code=503, detail="AgentNick not available")
    return request.app.state.agent_nick


def get_orchestrator(request: Request) -> Orchestrator:
    orchestrator = getattr(request.app.state, "orchestrator", None)
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator service is not available.")
    return orchestrator


@router.get("/list")
async def list_agents(agent_nick=Depends(get_agent_nick)):
    """List all available agents"""
    return {
        "agents": list(agent_nick.agents.keys()),
        "total": len(agent_nick.agents)
    }


class ReasonRequest(BaseModel):
    task: str
    max_rounds: int = 6
    # A grounded system should not answer from memory. Off only for explicitly
    # open-ended prompts where no tool could help.
    require_tool_use: bool = True


@router.get("/tools")
async def list_tools(agent_nick=Depends(get_agent_nick)):
    """Every tool AgentNick can call: the agents, governance, and the corpus."""
    tools = agent_nick.tools()
    return {
        "count": len(tools),
        "tools": [
            {
                "name": t.name,
                "description": t.description,
                "required": t.parameters.get("required", []),
            }
            for t in tools
        ],
    }


@router.post("/reason")
async def agent_nick_reason(req: ReasonRequest, agent_nick=Depends(get_agent_nick)):
    """Let AgentNick plan and act on a task by calling tools.

    Returns the answer AND the full trace — every tool call, its arguments, and
    what it returned. The trace is what makes the answer checkable: you can see
    which agent ran, which governed policy was fetched, and which corpus figures
    the answer was built from.
    """
    try:
        result = agent_nick.reason(
            req.task,
            max_rounds=req.max_rounds,
            require_tool_use=req.require_tool_use,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("AgentNick.reason failed")
        raise HTTPException(status_code=500, detail=str(exc))
    return result.to_dict()


@router.get("/status/{agent_name}")
async def get_agent_status(agent_name: str, agent_nick=Depends(get_agent_nick)):
    """Get status of specific agent"""
    if agent_name not in agent_nick.agents:
        raise HTTPException(status_code=404, detail=f"Agent {agent_name} not found")

    agent = agent_nick.agents[agent_name]
    return {
        "name": agent_name,
        "class": agent.__class__.__name__,
        "status": "ready"
    }


@router.get("/{agent_name}/manifest")
async def get_agent_manifest(agent_name: str, orchestrator: Orchestrator = Depends(get_orchestrator)):
    """Return the manifest describing the agent's responsibilities and knowledge."""

    try:
        return orchestrator.manifest_service.build_manifest(agent_name)
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Failed to build manifest for %s", agent_name)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/reload-policies")
async def reload_policies(agent_nick=Depends(get_agent_nick)):
    """Reload policy configurations"""
    try:
        agent_nick.policy_engine.reload_policies()
        return {"status": "success", "message": "Policies reloaded"}
    except Exception as e:
        logger.error(f"Failed to reload policies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reload-governance")
async def reload_governance(agent_nick=Depends(get_agent_nick)):
    """Reload both prompt and policy governance from the bp_ tables."""
    try:
        agent_nick.policy_engine.reload_policies()
        agent_nick.prompt_engine.refresh()
        from src.services.extraction_feedback.hint_store import HINT_STORE
        hints = HINT_STORE.refresh()
        return {
            "status": "success",
            "prompts": len(agent_nick.prompt_engine.all_prompts()),
            "policies": len(agent_nick.policy_engine.list_policies()),
            "extraction_vendor_hints": hints,
        }
    except Exception as e:  # pragma: no cover - defensive
        logger.error(f"Failed to reload governance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class AgentExecutionRequest(BaseModel):
    agent_type: str
    payload: Dict[str, Any] = {}


class DocumentProcessRequest(BaseModel):
    s3_prefix: Optional[str] = None
    s3_object_key: Optional[str] = None


@router.post("/process-document")
def process_document(
    req: DocumentProcessRequest, orchestrator: Orchestrator = Depends(get_orchestrator)
):
    """Convenience endpoint to run the document extraction workflow."""
    payload = {"s3_prefix": req.s3_prefix, "s3_object_key": req.s3_object_key}
    return orchestrator.execute_workflow("document_extraction", payload)


@router.post("/execute")
def execute_agent(
    req: AgentExecutionRequest,
    orchestrator: Orchestrator = Depends(get_orchestrator),
):
    """Execute a specified agent workflow."""
    if orchestrator.agent_nick is None:
        raise HTTPException(status_code=503, detail="Agent system not initialized")
    prs = orchestrator.agent_nick.process_routing_service
    if prs is None:
        raise HTTPException(status_code=503, detail="Process routing service unavailable")
    process_id = prs.log_process(
        process_name=req.agent_type,
        process_details=req.payload,
    )
    if process_id is None:
        raise HTTPException(status_code=500, detail="Failed to log process")
    action_id = prs.log_action(
        process_id=process_id,
        agent_type=req.agent_type,
        action_desc=req.payload,
        status="started",
    )
    try:
        result = orchestrator.execute_workflow(req.agent_type, req.payload)
        prs.log_action(
            process_id=process_id,
            agent_type=req.agent_type,
            action_desc=req.payload,
            process_output=result,
            status="completed",
            action_id=action_id,
        )
        prs.update_process_status(process_id, 1)
        return result
    except Exception as exc:  # pragma: no cover - defensive
        prs.log_action(
            process_id=process_id,
            agent_type=req.agent_type,
            action_desc=str(exc),
            status="failed",
            action_id=action_id,
        )
        prs.update_process_status(process_id, -1)
        raise HTTPException(status_code=500, detail=str(exc))
