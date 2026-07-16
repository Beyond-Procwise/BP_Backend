from fastapi import APIRouter, Depends, HTTPException, Request
from typing import Dict, Any, List, Optional
import copy
import json
import logging
import os
import re
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


def _do_reload_governance(agent_nick) -> Dict[str, Any]:
    """The one governance hot-reload: prompts + policies + extraction hints."""
    agent_nick.policy_engine.reload_policies()
    agent_nick.prompt_engine.refresh()
    from src.services.extraction_feedback.hint_store import HINT_STORE
    hints = HINT_STORE.refresh()
    return {
        "prompts": len(agent_nick.prompt_engine.all_prompts()),
        "policies": len(agent_nick.policy_engine.list_policies()),
        "extraction_vendor_hints": hints,
    }


@router.post("/reload-governance")
async def reload_governance(agent_nick=Depends(get_agent_nick)):
    """Reload both prompt and policy governance from the bp_ tables."""
    try:
        return {"status": "success", **_do_reload_governance(agent_nick)}
    except Exception as e:  # pragma: no cover - defensive
        logger.error(f"Failed to reload governance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Derived agents: a new catalogue identity on an existing backing class ──

_KEBAB_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def _kebab(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


class CreateAgentBody(BaseModel):
    name: str
    slug: Optional[str] = None
    description: str = ""
    backing_slug: str
    instructions: str
    capabilities: Optional[List[str]] = None


@router.get("/creatable-bases")
async def creatable_bases():
    """The honest options list for the UI: only agents that can actually
    execute (catalogue entries WITH a class_path) may back a derived agent."""
    from agents.definitions import load_agent_definitions
    return {
        "bases": [
            {
                "slug": a["slug"],
                "description": a.get("description", ""),
                "capabilities": list(a.get("capabilities") or []),
            }
            for a in load_agent_definitions()
            if a.get("class_path")
        ]
    }


@router.post("")
async def create_agent(
    body: CreateAgentBody, request: Request, agent_nick=Depends(get_agent_nick)
):
    """Create a DERIVED agent: a new catalogue entry backed by an existing
    agent class, plus a bp_prompt row carrying its instructions.

    No parallel mechanism: the result is a normal catalogue agent that the
    compiler validates, the engine resolves from the live registry, and
    governance links through prompt_linked_agents — all existing paths.
    """
    from agents.definitions import DEFINITIONS_PATH, load_agent_definitions

    name = (body.name or "").strip()
    if not name:
        raise HTTPException(status_code=422, detail="name must not be empty")
    instructions = body.instructions or ""
    if not instructions.strip():
        raise HTTPException(status_code=422, detail="instructions must not be empty")

    slug = (body.slug or _kebab(name)).strip()
    if not _KEBAB_RE.match(slug):
        raise HTTPException(
            status_code=422,
            detail=f"slug {slug!r} must be kebab-case (lowercase letters, digits, hyphens)",
        )

    definitions = load_agent_definitions()
    if any(a.get("slug") == slug for a in definitions):
        raise HTTPException(
            status_code=422, detail=f"slug {slug!r} already exists in the agent catalogue"
        )

    backing = next((a for a in definitions if a.get("slug") == body.backing_slug), None)
    if backing is None:
        raise HTTPException(
            status_code=422,
            detail=f"backing_slug {body.backing_slug!r} is not in the agent catalogue",
        )
    if not backing.get("class_path"):
        raise HTTPException(
            status_code=422,
            detail=(
                f"backing_slug {body.backing_slug!r} has no class_path and cannot execute; "
                "pick one of GET /agents/creatable-bases"
            ),
        )

    # The derived entry IS the backing entry (class_path, inputs, outputs,
    # elicit, dependencies, ...) under a new identity. Nothing is invented.
    entry = copy.deepcopy(backing)
    entry["agentId"] = max(int(a.get("agentId") or 0) for a in definitions) + 1
    entry["slug"] = slug
    entry["description"] = (body.description or "").strip() or backing.get("description", "")
    if body.capabilities is not None:
        entry["capabilities"] = list(body.capabilities)
    entry["derived_from"] = body.backing_slug
    entry["created_by"] = "workspace"

    # Governance token: PromptEngine tokenises linkage text on word characters
    # (a hyphen splits a token in two), so the kebab slug is stored underscored.
    gov_slug = slug.replace("-", "_")
    prompt_name = f"{gov_slug}_instructions"

    # 1) bp_prompt row first (most likely failure point): the instructions,
    #    verbatim, linked to the NEW slug. Same column set the governance
    #    loader reads (PromptEngine._DEFAULT_COLUMNS); prompt_id is identity.
    try:
        conn = agent_nick.get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO proc.bp_prompt "
                    "(prompt_name, prompt_type, prompt_linked_agents, prompts_desc, "
                    " prompts_status, version, created_by, last_modified_by) "
                    "VALUES (%s, 'agent_instructions', %s, to_jsonb(%s::text), 1, 1, "
                    "        'workspace', 'workspace') RETURNING prompt_id",
                    (prompt_name, gov_slug, instructions),
                )
                prompt_id = cur.fetchone()[0]
            conn.commit()
        finally:
            conn.close()
    except Exception as exc:
        logger.exception("create_agent: bp_prompt insert failed")
        raise HTTPException(status_code=500, detail=f"could not store instructions: {exc}")

    # 2) Append to agent_definitions.json atomically (tmp file + rename).
    try:
        with DEFINITIONS_PATH.open(encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            data.setdefault("agents", []).append(entry)
        else:
            data.append(entry)
        tmp = DEFINITIONS_PATH.with_name(DEFINITIONS_PATH.name + ".tmp")
        tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, DEFINITIONS_PATH)
    except Exception as exc:
        # A failed create must leave no trace: deactivate the prompt row.
        try:
            conn = agent_nick.get_db_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE proc.bp_prompt SET prompts_status = 0 WHERE prompt_id = %s",
                        (prompt_id,),
                    )
                conn.commit()
            finally:
                conn.close()
        except Exception:
            logger.exception(
                "create_agent: could not deactivate prompt %s after catalogue failure",
                prompt_id,
            )
        logger.exception("create_agent: catalogue write failed")
        raise HTTPException(status_code=500, detail=f"could not update agent catalogue: {exc}")

    # 3) Reload in-process — same registry-build startup does, same governance
    #    reload POST /agents/reload-governance does — so the new slug runs NOW.
    reload_report: Dict[str, Any] = {}
    try:
        from agents.auto_registry import AutoRegistry

        registry = getattr(agent_nick, "auto_registry", None)
        if registry is None:
            registry = AutoRegistry.from_json()
            registry.set_agent_nick(agent_nick)
            agent_nick.auto_registry = registry
        else:
            registry.refresh_from_json()
        instance = registry.get_agent(slug)  # stamps instance.governance_slug
        agent_nick.agents[slug] = instance  # same object the workflow engine resolves from

        # The orchestrator caches the catalogue (lru_cache) and the prompt /
        # policy catalogues per instance — clear them or a run with the new
        # slug is rejected against the stale view.
        from orchestration.orchestrator import Orchestrator

        Orchestrator._load_agent_definitions.cache_clear()
        orchestrator = getattr(request.app.state, "orchestrator", None)
        if orchestrator is not None:
            orchestrator._prompt_cache = None
            orchestrator._policy_cache = None

        reload_report = _do_reload_governance(agent_nick)
    except Exception as exc:
        # The agent exists on disk and in bp_prompt; a restart picks it up.
        logger.exception("create_agent: in-process reload failed")
        reload_report = {"error": str(exc)}

    return {
        "status": "created",
        "slug": slug,
        "agent_id": entry["agentId"],
        "derived_from": body.backing_slug,
        "prompt_id": prompt_id,
        "prompt_name": prompt_name,
        "registered": slug in agent_nick.agents,
        "governance": reload_report,
    }


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
