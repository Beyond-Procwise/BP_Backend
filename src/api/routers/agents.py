from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
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
os.environ.setdefault("OMP_NUM_THREADS", "8")

from api.auth import require_user
from api.endpoint_gate import require as gate

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
async def reload_policies(
    agent_nick=Depends(get_agent_nick), principal=Depends(require_user)
):
    """Reload policy configurations.

    Gated: this re-reads the rules that govern every other decision in the
    product, so it is the sharpest configure action there is.
    """
    gate("policy.reload", principal, agent="AgentsRouter")
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
async def reload_governance(
    agent_nick=Depends(get_agent_nick), principal=Depends(require_user)
):
    """Reload both prompt and policy governance from the bp_ tables."""
    gate("prompt.write", principal, agent="AgentsRouter")
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
    # Optional on purpose: the workspace console sends only a description —
    # the governed prompt is derived from it. Explicit instructions still win.
    instructions: str = ""
    capabilities: Optional[List[str]] = None
    # Both optional, and both meaning "leave it alone" when absent. An agent
    # created without them behaves exactly as one created before they existed:
    # standard model, governed by whatever already links to it.
    model_key: Optional[str] = None      # a key from proc.bp_model, never a model name
    prompt_ids: Optional[List[int]] = None
    policy_ids: Optional[List[int]] = None


@router.get("/governance-options")
def governance_options(agent_nick=Depends(get_agent_nick)):
    """Active prompts and policies a new agent can be linked to.

    Ids and names only. The prompt BODY is not returned: this list exists to pick
    from, and shipping every governed instruction to the browser to render a
    dropdown would put the system's operating instructions on the wire for no
    reason.
    """
    out: Dict[str, List[Dict[str, Any]]] = {"prompts": [], "policies": []}
    try:
        conn = agent_nick.get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT prompt_id, prompt_name, version FROM proc.bp_prompt "
                    "WHERE prompts_status = 1 ORDER BY prompt_name, version DESC"
                )
                out["prompts"] = [
                    {"id": r[0], "name": r[1], "version": r[2]} for r in cur.fetchall()
                ]
                cur.execute(
                    "SELECT policy_id, policy_name, version FROM proc.bp_policy "
                    "WHERE policy_status = 1 ORDER BY policy_name, version DESC"
                )
                out["policies"] = [
                    {"id": r[0], "name": r[1], "version": r[2]} for r in cur.fetchall()
                ]
        finally:
            conn.close()
    except Exception:  # noqa: BLE001
        # An unreadable governance table must not block agent creation — the
        # picker renders empty and the agent is created ungoverned, which is
        # exactly what it would have been before this option existed.
        logger.exception("governance options unavailable")
        return {"prompts": [], "policies": [], "error": "Governance list unavailable."}
    return out


def _link_governance(agent_nick, gov_slug: str, table: str, id_col: str,
                     link_col: str, ids: List[int]) -> List[int]:
    """Append ``gov_slug`` to the linked-agents column of each row.

    Appends rather than replaces: these rows govern OTHER agents too, and the
    workspace linking one to a new agent must never quietly un-govern the ones
    already relying on it. The token is added only when absent, so a repeated
    link is a no-op rather than a duplicate.
    """
    linked: List[int] = []
    if not ids:
        return linked
    conn = agent_nick.get_db_connection()
    try:
        with conn.cursor() as cur:
            for row_id in ids:
                cur.execute(
                    f"SELECT {link_col} FROM proc.{table} WHERE {id_col} = %s", (row_id,)
                )
                record = cur.fetchone()
                if record is None:
                    continue
                current = (record[0] or "").strip()
                tokens = [t.strip() for t in re.split(r"[,;]", current) if t.strip()]
                if gov_slug in tokens:
                    linked.append(row_id)
                    continue
                tokens.append(gov_slug)
                cur.execute(
                    f"UPDATE proc.{table} SET {link_col} = %s WHERE {id_col} = %s",
                    (", ".join(tokens), row_id),
                )
                linked.append(row_id)
        conn.commit()
    finally:
        conn.close()
    return linked


def _unlink_governance(agent_nick, gov_slug: str) -> int:
    """Remove ``gov_slug`` from every governance row that names it.

    The reverse of _link_governance, for delete. A deleted agent that stays
    listed in prompt_linked_agents leaves the governance badge of every other
    agent sharing that row citing something that no longer exists.
    """
    removed = 0
    conn = agent_nick.get_db_connection()
    try:
        with conn.cursor() as cur:
            for table, id_col, link_col in (
                ("bp_prompt", "prompt_id", "prompt_linked_agents"),
                ("bp_policy", "policy_id", "policy_linked_agents"),
            ):
                cur.execute(
                    f"SELECT {id_col}, {link_col} FROM proc.{table} "
                    f"WHERE {link_col} IS NOT NULL AND {link_col} <> ''"
                )
                for row_id, linked_agents in cur.fetchall():
                    tokens = [t.strip() for t in re.split(r"[,;]", linked_agents or "") if t.strip()]
                    if gov_slug not in tokens:
                        continue
                    kept = [t for t in tokens if t != gov_slug]
                    cur.execute(
                        f"UPDATE proc.{table} SET {link_col} = %s WHERE {id_col} = %s",
                        (", ".join(kept), row_id),
                    )
                    removed += 1
        conn.commit()
    finally:
        conn.close()
    return removed


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
    body: CreateAgentBody,
    request: Request,
    agent_nick=Depends(get_agent_nick),
    principal=Depends(require_user),
):
    """Create a DERIVED agent: a new catalogue entry backed by an existing
    agent class, plus a bp_prompt row carrying its instructions.

    No parallel mechanism: the result is a normal catalogue agent that the
    compiler validates, the engine resolves from the live registry, and
    governance links through prompt_linked_agents — all existing paths.
    """
    gate("agent.create", principal, agent="AgentsRouter",
         context={"name": (body.name or "").strip()})

    from agents.definitions import DEFINITIONS_PATH, load_agent_definitions

    name = (body.name or "").strip()
    if not name:
        raise HTTPException(status_code=422, detail="name must not be empty")
    instructions = (body.instructions or "").strip() or (body.description or "").strip()
    if not instructions:
        raise HTTPException(
            status_code=422,
            detail="instructions or description required: one of them becomes "
                   "the governed prompt that tells this agent what to do",
        )

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

    # An override is validated against proc.bp_model BEFORE anything is written:
    # a model key that is not offered, or one whose provider has no key, must be
    # refused at the door rather than stored and discovered at run time by an
    # agent that then silently falls back.
    model_ref: Optional[str] = None
    if body.model_key:
        from repositories import model_catalogue_repo as model_repo

        row = model_repo.get(body.model_key)
        if row is None or not row.get("model_status"):
            raise HTTPException(
                status_code=422,
                detail=f"model {body.model_key!r} is not offered on this installation",
            )
        if not row.get("selectable"):
            raise HTTPException(
                status_code=422,
                detail=f"model {row['display_name']!r} needs an API key to be configured first",
            )
        # The standard model resolves to None on purpose: "standard" means
        # "whatever every other agent uses". Pinning today's standard onto this
        # agent would leave it behind the next time the standard moves.
        model_ref = model_repo.resolve_ref(body.model_key)

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
    if model_ref:
        entry["model_key"] = body.model_key   # what was chosen, for the UI to echo back
        entry["model"] = model_ref            # what AgentNick calls (see get_agent_model)

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

        # A model chosen for THIS agent only matters if the resolver can see it.
        # The registry is built once and cached, so it has to be told.
        if hasattr(agent_nick, "refresh_agent_model_registry"):
            agent_nick.refresh_agent_model_registry()

        reload_report = _do_reload_governance(agent_nick)
    except Exception as exc:
        # The agent exists on disk and in bp_prompt; a restart picks it up.
        logger.exception("create_agent: in-process reload failed")
        reload_report = {"error": str(exc)}

    # Linking runs LAST and never fails the create: the agent is already real by
    # this point, and an agent that exists ungoverned is recoverable (link it
    # again), whereas a 500 after the catalogue write would leave the user
    # believing nothing had happened while the agent sat there.
    linked: Dict[str, Any] = {"prompts": [], "policies": []}
    try:
        linked["prompts"] = _link_governance(
            agent_nick, gov_slug, "bp_prompt", "prompt_id",
            "prompt_linked_agents", body.prompt_ids or [],
        )
        linked["policies"] = _link_governance(
            agent_nick, gov_slug, "bp_policy", "policy_id",
            "policy_linked_agents", body.policy_ids or [],
        )
        if linked["prompts"] or linked["policies"]:
            reload_report = _do_reload_governance(agent_nick)
    except Exception as exc:  # noqa: BLE001
        logger.exception("create_agent: governance linking failed")
        linked["error"] = str(exc)

    return {
        "status": "created",
        "slug": slug,
        "agent_id": entry["agentId"],
        "derived_from": body.backing_slug,
        "prompt_id": prompt_id,
        "prompt_name": prompt_name,
        "model_key": body.model_key or None,
        "registered": slug in agent_nick.agents,
        "governance": reload_report,
        "linked": linked,
    }


@router.delete("/{slug}")
async def delete_agent(
    slug: str,
    request: Request,
    agent_nick=Depends(get_agent_nick),
    principal=Depends(require_user),
):
    """Delete a DERIVED agent: the exact reverse of POST /agents.

    Removes the catalogue entry, its bp_prompt instructions row(s), the live
    registry instance/contract, and clears the same caches create warms. The
    14 built-in agents (no "derived_from" marker) are permanent and 403 here.
    A derived agent still referenced by a saved workflow 409s instead of
    cascade-deleting the workflow — workflows are never silently destroyed.
    """
    gate("agent.delete", principal, agent="AgentsRouter", context={"slug": slug})
    from agents.definitions import DEFINITIONS_PATH, load_agent_definitions

    definitions = load_agent_definitions()
    entry = next((a for a in definitions if a.get("slug") == slug), None)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"agent {slug!r} not found in catalogue")

    if not entry.get("derived_from"):
        raise HTTPException(
            status_code=403,
            detail=(
                f"agent {slug!r} is a built-in agent and cannot be deleted"
            ),
        )

    # Dependency check: never cascade-delete a saved workflow. A workflow's
    # graph nodes carry {"id", "agent_slug", ...} (see agent_workflows.py's
    # _describe_nodes) -- any node whose agent_slug matches blocks the delete.
    from repositories import agent_workflow_repo as wf_repo

    dependents: List[Dict[str, Any]] = []
    for wf in wf_repo.list_active():
        nodes = (wf.get("graph") or {}).get("nodes") or []
        if any(n.get("agent_slug") == slug for n in nodes):
            dependents.append({"id": wf["workflow_id"], "name": wf["name"]})
    if dependents:
        # A plain `raise HTTPException(409, detail={...})` would come back to the
        # client as a STRINGIFIED dict: main.py's global _safe_http_exception
        # handler does `str(exc.detail)` on every non-string HTTPException detail
        # (its job is to stop raw exception text leaking, and it cannot tell a
        # deliberate structured payload from an accidental one). Returning a
        # JSONResponse directly raises no exception, so it skips that handler
        # and only passes through OutputSafetyMiddleware, which scrubs string
        # leaves but preserves dict/list structure -- dependent_workflows stays
        # a real array the UI can read.
        return JSONResponse(
            status_code=409,
            content={
                "detail": (
                    f"agent {slug!r} is used by {len(dependents)} saved "
                    "workflow(s); delete them first"
                ),
                "dependent_workflows": dependents,
            },
        )

    # Same governance token create used to write the instructions row.
    gov_slug = slug.replace("-", "_")

    # 1) bp_prompt rows linked to this slug (mirrors create's step 1 insert).
    prompts_deleted = 0
    try:
        conn = agent_nick.get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM proc.bp_prompt WHERE prompt_linked_agents = %s",
                    (gov_slug,),
                )
                prompts_deleted = cur.rowcount
            conn.commit()
        finally:
            conn.close()
    except Exception as exc:
        logger.exception("delete_agent: bp_prompt delete failed")
        raise HTTPException(status_code=500, detail=f"could not remove instructions: {exc}")

    # 2) Remove from agent_definitions.json atomically (tmp file + rename),
    #    same shape create's step 2 writes (indent=2 + trailing newline).
    try:
        with DEFINITIONS_PATH.open(encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            remaining = [a for a in data.get("agents", []) if a.get("slug") != slug]
            data["agents"] = remaining
        else:
            data = [a for a in data if a.get("slug") != slug]
        tmp = DEFINITIONS_PATH.with_name(DEFINITIONS_PATH.name + ".tmp")
        tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, DEFINITIONS_PATH)
    except Exception as exc:
        logger.exception("delete_agent: catalogue write failed")
        raise HTTPException(status_code=500, detail=f"could not update agent catalogue: {exc}")

    # 3) Reload in-process — mirrors create's step 3 in reverse, so the slug
    #    stops resolving/executing NOW rather than only after a restart.
    reload_report: Dict[str, Any] = {}
    try:
        from agents.auto_registry import AutoRegistry  # noqa: F401  (parity with create's import)

        registry = getattr(agent_nick, "auto_registry", None)
        if registry is not None:
            registry.refresh_from_json()
            registry.remove(slug)
        agent_nick.agents.pop(slug, None)

        from orchestration.orchestrator import Orchestrator

        Orchestrator._load_agent_definitions.cache_clear()
        orchestrator = getattr(request.app.state, "orchestrator", None)
        if orchestrator is not None:
            orchestrator._prompt_cache = None
            orchestrator._policy_cache = None

        # A model chosen for this agent dies with it (the entry carried it), but
        # the cached registry still holds the mapping until it is rebuilt.
        if hasattr(agent_nick, "refresh_agent_model_registry"):
            agent_nick.refresh_agent_model_registry()

        reload_report = _do_reload_governance(agent_nick)
    except Exception as exc:
        # The agent is already gone from disk and bp_prompt; a restart
        # guarantees the in-process view catches up.
        logger.exception("delete_agent: in-process reload failed")
        reload_report = {"error": str(exc)}

    # Rows this agent was LINKED to (as opposed to the instructions row created
    # for it, deleted above) survive — they govern other agents. Only this
    # agent's token is removed, or every badge citing that row would keep naming
    # an agent that no longer exists.
    unlinked = 0
    try:
        unlinked = _unlink_governance(agent_nick, gov_slug)
        if unlinked:
            reload_report = _do_reload_governance(agent_nick)
    except Exception:  # noqa: BLE001
        logger.exception("delete_agent: governance unlink failed")

    return {
        "status": "deleted",
        "slug": slug,
        "prompts_deleted": prompts_deleted,
        "governance_unlinked": unlinked,
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
