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
from repositories import agent_definition_repo
from services import rbac
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
async def agent_nick_reason(req: ReasonRequest, agent_nick=Depends(get_agent_nick),
                            principal=Depends(require_user)):
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
            agent="agent_nick",
            # the requester, so an approval this call triggers is never decided by them
            user_id=getattr(principal, "subject", None) or None,
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
    """The one governance hot-reload: prompts + policies + rules + hints.

    Rules are in here for the same reason prompts are: a threshold you have to
    restart the service to change is a threshold nobody changes.
    """
    agent_nick.policy_engine.reload_policies()
    agent_nick.prompt_engine.refresh()
    from src.services.extraction_feedback.hint_store import HINT_STORE
    hints = HINT_STORE.refresh()

    # A rule book that was unreadable at boot is retried here, so recovering
    # from an outage does not need a restart either.
    rule_book = getattr(agent_nick, "rule_book", None)
    if rule_book is None:
        from engines.rule_book import load_rule_book

        rule_book = load_rule_book(agent_nick)
        agent_nick.rule_book = rule_book
    else:
        from engines.rule_book import RuleBookUnavailable

        try:
            rule_book.reload()
        except RuleBookUnavailable as exc:
            # Keep serving the rules we already hold rather than dropping to
            # none: stale detection beats silent no detection.
            logger.error("rule book reload failed, keeping cached rules: %s", exc)

    return {
        "prompts": len(agent_nick.prompt_engine.all_prompts()),
        "policies": len(agent_nick.policy_engine.list_policies()),
        "rules": len(rule_book.active_rules()) if rule_book else 0,
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
    # The tools this agent reasons with. Absent means it is not narrowed; an
    # empty list means it is given none. Keys come from GET /agents/tools.
    tools: Optional[List[str]] = None
    # The outputs this agent returns, by name: [{"name", "description"}]. Each
    # run fills them in from what its tools returned, and reports one the
    # evidence does not establish as missing. Absent, the agent answers in prose only.
    outputs: Optional[List[Dict[str, Any]]] = None


def _checked_tools(agent_nick, tools: List[str]) -> List[str]:
    """The list as it will be stored, or a 422 naming what cannot be in it.

    Refused here rather than dropped at run time: a list that names a tool the
    agent will never be handed reads, to whoever reviews it, as a statement of
    what the agent can do.
    """
    from agents.base_agent import NODE_TOOL_EXCLUSIONS

    wanted = list(dict.fromkeys(str(t).strip() for t in tools if str(t).strip()))
    known = {t.name for t in agent_nick.tools()}
    unknown = [t for t in wanted if t not in known]
    if unknown:
        raise HTTPException(
            status_code=422,
            detail=f"not a tool on this installation: {', '.join(unknown)}",
        )
    barred = [t for t in wanted if t in NODE_TOOL_EXCLUSIONS]
    if barred:
        raise HTTPException(
            status_code=422,
            detail=f"no agent can be given: {', '.join(barred)}",
        )
    return wanted


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


@router.get("/tool-options")
async def tool_options(agent_nick=Depends(get_agent_nick)):
    """The tools an agent may be given, in the shape the authoring console picks
    from: ``{key, name, description}``.

    Distinct from GET /agents/tools, which answers "what can AgentNick call" for
    diagnostics and keys on the raw handler name. This is the pickable list, so
    the key is what a caller sends back and the name is a label.

    No ``writes`` flag: services.tool_runtime.Tool carries no side-effect
    metadata, and inferring one from a tool's name would be a guess presented to
    a reviewer as a fact about what the agent may do.
    """
    from agents.base_agent import NODE_TOOL_EXCLUSIONS

    # Only what an agent can actually be given: offering the tool that sends
    # email here would invite a pick that the create and edit calls refuse.
    tools = [t for t in agent_nick.tools() if t.name not in NODE_TOOL_EXCLUSIONS]
    return {
        "count": len(tools),
        "tools": [
            {
                "key": t.name,
                "name": _tool_label(t.name),
                "description": t.description,
            }
            for t in tools
        ],
    }


def _tool_label(key: str) -> str:
    """`run_data_extraction` -> `Data extraction`. Presentation only: the key is
    unchanged and remains what the caller sends back."""
    stem = key[4:] if key.startswith("run_") else key
    words = stem.replace("-", " ").replace("_", " ").strip()
    return words[:1].upper() + words[1:] if words else key


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

    from agents.definitions import load_agent_definitions

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

    tools = _checked_tools(agent_nick, body.tools) if body.tools is not None else None

    output_spec: Optional[List[Dict[str, str]]] = None
    if body.outputs is not None:
        from agents import named_outputs

        try:
            output_spec = named_outputs.checked_spec(body.outputs)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

    # The derived entry IS the backing entry (class_path, inputs, outputs,
    # elicit, dependencies, ...) under a new identity. Nothing is invented.
    entry = copy.deepcopy(backing)
    entry["agentId"] = max(int(a.get("agentId") or 0) for a in definitions) + 1
    entry["slug"] = slug
    entry["description"] = (body.description or "").strip() or backing.get("description", "")
    if body.capabilities is not None:
        entry["capabilities"] = list(body.capabilities)
    entry["derived_from"] = body.backing_slug
    # Who made it. "workspace" was written here for every agent ever created,
    # which named a screen, not a person; it remains only for a server running
    # with sign-in off, where there is nobody to name.
    created_by = getattr(principal, "subject", None) or None
    entry["created_by"] = created_by or "workspace"
    # A new agent is a draft: its creator can run it, nobody else can, until a
    # second person publishes it (POST /agents/{slug}/publish). With sign-in
    # off there is no creator to keep it to and no second person to release
    # it, so it is published at once, as every agent was before drafts existed.
    entry["stage"] = "draft" if created_by else "published"
    entry.pop("published_by", None)
    entry.pop("draft_by", None)
    # A base's own list is never inherited: the base is not narrowed by one, and
    # a copy must not start life narrowed by a list nobody chose for it.
    entry.pop("tools", None)
    if tools is not None:
        entry["tools"] = tools
    # Likewise the base's named outputs. With its own, the copied "outputs" —
    # the base class's, which an agent that reasons never produces — are
    # replaced by the ones it was actually defined to return.
    entry.pop("output_spec", None)
    if output_spec is not None:
        entry["output_spec"] = output_spec
        entry["outputs"] = [o["name"] for o in output_spec]
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

    # 2) Store the agent. In the database, not the catalogue file: the file is
    #    tracked in git and a redeploy replaces it, taking with it every agent a
    #    person had made. The store records who made it and keeps each version.
    try:
        agent_definition_repo.create(
            slug, entry, derived_from=body.backing_slug, by=created_by
        )
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
        if isinstance(exc, agent_definition_repo.AgentExists):
            raise HTTPException(
                status_code=422, detail=f"slug {slug!r} already exists in the agent catalogue"
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
        "tools": tools,
        "outputs": output_spec,
        "stage": entry["stage"],
        "registered": slug in agent_nick.agents,
        "governance": reload_report,
        "linked": linked,
    }


class UpdateAgentBody(BaseModel):
    """A PARTIAL update: every field is optional and an omitted field is left
    alone. Field names deliberately match CreateAgentBody, and `tools` is spelled
    `tools` (not `tool_keys`) — both as the authoring console asked.

    Omitted and null are different answers, and the handler tells them apart with
    ``model_fields_set``: omitting ``model_key`` keeps the current model, while
    sending it as null moves the agent back to the standard model. An emptied
    governance list is sent as ``[]``, which unlinks; omitting it keeps the links.
    """

    name: Optional[str] = None
    description: Optional[str] = None
    instructions: Optional[str] = None
    backing_slug: Optional[str] = None
    model_key: Optional[str] = None
    tools: Optional[List[str]] = None
    # The named outputs, [{"name", "description"}]. Omitted keeps them; null
    # removes them, and the agent answers in prose only.
    outputs: Optional[List[Dict[str, Any]]] = None
    capabilities: Optional[List[str]] = None
    prompt_ids: Optional[List[int]] = None
    policy_ids: Optional[List[int]] = None
    # The governance half of an edit. Accepted rather than rejected as unknown
    # (which would 422 every save), and carried into the gate's context so they
    # land in proc.bp_agent_actions — the same log GET /spendiq/audit-logs reads.
    change_reason: Optional[str] = None
    authorised_by: Optional[str] = None


def _relink_governance(agent_nick, gov_slug: str, table: str, id_col: str,
                       link_col: str, name_col: Optional[str],
                       keep_name: Optional[str], ids: List[int]) -> List[int]:
    """Make ``ids`` the complete set of rows in ``table`` that govern this agent.

    _link_governance only ever appends, which is right for create but cannot
    express "this one is no longer linked". This removes the token from rows the
    caller did not list and adds it to the ones they did, so ``[]`` really does
    unlink everything.

    ``keep_name`` is the agent's OWN instructions row, which create wrote and
    which is not one of the optional links. Unlinking it because it was absent
    from prompt_ids would leave the agent with no instructions at all — a
    governance edit silently lobotomising the agent it was editing.
    """
    wanted = set(ids or [])
    conn = agent_nick.get_db_connection()
    try:
        with conn.cursor() as cur:
            select_cols = f"{id_col}, {link_col}" + (f", {name_col}" if name_col else "")
            cur.execute(
                f"SELECT {select_cols} FROM proc.{table} "
                f"WHERE {link_col} IS NOT NULL AND {link_col} <> ''"
            )
            for record in cur.fetchall():
                row_id, linked_agents = record[0], record[1]
                row_name = record[2] if name_col else None
                tokens = [t.strip() for t in re.split(r"[,;]", linked_agents or "") if t.strip()]
                if gov_slug not in tokens or row_id in wanted:
                    continue
                if keep_name and row_name == keep_name:
                    continue
                kept = [t for t in tokens if t != gov_slug]
                cur.execute(
                    f"UPDATE proc.{table} SET {link_col} = %s WHERE {id_col} = %s",
                    (", ".join(kept), row_id),
                )
        conn.commit()
    finally:
        conn.close()
    return _link_governance(agent_nick, gov_slug, table, id_col, link_col, list(wanted))


@router.patch("/{slug}")
async def update_agent(
    slug: str,
    body: UpdateAgentBody,
    request: Request,
    agent_nick=Depends(get_agent_nick),
    principal=Depends(require_user),
):
    """Edit a DERIVED agent in place, changing only the fields that were sent.

    The same boundary as DELETE /agents/{slug}: a slug that is not in the
    catalogue is a 404, and a built-in agent (no ``derived_from``) is a 403. The
    console hides the control in those cases, but the API is the boundary, not
    the screen.
    """
    sent = body.model_fields_set
    gate(
        "agent.update",
        principal,
        agent="AgentsRouter",
        context={
            "slug": slug,
            "fields": sorted(sent - {"change_reason", "authorised_by"}),
            "change_reason": (body.change_reason or "").strip() or None,
            "authorised_by": (body.authorised_by or "").strip() or None,
        },
    )

    from agents.definitions import load_agent_definitions

    definitions = load_agent_definitions()
    entry = next((a for a in definitions if a.get("slug") == slug), None)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"agent {slug!r} not found in catalogue")
    if not entry.get("derived_from"):
        raise HTTPException(
            status_code=403,
            detail=f"agent {slug!r} is a built-in agent and cannot be edited",
        )
    editor = getattr(principal, "subject", None) or None

    updated = copy.deepcopy(entry)
    applied: Dict[str, Any] = {}

    # --- backing class -----------------------------------------------------
    # Re-derives the executable half (class_path, inputs, outputs, elicit,
    # dependencies) from the new base while identity stays put, which is exactly
    # what create does — the difference is that slug and description survive.
    if "backing_slug" in sent:
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
                    f"backing_slug {body.backing_slug!r} has no class_path and cannot "
                    "execute; pick one of GET /agents/creatable-bases"
                ),
            )
        carried = copy.deepcopy(backing)
        for keep in ("agentId", "slug", "description", "created_by", "name",
                     "capabilities", "model", "model_key", "tools", "output_spec",
                     "stage", "draft_by", "published_by"):
            carried.pop(keep, None)
        updated.update(carried)
        updated["derived_from"] = body.backing_slug
        applied["backing_slug"] = body.backing_slug

    if "name" in sent:
        updated["name"] = (body.name or "").strip()
        applied["name"] = updated["name"]
    if "description" in sent:
        updated["description"] = (body.description or "").strip()
        applied["description"] = updated["description"]
    if "capabilities" in sent:
        updated["capabilities"] = list(body.capabilities or [])
        applied["capabilities"] = updated["capabilities"]

    # --- tools -------------------------------------------------------------
    # The list IS the agent's tools at run time (the registry stamps it on the
    # agent and the loop is built from it). A list is the whole answer: [] is
    # no tools. Null removes the list, and the agent is no longer narrowed.
    if "tools" in sent:
        if body.tools is None:
            updated.pop("tools", None)
            applied["tools"] = None
        else:
            updated["tools"] = _checked_tools(agent_nick, body.tools)
            applied["tools"] = updated["tools"]

    # --- named outputs -----------------------------------------------------
    if "outputs" in sent:
        if body.outputs is None:
            updated.pop("output_spec", None)
            updated["outputs"] = []
            applied["outputs"] = None
        else:
            from agents import named_outputs

            try:
                spec = named_outputs.checked_spec(body.outputs)
            except ValueError as exc:
                raise HTTPException(status_code=422, detail=str(exc))
            updated["output_spec"] = spec
            updated["outputs"] = [o["name"] for o in spec]
            applied["outputs"] = spec

    # --- model -------------------------------------------------------------
    if "model_key" in sent:
        if body.model_key is None:
            updated.pop("model_key", None)
            updated.pop("model", None)
            applied["model_key"] = None
        else:
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
            model_ref = model_repo.resolve_ref(body.model_key)
            if model_ref:
                updated["model_key"] = body.model_key
                updated["model"] = model_ref
            else:
                # "standard" resolves to no ref on purpose; pinning today's
                # standard would leave this agent behind when the standard moves.
                updated.pop("model_key", None)
                updated.pop("model", None)
            applied["model_key"] = body.model_key

    gov_slug = slug.replace("-", "_")
    prompt_name = f"{gov_slug}_instructions"

    # --- instructions ------------------------------------------------------
    # Only when explicitly sent. A description edit does NOT rewrite the governed
    # prompt: create seeds the prompt from the description, but after that they
    # are different things, and silently rewriting an agent's instructions
    # because someone tidied its one-line summary is not an edit anyone asked for.
    instructions_updated = 0
    if "instructions" in sent:
        text = (body.instructions or "").strip()
        if not text:
            raise HTTPException(
                status_code=422, detail="instructions must not be empty when sent"
            )
        try:
            conn = agent_nick.get_db_connection()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE proc.bp_prompt "
                        "SET prompts_desc = to_jsonb(%s::text), "
                        "    version = COALESCE(version, 0) + 1, "
                        "    last_modified_by = 'workspace' "
                        "WHERE prompt_name = %s AND prompt_linked_agents = %s",
                        (text, prompt_name, gov_slug),
                    )
                    instructions_updated = cur.rowcount
                conn.commit()
            finally:
                conn.close()
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("update_agent: bp_prompt update failed")
            raise HTTPException(status_code=500, detail=f"could not store instructions: {exc}")
        applied["instructions"] = True

    # --- back to draft -----------------------------------------------------
    # An edit changes what the agent does, so it has to be released again: were
    # it to stay published, one person could have a harmless agent published
    # and then rewrite it, and the second pair of eyes would have seen nothing
    # of what everyone now runs. Any edit counts — deciding which edits are
    # harmless is exactly the judgement the second person is there to make.
    # It is the EDITOR's draft: they can try it, and they cannot publish it.
    # With sign-in off there is nobody to hold it to, so its stage is left alone.
    if editor:
        updated["stage"] = "draft"
        updated["draft_by"] = editor
        updated.pop("published_by", None)
    applied["stage"] = updated.get("stage") or "published"

    # --- catalogue write ---------------------------------------------------
    # To the store, where created agents live, and recorded as a version. This
    # used to rewrite agent_definitions.json, which no longer holds them: the
    # edit reported success and changed nothing.
    try:
        version = agent_definition_repo.update(slug, updated, by=editor)
    except agent_definition_repo.AgentNotFound:
        raise HTTPException(status_code=404, detail=f"agent {slug!r} not found in catalogue")
    except Exception as exc:
        logger.exception("update_agent: catalogue write failed")
        raise HTTPException(status_code=500, detail=f"could not update agent catalogue: {exc}")

    # --- governance links --------------------------------------------------
    linked: Dict[str, List[int]] = {}
    if "prompt_ids" in sent:
        linked["prompts"] = _relink_governance(
            agent_nick, gov_slug, "bp_prompt", "prompt_id", "prompt_linked_agents",
            "prompt_name", prompt_name, body.prompt_ids or [],
        )
        applied["prompt_ids"] = linked["prompts"]
    if "policy_ids" in sent:
        linked["policies"] = _relink_governance(
            agent_nick, gov_slug, "bp_policy", "policy_id", "policy_linked_agents",
            None, None, body.policy_ids or [],
        )
        applied["policy_ids"] = linked["policies"]

    # --- in-process reload -------------------------------------------------
    reload_report: Dict[str, Any] = {}
    try:
        registry = getattr(agent_nick, "auto_registry", None)
        if registry is not None:
            registry.refresh_from_json()

        Orchestrator._load_agent_definitions.cache_clear()
        orchestrator = getattr(request.app.state, "orchestrator", None)
        if orchestrator is not None:
            orchestrator._prompt_cache = None
            orchestrator._policy_cache = None

        if hasattr(agent_nick, "refresh_agent_model_registry"):
            agent_nick.refresh_agent_model_registry()

        reload_report = _do_reload_governance(agent_nick)
    except Exception as exc:
        # Disk and bp_prompt already carry the change; a restart catches the
        # in-process view up.
        logger.exception("update_agent: in-process reload failed")
        reload_report = {"error": str(exc)}

    return {
        "status": "updated",
        "slug": slug,
        "applied": applied,
        "instructions_rows_updated": instructions_updated,
        # The saved list is what the agent reasons with from this moment.
        "tools_enforced": True,
        "stage": applied["stage"],
        "version": version,
        "change_reason": (body.change_reason or "").strip() or None,
        "authorised_by": (body.authorised_by or "").strip() or None,
        "governance": reload_report,
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
    from agents.definitions import load_agent_definitions

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

    # 2) Mark it deleted in the store. The row and its history stay; a copy an
    #    older version left in the catalogue file stays hidden behind it.
    try:
        agent_definition_repo.delete(
            slug, entry, derived_from=entry["derived_from"],
            by=getattr(principal, "subject", None) or None,
        )
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


@router.get("/{slug}/history")
async def agent_history(slug: str, principal=Depends(require_user)):
    """Every recorded change to one created agent, newest first: what changed,
    who changed it and when, and what the agent looked like afterwards.

    A built-in agent has no history here — a release changes it, not a person —
    and answers 404 like a slug that was never created.
    """
    versions = agent_definition_repo.history(slug)
    if not versions:
        raise HTTPException(status_code=404, detail=f"no created agent {slug!r}")
    return {"slug": slug, "versions": versions}


def _require_role(principal, minimum: str, what: str) -> str:
    """The caller's role, or a 403 naming the role ``what`` needs. Nobody
    signed in is a Viewer, so this refuses rather than waves through."""
    role = rbac.effective_role(principal)
    if rbac.role_rank(role) < rbac.role_rank(minimum):
        raise HTTPException(status_code=403, detail=f"{what} needs the {minimum} role or higher")
    return role


@router.post("/{slug}/publish")
async def publish_agent(
    slug: str,
    request: Request,
    agent_nick=Depends(get_agent_nick),
    principal=Depends(require_user),
):
    """Release a draft agent to everyone.

    Two conditions, and both are the point: the caller has approval rights, and
    the caller is not the person who created the agent. One person cannot both
    write an agent's task and decide it is fit for everyone else to run.
    """
    from agents.definitions import load_agent_definitions

    entry = next((a for a in load_agent_definitions() if a.get("slug") == slug), None)
    if entry is None or not entry.get("derived_from"):
        raise HTTPException(status_code=404, detail=f"no created agent {slug!r}")
    if entry.get("stage") != "draft":
        raise HTTPException(status_code=409, detail=f"agent {slug!r} is not a draft")

    _require_role(principal, "Approver", "publishing an agent")
    subject = getattr(principal, "subject", None) or None
    # Whose draft it is: whoever last edited it, or failing that whoever created it.
    author = entry.get("draft_by") or entry.get("created_by")
    if not subject or subject == author:
        raise HTTPException(
            status_code=403,
            detail="an agent cannot be published by the person who created or last edited it",
        )

    published = {**entry, "stage": "published", "published_by": subject}
    published.pop("draft_by", None)
    try:
        version = agent_definition_repo.publish(slug, published, by=subject)
    except agent_definition_repo.AgentNotFound:
        raise HTTPException(status_code=404, detail=f"no created agent {slug!r}")

    # The agent that is already running must stop being a draft NOW.
    try:
        registry = getattr(agent_nick, "auto_registry", None)
        if registry is not None:
            registry.refresh_from_json()
        Orchestrator._load_agent_definitions.cache_clear()
    except Exception:
        logger.exception("publish_agent: in-process reload failed; a restart picks it up")

    return {"status": "published", "slug": slug, "stage": "published",
            "published_by": subject, "version": version}


@router.get("/usage")
async def agent_usage(since: str, until: str, principal=Depends(require_user)):
    """Tokens used by created agents in a period, by agent, model and person.

    ``since`` is included and ``until`` is not (ISO dates or date-times), so
    one month's report ends where the next begins. Admin only: this is what a
    customer is charged from, and it names who ran what.
    """
    _require_role(principal, "Admin", "reading agent usage")
    from datetime import datetime as _dt

    from repositories import agent_run_repo

    try:
        start, end = _dt.fromisoformat(since), _dt.fromisoformat(until)
    except ValueError:
        raise HTTPException(status_code=422, detail="since and until must be ISO dates")
    if end <= start:
        raise HTTPException(status_code=422, detail="until must be after since")

    rows = agent_run_repo.usage_summary(start, end)
    totals = {
        key: sum(r[key] for r in rows)
        for key in ("runs", "model_calls", "prompt_tokens", "completion_tokens")
    }
    return {"since": since, "until": until, "rows": rows, "totals": totals}


@router.get("/runs/{run_id}")
async def agent_run(run_id: str, principal=Depends(require_user)):
    """The record of one reasoning run: what the agent was told (its system
    prompt, as sent), which model answered, each tool call with what it was
    asked and what it returned, and the answer.

    Asked of the same gate as running an agent: the record holds what the tools
    returned, which is as much as running the agent would show.
    """
    gate("agent.run", principal, agent="AgentsRouter", context={"run_id": run_id})
    from repositories import agent_run_repo

    run = agent_run_repo.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"no recorded run {run_id!r}")
    return run


@router.get("/{slug}/runs")
async def agent_runs(slug: str, limit: int = 50, principal=Depends(require_user)):
    """The recorded runs of one agent, newest first. A summary of each — who,
    when, which model, how it ended; GET /agents/runs/{run_id} has the rest."""
    gate("agent.run", principal, agent="AgentsRouter", context={"slug": slug})
    from repositories import agent_run_repo

    return {"slug": slug, "runs": agent_run_repo.list_for_agent(slug, limit)}


class AgentExecutionRequest(BaseModel):
    agent_type: str
    payload: Dict[str, Any] = {}


class DocumentProcessRequest(BaseModel):
    s3_prefix: Optional[str] = None
    s3_object_key: Optional[str] = None


@router.post("/process-document")
def process_document(
    req: DocumentProcessRequest, orchestrator: Orchestrator = Depends(get_orchestrator),
    principal=Depends(require_user),
):
    """Convenience endpoint to run the document extraction workflow."""
    payload = {"s3_prefix": req.s3_prefix, "s3_object_key": req.s3_object_key}
    return orchestrator.execute_workflow(
        "document_extraction", payload,
        user_id=getattr(principal, "subject", None) or None)


@router.post("/execute")
def execute_agent(
    req: AgentExecutionRequest,
    orchestrator: Orchestrator = Depends(get_orchestrator),
    principal=Depends(require_user),
):
    """Execute a specified agent workflow."""
    if orchestrator.agent_nick is None:
        raise HTTPException(status_code=503, detail="Agent system not initialized")
    prs = orchestrator.agent_nick.process_routing_service
    if prs is None:
        raise HTTPException(status_code=503, detail="Process routing service unavailable")
    # A person started this, so the routing row names them. Without it the
    # service filled created_by with settings.script_user ("AgentNick").
    subject = getattr(principal, "subject", None) or None
    process_id = prs.log_process(
        process_name=req.agent_type,
        process_details=req.payload,
        created_by=subject,
        user_id=subject,
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
        result = orchestrator.execute_workflow(req.agent_type, req.payload, user_id=subject)
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
