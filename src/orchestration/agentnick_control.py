"""AgentNick's control loop — the thing that puts AgentNick actually in charge.

Before this, "AgentNick" was two unrelated things sharing a name: a
dependency-injection hub class in base_agent.py with no `run()` method, and the
Ollama model family `BeyondProcwise/AgentNick:{extract,unified}`. Nothing joined
them. The component that would have — `ReasoningEngine`'s plan/observe loop — was
constructed at startup (api/main.py) and then never invoked; `RoutingEngine.
evaluate_routing()` had zero callsites. Routing was done entirely by hardcoded
edge conditions in the workflow graphs, and the LLM never chose anything.

Here AgentNick drives, by calling tools. Three families are in scope:

  * **the agents** — every instantiable agent in the registry, advertised from its
    existing contract (`AutoRegistry.tool_schemas()`). AgentNick decides which to
    run and with what input.
  * **governance** — `get_policy` / `get_prompt` / `list_governance`, so the rules
    it applies are the governed ones in `proc.bp_policy` / `proc.bp_prompt`, not
    assumptions baked into a prompt.
  * **corpus facts** — the real spend/supplier/invoice/quote numbers, so answers
    about the data come from the data.

Every call and result is captured in the trace, so what AgentNick did — and what
it knew when it did it — is inspectable afterwards rather than taken on faith.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Dict, List, Optional

from services.tool_runtime import Tool, ToolRunResult, run_tools

log = logging.getLogger(__name__)


_SYSTEM = """You are AgentNick, the controller of the ProcWise procurement platform.

You do not answer procurement questions from memory. You establish facts by calling
tools, then answer from what they return.

Rules:
- To DO something (extract a document, rank suppliers, evaluate quotes, decide an
  approval, draft an email), call the matching run_<agent> tool. Do not describe
  what an agent would do — run it.
- Before applying any rule, weight, or threshold, fetch it with get_policy. Never
  assume a number. If a policy does not exist, say so; do not invent one.
- For questions about spend, suppliers, invoices, quotes, POs or findings, call
  get_corpus_facts. The figures must come from the corpus, not from your priors.
- If a tool returns nothing, say the data is not there. Do NOT fill the gap with a
  plausible-sounding value. A missing number is a fact; a made-up one is a defect.
- Be concise and concrete. Cite the figures the tools returned.

Arithmetic:
- NEVER add amounts denominated in different currencies. £101,120 and $73,839 do
  not sum to 174,959 of anything. Report each currency separately, on its own line,
  with its own symbol. Only combine them if a tool gives you an explicit converted
  figure — and then say which rate it used.
- Do not total a truncated or "top N" list and present it as the whole. If you need
  a total, use the total the tool returned; if it did not return one, say so.
- Report the figures as given. Do not round, rescale, or restate them in other units."""


def _agent_tools(agent_nick: Any) -> List[Tool]:
    """Wrap every instantiable agent as a callable tool."""
    registry = getattr(agent_nick, "auto_registry", None)
    agents = getattr(agent_nick, "agents", None)
    if registry is None or not agents:
        return []

    tools: List[Tool] = []
    for schema in registry.tool_schemas():
        fn = schema["function"]
        slug = fn["name"][len("run_") :]

        def _make(slug: str):
            def _run(**kwargs: Any) -> Dict[str, Any]:
                # The model may pass extra fields as a JSON blob; merge them in.
                payload: Dict[str, Any] = {
                    k: v for k, v in kwargs.items() if k != "payload_json"
                }
                blob = kwargs.get("payload_json")
                if blob:
                    try:
                        extra = json.loads(blob)
                        if isinstance(extra, dict):
                            payload.update(extra)
                    except Exception:  # noqa: BLE001
                        log.debug("payload_json was not valid JSON for %s", slug)

                agent = agents.get(slug)
                if agent is None:
                    return {"error": f"agent '{slug}' is not registered"}

                from agents.base_agent import AgentContext

                ctx = AgentContext(
                    workflow_id=uuid.uuid4().hex,
                    agent_id=slug,
                    user_id="AgentNick",
                    input_data=payload,
                )
                output = agent.run(ctx)
                data = dict(output.data or {})
                # The agentic_plan is narration for a human reader; feeding it back
                # into the loop just burns context the model needs for facts.
                data.pop("agentic_plan", None)
                return {
                    "status": getattr(output.status, "value", str(output.status)),
                    "data": data,
                    "error": output.error,
                }

            return _run

        tools.append(
            Tool(
                name=fn["name"],
                description=fn["description"],
                parameters=fn["parameters"],
                handler=_make(slug),
            )
        )
    return tools


def _governance_tools() -> List[Tool]:
    """The governed prompts and policies, as tools."""
    try:
        from services.governance_tools import tools as GT
    except Exception:  # noqa: BLE001
        return []

    def _policy(query: str) -> Any:
        """Fetch a governed policy; on a miss, say what DOES exist.

        A tool that returns an empty dict teaches the model nothing. AgentNick
        asked for 'approval_thresholds' (plural), missed the real
        'approval_threshold' by one character, and concluded no approval policy
        existed at all — a false negative that looked exactly like correct
        caution. A miss now hands back the catalogue so the next round can
        self-correct, instead of leaving the model to guess the exact slug.
        """
        hit = GT.get_policy(str(query))
        if hit:
            return hit

        # Retry once on the singular/plural of the query before giving up — the
        # single most common near-miss.
        q = str(query).strip().lower()
        for variant in (q.rstrip("s"), q + "s"):
            if variant and variant != q:
                hit = GT.get_policy(variant)
                if hit:
                    return hit

        catalogue = GT.list_governance(None) or {}
        available = [
            p.get("slug")
            for p in (catalogue.get("policies") or [])
            if p.get("slug")
        ]
        return {
            "found": False,
            "note": (
                f"No governed policy matches '{query}'. Retry with one of the "
                "names in 'available'. Do NOT invent a rule or a number."
            ),
            "available": available,
        }

    return [
        Tool(
            name="list_governance",
            description="List the governed prompts and policies available, optionally for one agent.",
            parameters={
                "type": "object",
                "properties": {"agent": {"type": "string"}},
            },
            handler=lambda agent=None: GT.list_governance(agent),
        ),
        Tool(
            name="get_policy",
            description=(
                "Fetch a governed policy — its rules, weights and thresholds — by "
                "type, slug, or the agent it applies to. Call this before applying "
                "ANY rule or number. If it reports found=false it also lists the "
                "policies that DO exist: retry with one of those."
            ),
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            handler=_policy,
        ),
        Tool(
            name="get_prompt",
            description="Fetch a governed prompt template by name, type, or agent.",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
            handler=lambda query: GT.get_prompt(str(query)),
        ),
    ]


def _corpus_tools(agent_nick: Any) -> List[Tool]:
    """The real procurement corpus, as a tool."""
    try:
        from services import corpus_facts
    except Exception:  # noqa: BLE001
        return []

    def _facts(query: str) -> Any:
        facts = corpus_facts.fetch_facts(agent_nick, str(query))
        if not facts:
            # Say so plainly. An empty result must not read as "no spend".
            return {"facts": None, "note": "no corpus facts matched this query"}
        return facts

    return [
        Tool(
            name="get_corpus_facts",
            description=(
                "Look up the REAL procurement corpus: total spend, suppliers, "
                "invoices, quotes, purchase orders, deals, findings and policies. "
                "Use this for any question about the data. Figures must come from "
                "here, never from memory."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "e.g. 'total spend', 'top suppliers', 'open findings'",
                    }
                },
                "required": ["query"],
            },
            handler=_facts,
        ),
    ]


def build_tools(agent_nick: Any) -> List[Tool]:
    """Every tool AgentNick can reach: the agents, governance, and the corpus."""
    return _agent_tools(agent_nick) + _governance_tools() + _corpus_tools(agent_nick)


def reason(
    agent_nick: Any,
    task: str,
    *,
    max_rounds: int = 6,
    require_tool_use: bool = True,
    extra_system: Optional[str] = None,
) -> ToolRunResult:
    """Let AgentNick plan and act on ``task`` by calling tools.

    ``require_tool_use`` defaults to True: for a grounded system, an answer
    produced without consulting a single tool is a guess, and the loop nudges once
    before accepting it.
    """
    tools = build_tools(agent_nick)
    system = _SYSTEM if not extra_system else f"{_SYSTEM}\n\n{extra_system}"
    result = run_tools(
        task,
        tools,
        system,
        max_rounds=max_rounds,
        require_tool_use=require_tool_use,
    )
    log.info(
        "AgentNick.reason rounds=%s tools=%s error=%s",
        result.rounds,
        result.tools_used,
        result.error,
    )
    return result
