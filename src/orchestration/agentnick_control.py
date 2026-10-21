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

import contextvars
import json
import logging
import os
import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple

from services.tool_runtime import Tool, ToolRunResult, run_tools

log = logging.getLogger(__name__)


_SYSTEM = """You are AgentNick, the controller of the ProcWise procurement platform.

You do not answer procurement questions from memory. You establish facts by calling
tools, then answer from what they return.

Rules:
- To DO something (extract a document, rank suppliers, evaluate quotes, decide an
  approval, draft an email), call the matching run_<agent> tool. Do not describe
  what an agent would do — run it.
- To find out whether a purchase, PO, award, contract or payment is allowed, or who must
  approve it, call check_policy. It reads the approval policy and returns the verdict;
  do not work the verdict out from get_policy yourself.
- For a bigger job (a negotiation plan, an Incoterm choice, validating a saving, a full
  policy check), call list_skills and read the matching skill with use_skill first,
  and follow it.
- Before applying any other rule, weight, or threshold, fetch it with get_policy. Never
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


# The created agents currently reasoning on this thread, outermost first. One
# created agent may be asked from inside a run; the agent it was asked from
# cannot start another. Each loop is minutes of model time, and a chain nobody
# drew on a canvas has no natural end.
MAX_CREATED_AGENT_DEPTH = 1
_CREATED_AGENT_CHAIN: contextvars.ContextVar[Tuple[str, ...]] = contextvars.ContextVar(
    "created_agent_chain", default=()
)


def _agent_tools(
    agent_nick: Any,
    *,
    workflow_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> List[Tool]:
    """Wrap every instantiable agent as a callable tool.

    ``workflow_id`` / ``user_id`` tie the agent's routing rows to the run that
    called it (a workflow node reasoning with tools); without them each call
    gets its own id, as the Ask path always has.
    """
    registry = getattr(agent_nick, "auto_registry", None)
    agents = getattr(agent_nick, "agents", None)
    if registry is None or not agents:
        return []

    tools: List[Tool] = []
    for schema in registry.tool_schemas():
        fn = schema["function"]
        slug = fn["name"][len("run_") :]
        # A draft agent is not on offer to anyone but the person who created it.
        owner = getattr(agents.get(slug), "draft_owner", None)
        if owner is not None and (not user_id or user_id != owner):
            continue

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

                # A created agent is its task: run as the class it was copied
                # from, it would do that class's job and ignore what a person
                # wrote for it. So it reasons here exactly as it does on the canvas.
                created = bool(getattr(agent, "governance_slug", None))
                chain = _CREATED_AGENT_CHAIN.get()
                if created:
                    if slug in chain:
                        return {"error": f"agent '{slug}' cannot call itself"}
                    if len(chain) >= MAX_CREATED_AGENT_DEPTH:
                        return {
                            "error": (
                                f"agent '{slug}' was not run: this is too many created "
                                "agents deep. Answer with what you already have."
                            )
                        }

                from agents.base_agent import AgentContext

                ctx = AgentContext(
                    workflow_id=workflow_id or uuid.uuid4().hex,
                    agent_id=slug,
                    user_id=user_id,
                    input_data=payload,
                )
                # execute(), not run(): a tool call is an agent run like any other
                # and must leave the same proc.routing / bp_action record.
                if created:
                    token = _CREATED_AGENT_CHAIN.set(chain + (slug,))
                    try:
                        output = agent.execute(ctx, tool_loop=True)
                    finally:
                        _CREATED_AGENT_CHAIN.reset(token)
                else:
                    output = agent.execute(ctx)
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


def _with_check_policy_note(policy: Dict[str, Any]) -> Dict[str, Any]:
    """The approval limit is applied by check_policy, not by the model reading its numbers."""
    if isinstance(policy, dict) and policy.get("slug") == "approval_threshold_policy":
        return {**policy, "note": ("To check a request against this policy, call check_policy with the "
                                   "facts of the request. It applies the limit and returns the verdict; "
                                   "do not work the verdict out from these numbers yourself.")}
    return policy


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
            return _with_check_policy_note(hit)

        # Retry once on the singular/plural of the query before giving up — the
        # single most common near-miss.
        q = str(query).strip().lower()
        for variant in (q.rstrip("s"), q + "s"):
            if variant and variant != q:
                hit = GT.get_policy(variant)
                if hit:
                    return _with_check_policy_note(hit)

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


_MINER_SLUG = "opportunity_miner"


def _opportunity_tools(
    agent_nick: Any,
    *,
    workflow_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> List[Tool]:
    """The saving levers, as tools: what exists, what a bucket holds, and try one.

    The levers are the detection rules in proc.bp_rule, each bound to a calculation
    in the Opportunity Miner. An agent chooses which to apply to a supplier and item
    and explains the result; every figure comes from the calculation, never from
    the model.
    """
    agents = getattr(agent_nick, "agents", None) or {}
    miner = agents.get(_MINER_SLUG)
    if miner is None:
        return []

    def _catalogue() -> Dict[str, Dict[str, Any]]:
        return miner._get_policy_registry()

    def _levers() -> Any:
        try:
            registry = _catalogue()
        except Exception as exc:  # noqa: BLE001 - e.g. the rule book is unreadable
            # Not an empty list: "no levers" and "could not read them" differ.
            return {"levers": None, "note": f"the lever catalogue could not be read: {exc}"}
        return {
            "levers": [
                {
                    "lever": slug,
                    "name": cfg.get("policy_name"),
                    "kind": cfg.get("finding_type"),
                    "looks_at": cfg.get("scope"),
                    "needs": list(cfg.get("required_fields") or []),
                    "thresholds": dict(cfg.get("default_conditions") or {}),
                }
                for slug, cfg in registry.items()
            ],
            "note": (
                "Use evaluate_opportunity_lever to apply one to a supplier and item. "
                "A lever's saving comes from its calculation; do not estimate one."
            ),
        }

    def _bucket(supplier_id: str, item_id: str) -> Any:
        from src.services import opportunity_buckets

        found = opportunity_buckets.get_bucket(str(supplier_id), str(item_id))
        if found is None:
            return {
                "bucket": None,
                "note": (
                    "No lever with a unit price and a volume is recorded for this "
                    "supplier and item. That is not the same as 'no saving exists'."
                ),
            }
        return {"bucket": found}

    def _evaluate(lever: str, supplier_id: str, item_id: Optional[str] = None) -> Any:
        try:
            registry = _catalogue()
        except Exception as exc:  # noqa: BLE001
            return {"error": f"the lever catalogue could not be read: {exc}"}
        if lever not in registry:
            return {
                "found": False,
                "note": f"No lever is called '{lever}'. Retry with one of 'available'.",
                "available": sorted(registry),
            }
        from agents.base_agent import AgentContext

        conditions: Dict[str, Any] = {"supplier_id": str(supplier_id)}
        if item_id:
            conditions["item_id"] = str(item_id)
        ctx = AgentContext(
            workflow_id=workflow_id or uuid.uuid4().hex,
            agent_id=_MINER_SLUG,
            user_id=user_id,
            # One lever, one supplier (and item): a scoped run, so it retires nothing.
            input_data={"workflow": lever, "conditions": conditions},
        )
        output = miner.execute(ctx)
        data = dict(output.data or {})
        findings = data.get("findings") or []
        return {
            "lever": lever,
            "status": getattr(output.status, "value", str(output.status)),
            "error": output.error,
            "found_a_saving": bool(findings),
            "results": [
                {
                    "saving_gbp": f.get("financial_impact_gbp"),
                    "supplier_id": f.get("supplier_id"),
                    "item_id": f.get("item_id"),
                    "working": f.get("calculation_details"),
                    "evidence": f.get("source_records"),
                    "previously_rejected": bool(f.get("is_rejected")),
                }
                for f in findings
            ],
            # Why nothing fired, in the miner's own words (a missing input, a
            # threshold not met) -- so "no saving" is never a bare empty list.
            "events": (data.get("policy_events") or [])[-5:],
        }

    return [
        Tool(
            name="list_opportunity_levers",
            description=(
                "List the saving levers that exist (price against benchmark, volume "
                "consolidation, overbilling and so on), what each one needs and its "
                "thresholds. Call this before choosing levers for a supplier and item."
            ),
            parameters={"type": "object", "properties": {}},
            handler=_levers,
        ),
        Tool(
            name="get_opportunity_bucket",
            description=(
                "Fetch the opportunity bucket for one supplier and item: the unit "
                "price, the volume, and every lever already found on it with its "
                "saving, its stage, and whether new information is waiting on a "
                "rejected lever. The bucket's saving is its largest open lever; "
                "levers overlap and must not be added together."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "supplier_id": {"type": "string"},
                    "item_id": {"type": "string"},
                },
                "required": ["supplier_id", "item_id"],
            },
            handler=_bucket,
        ),
        Tool(
            name="evaluate_opportunity_lever",
            description=(
                "Apply ONE saving lever to a supplier (and item) and return the "
                "computed saving with its working and evidence. The lever name must "
                "come from list_opportunity_levers. The result is recorded on the "
                "supplier and item's bucket; a lever a person rejected stays rejected "
                "and is marked for review if the evidence has changed."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "lever": {"type": "string"},
                    "supplier_id": {"type": "string"},
                    "item_id": {"type": "string"},
                },
                "required": ["lever", "supplier_id"],
            },
            handler=_evaluate,
        ),
    ]


def _calculation_tools(*, workflow_id: Optional[str] = None) -> List[Tool]:
    """The registered formulas, as tools: what can be calculated, and run one.

    For the calculation nobody could schedule in advance — a "what if" in a
    conversation, a percentage an agent needs part-way through. A calculation an
    agent must ALWAYS have is not left to these: it is declared on the agent and
    run before it reasons (``BaseAgent.run_tool_loop``).
    """
    try:
        from services import calculation_tools as CT
    except Exception:  # noqa: BLE001
        log.exception("calculation tools unavailable")
        return []

    return [
        Tool(
            name="list_calculations",
            description=(
                "List the exact calculations that exist (scores, savings, price "
                "variances, negotiation positions and so on). With no argument: "
                "every calculation by name and purpose. With a family or a name: "
                "the inputs it needs. Call this before run_calculation."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "family": {"type": "string",
                               "description": "e.g. 'negotiation', 'supplier', 'deal'"},
                    "name": {"type": "string",
                             "description": "One calculation, e.g. 'deal.pct_change'"},
                },
            },
            handler=lambda family=None, name=None: CT.list_calculations(family, name),
        ),
        Tool(
            name="run_calculation",
            description=(
                "Run ONE exact calculation and return its value. Use this instead of "
                "doing arithmetic yourself. The name and the input names must come "
                "from list_calculations, and every input value must be one a tool "
                "returned or the task stated — never one you estimated. If it "
                "reports assessed=false, say the figure could not be calculated and "
                "why; do not supply one."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "inputs": {"type": "object",
                               "description": "Input names and values for the calculation."},
                    "rows": {"type": "array", "items": {"type": "object"},
                             "description": ("Only for a calculation that scores a whole "
                                             "set: one object per member.")},
                },
                "required": ["name"],
            },
            handler=lambda name=None, inputs=None, rows=None: CT.run_calculation(
                name, inputs, rows, trace_id=workflow_id),
        ),
    ]


def _skill_tools() -> List[Tool]:
    """Skills: playbooks the agent reads before a job. Reading one runs nothing."""
    from src.services import skills as SK

    def _refuse(note: str) -> Dict[str, Any]:
        return {"found": False, "note": note, "available": [s.name for s in SK.list_skills()]}

    def _list() -> Dict[str, Any]:
        return {"skills": [{"name": s.name, "description": s.description} for s in SK.list_skills()]}

    def _use(name: Any = None, part: Any = 1) -> Dict[str, Any]:
        if not isinstance(name, str):
            return _refuse("Give the skill name as text. Pick one from 'available'.")
        try:
            skill = SK.load_skill(name)
        except SK.SkillNotFound:
            return _refuse(f"No skill named '{name}'. Pick one from 'available'.")
        except SK.SkillInvalid as exc:
            return _refuse(f"Skill '{name}' is not usable: {exc}")
        pieces = SK.split_parts(skill.body)
        try:
            number = int(part)
        except (TypeError, ValueError):
            number = 0
        if not 1 <= number <= len(pieces):
            return _refuse(f"'{name}' has parts 1 to {len(pieces)}; part {part!r} does not exist.")
        out: Dict[str, Any] = {"found": True, "name": skill.name, "part": number,
                               "parts_total": len(pieces), "playbook": pieces[number - 1]}
        if number == 1:
            out["config"] = skill.config
            out["references_available"] = sorted(skill.references)
        if len(pieces) > 1:
            out["note"] = (f"This is part {number} of {len(pieces)}. Read all parts, in order, "
                           "before acting: call use_skill again with the next part number.")
        return out

    return [
        Tool(
            name="list_skills",
            description="List the skills (packaged playbooks) available, with when to use each.",
            parameters={"type": "object", "properties": {}},
            handler=_list,
        ),
        Tool(
            name="use_skill",
            description=(
                "Read a skill's playbook and its settings before doing the job it covers "
                "(policy checks, Incoterms, negotiation plans, savings validation). A skill is "
                "advice: it does not run anything, and every tool call, number and approval it "
                "mentions still has to come from the real tools and the approval flow. If it "
                "lists a tool as NOT AVAILABLE, say so; do not work around it with a guess."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "A name from list_skills."},
                    "part": {"type": "integer", "description": "Which part to read; start at 1."},
                },
                "required": ["name"],
            },
            handler=_use,
        ),
    ]


def _policy_check_tools() -> List[Tool]:
    """The procurement policy verdict, computed in code from the governed policy.

    The model supplies the facts of the request. It never supplies the limit, the role that
    approves or the verdict: those come from the policy row read here. Any failure to read
    governance gives "insufficient information", never a pass.
    """
    from src.services import policy_check as PC
    from src.services import skills as SK

    def _check(total_commitment: Any = None, currency: Any = None, annual_value: Any = None,
               term_months: Any = None, orders: Any = None, history_checked: Any = None,
               requester: Any = None, approver: Any = None, waiver_id: Any = None,
               fx_rate_to_policy_currency: Any = None, action_type: Any = None) -> Dict[str, Any]:
        request = {
            "total_commitment": total_commitment, "currency": currency, "annual_value": annual_value,
            "term_months": term_months, "orders": orders if isinstance(orders, list) else [],
            "history_checked": bool(history_checked), "requester": requester, "approver": approver,
            "waiver_id": waiver_id, "fx_rate_to_policy_currency": fx_rate_to_policy_currency,
            "action_type": action_type,
        }
        try:
            config = SK.load_skill("procurement-policy-compliance").config
        except Exception as exc:  # noqa: BLE001
            log.warning("check_policy: skill config unavailable: %s", exc)
            return PC.evaluate.evaluate(request, {"available": False}, {}, now=None)
        governance: Dict[str, Any] = {"available": False}
        try:
            from services.governance_tools import tools as GT

            GT.refresh(strict=True)
            if GT.active_policy_count() > 0:
                slug = (config.get("governed_policies") or {}).get("approval_limit", "approval_threshold_policy")
                catalogue = GT.list_governance(None) or {}
                row = GT.get_policy(slug) or None
                if row and row.get("slug") == slug:
                    governance = {"available": True, "approval_policy": row,
                                  "slugs": [p.get("slug") for p in (catalogue.get("policies") or [])]}
        except Exception as exc:  # noqa: BLE001 - an unreadable store is an outage, not "no rules"
            log.warning("check_policy: governance unreadable: %s", exc)
            governance = {"available": False}
        return PC.evaluate.evaluate(request, governance, config)

    return [
        Tool(
            name="check_policy",
            description=(
                "Check a procurement action (requisition, PO, award, contract, payment) against the "
                "governed approval policy and get a verdict, the required approver and any split-order or "
                "self-approval problem. Pass only the facts of the request: total_commitment over the whole "
                "term including renewals, currency, any orders (each {amount, source: planned or history}), "
                "requester and approver only if the user stated them. List related orders you found or the user mentioned; the split check covers only those. "
                "The limit, the approving role and the verdict come from the policy, never from you. Relay "
                "reply_text to the user as it is: do not change a number, add a policy version, name an "
                "approver, or call it approved. If it says information is needed, ask for exactly that."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "total_commitment": {"type": "number", "description": "Total over the full term, not the annual value."},
                    "currency": {"type": "string", "description": "ISO 4217 code, e.g. GBP."},
                    "orders": {"type": "array", "items": {"type": "object"},
                               "description": "Related orders: {amount, source: planned|history}."},
                    "history_checked": {"type": "boolean", "description": "Ignored; list the orders instead."},
                    "requester": {"type": "string"}, "approver": {"type": "string"},
                    "fx_rate_to_policy_currency": {"type": "number", "description": "Only if the user gave one."},
                    "waiver_id": {"type": "string"},
                    "action_type": {"type": "string"},
                    "annual_value": {"type": "number"}, "term_months": {"type": "integer"},
                },
            },
            handler=_check,
        ),
    ]


def _contract_tools() -> List[Tool]:
    """Find contracts, read their stored clauses, compare two wordings. All read-only."""
    from src.services.contract_tools import clauses as CL
    from src.services.contract_tools import redline as RL
    from src.services.contract_tools import search as SE

    return [
        Tool(
            name="search_contracts",
            description=(
                "Find contracts in the register by words in the title or type, supplier name, status, or "
                "expiring within N days. Returns dates, value, auto-renewal, renewal term, governing law and "
                "payment terms as stored. Covers only contracts that were uploaded: no match does not mean "
                "no contract exists. If it says the register could not be read, say so."
            ),
            parameters={"type": "object", "properties": {
                "query": {"type": "string"}, "supplier": {"type": "string"}, "status": {"type": "string"},
                "expires_within_days": {"type": "integer"}, "limit": {"type": "integer"}}},
            handler=SE.search,
        ),
        Tool(
            name="extract_clauses",
            description=(
                "Read the clauses already extracted from one uploaded contract, quoted word for word with "
                "clause number, type and parties. Pass the document_id (or contract_id). Optionally filter by "
                "clause_type (must_perform, entitled_to, triggered_by, penalised_by, risk_transfer) or by words "
                "the clause contains. Quote clauses as returned; a clause not listed may still be in the "
                "contract, so say 'not checked', not 'absent'."
            ),
            parameters={"type": "object", "properties": {
                "document_id": {"type": "string"}, "clause_type": {"type": "string"},
                "containing": {"type": "string"}}, "required": ["document_id"]},
            handler=CL.extract,
        ),
        Tool(
            name="redline_compare",
            description=(
                "Compare two wordings of a clause or contract and list what was added, removed or replaced, in "
                "reading order. Pass both texts exactly as the user gave them. It reports wording differences "
                "only; it does not say whether a change is acceptable."
            ),
            parameters={"type": "object", "properties": {
                "original": {"type": "string"}, "revised": {"type": "string"}},
                "required": ["original", "revised"]},
            handler=RL.compare,
        ),
    ]


def _ledger_tools() -> List[Tool]:
    """Savings as the value ledger states them. Read-only: logging a saving is a person's act."""
    from src.services import savings_tracker as ST

    return [
        Tool(
            name="savings_tracker",
            description=(
                "Read what has been saved so far from the value ledger: saved (avoided + recovered + "
                "realised), money still only claimed, and a month-by-month split, optionally between two "
                "dates. It is the same figure the dashboards show. It logs nothing; do not say a saving "
                "was recorded. If the ledger cannot be read, say so rather than reporting zero."
            ),
            parameters={"type": "object", "properties": {
                "since": {"type": "string", "description": "YYYY-MM-DD, optional."},
                "until": {"type": "string", "description": "YYYY-MM-DD, optional."}}},
            handler=ST.track,
        ),
    ]


def _spend_tools() -> List[Tool]:
    """Invoiced spend cut by one dimension. Read-only; same figures as the dashboards."""
    from src.services.analytics import spend_cube as SC

    return [
        Tool(
            name="build_spend_cube",
            description=(
                "Invoiced spend cut by ONE dimension for a period: by = supplier, month, currency, country, "
                "region, buyer, payment_terms or status. Defaults to the fiscal year to date and GBP. Returns the "
                "total, invoice count, and the top cells with share and as-billed amounts; the total equals the "
                "dashboard. There is no category or business unit. A currency with no rate is left out and "
                "named; quote that. If spend cannot be read, say so rather than reporting zero."
            ),
            parameters={"type": "object", "properties": {
                "by": {"type": "string", "enum": list(SC.DIMENSIONS)},
                "start": {"type": "string", "description": "YYYY-MM-DD, optional."},
                "end": {"type": "string", "description": "YYYY-MM-DD, optional."},
                "currency": {"type": "string", "description": "ISO code, default GBP."},
                "top_n": {"type": "integer"}}, "required": ["by"]},
            handler=SC.build,
        ),
    ]


def _supplier_tools() -> List[Tool]:
    """What the platform itself knows about suppliers. Read-only; no bank details or personal contacts."""
    from src.services import supplier_tools as ST

    return [
        Tool(
            name="supplier_discovery",
            description=(
                "List suppliers this organisation already deals with that have a PO, invoice or quote line "
                "matching an item or service, with the evidence counts, preferred flag, country, certifications "
                "on file and insurance status. It is NOT a market scan: suppliers it does not return may exist. "
                "Never present a result as a complete market view."
            ),
            parameters={"type": "object", "properties": {
                "item": {"type": "string", "description": "The item or service, e.g. 'laptop'."},
                "country": {"type": "string"}, "preferred_only": {"type": "boolean"},
                "limit": {"type": "integer"}}, "required": ["item"]},
            handler=ST.discover,
        ),
        Tool(
            name="supplier_lookup",
            description=(
                "A supplier's recorded profile by name or supplier_id: legal details, preferred flag, risk "
                "score (or 'not scored'), certifications on file, insurance, currency, Incoterms, lead time. "
                "It is the master record, not an outside check. If several suppliers match, ask which one."
            ),
            parameters={"type": "object", "properties": {"supplier": {"type": "string"}},
                        "required": ["supplier"]},
            handler=ST.lookup,
        ),
    ]


def build_tools(
    agent_nick: Any,
    *,
    exclude: Sequence[str] = (),
    only: Optional[Sequence[str]] = None,
    workflow_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> List[Tool]:
    """Every tool AgentNick can reach: the agents, governance, the corpus, the
    saving levers and the registered calculations.

    ``exclude`` names tools to leave out — a workflow node's loop, for one,
    must never be handed the tool that sends real email.

    ``only`` is the list of tools an agent was given. ``None`` means no list was
    ever set; an empty list means none at all, which is why it is not tested for
    truth. It narrows and never widens: a name in ``exclude`` stays out even
    when it is also in ``only``.
    """
    tools = (
        _agent_tools(agent_nick, workflow_id=workflow_id, user_id=user_id)
        + _governance_tools()
        + _corpus_tools(agent_nick)
        + _opportunity_tools(agent_nick, workflow_id=workflow_id, user_id=user_id)
        + _calculation_tools(workflow_id=workflow_id)
        + _skill_tools()
        + _policy_check_tools()
        + _contract_tools()
        + _ledger_tools()
        + _spend_tools()
        + _supplier_tools()
    )
    if exclude:
        blocked = set(exclude)
        tools = [t for t in tools if t.name not in blocked]
    if only is not None:
        given = set(only)
        tools = [t for t in tools if t.name in given]
    return tools


def _on(switch: str) -> bool:
    """A kill switch: on unless the environment says ``off``."""
    return os.getenv(switch, "on").strip().lower() != "off"


def _plan_context(task: str, tools: Sequence[Tool], extra_system: Optional[str]):
    """Route the request to a skill if one clearly covers it, and build the prompt in layers.

    Returns ``(context, skill_settings)``. It never raises: a router or loader fault just means no
    skill is injected, and the model can still read one itself. A skill is injected only when this
    agent was given every tool the skill tells it to call.
    """
    from . import context_builder, skill_router

    skill = None
    if _on("AGENT_SKILL_ROUTER") and tools:
        try:
            from src.services import skills as SK

            match = skill_router.Router(skill_router.load_entries()).route(task)
            if match is not None:
                loaded = SK.load_skill(match.skill)
                fits, why = context_builder.can_inject(loaded, {t.name for t in tools})
                if fits:
                    skill = loaded
                else:
                    log.info("skill %s not injected: %s", match.skill, why)
        except Exception:  # noqa: BLE001
            log.exception("skill routing failed; carrying on without a skill")
    ctx = context_builder.build(base=_SYSTEM, extra=extra_system, skill=skill)
    if ctx.skipped:
        log.info("skill not injected: %s", ctx.skipped)
    return ctx, (dict(skill.config) if ctx.skill and skill is not None else None)


def _verifier(task: str, skill_settings: Optional[Dict[str, Any]]):
    """The verify step for one run: what it checks is in ``orchestration.verifier``."""
    from . import verifier

    def _verify(answer: str, result: Any, retried: bool):
        return verifier.verify(answer, result.calls, task, skill_config=skill_settings, retried=retried)

    return _verify


def _argument_checker(task: str, skill_settings: Optional[Dict[str, Any]]):
    """Step 3b: refuse a tool call whose assumptions (rates, horizons, thresholds) nobody supplied."""
    from . import argument_check

    return argument_check.make_checker(task, skill_settings)


def reason(
    agent_nick: Any,
    task: str,
    *,
    max_rounds: int = 6,
    require_tool_use: bool = True,
    extra_system: Optional[str] = None,
    exclude: Sequence[str] = (),
    only: Optional[Sequence[str]] = None,
    workflow_id: Optional[str] = None,
    user_id: Optional[str] = None,
    agent: Optional[str] = None,
    model: Optional[str] = None,
) -> ToolRunResult:
    """Let AgentNick plan and act on ``task`` by calling tools.

    ``model`` is the model the calling agent was given, if it was given one.
    ``None`` is the standard model. It used to be impossible to pass: an agent's
    model was validated and stored at creation and then ignored here.

    ``agent`` names the calling agent (its slug) for the agent-policy gate, which sees every
    tool call with the agent, workflow and user it belongs to.

    ``require_tool_use`` defaults to True: for a grounded system, an answer
    produced without consulting a single tool is a guess, and the loop nudges once
    before accepting it.
    """
    tools = build_tools(
        agent_nick, exclude=exclude, only=only, workflow_id=workflow_id, user_id=user_id
    )
    if not tools:
        # An agent given no tools has nothing to consult; insisting that it
        # consult something would only buy a wasted round and a nudge.
        require_tool_use = False
    context, skill_settings = _plan_context(task, tools, extra_system)
    if context.skill:
        # The playbook is already in the prompt, so reading it again would only spend rounds.
        tools = [t for t in tools if t.name not in ("list_skills", "use_skill")]
    system = context.system
    result = run_tools(
        task,
        tools,
        system,
        max_rounds=max_rounds,
        require_tool_use=require_tool_use,
        model=model,
        agent=agent,
        workflow_id=workflow_id,
        user_id=user_id,
        verify=_verifier(task, skill_settings) if _on("AGENT_VERIFY") else None,
        check_arguments=_argument_checker(task, skill_settings) if _on("AGENT_VERIFY") else None,
    )
    result.context = context.describe()
    # The record of a run needs what it was actually given. The loop fills
    # these; anything that stands in for the loop may not.
    if not result.system:
        result.system = system
    if not result.tools_offered:
        result.tools_offered = [t.name for t in tools]
    log.info(
        "AgentNick.reason rounds=%s tools=%s error=%s layers=%s verification=%s",
        result.rounds,
        result.tools_used,
        result.error,
        [l["layer"] for l in result.context],
        result.verification,
    )
    return result
