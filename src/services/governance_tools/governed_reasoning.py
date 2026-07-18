"""Governed reasoning: AgentNick tool-use loop over the policy & prompt engines.

AgentNick pulls the applicable governed prompt/policy at runtime via tool-calls and
applies them, so its reasoning is governed dynamically and traceable. Read-only;
bounded; never raises to the caller. AgentNick stays the base model.
"""
from __future__ import annotations

import json
import logging
import os
import re

import requests

from src.services.governance_tools import tools as GT

log = logging.getLogger(__name__)

_OLLAMA_CHAT = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/") + "/api/chat"
_MODEL = os.getenv("GOVERNED_REASONING_MODEL", "BeyondProcwise/AgentNick:unified")
_MAX_ROUNDS = int(os.getenv("GOVERNED_REASONING_MAX_ROUNDS", "5"))

_TOOLS = [
    {"type": "function", "function": {
        "name": "list_governance",
        "description": "List the governed prompts and policies available (optionally for one agent).",
        "parameters": {"type": "object", "properties": {"agent": {"type": "string"}}}}},
    {"type": "function", "function": {
        "name": "get_policy",
        "description": "Fetch a governed policy (rules/weights/thresholds) by type, slug, or agent.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "get_prompt",
        "description": "Fetch a governed prompt template by name, type, or agent.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    # The platform's model of itself, held in the knowledge graph. Retrieved on demand,
    # deliberately NOT baked into the system prompt: the platform description is a few
    # thousand tokens, and in a SYSTEM block it would be charged on every call and eat a
    # third of an 8k context -- slowing every extraction to teach the model something
    # extraction does not need. Here it costs nothing until it is asked for.
    {"type": "function", "function": {
        "name": "describe_platform",
        "description": (
            "Look up how ProcWise itself works: the ingest pipeline and its stages, the "
            "promotion gates, the agents and the tables they read/write, the SpendIQ "
            "screens and the endpoints behind them, and the known gaps. Use this whenever "
            "a question is about the SYSTEM (how a document flows, which agent decides "
            "what, where a buyer sees something, what is broken) rather than about the "
            "procurement data itself."
        ),
        "parameters": {"type": "object", "properties": {
            "topic": {"type": "string",
                      "description": "e.g. 'upload', 'promotion', 'quote', 'supplier ranking', 'invoices screen'"}},
            "required": ["topic"]}}},
]

_SYSTEM = (
    "You are a procurement reasoning assistant whose prompts and policies are GOVERNED in a "
    "database. Do NOT rely on assumptions about rules or weights. For the task: FIRST call "
    "list_governance (optionally with the relevant agent) to see what governance exists, then "
    "call get_policy and/or get_prompt to fetch the applicable governed policy (rules/weights/"
    "thresholds) and prompt template, and APPLY them in your answer. Only reference governance "
    "you actually fetched via the tools; never name a policy or prompt you did not fetch. After "
    "your concise governed answer, end with a single final line, exactly:\n"
    'CITED: {"policies": ["<slug>", ...], "prompts": ["<prompt_name>", ...]}\n'
    "listing ONLY the governance you fetched with the tools (empty lists if none)."
)

_NUDGE = (
    "You answered without consulting the governed rules. First call list_governance and the "
    "relevant get_policy / get_prompt to fetch the applicable governance, then give your "
    "governed answer with the final CITED line."
)


def _split_citations(content: str) -> tuple[str, dict]:
    """Split a final answer into (prose, declared_citations).

    The model is asked to end with `CITED: {json}`. Parse it robustly; on any
    failure treat the whole content as prose with no declared citations.
    """
    cited = {"policies": [], "prompts": []}
    text = content or ""
    m = re.search(r"CITED\s*:", text, re.IGNORECASE)
    if not m:
        return text.strip(), cited
    prose = text[: m.start()].strip()
    tail = text[m.end():]
    brace = tail.find("{")
    if brace != -1:
        try:
            obj = json.loads(tail[brace: tail.rfind("}") + 1])
            if isinstance(obj, dict):
                for key in ("policies", "prompts"):
                    vals = obj.get(key) or []
                    if isinstance(vals, list):
                        cited[key] = [str(v) for v in vals if str(v).strip()]
        except Exception:  # noqa: BLE001
            pass
    return prose, cited


def _norm(s: str) -> str:
    """Lowercase and fold underscores/whitespace so `foo_bar_policy` and
    `Foo Bar Policy` compare equal."""
    return re.sub(r"[\s_]+", " ", str(s or "").lower()).strip()


def _prose_unfetched(prose: str, catalog: dict, fetched_policies: set,
                     fetched_prompts: set, seen_policies: set | None = None,
                     seen_prompts: set | None = None) -> dict:
    """Governance names that appear in the prose but were never fetched.

    Catches the "honest CITED, fabricated prose" case the citation cross-check
    misses. Conservative to avoid false positives: only DISTINCTIVE identifiers
    (2+ tokens) are scanned, matched as a whole normalized phrase; fetched names
    are never flagged even when named in prose.
    """
    norm_prose = _norm(prose)
    hits = {"policies": [], "prompts": []}
    for slug in catalog.get("policies", []):
        if str(slug).lower() in fetched_policies:
            continue
        if str(slug).lower() in (seen_policies or set()):
            continue
        n = _norm(slug)
        if len(n.split()) >= 2 and n in norm_prose:
            hits["policies"].append(slug)
    for name in catalog.get("prompts", []):
        if str(name).lower() in fetched_prompts:
            continue
        if str(name).lower() in (seen_prompts or set()):
            continue
        n = _norm(name)
        if len(n.split()) >= 2 and n in norm_prose:
            hits["prompts"].append(name)
    return hits


def _dedupe(values: list) -> list:
    """Order-preserving, case-insensitive de-duplication."""
    seen, out = set(), []
    for v in values:
        k = str(v).lower()
        if k not in seen:
            seen.add(k)
            out.append(v)
    return out


def _finalize(content: str, used: dict, fetched_policies: set, fetched_prompts: set,
              catalog: dict, rounds: int, listed_policies: set | None = None,
              listed_prompts: set | None = None) -> dict:
    """Ground the answer: cross-check declared citations AND scan the prose,
    both against what was actually fetched."""
    answer, cited = _split_citations(content)
    prose_hits = _prose_unfetched(answer, catalog, fetched_policies, fetched_prompts,
                                  listed_policies, listed_prompts)
    # "Unsupported" means the name came from nowhere: neither fetched in full via
    # get_policy/get_prompt nor seen in a list_governance catalog listing.
    ok_policies = fetched_policies | (listed_policies or set())
    ok_prompts = fetched_prompts | (listed_prompts or set())
    unsupported = {
        "policies": _dedupe([c for c in cited["policies"] if c.lower() not in ok_policies]
                            + prose_hits["policies"]),
        "prompts": _dedupe([c for c in cited["prompts"] if c.lower() not in ok_prompts]
                           + prose_hits["prompts"]),
    }
    if unsupported["policies"] or unsupported["prompts"]:
        log.warning("governed_reasoning ungrounded citations dropped: %s", unsupported)
    return {"answer": answer, "governance_used": used,
            "unsupported": unsupported, "rounds": rounds}


def _audit(result: dict, task: str, agent: str | None) -> dict:
    """Log the governed-reasoning run to proc.bp_agent_actions. Best-effort:
    record_action never raises, so a logging problem cannot break the answer."""
    try:
        from src.services.agent_actions import record_action

        gov = result.get("governance_used") or {}
        unsupported = result.get("unsupported") or {}
        grounded = not (unsupported.get("policies") or unsupported.get("prompts"))
        record_action(
            phase="governance",
            action_type="governed_reasoning",
            agent=agent or "agentnick",
            status="success" if grounded else "ungrounded_citations",
            summary=f"governed reasoning: {str(task)[:180]}",
            details={"task": str(task)[:500], "agent": agent,
                     "governance_used": gov, "unsupported": unsupported,
                     "rounds": result.get("rounds")},
        )
    except Exception:  # noqa: BLE001 - auditing must never break the caller
        log.debug("governed_reasoning audit failed", exc_info=True)
    return result


def _chat(messages: list[dict]) -> dict:
    r = requests.post(_OLLAMA_CHAT, json={
        "model": _MODEL, "messages": messages, "tools": _TOOLS,
        "stream": False, "think": False, "keep_alive": -1,
        "options": {"temperature": 0, "num_predict": 2048},
    }, timeout=180)
    r.raise_for_status()
    return r.json().get("message", {})


def govern(task: str, agent: str | None = None) -> dict:
    """Run AgentNick over a task with governance tools. Returns answer + governance_used."""
    GT.refresh()
    # Full catalog of governance that exists — the vocabulary the prose scan
    # checks against. Fail-open to empty (scan then no-ops).
    try:
        cat = GT.list_governance(None) or {}
        catalog = {
            "policies": [p.get("slug") for p in (cat.get("policies") or []) if p.get("slug")],
            "prompts": [p.get("prompt_name") for p in (cat.get("prompts") or []) if p.get("prompt_name")],
        }
    except Exception:  # noqa: BLE001
        catalog = {"policies": [], "prompts": []}
    used: dict = {"prompts": [], "policies": []}
    fetched_policies: set[str] = set()   # slugs actually returned by get_policy
    fetched_prompts: set[str] = set()    # names actually returned by get_prompt
    listed_policies: set[str] = set()    # slugs seen via list_governance (name-only)
    listed_prompts: set[str] = set()     # names seen via list_governance (name-only)
    any_tool_call = False                # did the model call any tool at all?
    nudged = False                       # corrective retry fired at most once
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": (f"Agent: {agent}\n" if agent else "") + f"Task: {task}"},
    ]
    rounds = 0
    for _ in range(_MAX_ROUNDS):
        rounds += 1
        try:
            msg = _chat(messages)
        except Exception as exc:  # noqa: BLE001
            log.warning("governed_reasoning chat failed: %s", exc)
            return {"answer": "", "governance_used": used, "unsupported": {"policies": [], "prompts": []},
                    "rounds": rounds, "error": str(exc)[:200]}
        messages.append(msg)
        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            # Model wants to answer. If it never consulted governance, nudge once.
            if not any_tool_call and not nudged:
                nudged = True
                messages.append({"role": "user", "content": _NUDGE})
                continue
            return _audit(_finalize(msg.get("content") or "", used, fetched_policies,
                                    fetched_prompts, catalog, rounds,
                                    listed_policies, listed_prompts), task, agent)
        any_tool_call = True
        for tc in tool_calls:
            fn = tc.get("function", {})
            name = fn.get("name")
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:  # noqa: BLE001
                    args = {}
            if name == "list_governance":
                res = GT.list_governance(args.get("agent"))
                # Names the model legitimately learned from the catalog listing. It
                # may name them without having fetched their details, so the prose
                # scan must not flag them as fabricated.
                for p in (res.get("policies") or []):
                    if p.get("slug"):
                        listed_policies.add(str(p["slug"]).lower())
                for p in (res.get("prompts") or []):
                    if p.get("prompt_name"):
                        listed_prompts.add(str(p["prompt_name"]).lower())
            elif name == "get_policy":
                res = GT.get_policy(str(args.get("query", "")))
                if res:
                    used["policies"].append({"policy_type": res.get("policy_type"), "slug": res.get("slug")})
                    if res.get("slug"):
                        fetched_policies.add(str(res["slug"]).lower())
            elif name == "get_prompt":
                res = GT.get_prompt(str(args.get("query", "")))
                if res:
                    used["prompts"].append({"prompt_name": res.get("prompt_name"), "prompt_type": res.get("prompt_type")})
                    if res.get("prompt_name"):
                        fetched_prompts.add(str(res["prompt_name"]).lower())
            elif name == "describe_platform":
                # Fail soft: if Neo4j is down the model should carry on answering the
                # procurement question rather than the whole turn dying over a lookup.
                try:
                    from src.services.platform_kg import describe

                    res = {"facts": describe(str(args.get("topic", "")))}
                except Exception as exc:  # noqa: BLE001
                    log.warning("describe_platform unavailable: %s", exc)
                    res = {"error": "platform knowledge graph unavailable", "facts": []}
            else:
                res = {"error": "unknown tool"}
            messages.append({"role": "tool", "name": name, "content": json.dumps(res, default=str)})
    # rounds exhausted → ask for a final answer
    messages.append({"role": "user", "content": "Give your final governed answer now, ending with the CITED line."})
    try:
        final = _chat(messages).get("content") or ""
    except Exception:  # noqa: BLE001
        final = ""
    return _audit(_finalize(final, used, fetched_policies, fetched_prompts, catalog,
                            rounds, listed_policies, listed_prompts), task, agent)
