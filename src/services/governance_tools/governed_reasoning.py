"""Governed reasoning: AgentNick tool-use loop over the policy & prompt engines.

AgentNick pulls the applicable governed prompt/policy at runtime via tool-calls and
applies them, so its reasoning is governed dynamically and traceable. Read-only;
bounded; never raises to the caller. AgentNick stays the base model.
"""
from __future__ import annotations

import json
import logging
import os

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
]

_SYSTEM = (
    "You are a procurement reasoning assistant whose prompts and policies are GOVERNED in a "
    "database. Do NOT rely on assumptions about rules or weights. For the task: FIRST call "
    "list_governance (optionally with the relevant agent) to see what governance exists, then "
    "call get_policy and/or get_prompt to fetch the applicable governed policy (rules/weights/"
    "thresholds) and prompt template, and APPLY them in your answer. State which governance you "
    "used. Then give a concise, governed answer."
)


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
    used: dict = {"prompts": [], "policies": []}
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
            return {"answer": "", "governance_used": used, "rounds": rounds, "error": str(exc)[:200]}
        messages.append(msg)
        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            return {"answer": msg.get("content") or "", "governance_used": used, "rounds": rounds}
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
            elif name == "get_policy":
                res = GT.get_policy(str(args.get("query", "")))
                if res:
                    used["policies"].append({"policy_type": res.get("policy_type"), "slug": res.get("slug")})
            elif name == "get_prompt":
                res = GT.get_prompt(str(args.get("query", "")))
                if res:
                    used["prompts"].append({"prompt_name": res.get("prompt_name"), "prompt_type": res.get("prompt_type")})
            else:
                res = {"error": "unknown tool"}
            messages.append({"role": "tool", "name": name, "content": json.dumps(res, default=str)})
    # rounds exhausted → ask for a final answer
    messages.append({"role": "user", "content": "Give your final governed answer now."})
    try:
        final = _chat(messages).get("content") or ""
    except Exception:  # noqa: BLE001
        final = ""
    return {"answer": final, "governance_used": used, "rounds": rounds}
