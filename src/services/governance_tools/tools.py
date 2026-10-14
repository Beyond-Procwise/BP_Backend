"""Backend-executed governance tools for AgentNick tool-use.

Expose the DB-governed prompt engine (proc.bp_prompt) and policy engine
(proc.bp_policy) as read-only tools so AgentNick can pull the applicable governed
prompt/policy at runtime. Reads through the existing engine classes (cached +
hot-reloadable via /agents/reload-governance). Fail-open (return {} / [] on error).
"""
from __future__ import annotations

import logging
import threading

log = logging.getLogger(__name__)

_lock = threading.Lock()
_pe = None   # PromptEngine
_pol = None  # PolicyEngine


def _engines():
    global _pe, _pol
    with _lock:
        if _pe is None or _pol is None:
            from src.services.db import get_conn
            from orchestration.prompt_engine import PromptEngine
            from engines.policy_engine import PolicyEngine
            _pe = PromptEngine(connection_factory=get_conn)
            _pol = PolicyEngine(connection_factory=get_conn)
    return _pe, _pol


def refresh(strict: bool = False) -> None:
    """Reload governance so edits + /agents/reload-governance are reflected.

    ``strict`` is for callers that must not proceed on governance they cannot
    prove is current. The default stays fail-open: a tool-call that cannot
    refresh should still answer from what it has, and AgentNick asking for a
    policy is not an authority decision.

    A failed reload leaves the engines holding whatever they last loaded, which
    is why a caller that cares has to be told rather than left to infer it from
    a result that looks entirely normal.
    """
    pe, pol = _engines()
    try:
        pe.refresh()
        pol.reload_policies()
    except Exception:  # noqa: BLE001
        if strict:
            raise
        log.debug("governance_tools.refresh failed", exc_info=True)


def active_policy_count() -> int:
    """How many active policies the policy engine is currently holding.

    The only signal that separates "the store answered, and nothing governs
    this agent" from "the store did not answer". ``PolicyEngine`` logs
    "Failed to load policies from database" and returns ``[]`` rather than
    raising, so there is no exception for a caller to catch — an outage and an
    empty governance set are the same value everywhere above it.

    Zero is treated as an outage by the callers that fail closed. This product
    has thirty-one active policies; it has never legitimately had none, and a
    deployment that genuinely has none has not been configured yet.
    """
    _, pol = _engines()
    try:
        return len(pol.list_policies() or [])
    except Exception:  # noqa: BLE001
        return 0


def _prompt_row(p: dict) -> dict:
    return {
        "prompt_name": p.get("promptName"),
        "prompt_type": p.get("promptType"),
        "linked_agents": p.get("linked_agents"),
        "template": p.get("template") or p.get("prompts_desc"),
    }


def _policy_row(p: dict) -> dict:
    return {
        "policy_name": p.get("policyName"),
        "policy_type": p.get("policy_type"),
        "slug": p.get("slug"),
        "linked_agents": p.get("policy_linked_agents"),
        "details": p.get("details"),
    }


def list_governance(agent: str | None = None) -> dict:
    """Summarise available governed prompts and policies (optionally for one agent)."""
    pe, pol = _engines()
    try:
        prompts = pe.all_prompts() or []
        policies = pol.list_policies() or []
    except Exception:  # noqa: BLE001
        return {"prompts": [], "policies": []}
    if agent:
        a = agent.lower()
        prompts = [p for p in prompts if a in str(p.get("linked_agents") or "").lower()]
        policies = [p for p in policies if a in str(p.get("policy_linked_agents") or "").lower()]
    return {
        "prompts": [{"prompt_name": p.get("promptName"), "prompt_type": p.get("promptType"),
                     "linked_agents": p.get("linked_agents")} for p in prompts],
        "policies": [{"policy_type": p.get("policy_type"), "slug": p.get("slug"),
                      "linked_agents": p.get("policy_linked_agents")} for p in policies],
    }


def get_prompt(query: str) -> dict:
    """Best-matched governed prompt for a name / type / linked-agent query."""
    pe, _ = _engines()
    q = (query or "").strip().lower()
    if not q:
        return {}
    prompts = pe.all_prompts() or []
    fallback = None
    for p in prompts:
        name = str(p.get("promptName") or "").lower()
        ptype = str(p.get("promptType") or "").lower()
        agents = str(p.get("linked_agents") or "").lower()
        if q == name or q == ptype:
            return _prompt_row(p)
        if fallback is None and (q in agents or q in name or q in ptype):
            fallback = p
    return _prompt_row(fallback) if fallback else {}


def get_policy(query: str) -> dict:
    """Best-matched governed policy for a slug / type / linked-agent query."""
    _, pol = _engines()
    q = (query or "").strip().lower()
    if not q:
        return {}
    policies = pol.list_policies() or []
    fallback = None
    for p in policies:
        ptype = str(p.get("policy_type") or "").lower()
        slug = str(p.get("slug") or "").lower()
        agents = str(p.get("policy_linked_agents") or "").lower()
        aliases = str(p.get("aliases") or "").lower()
        if q == slug or q == ptype:
            return _policy_row(p)
        if fallback is None and (q in agents or q in ptype or q in slug or q in aliases):
            fallback = p
    return _policy_row(fallback) if fallback else {}
