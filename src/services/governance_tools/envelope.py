"""Governance envelope: resolve the governed policy + prompt for a workflow.

Deterministic (no LLM) — reads the live governance via the governance tools, so
the orchestrator can inject a single governed source into every agentic workflow
and audit it. Fail-open: returns {} on any error so a governance problem never
breaks a workflow.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)

# Map a workflow name to its primary governed agent (for prompt/policy lookup).
_WORKFLOW_AGENT = {
    "supplier_ranking": "supplier_ranking_agent",
    "quote_evaluation": "quote_evaluation_agent",
    "opportunity_mining": "opportunity_miner_agent",
    "supplier_interaction": "supplier_interaction_agent",
    "negotiation": "negotiation_agent",
}


def resolve_governance(workflow_name: str, agent: str | None = None) -> dict:
    """Return {agent, policies:[...], prompts:[...]} governing this workflow."""
    try:
        from src.services.governance_tools import tools as GT
        GT.refresh()
        agent = agent or _WORKFLOW_AGENT.get(workflow_name, workflow_name)
        gov = GT.list_governance(agent) or {"prompts": [], "policies": []}
        policies = list(gov.get("policies") or [])
        # Also match a policy by the workflow name itself (policy_type), if distinct.
        pol = GT.get_policy(workflow_name)
        if pol and pol.get("policy_type") and not any(
            p.get("policy_type") == pol.get("policy_type") for p in policies
        ):
            policies.append({"policy_type": pol.get("policy_type"), "slug": pol.get("slug")})
        return {"agent": agent, "policies": policies, "prompts": list(gov.get("prompts") or [])}
    except Exception:  # noqa: BLE001 - fail open
        log.debug("resolve_governance failed for %s", workflow_name, exc_info=True)
        return {}
