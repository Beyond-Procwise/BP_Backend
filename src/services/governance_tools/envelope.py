"""Governance envelope: resolve the governed policy + prompt for a workflow.

Deterministic (no LLM) — reads the live governance via the governance tools, so
the orchestrator can inject a single governed source into every agentic workflow
and audit it.

This used to be fail-open: any exception was swallowed and ``{}`` returned. The
problem was not only that it ran a workflow ungoverned — it was that ``{}`` is
also what this returns for a workflow that legitimately has no governance at
all (quote_evaluation and requirements_to_ranking, live, today). One value for
two states means the caller cannot fail closed on one and proceed on the other,
however much it would like to. So the two are now different:

    a resolved envelope   -> a dict, possibly with empty policies/prompts
    a failed resolution   -> GovernanceUnavailable

Deciding what to do about either is the caller's business, not this module's.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)


class GovernanceUnavailable(RuntimeError):
    """The governance for a workflow could not be read.

    Distinct from "nothing governs this workflow", which is a resolved answer
    and comes back as an envelope with empty lists.
    """

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

        # strict: a reload that failed leaves the engines on whatever they last
        # held, and resolution would then carry on against arbitrarily old
        # governance with nothing anywhere saying so.
        GT.refresh(strict=True)

        # And the quieter one. PolicyEngine returns [] when the governance query
        # fails, so an outage reaches this module as "nothing governs anything"
        # -- which, without this check, reads as an ordinary ungoverned workflow
        # and runs. Nothing below raises; this is the only place the difference
        # can still be seen.
        if GT.active_policy_count() == 0:
            raise GovernanceUnavailable(
                "the governance store returned no active policies at all — "
                "treating this as unreadable rather than as ungoverned"
            )

        agent = agent or _WORKFLOW_AGENT.get(workflow_name, workflow_name)
        gov = GT.list_governance(agent) or {"prompts": [], "policies": []}
        policies: list[dict] = []
        seen: set = set()
        # Primary policy WITH details (rules/weights/thresholds) so agents can
        # actually consume the governed values, not just see a summary.
        pol = GT.get_policy(workflow_name)
        if pol and pol.get("policy_type"):
            policies.append({"policy_type": pol.get("policy_type"), "slug": pol.get("slug"),
                             "details": pol.get("details")})
            seen.add(pol.get("policy_type"))
        for p in gov.get("policies") or []:
            if p.get("policy_type") not in seen:
                policies.append({"policy_type": p.get("policy_type"), "slug": p.get("slug")})
                seen.add(p.get("policy_type"))
        return {"agent": agent, "policies": policies, "prompts": list(gov.get("prompts") or [])}
    except GovernanceUnavailable:
        raise  # already the right answer; do not wrap it in itself
    except Exception as exc:  # noqa: BLE001 - fail CLOSED, by telling the caller
        log.warning("resolve_governance failed for %s: %s", workflow_name, exc,
                    exc_info=True)
        raise GovernanceUnavailable(
            f"could not resolve governance for {workflow_name}: {exc}"
        ) from exc
