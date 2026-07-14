"""Which governed prompt and policy actually applied to this agent.

Only 5 of the 14 agents have any governance at all. The other 9 run on their
built-in defaults, and this says so plainly. A node that shows a governance tick
it did not earn is worse than a node that shows none.

``*_linked_agents`` is not guaranteed to hold a single agent name — it can list
several tokens (e.g. "supplier_ranking_agent, negotiation_agent"), exactly as
PromptEngine and PolicyEngine already assume when they *apply* governance
(see ``PromptEngine._coerce_linked_agents`` / ``PolicyEngine._coerce_linked_agents``).
This module must tokenize the column the same way those engines do, or the
badge here could disagree with what actually gets applied at run time.
"""

from __future__ import annotations

from typing import Any, Dict, List

from engines.policy_engine import PolicyEngine
from services.db import get_conn

# PolicyEngine._coerce_linked_agents and PromptEngine._coerce_linked_agents are
# byte-identical (both regex-split "*_linked_agents" text into slugified
# tokens). Reuse one rather than adding a third divergent copy here.
# PolicyEngine is picked over PromptEngine because it does not pull in the
# heavier ollama/torch imports that prompt_engine.py carries at module scope.
_coerce_linked_agents = PolicyEngine._coerce_linked_agents


def normalise_agent_name(slug: str) -> str:
    """Registry uses `supplier_ranking`; the governance tables use `supplier_ranking_agent`."""
    return slug if slug.endswith("_agent") else f"{slug}_agent"


def _bare_agent_name(slug: str) -> str:
    """The `_agent`-less form, tolerated because some rows store it bare."""
    linked = normalise_agent_name(slug)
    return linked[: -len("_agent")]


def _rows(sql: str, linked_form: str, bare_form: str) -> List[Dict[str, Any]]:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql)
        matched: List[Dict[str, Any]] = []
        for name, version, linked_agents in cur.fetchall():
            tokens = _coerce_linked_agents(linked_agents)
            if linked_form in tokens or bare_form in tokens:
                matched.append({"name": name, "version": version})
        return matched


def governance_for(slug: str) -> Dict[str, Any]:
    linked_form = normalise_agent_name(slug)
    bare_form = _bare_agent_name(slug)
    prompts = _rows(
        """SELECT prompt_name, version, prompt_linked_agents FROM proc.bp_prompt
            WHERE prompts_status = 1""",
        linked_form, bare_form,
    )
    policies = _rows(
        """SELECT policy_name, version, policy_linked_agents FROM proc.bp_policy
            WHERE policy_status = 1""",
        linked_form, bare_form,
    )
    return {"prompts": prompts, "policies": policies,
            "governed": bool(prompts or policies)}
