"""Which governed prompt and policy actually applied to this agent.

Only 5 of the 14 agents have any governance at all. The other 9 run on their
built-in defaults, and this says so plainly. A node that shows a governance tick
it did not earn is worse than a node that shows none.
"""

from __future__ import annotations

from typing import Any, Dict, List

from services.db import get_conn


def normalise_agent_name(slug: str) -> str:
    """Registry uses `supplier_ranking`; the governance tables use `supplier_ranking_agent`."""
    return slug if slug.endswith("_agent") else f"{slug}_agent"


def _rows(sql: str, name: str) -> List[Dict[str, Any]]:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, (name, name))
        return [{"name": r[0], "version": r[1]} for r in cur.fetchall()]


def governance_for(slug: str) -> Dict[str, Any]:
    linked = normalise_agent_name(slug)
    prompts = _rows(
        """SELECT prompt_name, version FROM proc.bp_prompt
            WHERE prompts_status = 1 AND (prompt_linked_agents = %s OR prompt_linked_agents = %s)""",
        linked,
    )
    policies = _rows(
        """SELECT policy_name, version FROM proc.bp_policy
            WHERE policy_status = 1 AND (policy_linked_agents = %s OR policy_linked_agents = %s)""",
        linked,
    )
    return {"prompts": prompts, "policies": policies,
            "governed": bool(prompts or policies)}
