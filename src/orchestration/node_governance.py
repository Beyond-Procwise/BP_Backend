"""Which governed prompt and policy are LINKED to this agent.

Only 5 of the 14 agents have any governance at all. The other 9 run on their
built-in defaults, and this says so plainly. A node that shows a governance tick
it did not earn is worse than a node that shows none.

Important: this is a STATIC LOOKUP over bp_prompt/bp_policy, not a resolution
through PromptEngine/PolicyEngine's selection logic. If several prompt or
policy rows match an agent, PromptEngine/PolicyEngine pick ONE at run time
(e.g. the highest version); this module has no way to know which one that
would be, so it must not claim any of them "applied". It reports linkage —
every row named here is real, active, and genuinely linked — and stops there.
See ``governance_for``'s ``status`` field for the exact wording to surface.

``*_linked_agents`` is not guaranteed to hold a single agent name — it can list
several tokens (e.g. "supplier_ranking_agent, negotiation_agent"), exactly as
PromptEngine and PolicyEngine already assume when they select governance to
apply (see ``PromptEngine._coerce_linked_agents`` / ``PolicyEngine._coerce_linked_agents``).
This module must tokenize the column the same way those engines do, or the
badge here could disagree with what is genuinely linked at run time.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from engines.policy_engine import PolicyEngine
from services.db import get_conn

# PolicyEngine._coerce_linked_agents and PromptEngine._coerce_linked_agents are
# byte-identical (both regex-split "*_linked_agents" text into slugified
# tokens). Reuse one rather than adding a third divergent copy here.
# PolicyEngine is picked over PromptEngine because it does not pull in the
# heavier ollama/torch imports that prompt_engine.py carries at module scope.
_coerce_linked_agents = PolicyEngine._coerce_linked_agents


def _token_form(slug: str) -> str:
    """The slug as it appears once ``_coerce_linked_agents`` has tokenised it.

    That tokeniser splits the column on anything outside ``[A-Za-z0-9_]``, so a
    hyphen becomes a token boundary: "advanced-probe" stored as a link is read
    back as "advanced_probe". Comparing the raw slug against those tokens could
    therefore never match for any agent created in the workspace, whose ids are
    kebab-case by definition — every one of them showed "built-in default" on the
    canvas no matter how much governance was genuinely linked to it. The run-time
    engines were never confused (they resolve on the underscored governance slug);
    only this badge was, which is the worse failure: governance silently
    under-reported reads as governance absent.
    """
    return re.sub(r"[^A-Za-z0-9]+", "_", (slug or "")).strip("_").lower()


def normalise_agent_name(slug: str) -> str:
    """Registry uses `supplier_ranking`; the governance tables use `supplier_ranking_agent`."""
    slug = _token_form(slug)
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
    """Real, active, linked governance for ``slug`` — never "applied".

    ``status`` is the honest label for display: "linked & active" when at
    least one prompt or policy row is linked (this function cannot prove any
    one of them is the row PromptEngine/PolicyEngine would actually select
    and apply at run time — see the module docstring), or "built-in default"
    when none are. Do not rename this to imply application without actually
    routing the lookup through PromptEngine/PolicyEngine's selection logic.
    """
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
    governed = bool(prompts or policies)
    return {"prompts": prompts, "policies": policies,
            "governed": governed,
            "status": "linked & active" if governed else "built-in default"}
