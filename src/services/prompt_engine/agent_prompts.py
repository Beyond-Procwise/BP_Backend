"""An agent's own instructions, kept in the prompt registry.

Creating an agent writes its instructions; editing it may rewrite them;
deleting it retires them. The agent routes call these so the registry holds
every wording an agent has run under and who wrote it.

Release of an agent's instructions is governed by the agent itself: a new or
edited agent is a draft only its author can run until a second person publishes
it. So these are recorded as in use straight away (``agent_saved``), not
promoted here.

Until rendering moves to the registry the agent still reads its instructions
from the older table, so a failure here must not fail the agent's save. It is
logged as an error, and scripts/import_prompts_to_registry.py brings the
registry back in line.
"""
from __future__ import annotations

import logging
from typing import Optional

from services.prompt_engine import registry

logger = logging.getLogger(__name__)

UNSIGNED = "workspace"   # who the older table records when sign-in is off


def record(prompt_name: str, text: str, *, by: Optional[str], linked_agent: str) -> Optional[int]:
    """Record ``text`` as the agent's instructions now in use. Returns the
    version, or None if it could not be recorded."""
    try:
        try:
            live = registry.resolve(prompt_name)
        except (registry.PointerUnresolved, registry.PromptNotFound):
            live = None
        if live is not None and live["template"] == text:
            return live["version"]
        return registry.adopt(
            prompt_name, template=text, by=(by or UNSIGNED), change="agent_saved",
            settings={registry.LEGACY_DESC: text, "legacy_type": "agent_instructions",
                      "legacy_linked_agents": linked_agent},
            change_note="written with its agent",
            note="set with its agent; the agent's draft-and-publish governs its release")
    except Exception:
        logger.error("instructions %s could not be recorded in the prompt registry",
                     prompt_name, exc_info=True)
        return None


def retire(prompt_name: str, *, by: Optional[str]) -> bool:
    try:
        return registry.retire(prompt_name, by=(by or UNSIGNED), note="its agent was deleted")
    except Exception:
        logger.error("instructions %s could not be retired in the prompt registry",
                     prompt_name, exc_info=True)
        return False
