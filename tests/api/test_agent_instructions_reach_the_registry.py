"""Creating, editing and deleting an agent each tell the prompt registry."""
from __future__ import annotations

import inspect
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import api.routers.agents as agents_mod


def _body(fn):
    return inspect.getsource(fn)


def test_each_agent_route_that_touches_instructions_calls_the_registry():
    """The three routes each write proc.bp_prompt directly; the registry is told
    in the same breath. The behaviour of the call is tested against a real
    database in tests/prompt_engine/test_agent_prompts_live.py."""
    assert "agent_prompts.record(prompt_name, instructions, by=created_by" in _body(
        agents_mod.create_agent)
    assert "agent_prompts.record(prompt_name, text, by=editor" in _body(agents_mod.update_agent)
    assert "agent_prompts.retire(" in _body(agents_mod.delete_agent)


def test_the_registry_is_told_only_after_the_agent_itself_was_stored():
    """A create that fails must leave no version behind: versions cannot be deleted."""
    src = _body(agents_mod.create_agent)
    assert src.index("agent_definition_repo.create(") < src.index("agent_prompts.record(")
