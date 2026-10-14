"""A new agent is a draft until a second person publishes it.

A created agent used to be usable by everyone the moment it was created: no
trial, no second pair of eyes. Now a new agent starts as a draft that only the
person who created it can run. Someone else, with approval rights, publishes it.
"""
from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents import base_agent
from agents.auto_registry import AutoRegistry
from agents.base_agent import AgentContext, AgentStatus
from engines.policy_engine import PolicyEngine
from orchestration import agentnick_control
from orchestration.prompt_engine import PromptEngine
from repositories import agent_run_repo
from services.tool_runtime import ToolRunResult


class _Registry:
    def tool_schemas(self):
        return [{"function": {"name": "run_overcharge-hunter", "description": "",
                              "parameters": {"type": "object", "properties": {}}}}]


class _Created(base_agent.BaseAgent):
    def run(self, ctx):
        return base_agent.AgentOutput(status=AgentStatus.SUCCESS, data={"ran": True})


def _nick():
    nick = SimpleNamespace(
        settings=SimpleNamespace(script_user="tester", extraction_model="m"),
        prompt_engine=PromptEngine(prompt_rows=[]),
        policy_engine=PolicyEngine(policy_rows=[]),
        learning_repository=None, auto_registry=_Registry(), agents={},
    )
    agent = _Created(nick)
    agent.governance_slug = "overcharge-hunter"
    agent.draft_owner = "sub-maker"
    nick.agents["overcharge-hunter"] = agent
    return nick


def _ctx(user):
    return AgentContext(workflow_id="wf-9", agent_id="overcharge-hunter", user_id=user,
                        input_data={})


@pytest.fixture(autouse=True)
def _quiet(monkeypatch):
    monkeypatch.setattr(base_agent.BaseAgent, "_record_context_example",
                        lambda self, ctx, out: None)
    monkeypatch.setattr(agent_run_repo, "record", lambda **kw: None)
    monkeypatch.setattr(
        agentnick_control, "run_tools",
        lambda task, tools, system, **kw: ToolRunResult(answer="ok", rounds=1, system=system),
    )


# --- who can run a draft ------------------------------------------------------

def test_the_person_who_created_a_draft_can_run_it():
    out = _nick().agents["overcharge-hunter"].execute(_ctx("sub-maker"), tool_loop=True)
    assert out.status == AgentStatus.SUCCESS
    assert out.data["answer"] == "ok"


@pytest.mark.parametrize("someone_else", ["sub-other", None, ""])
def test_nobody_else_can_run_a_draft(someone_else):
    agent = _nick().agents["overcharge-hunter"]
    for tool_loop in (True, False):
        out = agent.execute(_ctx(someone_else), tool_loop=tool_loop)
        assert out.status == AgentStatus.FAILED
        assert "draft" in out.error
        assert "ran" not in (out.data or {}) and "answer" not in (out.data or {})


def test_a_published_agent_runs_for_anyone():
    agent = _nick().agents["overcharge-hunter"]
    agent.draft_owner = None
    assert agent.execute(_ctx("sub-other"), tool_loop=True).status == AgentStatus.SUCCESS


def test_a_draft_is_offered_as_a_tool_only_to_its_creator():
    nick = _nick()
    mine = [t.name for t in agentnick_control.build_tools(nick, user_id="sub-maker")]
    theirs = [t.name for t in agentnick_control.build_tools(nick, user_id="sub-other")]
    nobodys = [t.name for t in agentnick_control.build_tools(nick)]

    assert "run_overcharge-hunter" in mine
    assert "run_overcharge-hunter" not in theirs
    assert "run_overcharge-hunter" not in nobodys


# --- the stage reaches the agent from the catalogue ---------------------------

def _registry(tmp_path, **entry_extra):
    path = tmp_path / "agent_definitions.json"
    entry = {"agentId": 1, "slug": "overcharge-hunter", "agentType": "X",
             "class_path": f"{__name__}._Created", "derived_from": "supplier_ranking",
             **entry_extra}
    path.write_text(json.dumps([entry]))
    registry = AutoRegistry.from_json(path)
    registry.set_agent_nick(_nick())
    return registry, path, entry


def test_a_draft_in_the_catalogue_is_stamped_with_its_creator(tmp_path):
    registry, _, _ = _registry(tmp_path, stage="draft", created_by="sub-maker")
    assert registry.get_agent("overcharge-hunter").draft_owner == "sub-maker"


def test_an_agent_with_no_stage_is_published(tmp_path):
    """Every agent created before drafts existed carries no stage, and must not
    stop working for the people already using it."""
    registry, _, _ = _registry(tmp_path, created_by="sub-maker")
    assert getattr(registry.get_agent("overcharge-hunter"), "draft_owner", None) is None


def test_publishing_frees_an_agent_that_is_already_running(tmp_path):
    registry, path, entry = _registry(tmp_path, stage="draft", created_by="sub-maker")
    agent = registry.get_agent("overcharge-hunter")

    path.write_text(json.dumps([{**entry, "stage": "published"}]))
    registry.refresh_from_json(path)

    assert agent.draft_owner is None


def test_after_an_edit_the_draft_is_the_editors_not_the_creators(tmp_path):
    registry, _, _ = _registry(tmp_path, stage="draft", created_by="sub-maker",
                               draft_by="sub-editor")
    assert registry.get_agent("overcharge-hunter").draft_owner == "sub-editor"
