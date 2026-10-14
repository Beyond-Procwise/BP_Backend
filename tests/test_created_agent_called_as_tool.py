"""A created agent does what it was told wherever it is run from, with only the
tools it was given.

Two faults, both invisible from the canvas:

* Called as a tool (by AgentNick in Ask, or by another agent's loop) a created
  agent ran as the built-in class it was copied from: ``execute(ctx)`` with no
  ``tool_loop``, so the task a person wrote for it was never read.
* The list of tools saved on a created agent was read by nothing. Every created
  agent reasoned with every tool except the one that sends email.
"""
from __future__ import annotations

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
from services.tool_runtime import ToolRunResult

INSTRUCTIONS = "Find suppliers whose invoices exceed their PO by more than 5%."


class _Registry:
    slugs = ("supplier_ranking", "email_dispatch", "overcharge-hunter", "second-opinion")

    def tool_schemas(self):
        return [
            {"function": {"name": f"run_{slug}", "description": slug,
                          "parameters": {"type": "object", "properties": {}}}}
            for slug in self.slugs
        ]


class _Builtin(base_agent.BaseAgent):
    def run(self, ctx):
        return base_agent.AgentOutput(status=AgentStatus.SUCCESS, data={"ranked": []})


class _Created(base_agent.BaseAgent):
    """A created agent: the backing class under its own slug."""

    def run(self, ctx):
        return base_agent.AgentOutput(status=AgentStatus.SUCCESS, data={"ran_as": "base"})


def _nick():
    prompt_rows = [
        {"prompt_id": 7, "prompt_name": "overcharge_hunter_instructions",
         "prompt_type": "agent_instructions",
         "prompt_linked_agents": "overcharge_hunter", "prompts_desc": INSTRUCTIONS},
        {"prompt_id": 8, "prompt_name": "second_opinion_instructions",
         "prompt_type": "agent_instructions",
         "prompt_linked_agents": "second_opinion", "prompts_desc": "Check the first answer."},
    ]
    nick = SimpleNamespace(
        settings=SimpleNamespace(script_user="tester", extraction_model="m"),
        prompt_engine=PromptEngine(prompt_rows=prompt_rows),
        policy_engine=PolicyEngine(policy_rows=[]),
        learning_repository=None,
        auto_registry=_Registry(),
        agents={},
    )
    nick.agents["supplier_ranking"] = _Builtin(nick)
    nick.agents["email_dispatch"] = _Builtin(nick)
    for slug in ("overcharge-hunter", "second-opinion"):
        created = _Created(nick)
        created.governance_slug = slug
        nick.agents[slug] = created
    return nick


@pytest.fixture(autouse=True)
def _no_training_files(monkeypatch):
    """An agent run records a training example to disk; these runs are not examples."""
    monkeypatch.setattr(base_agent.BaseAgent, "_record_context_example",
                        lambda self, ctx, out: None)


@pytest.fixture()
def loops(monkeypatch):
    """Every tool loop that starts, in order: its system prompt and its tools."""
    started = []

    def fake_run_tools(task, tools, system, **kw):
        started.append({"task": task, "system": system, "kw": kw,
                        "tools": {t.name: t for t in tools}})
        script = fake_run_tools.script
        if script:
            return script.pop(0)(started[-1])
        return ToolRunResult(answer="Two suppliers billed above their PO.", rounds=1)

    fake_run_tools.script = []
    monkeypatch.setattr(agentnick_control, "run_tools", fake_run_tools)
    started_api = SimpleNamespace(started=started, script=fake_run_tools.script)
    return started_api


def _tools(nick, **kw):
    return {t.name: t for t in agentnick_control.build_tools(nick, **kw)}


# --- a created agent follows its own task when it is called as a tool ---------

def test_a_created_agent_called_as_a_tool_follows_its_own_task(loops):
    out = _tools(_nick())["run_overcharge-hunter"].handler()

    assert out["status"] == "success"
    assert out["data"]["answer"] == "Two suppliers billed above their PO."
    assert "ran_as" not in out["data"]          # the backing class's run() was not used
    assert INSTRUCTIONS in loops.started[0]["system"]


def test_a_built_in_agent_called_as_a_tool_still_runs_as_itself(loops):
    out = _tools(_nick())["run_supplier_ranking"].handler()

    assert out["data"]["ranked"] == []
    assert "answer" not in out["data"]
    assert loops.started == []                   # no reasoning loop was started


def test_a_created_agent_cannot_call_itself(loops):
    """It is not offered its own tool; should a caller hand it one anyway, the
    call is refused rather than run."""
    tools = _tools(_nick())

    def call_self(loop):
        assert "run_overcharge-hunter" not in loop["tools"]
        return ToolRunResult(answer=str(tools["run_overcharge-hunter"].handler()), rounds=1)

    loops.script.append(call_self)
    out = tools["run_overcharge-hunter"].handler()

    assert "cannot call itself" in out["data"]["answer"]
    assert len(loops.started) == 1               # the second loop never began


def test_created_agents_do_not_chain_without_end(loops):
    """A created agent run as a tool cannot start another created agent: each
    loop is minutes of model time, and a chain nobody drew has no end."""
    def call_other(loop):
        return ToolRunResult(answer=str(loop["tools"]["run_second-opinion"].handler()), rounds=1)

    loops.script.append(call_other)
    out = _tools(_nick())["run_overcharge-hunter"].handler()

    assert len(loops.started) == 1               # second-opinion's loop never began
    assert "too many created agents deep" in out["data"]["answer"]


def test_a_created_agent_is_never_offered_itself_as_a_tool(loops):
    nick = _nick()
    nick.agents["overcharge-hunter"].execute(
        AgentContext(workflow_id="wf-9", agent_id="overcharge-hunter", user_id="u1",
                     input_data={}),
        tool_loop=True,
    )
    assert "run_overcharge-hunter" not in loops.started[0]["tools"]
    assert "run_second-opinion" in loops.started[0]["tools"]


def test_the_depth_is_given_back_after_a_call(loops):
    tools = _tools(_nick())
    tools["run_overcharge-hunter"].handler()
    tools["run_overcharge-hunter"].handler()
    assert len(loops.started) == 2               # the second call was not refused


# --- a created agent reasons with only the tools it was given ----------------

def _created(nick, slug="overcharge-hunter"):
    return nick.agents[slug]


def _ctx():
    return AgentContext(workflow_id="wf-9", agent_id="overcharge-hunter", user_id="u1",
                        input_data={"deal_id": "D1"})


def test_an_agent_given_a_tool_list_gets_exactly_those_tools(loops):
    nick = _nick()
    _created(nick).allowed_tools = ["get_corpus_facts", "run_supplier_ranking"]

    _created(nick).execute(_ctx(), tool_loop=True)

    assert sorted(loops.started[0]["tools"]) == ["get_corpus_facts", "run_supplier_ranking"]


def test_a_tool_list_can_never_grant_the_tool_that_sends_email(loops):
    nick = _nick()
    _created(nick).allowed_tools = ["get_corpus_facts", "run_email_dispatch"]

    _created(nick).execute(_ctx(), tool_loop=True)

    assert sorted(loops.started[0]["tools"]) == ["get_corpus_facts"]


def test_an_empty_tool_list_means_no_tools(loops):
    nick = _nick()
    _created(nick).allowed_tools = []

    out = _created(nick).execute(_ctx(), tool_loop=True)

    assert out.status == AgentStatus.SUCCESS
    assert loops.started[0]["tools"] == {}
    # With nothing to consult, the loop must not insist that something is consulted.
    assert loops.started[0]["kw"]["require_tool_use"] is False


def test_an_agent_never_given_a_tool_list_keeps_what_it_had(loops):
    nick = _nick()
    _created(nick).execute(_ctx(), tool_loop=True)

    names = set(loops.started[0]["tools"])
    assert {"run_supplier_ranking", "get_corpus_facts", "get_policy"} <= names
    assert "run_email_dispatch" not in names
    assert loops.started[0]["kw"]["require_tool_use"] is True


# --- the list reaches the agent from the catalogue ---------------------------

def _registry(tmp_path, entry_extra):
    import json

    entry = {"agentId": 1, "slug": "overcharge-hunter", "agentType": "X",
             "class_path": f"{__name__}._Created", "derived_from": "supplier_ranking",
             **entry_extra}
    path = tmp_path / "agent_definitions.json"
    path.write_text(json.dumps([entry]))
    registry = AutoRegistry.from_json(path)
    registry.set_agent_nick(_nick())
    return registry


def test_the_catalogue_tool_list_is_stamped_on_the_agent(tmp_path):
    agent = _registry(tmp_path, {"tools": ["get_policy"]}).get_agent("overcharge-hunter")
    assert agent.allowed_tools == ["get_policy"]


def test_an_empty_catalogue_list_is_not_mistaken_for_no_list(tmp_path):
    agent = _registry(tmp_path, {"tools": []}).get_agent("overcharge-hunter")
    assert agent.allowed_tools == []


def test_an_entry_with_no_list_leaves_the_agent_unrestricted(tmp_path):
    agent = _registry(tmp_path, {}).get_agent("overcharge-hunter")
    assert getattr(agent, "allowed_tools", None) is None


def test_narrowing_the_list_narrows_an_agent_that_is_already_running(tmp_path):
    import json

    registry = _registry(tmp_path, {"tools": ["get_policy", "get_corpus_facts"]})
    agent = registry.get_agent("overcharge-hunter")
    path = tmp_path / "agent_definitions.json"

    entry = json.loads(path.read_text())[0]
    entry["tools"] = ["get_policy"]
    path.write_text(json.dumps([entry]))
    registry.refresh_from_json(path)
    assert agent.allowed_tools == ["get_policy"]

    del entry["tools"]
    path.write_text(json.dumps([entry]))
    registry.refresh_from_json(path)
    assert agent.allowed_tools is None
