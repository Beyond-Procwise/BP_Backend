"""Programme item A4 — a canvas-made agent reasons with tools, under its own
instructions.

Before this, the instructions typed into the workspace were stored in
bp_prompt and never read: a derived agent ran exactly as the built-in class it
was copied from. Now ``execute(context, tool_loop=True)`` routes through
``BaseAgent.reason()`` — the governed AgentNick loop (corpus facts, policies,
agents-as-tools) with the agent's instructions in the system prompt.
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents import base_agent
from agents.base_agent import AgentContext, AgentStatus
from engines.policy_engine import PolicyEngine
from orchestration import agentnick_control
from orchestration.prompt_engine import PromptEngine
from services.tool_runtime import ToolCall, ToolRunResult

INSTRUCTIONS = "Find suppliers whose invoices exceed their PO by more than 5%."


class _Registry:
    def tool_schemas(self):
        return [
            {"function": {"name": f"run_{slug}", "description": slug,
                          "parameters": {"type": "object", "properties": {}}}}
            for slug in ("supplier_ranking", "email_dispatch")
        ]


class _Backing:
    """A registered built-in agent, as the loop would call it."""

    def __init__(self):
        self.contexts = []

    def run(self, ctx):
        self.contexts.append(ctx)
        return base_agent.AgentOutput(status=AgentStatus.SUCCESS, data={"ranked": []})


def _agent_nick():
    prompt_rows = [{
        "prompt_id": 7,
        "prompt_name": "overcharge_hunter_instructions",
        "prompt_type": "agent_instructions",
        "prompt_linked_agents": "overcharge_hunter",
        "prompts_desc": INSTRUCTIONS,
    }]
    return SimpleNamespace(
        settings=SimpleNamespace(script_user="tester", extraction_model="m"),
        prompt_engine=PromptEngine(prompt_rows=prompt_rows),
        policy_engine=PolicyEngine(policy_rows=[]),
        learning_repository=None,
        auto_registry=_Registry(),
        agents={"supplier_ranking": _Backing(), "email_dispatch": _Backing()},
    )


class _Derived(base_agent.BaseAgent):
    """Stands in for a derived agent: the backing class under a new slug."""

    def run(self, *a, **k):  # pragma: no cover - must never be reached
        raise AssertionError("run() must not be called for a tool-loop node")


@pytest.fixture()
def loop(monkeypatch):
    """Capture what the tool loop is asked, and script what it answers."""
    captured = {}

    def fake_run_tools(task, tools, system, **kw):
        captured.update(task=task, tools=[t.name for t in tools], system=system, kw=kw)
        return captured.get("reply") or ToolRunResult(
            answer="Two suppliers billed above their PO.",
            rounds=2,
            calls=[ToolCall(name="get_corpus_facts", arguments={"query": "overbilling"},
                            ok=True, result={"rows": [{"secret": "raw"}]})],
        )

    monkeypatch.setattr(agentnick_control, "run_tools", fake_run_tools)
    return captured


def _derived():
    agent = _Derived(_agent_nick())
    agent.governance_slug = "overcharge_hunter"
    return agent


def _context():
    return AgentContext(
        workflow_id="wf-9", agent_id="overcharge_hunter", user_id="u1",
        input_data={"deal_id": "D1", "prompts": [{"promptId": 7}], "policies": []},
    )


def test_execute_with_tool_loop_reasons_instead_of_running(loop):
    out = _derived().execute(_context(), tool_loop=True)

    assert out.status == AgentStatus.SUCCESS
    assert out.data["answer"] == "Two suppliers billed above their PO."
    assert out.data["tools_used"] == ["get_corpus_facts"]
    assert out.data["rounds"] == 2
    # The trace says what was called with what — never what came back.
    assert out.data["trace"][0]["name"] == "get_corpus_facts"
    assert out.data["trace"][0]["arguments"] == {"query": "overbilling"}
    assert "secret" not in str(out.data)


def test_the_agents_own_instructions_govern_the_loop(loop):
    _derived().execute(_context(), tool_loop=True)
    assert INSTRUCTIONS in loop["system"]


def test_the_task_is_the_workflow_inputs_not_the_governance_blobs(loop):
    _derived().execute(_context(), tool_loop=True)
    assert "D1" in loop["task"]
    assert "promptId" not in loop["task"]


def test_a_node_loop_can_run_agents_but_never_send_email(loop):
    _derived().execute(_context(), tool_loop=True)
    assert "run_supplier_ranking" in loop["tools"]
    assert "run_email_dispatch" not in loop["tools"]


def test_a_failed_loop_is_a_failed_node(loop):
    loop["reply"] = ToolRunResult(answer="", rounds=1, error="ollama unreachable")
    out = _derived().execute(_context(), tool_loop=True)
    assert out.status == AgentStatus.FAILED
    assert "ollama unreachable" in (out.error or "")


def test_agents_called_as_tools_run_under_the_workflow():
    """A tool call from inside a run is part of that run — its routing rows
    must carry the run's workflow id, not a random one."""
    nick = _agent_nick()
    tools = {t.name: t for t in agentnick_control.build_tools(nick, workflow_id="wf-9", user_id="u1")}
    tools["run_supplier_ranking"].handler()
    ctx = nick.agents["supplier_ranking"].contexts[0]
    assert ctx.workflow_id == "wf-9"
    assert ctx.user_id == "u1"
