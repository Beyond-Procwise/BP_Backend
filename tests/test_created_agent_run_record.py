"""Every run of a created agent leaves a record that can explain it later:
what it was told, which model answered, what each tool was asked and returned,
and what it said.

Before this a run left its answer and the names of the tools it used. The
system prompt it ran under — its instructions and the policies in front of it
at that moment — was gone when the run ended.
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
from repositories import agent_run_repo
from services import tool_runtime
from services.tool_runtime import Tool, ToolCall, ToolRunResult

INSTRUCTIONS = "Find suppliers whose invoices exceed their PO by more than 5%."


class _Registry:
    def tool_schemas(self):
        return []


class _Created(base_agent.BaseAgent):
    def run(self, *a, **k):  # pragma: no cover
        raise AssertionError("run() must not be called for a tool-loop node")


def _agent():
    nick = SimpleNamespace(
        settings=SimpleNamespace(script_user="tester", extraction_model="m"),
        prompt_engine=PromptEngine(prompt_rows=[{
            "prompt_id": 7, "prompt_name": "overcharge_hunter_instructions",
            "prompt_type": "agent_instructions",
            "prompt_linked_agents": "overcharge_hunter", "prompts_desc": INSTRUCTIONS}]),
        policy_engine=PolicyEngine(policy_rows=[]),
        learning_repository=None, auto_registry=_Registry(), agents={},
    )
    agent = _Created(nick)
    agent.governance_slug = "overcharge-hunter"
    return agent


def _ctx():
    return AgentContext(workflow_id="wf-9", agent_id="overcharge-hunter", user_id="u1",
                        input_data={"deal_id": "D1"})


@pytest.fixture(autouse=True)
def _no_training_files(monkeypatch):
    monkeypatch.setattr(base_agent.BaseAgent, "_record_context_example",
                        lambda self, ctx, out: None)


@pytest.fixture()
def recorded(monkeypatch):
    rows = []
    monkeypatch.setattr(agent_run_repo, "record", lambda **kw: rows.append(kw))
    return rows


@pytest.fixture()
def loop(monkeypatch):
    def fake_run_tools(task, tools, system, **kw):
        return ToolRunResult(
            answer="Two suppliers billed above their PO.", rounds=2,
            calls=[ToolCall(name="get_corpus_facts", arguments={"query": "overbilling"},
                            ok=True, result={"rows": [{"supplier": "Acme", "over": 812.5}]})],
            model="BeyondProcwise/AgentNick:unified", system=system,
            tools_offered=[t.name for t in tools],
        )

    monkeypatch.setattr(agentnick_control, "run_tools", fake_run_tools)


def test_a_run_is_recorded_with_what_the_agent_was_told(loop, recorded):
    out = _agent().execute(_ctx(), tool_loop=True)

    assert out.status == AgentStatus.SUCCESS
    (row,) = recorded
    assert row["agent_slug"] == "overcharge-hunter"
    assert row["workflow_id"] == "wf-9" and row["user_id"] == "u1"
    assert row["model"] == "BeyondProcwise/AgentNick:unified"
    assert INSTRUCTIONS in row["system_prompt"]
    assert "D1" in row["task"]
    assert "get_corpus_facts" in row["tools_offered"]
    assert row["answer"] == "Two suppliers billed above their PO."
    assert row["rounds"] == 2
    assert row["started_at"] <= row["finished_at"]


def test_the_record_keeps_what_each_tool_returned(loop, recorded):
    _agent().execute(_ctx(), tool_loop=True)
    (call,) = recorded[0]["calls"]
    assert call.name == "get_corpus_facts"
    assert call.result == {"rows": [{"supplier": "Acme", "over": 812.5}]}


def test_the_run_names_its_record_and_still_hides_what_tools_returned(loop, recorded):
    out = _agent().execute(_ctx(), tool_loop=True)

    assert out.data["run_id"] == recorded[0]["run_id"]
    assert out.data["recorded"] is True
    assert "Acme" not in str(out.data)             # the node result never carries raw rows


def test_a_failed_run_is_recorded_too(monkeypatch, recorded):
    monkeypatch.setattr(
        agentnick_control, "run_tools",
        lambda task, tools, system, **kw: ToolRunResult(
            answer="", rounds=1, error="ollama unreachable", system=system, model="m"),
    )
    out = _agent().execute(_ctx(), tool_loop=True)

    assert out.status == AgentStatus.FAILED
    assert recorded[0]["error"] == "ollama unreachable"


def test_a_run_whose_record_could_not_be_written_says_so(loop, monkeypatch):
    """The answer is still returned — the work was done — but it must not look
    like a run that can be explained later."""
    def broken(**kw):
        raise RuntimeError("database is down")

    monkeypatch.setattr(agent_run_repo, "record", broken)
    out = _agent().execute(_ctx(), tool_loop=True)

    assert out.status == AgentStatus.SUCCESS
    assert out.data["recorded"] is False
    assert out.data["answer"] == "Two suppliers billed above their PO."


# --- the loop reports what it actually ran with -------------------------------

def test_the_loop_reports_its_model_prompt_and_tools(monkeypatch):
    monkeypatch.setattr(
        tool_runtime, "_chat",
        lambda messages, schemas, model, timeout_s: {"role": "assistant", "content": "done"},
    )
    monkeypatch.setattr(tool_runtime, "_gate", lambda content, *a, **k: content)
    tool = Tool(name="get_policy", description="", parameters={"type": "object"},
                handler=lambda **kw: {})
    result = tool_runtime.run_tools("task", [tool], "SYSTEM TEXT", model="some-model")

    assert result.model == "some-model"
    assert result.system == "SYSTEM TEXT"
    assert result.tools_offered == ["get_policy"]
    assert "SYSTEM TEXT" not in str(result.to_dict())   # never part of what a caller is shown


# --- what is stored about a tool's result ------------------------------------

def test_a_long_result_is_kept_in_part_with_the_hash_of_the_whole():
    big = {"rows": ["x" * 100] * 1000}
    stored = agent_run_repo._call_record(
        ToolCall(name="get_corpus_facts", arguments={}, ok=True, result=big))

    assert stored["result_truncated"] is True
    assert len(stored["result"]) == agent_run_repo.RESULT_MAX_CHARS
    assert len(stored["result_sha256"]) == 64


def test_a_short_result_is_kept_whole():
    stored = agent_run_repo._call_record(
        ToolCall(name="get_policy", arguments={"query": "x"}, ok=True, result={"a": 1}))
    assert stored["result"] == '{"a": 1}'
    assert stored["result_truncated"] is False
