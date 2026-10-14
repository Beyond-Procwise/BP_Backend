"""The model chosen for a created agent is the model it runs on, and every run
counts the tokens it used.

POST /agents validated a model choice against the offered models and stored it.
The run then called the tool loop with no model at all, which silently
substituted the default: the picker did nothing. And nothing counted what a
run cost, so nothing could be charged for.
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents import base_agent, named_outputs
from agents.base_agent import AgentContext, AgentStatus
from engines.policy_engine import PolicyEngine
from orchestration import agentnick_control
from orchestration.prompt_engine import PromptEngine
from repositories import agent_run_repo
from services import tool_runtime
from services.tool_runtime import ToolRunResult


class _Registry:
    def tool_schemas(self):
        return []


class _Created(base_agent.BaseAgent):
    def run(self, *a, **k):  # pragma: no cover
        raise AssertionError("run() must not be called for a tool-loop node")


def _agent(own_model=None):
    nick = SimpleNamespace(
        settings=SimpleNamespace(script_user="tester", extraction_model="m"),
        prompt_engine=PromptEngine(prompt_rows=[]),
        policy_engine=PolicyEngine(policy_rows=[]),
        learning_repository=None, auto_registry=_Registry(), agents={},
        get_agent_model=lambda slug, fallback=None, strict=False: (
            own_model if slug == "overcharge-hunter" else None),
    )
    agent = _Created(nick)
    agent.governance_slug = "overcharge-hunter"
    return agent


def _ctx():
    return AgentContext(workflow_id="wf-9", agent_id="overcharge-hunter", user_id="u1",
                        input_data={"deal_id": "D1"})


@pytest.fixture(autouse=True)
def recorded(monkeypatch):
    rows = []
    monkeypatch.setattr(base_agent.BaseAgent, "_record_context_example",
                        lambda self, ctx, out: None)
    monkeypatch.setattr(agent_run_repo, "record", lambda **kw: rows.append(kw))
    return rows


@pytest.fixture()
def loop(monkeypatch):
    seen = {}

    def fake_run_tools(task, tools, system, **kw):
        seen.update(kw)
        tool_runtime.note_usage({"prompt_eval_count": 900, "eval_count": 120})
        tool_runtime.note_usage({"prompt_eval_count": 1400, "eval_count": 60})
        return ToolRunResult(answer="ok", rounds=2, model=kw.get("model") or "the-default",
                             system=system)

    monkeypatch.setattr(agentnick_control, "run_tools", fake_run_tools)
    return seen


# --- the model ---------------------------------------------------------------

def test_an_agent_given_a_model_runs_on_it(loop, recorded):
    _agent(own_model="BeyondProcwise/AgentNick:other").execute(_ctx(), tool_loop=True)

    assert loop["model"] == "BeyondProcwise/AgentNick:other"
    assert recorded[0]["model"] == "BeyondProcwise/AgentNick:other"


def test_an_agent_given_no_model_runs_on_the_standard_one(loop, recorded):
    _agent(own_model=None).execute(_ctx(), tool_loop=True)

    assert loop["model"] is None            # the loop's own default applies, as before
    assert recorded[0]["model"] == "the-default"


def test_the_named_outputs_are_filled_by_the_same_model(loop, monkeypatch):
    asked = {}
    monkeypatch.setattr(named_outputs, "structured",
                        lambda system, user, schema, **kw: asked.update(kw) or {})
    agent = _agent(own_model="BeyondProcwise/AgentNick:other")
    agent.output_spec = [{"name": "supplier", "description": ""}]
    agent.execute(_ctx(), tool_loop=True)

    assert asked["model"] == "BeyondProcwise/AgentNick:other"


# --- the tokens --------------------------------------------------------------

def test_a_run_counts_the_tokens_of_every_model_call_it_made(loop, recorded):
    out = _agent().execute(_ctx(), tool_loop=True)

    assert out.status == AgentStatus.SUCCESS
    assert recorded[0]["prompt_tokens"] == 2300
    assert recorded[0]["completion_tokens"] == 180
    assert recorded[0]["model_calls"] == 2
    assert out.data["usage"] == {"prompt_tokens": 2300, "completion_tokens": 180,
                                 "model_calls": 2}


def test_filling_the_named_outputs_is_counted_too(loop, recorded, monkeypatch):
    def fake_structured(system, user, schema, **kw):
        tool_runtime.note_usage({"prompt_eval_count": 500, "eval_count": 40})
        return {}

    monkeypatch.setattr(named_outputs, "structured", fake_structured)
    agent = _agent()
    agent.output_spec = [{"name": "supplier", "description": ""}]
    agent.execute(_ctx(), tool_loop=True)

    assert recorded[0]["prompt_tokens"] == 2800
    assert recorded[0]["model_calls"] == 3


def test_a_run_started_inside_another_counts_toward_both():
    with tool_runtime.count_tokens() as outer:
        tool_runtime.note_usage({"prompt_eval_count": 100, "eval_count": 10})
        with tool_runtime.count_tokens() as inner:
            tool_runtime.note_usage({"prompt_eval_count": 30, "eval_count": 3})
        tool_runtime.note_usage({"prompt_eval_count": 1, "eval_count": 1})

    assert inner == {"prompt_tokens": 30, "completion_tokens": 3, "model_calls": 1}
    assert outer == {"prompt_tokens": 131, "completion_tokens": 14, "model_calls": 3}


def test_tokens_noted_outside_any_run_are_dropped_quietly():
    tool_runtime.note_usage({"prompt_eval_count": 5, "eval_count": 5})   # must not raise


def test_a_reply_that_reports_no_counts_still_counts_as_a_call():
    with tool_runtime.count_tokens() as usage:
        tool_runtime.note_usage({})
    assert usage == {"prompt_tokens": 0, "completion_tokens": 0, "model_calls": 1}


def test_the_loop_notes_the_counts_the_model_server_reports(monkeypatch):
    class _Reply:
        def raise_for_status(self):
            pass

        def json(self):
            return {"message": {"role": "assistant", "content": "done"},
                    "prompt_eval_count": 321, "eval_count": 12}

    monkeypatch.setattr(tool_runtime.egress, "post", lambda url, **kw: _Reply())
    monkeypatch.setattr(tool_runtime, "_gate", lambda content, *a, **k: content)
    with tool_runtime.count_tokens() as usage:
        tool_runtime.run_tools("task", [], "SYS", model="m")

    assert usage == {"prompt_tokens": 321, "completion_tokens": 12, "model_calls": 1}


def test_the_structured_call_notes_its_counts(monkeypatch):
    class _Reply:
        status_code = 200
        text = ""

        def raise_for_status(self):
            pass

        def json(self):
            return {"message": {"content": "{}"}, "prompt_eval_count": 77, "eval_count": 5}

    monkeypatch.setattr(tool_runtime.egress, "post", lambda url, **kw: _Reply())
    with tool_runtime.count_tokens() as usage:
        tool_runtime.structured("SYS", "USER", {"type": "object"})

    assert usage == {"prompt_tokens": 77, "completion_tokens": 5, "model_calls": 1}
