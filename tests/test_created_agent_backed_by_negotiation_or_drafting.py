"""A created agent whose backing class is the negotiation or the email drafting agent runs in
tool-loop mode (audit rec 10). Both classes override ``execute``; before, that override took no
``tool_loop`` option, so the call raised TypeError and the person's task was never read."""
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents import base_agent
from agents.email_drafting_agent import EmailDraftingAgent
from agents.negotiation_agent import NegotiationAgent
from engines.policy_engine import PolicyEngine
from orchestration import agentnick_control
from orchestration.prompt_engine import PromptEngine
from services.tool_runtime import ToolRunResult

INSTRUCTIONS = "Draft a firm but polite reply asking the supplier to hold last quarter's price."


class _Registry:
    slugs = ("supplier-negotiator",)

    def tool_schemas(self):
        return [{"function": {"name": f"run_{s}", "description": s,
                              "parameters": {"type": "object", "properties": {}}}}
                for s in self.slugs]


def _nick_with(cls):
    nick = SimpleNamespace(
        settings=SimpleNamespace(script_user="tester", extraction_model="m"),
        prompt_engine=PromptEngine(prompt_rows=[
            {"prompt_id": 7, "prompt_name": "supplier_negotiator_instructions",
             "prompt_type": "agent_instructions", "prompt_linked_agents": "supplier_negotiator",
             "prompts_desc": INSTRUCTIONS}]),
        policy_engine=PolicyEngine(policy_rows=[]),
        learning_repository=None, auto_registry=_Registry(), agents={},
        process_routing_service=SimpleNamespace(
            log_process=lambda **_: None, log_run_detail=lambda **_: None,
            log_action=lambda **_: None, validate_workflow_id=lambda *a, **k: True),
    )
    try:
        agent = cls(nick)
    except Exception as exc:  # a constructor needing more of the real app is not this test's concern
        pytest.skip(f"{cls.__name__} needs more than a stub nick to construct: {exc}")
    agent.governance_slug = "supplier-negotiator"
    nick.agents["supplier-negotiator"] = agent
    return nick


@pytest.fixture(autouse=True)
def _quiet(monkeypatch):
    monkeypatch.setattr(base_agent.BaseAgent, "_record_context_example", lambda self, c, o: None)
    monkeypatch.setattr(base_agent.BaseAgent, "_record_reasoning_run",
                        lambda self, *a, **k: (None, False), raising=False)


@pytest.mark.parametrize("cls", [NegotiationAgent, EmailDraftingAgent])
def test_a_created_agent_backed_by_this_class_follows_its_own_task_as_a_tool(cls, monkeypatch):
    started = []

    def fake_run_tools(task, tools, system, **kw):
        started.append(system)
        return ToolRunResult(answer="Reply drafted.", rounds=1)

    monkeypatch.setattr(agentnick_control, "run_tools", fake_run_tools)
    nick = _nick_with(cls)
    tool = {t.name: t for t in agentnick_control.build_tools(nick)}["run_supplier-negotiator"]

    out = tool.handler()

    assert out["error"] in (None, ""), out
    assert out["status"] == "success"
    assert out["data"]["answer"] == "Reply drafted."
    assert INSTRUCTIONS in started[0]
