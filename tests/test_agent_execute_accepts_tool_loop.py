"""Every agent can run in tool-loop mode.

The workflow engine and the agent-as-tool path call ``agent.execute(ctx, tool_loop=True)`` on
any agent, including a derived agent whose backing class is the negotiation or the email
drafting agent. Those two classes overrode ``execute(self, context)`` with no ``tool_loop``
option, so such a call raised TypeError, and nothing tested it.
"""
import inspect
import os
import sys

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents import base_agent
from agents.base_agent import AgentContext, AgentOutput, AgentStatus, BaseAgent
from agents.email_drafting_agent import EmailDraftingAgent
from agents.negotiation_agent import NegotiationAgent


def _all_agent_classes():
    import importlib
    import pkgutil

    import agents

    for mod in pkgutil.iter_modules(agents.__path__):
        try:
            importlib.import_module(f"agents.{mod.name}")
        except Exception:  # a module that cannot import here is not this test's business
            continue
    seen, stack = [], [BaseAgent]
    while stack:
        cls = stack.pop()
        for sub in cls.__subclasses__():
            if sub not in seen:
                seen.append(sub)
                stack.append(sub)
    return seen


def _accepts_tool_loop(fn) -> bool:
    params = inspect.signature(fn).parameters
    return "tool_loop" in params or any(p.kind is inspect.Parameter.VAR_KEYWORD
                                        for p in params.values())


def test_every_execute_override_accepts_tool_loop():
    offenders = [f"{c.__module__}.{c.__name__}" for c in _all_agent_classes()
                 if "execute" in c.__dict__ and not _accepts_tool_loop(c.__dict__["execute"])]
    assert offenders == []


@pytest.mark.parametrize("cls", [NegotiationAgent, EmailDraftingAgent])
def test_the_two_known_agents_pass_tool_loop_through(cls, monkeypatch):
    seen = {}

    def fake_execute(self, context, *, tool_loop=False):
        seen["tool_loop"] = tool_loop
        return AgentOutput(status=AgentStatus.SUCCESS, data={})

    monkeypatch.setattr(BaseAgent, "execute", fake_execute)
    agent = object.__new__(cls)  # no constructor: only the call path is under test
    ctx = AgentContext(workflow_id="w", agent_id="a", user_id="u", input_data={})
    out = cls.execute(agent, ctx, tool_loop=True)
    assert out.status == AgentStatus.SUCCESS
    assert seen["tool_loop"] is True


@pytest.mark.parametrize("cls", [NegotiationAgent, EmailDraftingAgent])
def test_a_plain_call_still_works(cls, monkeypatch):
    seen = {}

    def fake_execute(self, context, *, tool_loop=False):
        seen["tool_loop"] = tool_loop
        return AgentOutput(status=AgentStatus.SUCCESS, data={})

    monkeypatch.setattr(BaseAgent, "execute", fake_execute)
    ctx = AgentContext(workflow_id="w", agent_id="a", user_id="u", input_data={})
    cls.execute(object.__new__(cls), ctx)
    assert seen["tool_loop"] is False
