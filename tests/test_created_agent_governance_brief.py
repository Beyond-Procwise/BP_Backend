"""The prompts and policies linked to a created agent are put in front of it.

Linking a prompt or a policy to a created agent recorded the link and nothing
else. The agent's loop was given its own instructions; anything else governing
it sat behind ``get_prompt`` / ``get_policy``, to be fetched if the model
happened to ask. A rule the agent is never shown is a rule it follows by luck.
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
from services.tool_runtime import ToolRunResult

INSTRUCTIONS = "Find suppliers whose invoices exceed their PO by more than 5%."
TONE = "Write for a finance director: figures first, no adjectives."
OTHER = "Rank suppliers on price, then delivery."

PROMPTS = [
    {"prompt_id": 7, "prompt_name": "overcharge_hunter_instructions",
     "prompt_type": "agent_instructions",
     "prompt_linked_agents": "overcharge_hunter", "prompts_desc": INSTRUCTIONS},
    {"prompt_id": 8, "prompt_name": "finance_tone", "prompt_type": "style",
     "prompt_linked_agents": "email_drafting, overcharge_hunter", "prompts_desc": TONE},
    {"prompt_id": 9, "prompt_name": "ranking_method", "prompt_type": "method",
     "prompt_linked_agents": "supplier_ranking", "prompts_desc": OTHER},
]

POLICIES = [
    {"policy_id": 21, "policy_name": "OverbillingTolerance",
     "policy_desc": "An invoice may exceed its purchase order by at most this much.",
     "policy_linked_agents": "overcharge_hunter",
     "policy_details": {"rules": {"tolerance_pct": 5, "escalate_above_gbp": 10000}}},
    {"policy_id": 22, "policy_name": "SupplierWeights",
     "policy_desc": "Weights for ranking suppliers.",
     "policy_linked_agents": "supplier_ranking",
     "policy_details": {"rules": {"price": 0.6}}},
]


class _Registry:
    def tool_schemas(self):
        return []


class _Created(base_agent.BaseAgent):
    def run(self, *a, **k):  # pragma: no cover - must never be reached
        raise AssertionError("run() must not be called for a tool-loop node")


def _agent(prompts=PROMPTS, policies=POLICIES, slug="overcharge-hunter"):
    nick = SimpleNamespace(
        settings=SimpleNamespace(script_user="tester", extraction_model="m"),
        prompt_engine=PromptEngine(prompt_rows=prompts),
        policy_engine=PolicyEngine(policy_rows=policies),
        learning_repository=None,
        auto_registry=_Registry(),
        agents={},
    )
    agent = _Created(nick)
    agent.governance_slug = slug
    return agent


@pytest.fixture(autouse=True)
def _no_training_files(monkeypatch):
    monkeypatch.setattr(base_agent.BaseAgent, "_record_context_example",
                        lambda self, ctx, out: None)


@pytest.fixture()
def system(monkeypatch):
    """The system prompt the loop was started with."""
    seen = {}

    def fake_run_tools(task, tools, system, **kw):
        seen["system"] = system
        return ToolRunResult(answer="ok", rounds=1)

    monkeypatch.setattr(agentnick_control, "run_tools", fake_run_tools)
    return seen


def _run(agent):
    out = agent.execute(
        AgentContext(workflow_id="wf-9", agent_id="overcharge-hunter", user_id="u1",
                     input_data={"deal_id": "D1"}),
        tool_loop=True,
    )
    assert out.status == AgentStatus.SUCCESS
    return out


def test_a_linked_prompt_is_in_front_of_the_agent(system):
    _run(_agent())
    assert TONE in system["system"]
    assert "finance_tone" in system["system"]


def test_a_linked_policy_is_in_front_of_the_agent_with_its_values(system):
    _run(_agent())
    text = system["system"]
    assert "OverbillingTolerance" in text
    assert "An invoice may exceed its purchase order by at most this much." in text
    assert "tolerance_pct" in text and "5" in text
    assert "escalate_above_gbp" in text and "10000" in text


def test_what_governs_other_agents_is_not_shown(system):
    _run(_agent())
    assert OTHER not in system["system"]
    assert "SupplierWeights" not in system["system"]


def test_the_agents_own_instructions_are_not_repeated_as_a_linked_prompt(system):
    _run(_agent())
    assert system["system"].count(INSTRUCTIONS) == 1


def test_policies_are_stated_as_binding_and_prompts_as_guidance(system):
    _run(_agent())
    text = system["system"]
    assert text.index("must not") < text.index("OverbillingTolerance")
    # The agent's own task comes first, then what governs it.
    assert text.index(INSTRUCTIONS) < text.index("OverbillingTolerance")


def test_an_agent_with_nothing_linked_is_told_nothing_extra(system):
    _run(_agent(prompts=PROMPTS[:1], policies=[]))
    assert "Policies that bind you" not in system["system"]
    assert "Guidance linked to you" not in system["system"]


def test_what_does_not_fit_is_named_not_dropped(system):
    """A rule left out without a word is the fault this fixes. What cannot be
    shown in full is named, with the tool that fetches it."""
    long_policies = [
        {"policy_id": 30 + i, "policy_name": f"LongPolicy{i}",
         "policy_desc": "x" * 3000, "policy_linked_agents": "overcharge_hunter",
         "policy_details": {"rules": {"n": i}}}
        for i in range(6)
    ]
    _run(_agent(policies=long_policies))
    text = system["system"]
    for i in range(6):
        assert f"LongPolicy{i}" in text            # every one is at least named
    assert "get_policy" in text
    assert len(text) < 20000


def test_a_built_in_agent_is_not_given_the_brief():
    """Only created agents reason this way; the brief is theirs."""
    agent = _agent()
    agent.governance_slug = None
    assert agent.governance_brief() is None
