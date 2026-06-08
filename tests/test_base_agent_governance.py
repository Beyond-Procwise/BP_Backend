import os
import sys
from types import SimpleNamespace

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents import base_agent
from orchestration.prompt_engine import PromptEngine
from engines.policy_engine import PolicyEngine


def _make_agent():
    prompt_rows = [
        {
            "prompt_id": 1,
            "prompt_name": "supplier_ranking_justification",
            "prompt_type": "justification",
            "prompt_linked_agents": "supplier_ranking_agent",
            "prompts_desc": '{"prompt_template": "Score for {name} is {score}."}',
        },
        {
            "prompt_id": 2,
            "prompt_name": "global_only",
            "prompt_type": "info",
            "prompt_linked_agents": "",
            "prompts_desc": '{"prompt_template": "Global template."}',
        },
    ]
    policy_rows = [
        {
            "policy_id": 1,
            "policy_name": "WeightAllocationPolicy",
            "policy_type": "supplier_ranking",
            "policy_desc": "weights",
            "policy_details": '{"rules": {"default_weights": {"price": 1.0}}}',
            "policy_linked_agents": "supplier_ranking_agent",
        },
        {
            "policy_id": 2,
            "policy_name": "UnrelatedPolicy",
            "policy_type": "other",
            "policy_desc": "n/a",
            "policy_details": "{}",
            "policy_linked_agents": "some_other_agent",
        },
    ]
    agent_nick = SimpleNamespace(
        settings=SimpleNamespace(script_user="tester", extraction_model="gpt-oss"),
        prompt_engine=PromptEngine(prompt_rows=prompt_rows),
        policy_engine=PolicyEngine(policy_rows=policy_rows),
        learning_repository=None,
    )

    class SupplierRankingAgent(base_agent.BaseAgent):
        def run(self, *a, **k):
            return None

    return SupplierRankingAgent(agent_nick)


def test_resolve_prompt_agent_scoped_with_formatting():
    agent = _make_agent()
    out = agent.resolve_prompt("supplier_ranking_justification", name="ACME", score=9)
    assert out == "Score for ACME is 9."


def test_resolve_prompt_falls_back_to_global_then_none():
    agent = _make_agent()
    assert agent.resolve_prompt("global_only") == "Global template."
    assert agent.resolve_prompt("does_not_exist") is None


def test_governing_policies_returns_only_linked():
    agent = _make_agent()
    names = {p.get("policyName") for p in agent.governing_policies()}
    assert names == {"WeightAllocationPolicy"}


def test_governing_policy_by_name():
    agent = _make_agent()
    policy = agent.governing_policy("weight_allocation_policy")
    assert policy is not None
    assert policy.get("policyName") == "WeightAllocationPolicy"
    assert agent.governing_policy("nonexistent_policy") is None
