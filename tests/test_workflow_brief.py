"""Phase 2 — shared knowledge brief attached to the blackboard.

The orchestrator enriches each workflow's WorkflowContext with a procurement
brief (lifecycle + learned patterns + policies) fed from AgentNick's stores, so
every agent starts knowledgeable. Enrichment is best-effort and never fatal.
"""
import os
import sys
from types import SimpleNamespace

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from orchestration.orchestrator import Orchestrator
from agents.base_agent import AgentContext, AgentOutput, AgentStatus


class _StubPatternService:
    def __init__(self, patterns):
        self._patterns = patterns
        self.seen_category = "unset"

    def get_patterns(self, category=None, min_confidence=0.0):
        self.seen_category = category
        return list(self._patterns)


def _nick(agents, pattern_service=None):
    return SimpleNamespace(
        settings=SimpleNamespace(script_user="tester", max_workers=1, parallel_processing=False),
        agents=agents,
        policy_engine=SimpleNamespace(),
        query_engine=SimpleNamespace(),
        routing_engine=SimpleNamespace(routing_model=None),
        pattern_service=pattern_service,
    )


class _Agent:
    def __init__(self):
        self.saw = None

    def set_workflow_context(self, wf):
        self._wf = wf

    def execute(self, context):
        self.saw = context.input_data.get("_workflow_blackboard")
        return AgentOutput(status=AgentStatus.SUCCESS, data={"ok": True})


def test_brief_is_attached_with_patterns_and_policies():
    ps = _StubPatternService([
        {"pattern_text": "suppliers in EU bill in EUR", "category": "Raw Materials",
         "confidence": 0.9},
    ])
    agent = _Agent()
    orch = Orchestrator(_nick({"a": agent}, pattern_service=ps))
    ctx = AgentContext(
        workflow_id="wf-brief", agent_id="a", user_id="t",
        input_data={"workflow": "supplier_ranking", "category": "Raw Materials"},
        policy_context=[{"policyName": "VolumeDiscount"}],
    )

    orch._execute_agent("a", ctx)

    wf = orch._wf_contexts["wf-brief"]
    brief = wf.get_procurement_brief()
    assert brief is not None
    assert "suppliers in EU bill in EUR" in brief.get("patterns", [])
    assert "VolumeDiscount" in brief.get("active_policies", [])
    # the pattern store was queried scoped to the workflow's category
    assert ps.seen_category == "Raw Materials"
    # the agent saw the brief on its blackboard view
    assert agent.saw["procurement_brief"] is not None


def test_enrichment_is_best_effort_when_pattern_service_raises():
    class Boom:
        def get_patterns(self, **_):
            raise RuntimeError("store down")

    agent = _Agent()
    orch = Orchestrator(_nick({"a": agent}, pattern_service=Boom()))
    ctx = AgentContext(workflow_id="wf-safe", agent_id="a", user_id="t",
                       input_data={"workflow": "demo"})

    # must not raise despite the failing pattern store
    result = orch._execute_agent("a", ctx)
    assert result.data == {"ok": True}
    # a (pattern-less) brief is still attached
    assert orch._wf_contexts["wf-safe"].get_procurement_brief() is not None
