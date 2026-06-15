"""End-to-end: the declarative WorkflowEngine (the default path) shares the same
agentic blackboard as the legacy path via the orchestrator's wiring adapter.

This guards the live-analysis finding that the engine calls agent.execute()
directly and would otherwise bypass the Phase 1-4 interconnection.
"""
import os
import sys
from types import SimpleNamespace

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from orchestration.orchestrator import Orchestrator
from orchestration.workflow_engine import WorkflowEngine, WorkflowGraph, WorkflowNode
from orchestration.workflow_context import SignalType
from agents.base_agent import AgentOutput, AgentStatus


class _Agent:
    def __init__(self, payload, emit=False):
        self.payload = payload
        self.emit = emit
        self._wf = None
        self.saw = None

    def set_workflow_context(self, wf):
        self._wf = wf

    def execute(self, context):
        self.saw = context.input_data.get("_workflow_blackboard")
        if self.emit and self._wf is not None:
            self._wf.emit_signal(agent="Node", signal_type=SignalType.CONFIDENCE_LOW,
                                 message="weak")
        return AgentOutput(status=AgentStatus.SUCCESS, data=self.payload,
                           pass_fields=self.payload)


def _orch(agents, pattern_service=None):
    nick = SimpleNamespace(
        settings=SimpleNamespace(script_user="tester", max_workers=1, parallel_processing=False),
        agents=agents,
        policy_engine=SimpleNamespace(),
        query_engine=SimpleNamespace(),
        routing_engine=SimpleNamespace(routing_model=None),
        pattern_service=pattern_service,
    )
    return Orchestrator(nick)


def _two_node_graph():
    g = WorkflowGraph(name="t", description="d")
    g.add_node(WorkflowNode(name="first", agent_type="first",
                            output_to_shared=["a"], required=True))
    g.add_node(WorkflowNode(name="second", agent_type="second", required=False))
    g.add_edge("first", "second")
    return g


def test_engine_attaches_shared_blackboard_and_records_results():
    first = _Agent({"a": 1})
    second = _Agent({"b": 2})
    orch = _orch({"first": first, "second": second})

    state = orch._workflow_engine.execute(_two_node_graph(), workflow_id="wf-eng",
                                          input_data={"workflow": "t"})

    assert "completed" in str(getattr(state.status, "value", state.status)).lower()
    # both nodes got a blackboard view; second saw the first node's recorded result
    assert first.saw is not None and second.saw is not None
    assert second.saw["prior_results"].get("first") == {"a": 1}


def test_engine_signals_land_on_shared_context_for_learning():
    recorded = []

    class PS:
        def get_patterns(self, **_):
            return []

        def record_pattern(self, pattern_type, pattern_text, category="", confidence=0.5):
            recorded.append((pattern_type, pattern_text))

        def reinforce_pattern(self, *a, **k):
            pass

    node = _Agent({"a": 1}, emit=True)
    orch = _orch({"first": node, "second": _Agent({"b": 2})}, pattern_service=PS())

    g = WorkflowGraph(name="t", description="d")
    g.add_node(WorkflowNode(name="first", agent_type="first", required=True))

    from agents.base_agent import AgentContext
    ctx = AgentContext(workflow_id="wf-sig", agent_id="t", user_id="u",
                       input_data={"workflow": "t"})
    orch._workflow_engine.execute(g, workflow_id="wf-sig", input_data={"workflow": "t"})

    # the CONFIDENCE_LOW signal emitted inside the engine reached the shared
    # blackboard, so the learning loop can record it
    wf = orch._wf_contexts.get("wf-sig")
    assert wf is not None
    assert wf.has_signal(SignalType.CONFIDENCE_LOW)
    orch._learn_from_workflow("wf-sig", "t", ctx, {"ok": True}, success=True)
    assert any(r[0] == "workflow_concern" for r in recorded)
