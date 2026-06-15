"""Phase 3 — bounded signal-driven feedback loops in the orchestrator.

Agents emit signals onto the shared blackboard; the sequential executor turns
SUGGEST_AGENT into a bounded dynamic next-step and escalation signals into a
single route to ``approvals``. Agents that emit no signals run unchanged.
"""
import os
import sys
from types import SimpleNamespace

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from orchestration.orchestrator import Orchestrator
from orchestration.workflow_context import SignalType
from agents.base_agent import AgentContext, AgentOutput, AgentStatus


def _nick(agents):
    return SimpleNamespace(
        settings=SimpleNamespace(script_user="tester", max_workers=1, parallel_processing=False),
        agents=agents,
        policy_engine=SimpleNamespace(),
        query_engine=SimpleNamespace(),
        routing_engine=SimpleNamespace(routing_model=None),
    )


class _SignallingAgent:
    """Emits one SUGGEST_AGENT signal toward ``suggest`` on its first run."""

    def __init__(self, suggest=None, escalate=False):
        self.suggest = suggest
        self.escalate = escalate
        self._wf = None
        self.ran = 0

    def set_workflow_context(self, wf):
        self._wf = wf

    def execute(self, context):
        self.ran += 1
        if self._wf is not None and self.suggest:
            self._wf.emit_signal(agent="src", signal_type=SignalType.SUGGEST_AGENT,
                                 message="do next", data={"agent": self.suggest})
        if self._wf is not None and self.escalate:
            self._wf.emit_signal(agent="src", signal_type=SignalType.RECOMMEND_ESCALATION,
                                 message="needs human")
        return AgentOutput(status=AgentStatus.SUCCESS, data={"ran": self.ran})


class _PlainAgent:
    def __init__(self):
        self.ran = 0

    def execute(self, context):
        self.ran += 1
        return AgentOutput(status=AgentStatus.SUCCESS, data={"ran": self.ran})


def _ctx(orch):
    return AgentContext(workflow_id="wf-fb", agent_id="root", user_id="t",
                        input_data={"workflow": "demo"})


def test_suggest_agent_signal_adds_bounded_next_step():
    target = _PlainAgent()
    src = _SignallingAgent(suggest="target")
    orch = Orchestrator(_nick({"src": src, "target": target}))

    results = orch._execute_sequential_agents(["src"], _ctx(orch), {})

    assert "src" in results and "target" in results   # target was dynamically run
    assert target.ran == 1


def test_dynamic_additions_are_capped():
    # one source that re-suggests every time it runs would loop; the cap stops it.
    src = _SignallingAgent(suggest="target")
    # 'target' itself also suggests another agent each run
    target = _SignallingAgent(suggest="target2")
    target2 = _SignallingAgent(suggest="target3")
    target3 = _SignallingAgent(suggest="target4")
    target4 = _PlainAgent()
    agents = {"src": src, "target": target, "target2": target2,
              "target3": target3, "target4": target4}
    orch = Orchestrator(_nick(agents))
    orch.MAX_DYNAMIC_AGENTS = 2  # only 2 dynamic additions allowed

    results = orch._execute_sequential_agents(["src"], _ctx(orch), {})

    # src + exactly 2 dynamically-added agents
    assert len(results) == 3
    assert "src" in results and "target" in results and "target2" in results
    assert "target3" not in results


def test_no_agent_runs_twice_via_signals():
    # src suggests itself; the executed-set must prevent re-running it.
    src = _SignallingAgent(suggest="src")
    orch = Orchestrator(_nick({"src": src}))

    orch._execute_sequential_agents(["src"], _ctx(orch), {})

    assert src.ran == 1


def test_escalation_signal_routes_to_approvals_once():
    approvals = _PlainAgent()
    src = _SignallingAgent(escalate=True)
    orch = Orchestrator(_nick({"src": src, "approvals": approvals}))

    results = orch._execute_sequential_agents(["src"], _ctx(orch), {})

    assert "approvals" in results
    assert approvals.ran == 1


def test_plain_agents_unaffected_when_no_signals():
    a = _PlainAgent()
    b = _PlainAgent()
    orch = Orchestrator(_nick({"a": a, "b": b}))

    results = orch._execute_sequential_agents(["a", "b"], _ctx(orch), {})

    assert set(results) == {"a", "b"}
    assert a.ran == 1 and b.ran == 1
