"""Phase 1 — agentic interconnection: the shared WorkflowContext blackboard.

Verifies that the orchestrator wires a single WorkflowContext per workflow_id
into every agent it runs: the agent gets live signal access, a read-only view of
prior results, and its own result is recorded for downstream agents.
"""
import os
import sys
import threading
from types import SimpleNamespace

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from orchestration.orchestrator import Orchestrator
from orchestration.workflow_context import WorkflowContext, SignalType
from agents.base_agent import AgentContext, AgentOutput, AgentStatus


def _nick(agents):
    return SimpleNamespace(
        settings=SimpleNamespace(script_user="tester", max_workers=2, parallel_processing=False),
        agents=agents,
        policy_engine=SimpleNamespace(),
        query_engine=SimpleNamespace(),
        routing_engine=SimpleNamespace(routing_model=None),
    )


class _BlackboardAgent:
    """Minimal agent surface: receives the workflow context, reads the blackboard,
    emits a signal, and returns a result dict."""

    def __init__(self, payload):
        self.payload = payload
        self._wf = None
        self.saw_blackboard = None

    def set_workflow_context(self, wf):
        self._wf = wf

    def execute(self, context):
        self.saw_blackboard = context.input_data.get("_workflow_blackboard")
        if self._wf is not None:
            self._wf.emit_signal(
                agent="BlackboardAgent",
                signal_type=SignalType.SUGGEST_AGENT,
                message="next please",
                data={"agent": "approvals"},
            )
        return AgentOutput(status=AgentStatus.SUCCESS, data=self.payload)


def test_execute_agent_records_result_and_injects_blackboard():
    agent = _BlackboardAgent({"k": "v"})
    orch = Orchestrator(_nick({"a": agent}))
    ctx = AgentContext(workflow_id="wf-1", agent_id="a", user_id="t",
                       input_data={"workflow": "demo"})

    result = orch._execute_agent("a", ctx)

    assert result.data == {"k": "v"}
    wf = orch._wf_contexts["wf-1"]
    # downstream agents can read this agent's output
    assert wf.get_prior_result("a") == {"k": "v"}
    # the agent received a read-only blackboard view
    assert agent.saw_blackboard is not None
    assert agent.saw_blackboard["workflow_id"] == "wf-1"
    # emit_signal was live (context was attached)
    assert wf.has_signal(SignalType.SUGGEST_AGENT)


def test_blackboard_carries_prior_results_to_later_agents():
    first = _BlackboardAgent({"from_first": 1})
    second = _BlackboardAgent({"from_second": 2})
    orch = Orchestrator(_nick({"first": first, "second": second}))
    ctx1 = AgentContext(workflow_id="wf-2", agent_id="first", user_id="t",
                        input_data={"workflow": "demo"})
    ctx2 = AgentContext(workflow_id="wf-2", agent_id="second", user_id="t",
                        input_data={"workflow": "demo"})

    orch._execute_agent("first", ctx1)
    orch._execute_agent("second", ctx2)

    # second agent's blackboard view includes the first agent's recorded result
    assert second.saw_blackboard["prior_results"].get("first") == {"from_first": 1}


def test_wf_context_registry_is_released_per_workflow():
    orch = Orchestrator(_nick({"a": _BlackboardAgent({"x": 1})}))
    orch._release_wf_context("nope")  # no-op on unknown id
    ctx = AgentContext(workflow_id="wf-3", agent_id="a", user_id="t",
                       input_data={"workflow": "demo"})
    orch._execute_agent("a", ctx)
    assert "wf-3" in orch._wf_contexts
    orch._release_wf_context("wf-3")
    assert "wf-3" not in orch._wf_contexts


def test_workflow_context_is_threadsafe_under_parallel_writes():
    wf = WorkflowContext(goal="g", workflow_id="wf-par")

    def worker(i):
        wf.record_result(f"agent{i}", {"i": i})
        wf.emit_signal(agent=f"agent{i}", signal_type=SignalType.CONFIDENCE_LOW,
                       message="m")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(64)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(wf.agent_results) == 64
    assert len(wf.get_signals(SignalType.CONFIDENCE_LOW)) == 64
