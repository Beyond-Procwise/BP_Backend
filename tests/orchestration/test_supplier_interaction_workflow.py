"""The supplier_interaction graph must actually reach its negotiate node.

The graph declared `output_to_shared=["responses"]` on watch_responses, but
`EmailWatcherAgent` returns `data["supplier_responses"]`
(src/agents/email_watcher_agent.py:572). The engine copies with
`result.data.get(output_field)` (workflow_engine.py:613), so shared_data was
never populated, `_has_responses` was permanently False, and the negotiate and
compare_quotes nodes never executed.

These tests pin the three places the key appears against the two contracts it
has to satisfy: what EmailWatcherAgent emits, and what
`NegotiationAgent._extract_batch_inputs` reads.
"""

from unittest.mock import MagicMock

from agents.base_agent import AgentOutput, AgentStatus
from agents.negotiation_agent import NegotiationAgent
from orchestration.workflow_definitions import (
    build_supplier_interaction_workflow,
    _has_responses,
)
from orchestration.workflow_engine import (
    NodeStatus,
    WorkflowEngine,
    WorkflowState,
)

#: The shape EmailWatcherAgent actually returns — see its `data` dict at
#: src/agents/email_watcher_agent.py:572. Only the key name matters here.
WATCHER_DATA = {
    "workflow_id": "wf-1",
    "round": 1,
    "supplier_responses": [
        {"supplier_id": "SUP-1", "current_offer": 100.0, "target_price": 80.0},
        {"supplier_id": "SUP-2", "current_offer": 120.0, "target_price": 90.0},
    ],
    "status": "complete",
}


def _success_agent(data: dict) -> MagicMock:
    agent = MagicMock()
    agent.execute.return_value = AgentOutput(
        status=AgentStatus.SUCCESS, data=data, pass_fields={}
    )
    return agent


def _state_after_watch_responses() -> WorkflowState:
    """A state populated exactly the way the engine populates it."""
    graph = build_supplier_interaction_workflow()
    state = WorkflowState(
        workflow_id="wf-1", workflow_name="supplier_interaction", user_id="u"
    )
    for field in graph.nodes["watch_responses"].output_to_shared:
        value = WATCHER_DATA.get(field)
        if value:
            state.shared_data[field] = value
    state.node_results = {"watch_responses": dict(WATCHER_DATA)}
    return state


def test_watch_responses_publishes_a_key_the_watcher_actually_emits():
    """output_to_shared naming a key EmailWatcherAgent does not return is a
    silent no-op — the blackboard stays empty and every downstream gate fails."""
    graph = build_supplier_interaction_workflow()
    published = graph.nodes["watch_responses"].output_to_shared

    assert published, "watch_responses must publish something to the blackboard"
    for field in published:
        assert field in WATCHER_DATA, (
            f"watch_responses publishes {field!r}, which EmailWatcherAgent never "
            f"returns; it emits {sorted(WATCHER_DATA)}"
        )


def test_gate_traverses_on_the_watcher_s_real_output():
    """_has_responses must be True once the watcher has produced responses."""
    assert _has_responses(_state_after_watch_responses()) is True


def test_gate_stays_closed_when_no_responses_arrived():
    """The gate must still refuse an empty round — it is a real condition."""
    state = WorkflowState(
        workflow_id="wf-1", workflow_name="supplier_interaction", user_id="u"
    )
    assert _has_responses(state) is False


def test_negotiate_receives_a_batch_the_negotiation_agent_can_read():
    """The graph→agent seam: whatever key the node hands over,
    NegotiationAgent._extract_batch_inputs has to find the responses in it.
    Anything else drops the agent onto its single-negotiation path, which has
    no HITL checkpoint and no supplier offer to counter."""
    graph = build_supplier_interaction_workflow()
    input_data = graph.nodes["negotiate"].build_input_data(
        _state_after_watch_responses()
    )

    entries, shared = NegotiationAgent._extract_batch_inputs(None, input_data)

    assert entries, (
        "negotiate node produced no batch entries; the agent would fall through "
        "to _run_single_negotiation and skip the approval checkpoint"
    )
    assert len(entries) == 2
    assert {e["supplier_id"] for e in entries} == {"SUP-1", "SUP-2"}
    assert shared.get("negotiation_batch") is True


def test_negotiate_node_executes_end_to_end():
    """The whole point: with every agent registered and the watcher returning
    its real shape, the negotiate node runs instead of being gated out."""
    negotiation_agent = _success_agent({"negotiation_result": {"status": "drafted"}})
    agents = {
        "email_drafting": _success_agent({"drafts": [{"supplier_id": "SUP-1"}]}),
        "email_dispatch": _success_agent({"dispatch_results": {"sent": 1}}),
        "email_watcher": _success_agent(dict(WATCHER_DATA)),
        "negotiation": negotiation_agent,
        "quote_comparison": _success_agent({"comparison": {}, "recommendation": {}}),
    }
    settings = MagicMock()
    settings.parallel_processing = False
    engine = WorkflowEngine(agent_registry=agents, settings=settings)

    state = engine.execute(
        build_supplier_interaction_workflow(), input_data={}, user_id="u"
    )

    assert state.node_statuses.get("negotiate") == NodeStatus.COMPLETED, (
        f"negotiate did not run; statuses were {state.node_statuses}"
    )
    context = negotiation_agent.execute.call_args[0][0]
    assert context.input_data.get("supplier_responses"), (
        "negotiate ran but received no supplier responses"
    )
