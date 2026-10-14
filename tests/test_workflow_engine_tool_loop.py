"""Programme item A4 — the engine hands a tool-loop node to the agent as such.

The engine does not run the loop itself: it tells the agent's ``execute`` that
this node reasons with tools, so process logging, retries, blackboard wiring
and result handling stay exactly the path every other node takes.
"""
from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.base_agent import AgentOutput, AgentStatus
from orchestration.workflow_engine import WorkflowEngine, WorkflowGraph, WorkflowNode


def _engine(agent):
    settings = MagicMock()
    settings.parallel_processing = False
    return WorkflowEngine(agent_registry={"a": agent}, settings=settings)


def _agent():
    agent = MagicMock()
    agent.execute.return_value = AgentOutput(status=AgentStatus.SUCCESS, data={"answer": "ok"})
    return agent


def _run(node):
    g = WorkflowGraph(name="t")
    g.add_node(node)
    agent = _agent()
    state = _engine(agent).execute(g, input_data={"deal_id": "D1"})
    return agent, state


def test_a_tool_loop_node_executes_with_the_tool_loop_flag():
    agent, state = _run(WorkflowNode(name="n1", agent_type="a", tool_loop=True))
    _, kwargs = agent.execute.call_args
    assert kwargs.get("tool_loop") is True
    assert state.node_results["n1"] == {"answer": "ok"}


def test_an_ordinary_node_executes_exactly_as_before():
    """No new keyword reaches an ordinary agent — a subclass that overrides
    ``execute(context)`` without the flag must keep working."""
    agent, _ = _run(WorkflowNode(name="n1", agent_type="a"))
    args, kwargs = agent.execute.call_args
    assert len(args) == 1
    assert kwargs == {}
