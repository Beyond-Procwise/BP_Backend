"""Programme item A4 — the compiler marks which nodes reason with tools.

A node built from a canvas-made ("derived") agent carries human instructions
and must run through the governed tool loop; a built-in agent keeps its own
deterministic logic. The compiler is where a drawn node learns which it is.
"""
from __future__ import annotations

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import orchestration.workflow_compiler as wc

DEFS = [
    {"slug": "supplier_ranking", "class_path": "agents.x.Y", "outputs": ["ranked_suppliers"]},
    {
        "slug": "overcharge_hunter",
        "class_path": "agents.x.Y",
        "outputs": ["ranked_suppliers"],
        "derived_from": "supplier_ranking",
    },
]


def _compile(monkeypatch, slug):
    monkeypatch.setattr(wc, "load_agent_definitions", lambda: DEFS)
    graph = {"nodes": [{"id": "n1", "agent_slug": slug}], "edges": []}
    return wc.compile_graph("t", graph).nodes["n1"]


def test_a_built_in_agent_node_does_not_reason_with_tools(monkeypatch):
    node = _compile(monkeypatch, "supplier_ranking")
    assert node.tool_loop is False


def test_a_derived_agent_node_reasons_with_tools(monkeypatch):
    node = _compile(monkeypatch, "overcharge_hunter")
    assert node.tool_loop is True


def test_a_derived_node_publishes_its_answer_downstream(monkeypatch):
    """The loop's answer is the thing the next step needs to see; the backing
    agent's declared outputs stay published too."""
    node = _compile(monkeypatch, "overcharge_hunter")
    assert "answer" in node.output_to_shared
    assert "ranked_suppliers" in node.output_to_shared
