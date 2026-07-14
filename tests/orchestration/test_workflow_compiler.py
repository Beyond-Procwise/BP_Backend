import pytest
from orchestration.workflow_compiler import (
    compile_graph, validate_saved_graph, GraphValidationError,
)

TWO_NODE = {
    "nodes": [
        {"id": "n1", "agent_slug": "data_extraction", "x": 0, "y": 0},
        {"id": "n2", "agent_slug": "supplier_ranking", "x": 200, "y": 0},
    ],
    "edges": [{"source": "n1", "target": "n2"}],
}


def test_compiles_nodes_and_edges():
    g = compile_graph("Quote to PO", TWO_NODE)
    assert g.name == "Quote to PO"
    assert set(g.nodes) == {"n1", "n2"}
    assert g.nodes["n1"].agent_type == "data_extraction"
    assert g.entry_node == "n1"
    assert [(e.source, e.target) for e in g.edges] == [("n1", "n2")]


def test_edges_are_unconditional():
    """A drawn connector means 'then'. It carries no condition."""
    g = compile_graph("w", TWO_NODE)
    assert g.edges[0].condition is None


def test_node_outputs_are_published_to_the_blackboard():
    """Downstream nodes can only consume what upstream nodes publish."""
    g = compile_graph("w", TWO_NODE)
    assert "details" in g.nodes["n1"].output_to_shared      # data_extraction outputs
    assert "ranking" in g.nodes["n2"].output_to_shared      # supplier_ranking outputs


def test_compiled_graph_passes_the_engine_validator():
    ok, issues = compile_graph("w", TWO_NODE).validate()
    assert ok, issues


def test_rejects_a_cycle():
    bad = {"nodes": [{"id": "a", "agent_slug": "rag"}, {"id": "b", "agent_slug": "rag"}],
           "edges": [{"source": "a", "target": "b"}, {"source": "b", "target": "a"}]}
    with pytest.raises(GraphValidationError, match="cycle"):
        validate_saved_graph(bad)


def test_rejects_two_entry_nodes():
    bad = {"nodes": [{"id": "a", "agent_slug": "rag"}, {"id": "b", "agent_slug": "rag"},
                     {"id": "c", "agent_slug": "rag"}],
           "edges": [{"source": "a", "target": "c"}, {"source": "b", "target": "c"}]}
    with pytest.raises(GraphValidationError, match="one entry node"):
        validate_saved_graph(bad)


def test_rejects_an_unreachable_node():
    bad = {"nodes": [{"id": "a", "agent_slug": "rag"}, {"id": "b", "agent_slug": "rag"},
                     {"id": "orphan", "agent_slug": "rag"}],
           "edges": [{"source": "a", "target": "b"}]}
    with pytest.raises(GraphValidationError, match="unreachable|one entry node"):
        validate_saved_graph(bad)


def test_rejects_an_unknown_agent_slug():
    bad = {"nodes": [{"id": "a", "agent_slug": "not_a_real_agent"}], "edges": []}
    with pytest.raises(GraphValidationError, match="unknown agent"):
        validate_saved_graph(bad)


def test_rejects_an_empty_graph():
    with pytest.raises(GraphValidationError, match="at least one node"):
        validate_saved_graph({"nodes": [], "edges": []})


def test_single_node_graph_is_valid():
    validate_saved_graph({"nodes": [{"id": "a", "agent_slug": "rag"}], "edges": []})
