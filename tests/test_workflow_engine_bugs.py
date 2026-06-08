# tests/test_workflow_engine_bugs.py
"""Unit tests for Bug 1 (SKIPPED predecessor cascade) and Bug 2 (output_to_shared empty-list overwrite)."""

import pytest
from unittest.mock import MagicMock

from orchestration.workflow_engine import (
    WorkflowEngine,
    WorkflowGraph,
    WorkflowNode,
    WorkflowState,
    NodeStatus,
)
from agents.base_agent import AgentOutput, AgentStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_engine(agents: dict) -> WorkflowEngine:
    settings = MagicMock()
    settings.parallel_processing = False
    return WorkflowEngine(agent_registry=agents, settings=settings)


def _success_agent(data: dict, output_fields: dict = None) -> MagicMock:
    """Return a mock agent whose execute() always succeeds with the given data."""
    agent = MagicMock()
    out = AgentOutput(
        status=AgentStatus.SUCCESS,
        data=data,
        pass_fields=output_fields or {},
    )
    agent.execute.return_value = out
    return agent


# ---------------------------------------------------------------------------
# Bug 1 — SKIPPED predecessor should NOT cascade skip to downstream nodes
# ---------------------------------------------------------------------------

class TestBug1SkippedPredecessorCascade:
    """mine_opportunities (SKIPPED) -> rank_suppliers: rank should still execute."""

    def _build_graph(self) -> WorkflowGraph:
        g = WorkflowGraph(name="supplier_flow")
        g.add_node(WorkflowNode(name="mine_opportunities", agent_type="opportunity_miner", required=False))
        g.add_node(WorkflowNode(
            name="rank_suppliers",
            agent_type="supplier_ranking",
            output_to_shared=["ranked_suppliers"],
        ))
        g.add_edge("mine_opportunities", "rank_suppliers")
        return g

    def test_skipped_source_allows_downstream_execution(self):
        """rank_suppliers must execute when mine_opportunities is SKIPPED (no agent registered)."""
        ranked = ["Supplier A", "Supplier B"]
        ranking_agent = _success_agent({"ranked_suppliers": ranked})

        # opportunity_miner is intentionally absent from registry — causes SKIPPED
        engine = _make_engine({"supplier_ranking": ranking_agent})
        graph = self._build_graph()

        state = engine.execute(graph, input_data={}, user_id="test")

        # mine_opportunities should be SKIPPED (agent missing, required=False)
        assert state.node_statuses.get("mine_opportunities") == NodeStatus.SKIPPED, (
            "Expected mine_opportunities to be SKIPPED"
        )
        # rank_suppliers must NOT be skipped — it should have run
        assert state.node_statuses.get("rank_suppliers") == NodeStatus.COMPLETED, (
            "Expected rank_suppliers to be COMPLETED; cascade skip bug present"
        )
        assert ranking_agent.execute.called

    def test_skipped_source_with_supplier_candidates_in_shared_data(self):
        """Downstream node receives supplier_candidates already in shared_data when source is SKIPPED."""
        pre_supplied = ["Vendor X", "Vendor Y"]
        ranking_agent = _success_agent({"ranked_suppliers": pre_supplied})

        engine = _make_engine({"supplier_ranking": ranking_agent})
        graph = self._build_graph()

        # caller pre-populates shared_data with candidates
        state = engine.execute(
            graph,
            input_data={"supplier_candidates": pre_supplied},
            user_id="test",
        )

        assert state.node_statuses.get("mine_opportunities") == NodeStatus.SKIPPED
        assert state.node_statuses.get("rank_suppliers") == NodeStatus.COMPLETED
        # The pre-supplied candidates must still be in shared_data after execution
        assert state.shared_data.get("supplier_candidates") == pre_supplied

    def test_three_node_chain_with_skipped_first_node(self):
        """mine_opp (SKIPPED) -> rank -> email: both rank and email should execute."""
        ranking_agent = _success_agent({"ranked_suppliers": ["S1"]})
        email_agent = _success_agent({"draft": "Dear Supplier"})

        engine = _make_engine({"supplier_ranking": ranking_agent, "email_drafting": email_agent})
        g = WorkflowGraph(name="chain_flow")
        g.add_node(WorkflowNode(name="mine_opp", agent_type="opportunity_miner", required=False))
        g.add_node(WorkflowNode(name="rank", agent_type="supplier_ranking"))
        g.add_node(WorkflowNode(name="email", agent_type="email_drafting"))
        g.add_edge("mine_opp", "rank")
        g.add_edge("rank", "email")

        state = engine.execute(g, input_data={}, user_id="test")

        assert state.node_statuses.get("mine_opp") == NodeStatus.SKIPPED
        assert state.node_statuses.get("rank") == NodeStatus.COMPLETED
        assert state.node_statuses.get("email") == NodeStatus.COMPLETED


# ---------------------------------------------------------------------------
# Bug 2 — output_to_shared must NOT overwrite caller data with empty list
# ---------------------------------------------------------------------------

class TestBug2OutputToSharedEmptyListOverwrite:
    """A zero-result node returning [] must not wipe pre-supplied shared_data."""

    def _build_graph(self) -> WorkflowGraph:
        """Returns a mine -> rank graph."""
        g = WorkflowGraph(name="ranking_flow")
        g.add_node(WorkflowNode(
            name="mine",
            agent_type="miner",
            output_to_shared=["supplier_candidates"],
        ))
        g.add_node(WorkflowNode(name="rank", agent_type="ranker"))
        g.add_edge("mine", "rank")
        return g

    def test_empty_list_does_not_overwrite_caller_data(self):
        """When miner returns [], caller-supplied supplier_candidates must survive."""
        caller_candidates = ["Vendor A", "Vendor B"]

        # miner returns an empty list — simulates a zero-result run
        miner_agent = _success_agent({"supplier_candidates": []})
        ranker_agent = _success_agent({"ranked": []})

        engine = _make_engine({"miner": miner_agent, "ranker": ranker_agent})
        g = self._build_graph()

        state = engine.execute(
            g,
            input_data={"supplier_candidates": caller_candidates},
            user_id="test",
        )

        # The caller data must NOT be wiped by the empty-list output
        assert state.shared_data.get("supplier_candidates") == caller_candidates, (
            "Bug 2: empty list from miner overwrote caller-supplied supplier_candidates"
        )

    def test_truthy_value_does_overwrite(self):
        """When miner returns a non-empty list it SHOULD update shared_data."""
        new_candidates = ["Vendor C", "Vendor D"]
        old_candidates = ["Old Vendor"]

        miner_agent = _success_agent({"supplier_candidates": new_candidates})
        ranker_agent = _success_agent({"ranked": new_candidates})

        engine = _make_engine({"miner": miner_agent, "ranker": ranker_agent})
        g = WorkflowGraph(name="ranking_flow2")
        g.add_node(WorkflowNode(
            name="mine",
            agent_type="miner",
            output_to_shared=["supplier_candidates"],
        ))
        g.add_node(WorkflowNode(name="rank", agent_type="ranker"))
        g.add_edge("mine", "rank")

        state = engine.execute(
            g,
            input_data={"supplier_candidates": old_candidates},
            user_id="test",
        )

        # Non-empty result should overwrite
        assert state.shared_data.get("supplier_candidates") == new_candidates

    def test_none_value_does_not_overwrite(self):
        """When output field is absent (None), caller data must be preserved."""
        caller_candidates = ["Vendor Z"]

        # miner result does not include supplier_candidates key at all
        miner_agent = _success_agent({})
        ranker_agent = _success_agent({})

        engine = _make_engine({"miner": miner_agent, "ranker": ranker_agent})
        g = WorkflowGraph(name="ranking_flow3")
        g.add_node(WorkflowNode(
            name="mine",
            agent_type="miner",
            output_to_shared=["supplier_candidates"],
        ))
        g.add_node(WorkflowNode(name="rank", agent_type="ranker"))
        g.add_edge("mine", "rank")

        state = engine.execute(
            g,
            input_data={"supplier_candidates": caller_candidates},
            user_id="test",
        )

        assert state.shared_data.get("supplier_candidates") == caller_candidates
