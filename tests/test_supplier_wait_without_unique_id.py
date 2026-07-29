"""wait_for_response must survive a call with no unique_id.

Regression: the method referenced ``draft_match``, ``draft_context`` and
``watch_candidates`` — locals belonging to ``run()``, not to this method. They
sat behind ``elif`` branches reached only when no unique_id was supplied, so the
call raised NameError instead of waiting. All three call sites can pass None
(``context.get("unique_id")`` / ``target.get("unique_id")``), so the crash was
reachable in production whenever a dispatch record carried no unique_id.
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.supplier_interaction_agent import SupplierInteractionAgent


def _agent(recorder):
    """Minimal agent exercising only wait_for_response's own logic."""
    agent = SupplierInteractionAgent.__new__(SupplierInteractionAgent)
    agent.process_routing_service = None

    def _await_dispatch_ready(**kwargs):
        recorder["dispatch"] = kwargs
        return {"complete": True, "unique_ids": []}

    def _await_supplier_response_rows(workflow_id, **kwargs):
        recorder["rows"] = {"workflow_id": workflow_id, **kwargs}
        return []

    agent._await_dispatch_ready = _await_dispatch_ready
    agent._await_supplier_response_rows = _await_supplier_response_rows
    return agent


def test_wait_for_response_without_unique_id_does_not_crash():
    recorder = {}
    agent = _agent(recorder)

    result = agent.wait_for_response(
        workflow_id="wf-1",
        supplier_id="SUP-1",
        unique_id=None,
        timeout=5,
        poll_interval=1,
    )

    # No responses stored yet -> None, but reached the wait without NameError.
    assert result is None
    assert recorder["dispatch"]["workflow_id"] == "wf-1"
    # With no unique_id to seed, the unique IDs are discovered from the workflow.
    assert list(recorder["dispatch"]["unique_ids"]) == []
    # Still awaits one reply rather than zero.
    assert recorder["dispatch"]["expected_total"] == 1
    assert recorder["rows"]["unique_filter"] is None
    assert recorder["rows"]["supplier_filter"] == {"SUP-1"}


def test_wait_for_response_with_unique_id_still_seeds_it():
    recorder = {}
    agent = _agent(recorder)

    agent.wait_for_response(
        workflow_id="wf-2",
        supplier_id="SUP-2",
        unique_id="UNQ-2",
        timeout=5,
        poll_interval=1,
    )

    assert list(recorder["dispatch"]["unique_ids"]) == ["UNQ-2"]
    assert recorder["dispatch"]["expected_total"] == 1
    assert recorder["rows"]["unique_filter"] == {"UNQ-2"}
