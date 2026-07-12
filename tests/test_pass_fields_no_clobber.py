"""An empty pass_field must not erase a value already on the blackboard.

WorkflowEngine merged agent pass_fields with a blanket ``state.shared_data.update(...)``,
while the ``output_to_shared`` copy immediately above it guarded with ``if value:``.

The asymmetry had a concrete consequence. OpportunityMinerAgent returns
``supplier_candidates: []`` when it finds none. That empty list overwrote a caller-supplied
candidate list on the blackboard, so ``_has_supplier_candidates`` was always False, the
``mine_opportunities -> rank_suppliers`` edge never traversed, and **SupplierRankingAgent was
skipped on every run** — which is why nothing it produced was ever persisted or displayed.

A node may introduce a new key with any value. It may not blank out an existing one.
"""
from __future__ import annotations

import pytest

from orchestration.workflow_definitions import _has_supplier_candidates
from orchestration.workflow_engine import WorkflowState


def _state(shared: dict) -> WorkflowState:
    st = WorkflowState(
        workflow_id="t",
        workflow_name="supplier_ranking",
        user_id="u",
        started_at="",
    )
    st.shared_data = dict(shared)
    return st


def _merge_pass_fields(state: WorkflowState, pass_fields: dict) -> None:
    """The merge as WorkflowEngine._execute_node now performs it."""
    for k, v in (pass_fields or {}).items():
        if v or k not in state.shared_data:
            state.shared_data[k] = v


def test_empty_pass_field_does_not_erase_existing_value():
    state = _state({"supplier_candidates": ["SUP-A", "SUP-B"]})
    # The opportunity miner found nothing this run.
    _merge_pass_fields(state, {"supplier_candidates": [], "findings": []})

    assert state.shared_data["supplier_candidates"] == ["SUP-A", "SUP-B"], (
        "an empty pass_field erased the caller's candidate list — rank_suppliers "
        "would be skipped and SupplierRankingAgent would never run"
    )
    assert _has_supplier_candidates(state) is True


def test_new_key_is_still_introduced_even_when_empty():
    """A node may introduce a key it owns, empty or not — it just may not blank an existing one."""
    state = _state({})
    _merge_pass_fields(state, {"findings": []})
    assert "findings" in state.shared_data
    assert state.shared_data["findings"] == []


def test_non_empty_pass_field_still_overwrites():
    state = _state({"supplier_candidates": ["SUP-A"]})
    _merge_pass_fields(state, {"supplier_candidates": ["SUP-X", "SUP-Y"]})
    assert state.shared_data["supplier_candidates"] == ["SUP-X", "SUP-Y"]
