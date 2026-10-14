"""Draft-selection helpers on the Orchestrator.

The four workflow tests that used to live here drove
`_execute_supplier_interaction_workflow`, a second hand-rolled implementation of
supplier_interaction that the declarative engine shadowed. They had already
stopped exercising it -- the stub agent was never called -- and were failing
unnoticed. Both they and the method are gone; see
tests/orchestration/test_supplier_interaction_needs_the_engine.py for what
replaces them. The helpers below are still on the Orchestrator and still used.
"""
import os
import sys

from orchestration.orchestrator import Orchestrator

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_filter_drafts_for_workflow_respects_workflow_id():
    drafts = [
        {"workflow_id": "wf-keep", "unique_id": "uid-1"},
        {"metadata": {"workflow_id": "wf-keep"}, "unique_id": "uid-2"},
        {"workflow_id": "wf-drop", "unique_id": "uid-3"},
        {
            "unique_id": "uid-4",
            "metadata": {"context": {"workflow_id": "wf-drop"}},
        },
    ]

    filtered = Orchestrator._filter_drafts_for_workflow(drafts, "wf-keep")

    assert len(filtered) == 4
    for draft in filtered:
        workflow_value = draft.get("workflow_id")
        if not workflow_value and isinstance(draft.get("metadata"), dict):
            workflow_value = draft["metadata"].get("workflow_id")
        assert workflow_value == "wf-keep"


def test_filter_drafts_realigns_conflicting_workflow_ids():
    drafts = [
        {"workflow_id": "wf-a", "unique_id": "a1"},
        {
            "unique_id": "a2",
            "metadata": {"workflow_id": "wf-b", "context": {"workflow_id": "wf-b"}},
        },
    ]

    filtered = Orchestrator._filter_drafts_for_workflow(drafts, "wf-parent")

    assert len(filtered) == 2
    for draft in filtered:
        assert draft.get("workflow_id") == "wf-parent"
        metadata = draft.get("metadata") or {}
        assert metadata.get("workflow_id") == "wf-parent"
        context_meta = metadata.get("context") or {}
        assert context_meta.get("workflow_id") == "wf-parent"


def test_select_workflow_identifier_prefers_unique_draft_id():
    drafts = [
        {"workflow_id": "wf-dispatch", "unique_id": "uid-1"},
        {"unique_id": "uid-2", "metadata": {"workflow_id": "wf-dispatch"}},
    ]

    result = Orchestrator._select_workflow_identifier(drafts, "generated-workflow")

    assert result == "wf-dispatch"

