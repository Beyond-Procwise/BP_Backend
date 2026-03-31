"""Orchestration helpers for procurement workflows.

This package exposes a legacy star-import surface while avoiding eager imports
that previously triggered circular dependencies during test collection.  The
``__getattr__`` shim lazily resolves heavy modules (such as the main
``WorkflowOrchestrator`` implementation) only when they are first accessed.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "DraftEmail",
    "DispatchRecord",
    "EmailDraftingAgent",
    "EmailThread",
    "MockDatabaseConnection",
    "NegotiationAgent",
    "NegotiationSession",
    "QuoteComparisonAgent",
    "SupplierInteractionAgent",
    "WorkflowContextManager",
    "WorkflowOrchestrator",
    "build_status_request_id",
    "handle_supplier_response",
    "validate_implementation",
]


def __getattr__(name: str) -> Any:  # pragma: no cover - import shim
    if name == "WorkflowOrchestrator":
        module = import_module("orchestration.orchestrator")
        return getattr(module, name)

    workflow_exports = {
        "DraftEmail",
        "DispatchRecord",
        "EmailDraftingAgent",
        "EmailThread",
        "MockDatabaseConnection",
        "NegotiationAgent",
        "NegotiationSession",
        "QuoteComparisonAgent",
        "SupplierInteractionAgent",
        "WorkflowContextManager",
        "build_status_request_id",
        "handle_supplier_response",
        "validate_implementation",
    }
    if name in workflow_exports:
        module = import_module("orchestration.procurement_workflow")
        return getattr(module, name)

    # Lazy-load new orchestration modules
    module_map = {
        "dag_scheduler": "orchestration.dag_scheduler",
        "task_dispatcher": "orchestration.task_dispatcher",
        "result_collector": "orchestration.result_collector",
        "state_manager": "orchestration.state_manager",
        "worker": "orchestration.worker",
        "worker_context": "orchestration.worker_context",
        "message_protocol": "orchestration.message_protocol",
    }
    if name in module_map:
        module = import_module(module_map[name])
        return module

    raise AttributeError(f"module 'orchestration' has no attribute {name!r}")
