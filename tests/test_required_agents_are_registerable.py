"""Every agent a workflow marks `required` must be instantiable.

`email_watcher` shipped with `"class_path": null`, so `AutoRegistry` loaded its
contract but `main.py` (which guards on `if contract.class_path:`) never put it
in the live registry. Meanwhile `supplier_interaction` still declared a
`watch_responses` node with `agent_type="email_watcher", required=True`, and
`WorkflowEngine` fails a node outright when a required agent is absent. Net
effect: the supplier_interaction workflow died at node 3 of 5 on every single
run, and nothing in the test suite noticed.

This pins the invariant so a null class_path can never silently break a
workflow again.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

import pytest

from agents.auto_registry import AutoRegistry
from orchestration.workflow_definitions import WORKFLOW_REGISTRY


@pytest.fixture(scope="module")
def registry() -> AutoRegistry:
    return AutoRegistry.from_json()


def _required_agent_types() -> set[str]:
    """Agent types every workflow graph declares as required=True."""
    required: set[str] = set()
    for build in WORKFLOW_REGISTRY.values():
        for node in build().nodes.values():
            if getattr(node, "required", False) and node.agent_type:
                required.add(node.agent_type)
    return required


def test_every_required_agent_has_a_class_path(registry):
    missing = sorted(
        agent_type
        for agent_type in _required_agent_types()
        if not (
            (contract := registry.get_contract(agent_type)) and contract.class_path
        )
    )
    assert not missing, (
        "These agents are declared required=True by a workflow but have no "
        f"class_path, so they never enter the registry and their node always "
        f"fails: {missing}"
    )


def test_email_watcher_is_registerable(registry):
    """The specific regression: email_watcher must be instantiable."""
    contract = registry.get_contract("email_watcher")
    assert contract is not None
    assert contract.class_path == "agents.email_watcher_agent.EmailWatcherAgent"
