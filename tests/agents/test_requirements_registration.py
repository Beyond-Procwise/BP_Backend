"""The requirements agent must be registered in the one canonical catalogue.

This used to assert against ``agent_factory``'s ``_AGENT_MODULE_MAP`` /
``_AGENT_CLASS_MAP`` / ``AGENT_CONTRACTS`` — a second, hand-maintained copy of
what ``agent_definitions.json`` already declares. Those maps (and the
``AgentInterface`` abstraction no agent ever implemented) have been removed;
``AutoRegistry`` is the single source of truth, so the assertions now go
through it.
"""

from src.agents.auto_registry import AutoRegistry
from src.agents.definitions import load_agent_definitions


def test_agent_definition_registered():
    entry = next(
        (d for d in load_agent_definitions() if d.get("slug") == "requirements"), None
    )
    assert entry is not None
    assert entry["class_path"] == "agents.requirements_agent.RequirementsAgent"
    assert "requirements_gathering" in entry["capabilities"]
    assert "message" in entry["inputs"]["optional"]
    assert "requirement_id" in entry["outputs"]


def test_registry_resolves_requirements_slug():
    registry = AutoRegistry.from_json()
    contract = registry.get_contract("requirements")
    assert contract is not None
    assert contract.class_path == "agents.requirements_agent.RequirementsAgent"
    assert "requirements_gathering" in contract.capabilities
