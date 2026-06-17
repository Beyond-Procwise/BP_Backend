import json
from pathlib import Path

from src.agents.agent_interface import AgentCapability, CAPABILITY_ROLES, AgentRole


def test_capability_enum_present():
    assert AgentCapability.REQUIREMENTS_GATHERING.value == "requirements_gathering"
    assert CAPABILITY_ROLES[AgentCapability.REQUIREMENTS_GATHERING] == AgentRole.SOURCE


def test_agent_definition_registered():
    doc = json.loads(Path("agent_definitions.json").read_text())
    defs = doc["agents"] if isinstance(doc, dict) else doc
    entry = next((d for d in defs if d.get("slug") == "requirements"), None)
    assert entry is not None
    assert entry["class_path"] == "agents.requirements_agent.RequirementsAgent"
    assert "requirements_gathering" in entry["capabilities"]
    assert "message" in entry["inputs"]["optional"]
    assert "requirement_id" in entry["outputs"]


def test_factory_maps_resolve_requirements_slug():
    from src.agents.agent_factory import (
        _AGENT_MODULE_MAP, _AGENT_CLASS_MAP, AGENT_CONTRACTS,
    )
    assert _AGENT_MODULE_MAP["requirements"] == "agents.requirements_agent"
    assert _AGENT_CLASS_MAP["requirements"] == "RequirementsAgent"
    assert "requirements" in AGENT_CONTRACTS
