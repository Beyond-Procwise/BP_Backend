"""The catalogue is the built-in agents from the file plus the agents people
created, which live in the database.

Created agents used to be appended to agent_definitions.json, a file tracked in
git: a redeploy replaced the file and every created agent with it.
"""
from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents import definitions
from agents.auto_registry import AutoRegistry
from orchestration.orchestrator import Orchestrator

BUILTIN = {"agentId": 1, "slug": "negotiation", "agentType": "NegotiationAgent",
           "class_path": "agents.negotiation.NegotiationAgent"}
LEFT_IN_FILE = {"agentId": 2, "slug": "old-copy", "agentType": "NegotiationAgent",
                "class_path": "agents.negotiation.NegotiationAgent",
                "derived_from": "negotiation", "description": "the file's copy"}
CREATED = {"agentId": 9, "slug": "eu-negotiator", "agentType": "NegotiationAgent",
           "class_path": "agents.negotiation.NegotiationAgent",
           "derived_from": "negotiation", "tools": ["get_policy"]}


@pytest.fixture()
def catalogue(tmp_path, monkeypatch):
    path = tmp_path / "agent_definitions.json"
    path.write_text(json.dumps({"agents": [BUILTIN, LEFT_IN_FILE]}))
    monkeypatch.setattr(definitions, "DEFINITIONS_PATH", path)
    import agents.auto_registry as registry_mod
    monkeypatch.setattr(registry_mod, "_DEFAULT_JSON_PATH", path)
    Orchestrator._load_agent_definitions.cache_clear()
    yield path
    definitions.set_created_agent_source(None)
    Orchestrator._load_agent_definitions.cache_clear()


def _slugs():
    return [a["slug"] for a in definitions.load_agent_definitions()]


def test_with_no_source_the_catalogue_is_the_file(catalogue):
    assert _slugs() == ["negotiation", "old-copy"]


def test_created_agents_are_joined_onto_the_built_ins(catalogue):
    definitions.set_created_agent_source(lambda: ([CREATED], set()))
    assert _slugs() == ["negotiation", "old-copy", "eu-negotiator"]


def test_the_stored_agent_wins_over_a_copy_left_in_the_file(catalogue):
    stored = {**LEFT_IN_FILE, "description": "the stored one"}
    definitions.set_created_agent_source(lambda: ([stored], set()))

    found = [a for a in definitions.load_agent_definitions() if a["slug"] == "old-copy"]
    assert [a["description"] for a in found] == ["the stored one"]


def test_a_deleted_agent_does_not_come_back_from_the_file(catalogue):
    definitions.set_created_agent_source(lambda: ([], {"old-copy"}))
    assert _slugs() == ["negotiation"]


def test_an_unreadable_store_still_serves_the_built_ins(catalogue):
    def broken():
        raise RuntimeError("database is down")

    definitions.set_created_agent_source(broken)
    assert "negotiation" in _slugs()


def test_a_named_file_is_that_file_and_nothing_else(catalogue):
    definitions.set_created_agent_source(lambda: ([CREATED], set()))
    slugs = [a["slug"] for a in definitions.load_agent_definitions(catalogue)]
    assert slugs == ["negotiation", "old-copy"]


def test_the_registry_is_built_with_the_created_agents(catalogue):
    definitions.set_created_agent_source(lambda: ([CREATED], {"old-copy"}))
    registry = AutoRegistry.from_json()

    assert registry.agent_ids == ["eu-negotiator", "negotiation"]
    assert registry.get_contract("eu-negotiator").tools == ["get_policy"]


def test_a_registry_refresh_picks_up_a_newly_created_agent(catalogue):
    created = []
    definitions.set_created_agent_source(lambda: (list(created), set()))
    registry = AutoRegistry.from_json()
    assert "eu-negotiator" not in registry.agent_ids

    created.append(CREATED)
    registry.refresh_from_json()
    assert "eu-negotiator" in registry.agent_ids


def test_the_orchestrator_reads_the_same_catalogue(catalogue):
    definitions.set_created_agent_source(lambda: ([CREATED], set()))
    defs = Orchestrator._load_agent_definitions()
    assert defs["9"] == "NegotiationAgent"      # the created agent's id resolves
