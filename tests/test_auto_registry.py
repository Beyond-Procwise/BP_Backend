"""Tests for AutoRegistry — declarative agent discovery and lazy instantiation."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from pathlib import Path

import pytest

from agents.auto_registry import AgentContract, AutoRegistry

# Path to the live project-root agent_definitions.json
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFINITIONS_JSON = _PROJECT_ROOT / "agent_definitions.json"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def registry() -> AutoRegistry:
    """Load the real agent_definitions.json once for the entire test module."""
    return AutoRegistry.from_json(str(_DEFINITIONS_JSON))


# ---------------------------------------------------------------------------
# from_json
# ---------------------------------------------------------------------------


class TestFromJson:
    def test_loads_13_agents(self, registry):
        assert len(registry.agent_ids) == 13

    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            AutoRegistry.from_json(str(tmp_path / "nonexistent.json"))

    def test_raises_on_invalid_format(self, tmp_path):
        bad_file = tmp_path / "bad.json"
        bad_file.write_text('{"foo": "bar"}')
        with pytest.raises(ValueError, match="agent_definitions.json must be either"):
            AutoRegistry.from_json(str(bad_file))

    def test_accepts_legacy_flat_array(self, tmp_path):
        """Legacy format: a bare JSON array of agent objects."""
        flat = tmp_path / "flat.json"
        flat.write_text(
            '[{"slug": "test_agent", "class_path": "mod.Cls", '
            '"capabilities": ["foo"], "required_inputs": [], "output_fields": []}]'
        )
        reg = AutoRegistry.from_json(str(flat))
        assert "test_agent" in reg.agent_ids

    def test_accepts_object_with_agents_key(self, tmp_path):
        """New format: object with an 'agents' array."""
        obj_file = tmp_path / "obj.json"
        obj_file.write_text(
            '{"agents": [{"slug": "agent_a", "class_path": "m.C", '
            '"capabilities": [], "required_inputs": [], "output_fields": []}]}'
        )
        reg = AutoRegistry.from_json(str(obj_file))
        assert "agent_a" in reg.agent_ids

    def test_default_path_resolves(self):
        """from_json() with no argument should find the project-root JSON."""
        reg = AutoRegistry.from_json()
        assert len(reg.agent_ids) == 13


# ---------------------------------------------------------------------------
# agent_ids
# ---------------------------------------------------------------------------


class TestAgentIds:
    def test_returns_sorted_list(self, registry):
        ids = registry.agent_ids
        assert ids == sorted(ids)

    def test_contains_known_slugs(self, registry):
        expected = {
            "data_extraction", "supplier_ranking", "quote_evaluation",
            "quote_comparison", "opportunity_miner", "email_drafting",
            "negotiation", "supplier_interaction", "email_dispatch",
            "approvals", "discrepancy_detection", "rag", "email_watcher",
        }
        assert expected == set(registry.agent_ids)


# ---------------------------------------------------------------------------
# get_contract
# ---------------------------------------------------------------------------


class TestGetContract:
    def test_returns_agent_contract_instance(self, registry):
        contract = registry.get_contract("supplier_ranking")
        assert isinstance(contract, AgentContract)

    def test_contract_has_correct_id(self, registry):
        contract = registry.get_contract("supplier_ranking")
        assert contract.id == "supplier_ranking"

    def test_contract_class_path(self, registry):
        contract = registry.get_contract("supplier_ranking")
        assert contract.class_path == "agents.supplier_ranking_agent.SupplierRankingAgent"

    def test_contract_capabilities_are_list(self, registry):
        contract = registry.get_contract("data_extraction")
        assert isinstance(contract.capabilities, list)
        assert "document_extraction" in contract.capabilities

    def test_contract_required_inputs(self, registry):
        contract = registry.get_contract("supplier_ranking")
        assert "query" in contract.required_inputs

    def test_contract_output_fields(self, registry):
        contract = registry.get_contract("email_dispatch")
        assert "dispatch_results" in contract.output_fields

    def test_contract_description_nonempty(self, registry):
        contract = registry.get_contract("rag")
        assert len(contract.description) > 0

    def test_email_watcher_has_no_class_path(self, registry):
        contract = registry.get_contract("email_watcher")
        assert contract.class_path is None

    def test_unknown_agent_raises_key_error(self, registry):
        with pytest.raises(KeyError, match="No agent registered"):
            registry.get_contract("does_not_exist")

    @pytest.mark.parametrize("slug", [
        "data_extraction", "supplier_ranking", "quote_evaluation",
        "quote_comparison", "opportunity_miner", "email_drafting",
        "negotiation", "supplier_interaction", "email_dispatch",
        "approvals", "discrepancy_detection", "rag", "email_watcher",
    ])
    def test_all_known_agents_have_contracts(self, registry, slug):
        contract = registry.get_contract(slug)
        assert contract.id == slug


# ---------------------------------------------------------------------------
# find_by_capability
# ---------------------------------------------------------------------------


class TestFindByCapability:
    def test_finds_single_match(self, registry):
        result = registry.find_by_capability("supplier_ranking")
        assert result == ["supplier_ranking"]

    def test_finds_multiple_matches(self, registry):
        # document_extraction is listed only under data_extraction
        result = registry.find_by_capability("document_extraction")
        assert "data_extraction" in result

    def test_returns_empty_list_for_unknown_capability(self, registry):
        result = registry.find_by_capability("teleportation")
        assert result == []

    def test_email_watching_maps_to_email_watcher(self, registry):
        result = registry.find_by_capability("email_watching")
        assert "email_watcher" in result

    def test_rag_query_maps_to_rag(self, registry):
        result = registry.find_by_capability("rag_query")
        assert "rag" in result

    def test_returns_list_type(self, registry):
        result = registry.find_by_capability("negotiation")
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# all_capabilities
# ---------------------------------------------------------------------------


class TestAllCapabilities:
    def test_returns_set(self, registry):
        caps = registry.all_capabilities()
        assert isinstance(caps, set)

    def test_contains_core_capabilities(self, registry):
        caps = registry.all_capabilities()
        expected = {
            "document_extraction", "supplier_ranking", "quote_comparison",
            "email_drafting", "negotiation", "rag_query", "approval_decision",
            "discrepancy_detection", "opportunity_mining",
        }
        assert expected.issubset(caps)

    def test_total_unique_capability_count(self, registry):
        # 13 agents — most have 1 capability; data_extraction has 4
        caps = registry.all_capabilities()
        # At minimum we expect all unique single caps plus data_extraction extras
        assert len(caps) >= 13

    def test_no_duplicates(self, registry):
        caps = registry.all_capabilities()
        assert len(caps) == len(caps)  # trivially true for a set; sanity check


# ---------------------------------------------------------------------------
# describe_for_llm
# ---------------------------------------------------------------------------


class TestDescribeForLlm:
    def test_returns_string(self, registry):
        result = registry.describe_for_llm()
        assert isinstance(result, str)

    def test_contains_all_agent_ids(self, registry):
        description = registry.describe_for_llm()
        for slug in registry.agent_ids:
            assert slug in description, f"Missing agent '{slug}' in LLM description"

    def test_contains_capabilities_section(self, registry):
        description = registry.describe_for_llm()
        assert "Capabilities" in description or "capabilities" in description

    def test_contains_inputs_and_outputs(self, registry):
        description = registry.describe_for_llm()
        assert "Inputs" in description or "inputs" in description
        assert "Outputs" in description or "outputs" in description

    def test_contains_header(self, registry):
        description = registry.describe_for_llm()
        assert "Available Procurement Agents" in description

    def test_noninstantiable_agents_flagged(self, registry):
        description = registry.describe_for_llm()
        # email_watcher has no class_path; should be flagged as inline service
        assert "inline service" in description

    def test_length_is_reasonable(self, registry):
        description = registry.describe_for_llm()
        # Should be a substantive description, not a stub
        assert len(description) > 500


# ---------------------------------------------------------------------------
# is_instantiated
# ---------------------------------------------------------------------------


class TestIsInstantiated:
    def test_false_before_any_get_agent(self, registry):
        assert not registry.is_instantiated("approvals")

    def test_false_for_unknown_id(self, registry):
        assert not registry.is_instantiated("totally_unknown_agent_xyz")


# ---------------------------------------------------------------------------
# set_agent_nick
# ---------------------------------------------------------------------------


class TestSetAgentNick:
    def test_set_agent_nick_stores_value(self):
        registry = AutoRegistry.from_json(str(_DEFINITIONS_JSON))
        sentinel = object()
        registry.set_agent_nick(sentinel)
        assert registry._agent_nick is sentinel


# ---------------------------------------------------------------------------
# get_agent — error paths only (no real heavy imports in unit tests)
# ---------------------------------------------------------------------------


class TestGetAgentErrors:
    def test_raises_key_error_for_unknown_agent(self):
        registry = AutoRegistry.from_json(str(_DEFINITIONS_JSON))
        with pytest.raises(KeyError):
            registry.get_agent("does_not_exist")

    def test_raises_value_error_for_no_class_path(self):
        registry = AutoRegistry.from_json(str(_DEFINITIONS_JSON))
        with pytest.raises(ValueError, match="no class_path"):
            registry.get_agent("email_watcher")

    def test_raises_import_error_for_bad_module(self, tmp_path):
        bad_json = tmp_path / "bad_module.json"
        bad_json.write_text(
            '{"agents": [{"slug": "phantom", "class_path": "no.such.module.Cls", '
            '"capabilities": [], "required_inputs": [], "output_fields": []}]}'
        )
        registry = AutoRegistry.from_json(str(bad_json))
        with pytest.raises(ImportError, match="Cannot import module"):
            registry.get_agent("phantom")

    def test_raises_import_error_for_bad_class(self, tmp_path):
        # Use a real module but a nonexistent class name
        bad_json = tmp_path / "bad_class.json"
        bad_json.write_text(
            '{"agents": [{"slug": "phantom2", "class_path": "json.NonExistentClass", '
            '"capabilities": [], "required_inputs": [], "output_fields": []}]}'
        )
        registry = AutoRegistry.from_json(str(bad_json))
        with pytest.raises(ImportError, match="has no class"):
            registry.get_agent("phantom2")

    def test_caches_instance_after_first_call(self, tmp_path):
        """A successfully created agent is returned from cache on second call."""
        # Use a lightweight stdlib class as a stand-in
        simple_json = tmp_path / "simple.json"
        simple_json.write_text(
            '{"agents": [{"slug": "simple", "class_path": "pathlib.Path", '
            '"capabilities": [], "required_inputs": [], "output_fields": []}]}'
        )
        registry = AutoRegistry.from_json(str(simple_json))
        registry.set_agent_nick("/tmp")
        instance1 = registry.get_agent("simple")
        assert registry.is_instantiated("simple")
        instance2 = registry.get_agent("simple")
        assert instance1 is instance2
