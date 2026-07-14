import pytest

from orchestration.node_governance import governance_for, normalise_agent_name

pytestmark = pytest.mark.integration


def test_normalises_both_directions():
    assert normalise_agent_name("supplier_ranking") == "supplier_ranking_agent"
    assert normalise_agent_name("supplier_ranking_agent") == "supplier_ranking_agent"


def test_a_governed_agent_reports_its_prompts_and_policies():
    g = governance_for("supplier_ranking")
    assert g["governed"] is True
    assert {p["name"] for p in g["prompts"]} >= {"rank_by_criteria"}
    assert {p["name"] for p in g["policies"]} >= {"WeightAllocationPolicy"}


def test_an_ungoverned_agent_says_so_rather_than_faking_it():
    g = governance_for("data_extraction")
    assert g["governed"] is False
    assert g["prompts"] == []
    assert g["policies"] == []
