import json
import pathlib
import pytest
from agents.definitions import get_elicit, load_agent_definitions

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_every_agent_has_an_elicit_key():
    agents = json.loads((REPO_ROOT / "agent_definitions.json").read_text())["agents"]
    missing = [a["slug"] for a in agents if "elicit" not in a]
    assert missing == [], f"agents with no elicit contract: {missing}"


def test_data_extraction_asks_for_documents():
    """The case the old manifest silently skipped: required_inputs is [] and the
    document inputs are optional, so nothing would ever be asked for."""
    groups = get_elicit("data_extraction")
    assert len(groups) == 1
    g = groups[0]
    # s3_object_keys (an explicit LIST of exact S3 keys) must be FIRST:
    # pending_requests uses any_of[0] as the canonical required_field, and the
    # document picker (the only UI path that actually knows what it uploaded)
    # must answer into this field — never a computed common prefix over a
    # shared upload folder (CRITICAL 1).
    # document_ids is gone: it was never read by the agent and could never be
    # answered (any_of[0] is always the required_field), so a payload/answer
    # keyed on it silently satisfied the group and swept the entire default
    # corpus (CRITICAL 2).
    assert g["any_of"] == ["s3_object_keys", "s3_prefix", "s3_object_key"]
    assert g["type"] == "document_ids"
    assert g["prompt"]


def test_supplier_ranking_asks_for_a_query():
    groups = get_elicit("supplier_ranking")
    assert any("query" in g["any_of"] for g in groups)


def test_unknown_agent_returns_empty():
    assert get_elicit("no_such_agent") == []


@pytest.mark.parametrize("slug", [
    "data_extraction", "supplier_ranking", "quote_comparison", "opportunity_miner",
    "email_drafting", "negotiation", "supplier_interaction", "approvals",
    "quote_evaluation", "email_dispatch", "email_watcher", "discrepancy_detection",
    "rag", "requirements",
])
def test_elicit_groups_are_well_formed(slug):
    for g in get_elicit(slug):
        assert isinstance(g["any_of"], list) and g["any_of"], f"{slug}: empty any_of"
        assert isinstance(g["type"], str) and g["type"]
        assert isinstance(g["prompt"], str) and g["prompt"]
