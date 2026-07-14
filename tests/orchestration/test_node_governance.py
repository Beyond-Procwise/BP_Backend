import pytest

from orchestration.node_governance import governance_for, normalise_agent_name
from services.db import get_conn

pytestmark = pytest.mark.integration

MULTI_AGENT_PROMPT_NAME = "test_node_governance_multi_agent_prompt"


def _sweep_multi_agent_prompt():
    """Belt-and-braces: remove anything left behind by this test's namespace,
    including rows orphaned by an earlier crashed/interrupted run."""
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "DELETE FROM proc.bp_prompt WHERE prompt_name = %s",
            (MULTI_AGENT_PROMPT_NAME,),
        )
        cur.close()


@pytest.fixture
def multi_agent_prompt_row():
    """Insert a temporary proc.bp_prompt row whose prompt_linked_agents lists
    TWO agents, and hard-delete it unconditionally afterwards — on success AND
    on failure — since this touches the LIVE production table (bp_sqldb) and
    repo-level soft-delete conventions do not apply here (there is none; this
    is a raw DELETE, mirroring tests/orchestration/test_agent_workflow_repo.py).
    """
    _sweep_multi_agent_prompt()

    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO proc.bp_prompt "
            "(prompt_name, prompt_type, prompt_linked_agents, prompts_desc, "
            " prompts_status, version, created_by, last_modified_by) "
            "VALUES (%s, 'test_fixture', %s, %s::jsonb, 1, 1, 'pytest', 'pytest') "
            "RETURNING prompt_id",
            (
                MULTI_AGENT_PROMPT_NAME,
                "supplier_ranking_agent, negotiation_agent",
                '{"note": "temporary row for test_node_governance multi-agent coverage"}',
            ),
        )
        prompt_id = cur.fetchone()[0]
        cur.close()

    try:
        yield prompt_id
    finally:
        try:
            with get_conn() as conn:
                cur = conn.cursor()
                cur.execute(
                    "DELETE FROM proc.bp_prompt WHERE prompt_id = %s", (prompt_id,)
                )
                cur.close()
        finally:
            # Belt and braces: sweep by namespace too, so a crash before/after
            # the id was captured can't leave an orphaned row behind.
            _sweep_multi_agent_prompt()


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


def test_a_row_linking_multiple_agents_governs_all_of_them(multi_agent_prompt_row):
    """prompt_linked_agents can list SEVERAL agent tokens in one row (as
    PromptEngine/PolicyEngine already assume when applying governance). A `=`
    match on the whole column would miss "negotiation_agent" inside
    "supplier_ranking_agent, negotiation_agent" and wrongly report the agent
    as ungoverned — which is exactly the bug this test guards against.
    """
    g = governance_for("negotiation")
    assert g["governed"] is True
    assert MULTI_AGENT_PROMPT_NAME in {p["name"] for p in g["prompts"]}
