import pytest

from scripts.testdata.guards import UnsafeTargetError
from scripts.testdata.reference import (
    REFERENCE_TABLES,
    copy_reference,
    load_taxonomy,
)


def test_reference_tables_cover_fx_governance_and_taxonomy():
    bp_tables = REFERENCE_TABLES["bp_sqldb"]
    assert "bp_fx_rates" in bp_tables
    assert "bp_policy" in bp_tables
    assert "bp_prompt" in bp_tables
    assert "bp_admin_config" in bp_tables

    ui_tables = REFERENCE_TABLES["uicanvas"]
    assert "bp_category" in ui_tables
    assert "category" in ui_tables
    assert "category_mapping" in ui_tables


def test_copy_reference_refuses_live_targets():
    with pytest.raises(UnsafeTargetError):
        copy_reference("bp_sqldb", "bp_sqldb")


@pytest.mark.integration
def test_taxonomy_has_six_families_and_five_populated_levels():
    leaves = load_taxonomy("uicanvas")
    assert len(leaves) >= 240

    families = {leaf.l1 for leaf in leaves}
    assert families == {
        "IT & Technology",
        "Marketing & Media",
        "Facilities & Real Estate",
        "Professional Services",
        "Logistics & Supply Chain",
        "Office & Administrative Supplies",
    }

    for leaf in leaves:
        assert leaf.l1 and leaf.l2 and leaf.l3 and leaf.l4 and leaf.l5


@pytest.mark.integration
def test_copy_reference_reproduces_fx_row_count():
    from scripts.testdata.db import connect

    written = copy_reference("bp_sqldb", "bp_testdb")
    assert written["bp_fx_rates"] > 0

    source = connect("bp_sqldb")
    target = connect("bp_testdb")
    try:
        with source.cursor() as cur:
            cur.execute("select count(*) from proc.bp_fx_rates")
            expected = cur.fetchone()[0]
        with target.cursor() as cur:
            cur.execute("select count(*) from proc.bp_fx_rates")
            assert cur.fetchone()[0] == expected
    finally:
        source.close()
        target.close()
