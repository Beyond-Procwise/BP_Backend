import pytest

from scripts.testdata.catalogue import build_catalogue
from scripts.testdata.org import build_business_units, build_cost_centres
from scripts.testdata.persist_org import (
    COLUMNS,
    REQUIRED,
    MissingRequiredValue,
    check_required,
    rows_for_org,
)
from scripts.testdata.reference import TaxonomyLeaf


def _leaves(count: int = 246) -> list[TaxonomyLeaf]:
    return [
        TaxonomyLeaf(
            l1="IT & Technology", l2="Software", l3="ERP", l4=f"Sub{i}", l5=f"Leaf{i}",
            l1_id="C-2000", l2_id="C-3000", l3_id="C-4000",
            l4_id=f"C-45{i:02d}", l5_id=f"C-51{i:02d}",
            unspsc_code=str(10000000 + i), esg_impact="Low", category_status="Active",
            spend_classification="Direct", category_risk_rating="Minimal",
            audit_frequency="Annually", policy_coverage="Full",
        )
        for i in range(count)
    ]


@pytest.fixture(scope="module")
def built():
    leaves = _leaves()
    units = build_business_units(42)
    centres = build_cost_centres(42, units, leaves)
    items = build_catalogue(42, leaves, [f"SUP-S{i}" for i in range(500)])
    return rows_for_org(units, centres, items), leaves


def test_three_tables_are_mapped():
    assert set(COLUMNS) == {"business_unit", "cost_centre", "item"}


def test_column_lists_match_the_live_schema_widths():
    assert len(COLUMNS["business_unit"]) == 16
    assert len(COLUMNS["cost_centre"]) == 26
    assert len(COLUMNS["item"]) == 15


def test_required_names_are_a_subset_of_columns():
    for table, required in REQUIRED.items():
        unknown = [c for c in required if c not in COLUMNS[table]]
        assert not unknown, f"{table}: {unknown}"


def test_row_counts_match_the_generated_population(built):
    rows, _ = built
    assert len(rows["business_unit"]) == 400
    assert len(rows["cost_centre"]) == 500
    assert len(rows["item"]) == 5000


def test_every_row_is_as_wide_as_its_column_list(built):
    rows, _ = built
    for table, table_rows in rows.items():
        width = len(COLUMNS[table])
        for row in table_rows:
            assert len(row) == width, table


def test_cost_centres_carry_six_levels_and_a_real_business_unit(built):
    rows, _ = built
    bu_ids = {
        row[COLUMNS["business_unit"].index("business_unit_id")]
        for row in rows["business_unit"]
    }
    columns = COLUMNS["cost_centre"]
    for row in rows["cost_centre"]:
        assert row[columns.index("cost_centre_level_6")]
        assert row[columns.index("business_unit_id")] in bu_ids


def test_cost_centre_document_links_are_left_null(built):
    """po_id and invoice_id are filled by stage S3, not invented here."""
    rows, _ = built
    columns = COLUMNS["cost_centre"]
    for row in rows["cost_centre"][:20]:
        assert row[columns.index("po_id")] is None
        assert row[columns.index("invoice_id")] is None


def test_items_carry_the_real_category_level_5_id(built):
    rows, leaves = built
    valid = {leaf.l5_id for leaf in leaves}
    columns = COLUMNS["item"]
    for row in rows["item"]:
        assert row[columns.index("category_id")] in valid
        assert row[columns.index("standard_price")] > 0


def test_business_unit_level_1_ids_are_stable_and_one_per_function(built):
    rows, _ = built
    columns = COLUMNS["business_unit"]
    pairs = {
        (row[columns.index("business_unit_level_1")],
         row[columns.index("business_unit_level_1_id")])
        for row in rows["business_unit"]
    }
    by_name = {name: ident for name, ident in pairs}
    assert len(pairs) == len(by_name) == 6


def test_generated_rows_satisfy_their_own_required_sets(built):
    rows, _ = built
    for table, table_rows in rows.items():
        check_required(table, table_rows)


def test_a_missing_required_value_is_an_error_not_a_silent_null():
    """These tables have almost no NOT NULL constraints; the DB will not catch it."""
    blank = [None] * len(COLUMNS["item"])
    with pytest.raises(MissingRequiredValue, match="item.item_id"):
        check_required("item", [blank])


def test_check_required_names_the_row_that_failed():
    good = ["ITM000001", "name", "C-5101", "each", 1, "GBP", "SUP-X"] + [None] * 8
    bad = list(good)
    bad[COLUMNS["item"].index("category_id")] = None
    with pytest.raises(MissingRequiredValue, match="row 2"):
        check_required("item", [good, bad])


def test_mapping_is_deterministic():
    leaves = _leaves()
    units = build_business_units(42)
    centres = build_cost_centres(42, units, leaves)
    items = build_catalogue(42, leaves, [f"SUP-S{i}" for i in range(500)])
    first = rows_for_org(units, centres, items)
    second = rows_for_org(units, centres, items)
    assert first == second
