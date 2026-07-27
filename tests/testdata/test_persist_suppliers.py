import json

import pytest

from scripts.testdata.persist_suppliers import (
    COLUMNS,
    REQUIRED,
    TARGET_TABLE,
    MissingRequiredValue,
    check_required,
    rows_for_suppliers,
)
from scripts.testdata.reference import TaxonomyLeaf
from scripts.testdata.suppliers import SUPPLIER_COLUMNS, TIERS, build_suppliers


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
    suppliers = build_suppliers(42, _leaves())
    return rows_for_suppliers(suppliers), suppliers


def test_five_logical_tables_are_mapped():
    assert set(COLUMNS) == {
        "bp_supplier_uicanvas", "bp_tprm_supplier", "esg_data", "contact", "bp_contact"
    }


def test_every_logical_table_names_a_physical_target():
    assert set(TARGET_TABLE) == set(COLUMNS)
    assert TARGET_TABLE["bp_supplier_uicanvas"] == "bp_supplier"


def test_column_lists_match_the_live_schema_widths():
    assert len(COLUMNS["bp_supplier_uicanvas"]) == 51
    assert len(COLUMNS["bp_tprm_supplier"]) == 10
    assert len(COLUMNS["esg_data"]) == 15
    assert len(COLUMNS["contact"]) == 14
    assert len(COLUMNS["bp_contact"]) == 14


def test_required_names_are_a_subset_of_columns():
    for table, required in REQUIRED.items():
        unknown = [c for c in required if c not in COLUMNS[table]]
        assert not unknown, f"{table}: {unknown}"


def test_every_row_is_as_wide_as_its_column_list(built):
    rows, _ = built
    for table, table_rows in rows.items():
        width = len(COLUMNS[table])
        for row in table_rows:
            assert len(row) == width, table


def test_esg_and_contacts_cover_every_supplier(built):
    rows, suppliers = built
    assert len(rows["esg_data"]) == len(suppliers) == 5000
    assert len(rows["contact"]) == 5000
    assert len(rows["bp_contact"]) == 5000


def test_uicanvas_master_uses_the_bp_identifier_convention(built):
    rows, _ = built
    position = SUPPLIER_COLUMNS.index("supplier_id")
    assert len(rows["bp_supplier_uicanvas"]) == 5000
    for row in rows["bp_supplier_uicanvas"][:50]:
        assert row[position].startswith("SUP-")


def test_third_party_risk_covers_only_the_strategic_tier(built):
    rows, suppliers = built
    strategic = next(tier.count for tier in TIERS if tier.name == "Strategic")
    assert len(rows["bp_tprm_supplier"]) == strategic == 120


def test_tprm_payload_is_valid_json_naming_its_supplier(built):
    rows, _ = built
    columns = COLUMNS["bp_tprm_supplier"]
    for row in rows["bp_tprm_supplier"][:20]:
        payload = json.loads(row[columns.index("payload")])
        assert payload["id"] == row[columns.index("tp_id")]
        assert payload["name"] == row[columns.index("name")]
        assert len(payload["dom"]) == 8


def test_esg_figures_agree_with_the_supplier_master(built):
    """A supplier holding ISO 14001 must not also report the worst score."""
    rows, suppliers = built
    columns = COLUMNS["esg_data"]
    by_id = {s.uicanvas_supplier_id: s for s in suppliers}
    for row in rows["esg_data"]:
        supplier = by_id[row[columns.index("supplier_id")]]
        if supplier.columns["esg_cert_iso14001"]:
            assert row[columns.index("esg_score")] >= 62
            assert row[columns.index("renewable_energy_use_perc")] >= 45
            assert "ISO 14001" in row[columns.index("certifications")]


def test_emissions_total_is_the_sum_of_its_scopes(built):
    rows, _ = built
    columns = COLUMNS["esg_data"]
    for row in rows["esg_data"][:200]:
        total = row[columns.index("carbon_emission_tco2")]
        parts = sum(
            row[columns.index(name)]
            for name in ("scope_1_emissions", "scope_2_emissions", "scope_3_emissions")
        )
        assert abs(total - round(parts, 2)) < 0.01


def test_contacts_come_from_the_supplier_master_record(built):
    rows, suppliers = built
    columns = COLUMNS["contact"]
    by_id = {s.uicanvas_supplier_id: s for s in suppliers}
    for row in rows["contact"][:100]:
        supplier = by_id[row[columns.index("supplier_id")]]
        assert row[columns.index("contact_email")] == supplier.columns["contact_email_1"]
        assert row[columns.index("is_primary_contact")] is True


def test_contact_ids_are_unique(built):
    rows, _ = built
    position = COLUMNS["contact"].index("contact_id")
    ids = [row[position] for row in rows["contact"]]
    assert len(set(ids)) == len(ids)


def test_generated_rows_satisfy_their_own_required_sets(built):
    rows, _ = built
    for table, table_rows in rows.items():
        check_required(table, table_rows)


def test_a_missing_required_value_is_an_error_not_a_silent_null():
    blank = [None] * len(COLUMNS["esg_data"])
    with pytest.raises(MissingRequiredValue, match="esg_data.supplier_id"):
        check_required("esg_data", [blank])


def test_mapping_is_deterministic():
    suppliers = build_suppliers(42, _leaves())
    assert rows_for_suppliers(suppliers) == rows_for_suppliers(suppliers)


@pytest.mark.integration
def test_supplier_reference_loads_into_the_scratch_database(scratch_uicanvas_schema, built):
    from scripts.testdata.db import connect
    from scripts.testdata.loader import load_tables
    from tests.testdata import SCRATCH_UICANVAS_DB

    rows, _ = built
    uicanvas_tables = ("bp_supplier_uicanvas", "esg_data", "contact", "bp_contact")
    written = load_tables(
        SCRATCH_UICANVAS_DB,
        {TARGET_TABLE[t]: COLUMNS[t] for t in uicanvas_tables},
        {TARGET_TABLE[t]: REQUIRED[t] for t in uicanvas_tables},
        {TARGET_TABLE[t]: rows[t] for t in uicanvas_tables},
    )
    assert written == {
        "bp_supplier": 5000, "esg_data": 5000, "contact": 5000, "bp_contact": 5000
    }

    conn = connect(SCRATCH_UICANVAS_DB)
    try:
        with conn.cursor() as cur:
            cur.execute("select count(*) from proc.esg_data where esg_score is null")
            assert cur.fetchone()[0] == 0
            cur.execute("select count(*) from proc.contact where is_primary_contact")
            assert cur.fetchone()[0] == 5000
    finally:
        conn.close()
