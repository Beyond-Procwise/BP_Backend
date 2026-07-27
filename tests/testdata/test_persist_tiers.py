import json

import pytest

from scripts.testdata.catalogue import build_catalogue
from scripts.testdata.documents import build_chains
from scripts.testdata.org import build_business_units, build_cost_centres
from scripts.testdata.persist import COLUMNS as TRGT_COLUMNS
from scripts.testdata.persist import rows_for
from scripts.testdata.persist_tiers import (
    COLUMNS,
    LOAD_ORDER,
    REQUIRED,
    MissingRequiredValue,
    check_required,
    rows_for_tiers,
    source_file_for,
)
from scripts.testdata.reference import TaxonomyLeaf
from scripts.testdata.suppliers import build_suppliers

FX = {"USD": 1.0, "GBP": 0.79, "EUR": 0.92, "INR": 83.2, "AED": 3.6725}


def _chains(count: int = 60):
    leaves = [
        TaxonomyLeaf(
            l1="IT & Technology", l2=f"G{i}", l3=f"C{i}", l4=f"S{i}", l5=f"Leaf{i}",
            l1_id="C-2000", l2_id="C-3000", l3_id="C-4000",
            l4_id=f"C-45{i:02d}", l5_id=f"C-51{i:02d}",
            unspsc_code=str(40000000 + i), esg_impact="Low", category_status="Active",
            spend_classification="Direct", category_risk_rating="Minimal",
            audit_frequency="Annually", policy_coverage="Full",
        )
        for i in range(60)
    ]
    suppliers = build_suppliers(42, leaves)
    items = build_catalogue(42, leaves, [s.bp_supplier_id for s in suppliers])
    centres = build_cost_centres(42, build_business_units(42), leaves)
    return build_chains(42, suppliers, items, centres, fx=FX, count=count)


@pytest.fixture(scope="module")
def built():
    chains = _chains(60)
    return rows_for_tiers(chains), chains


def test_twelve_tier_tables_are_mapped():
    assert len(COLUMNS) == 12
    assert set(LOAD_ORDER) == set(COLUMNS)


def test_raw_loads_before_its_lines_and_staging_after():
    order = list(LOAD_ORDER)
    assert order.index("bp_quote_raw") < order.index("bp_quote_line_items_raw")
    assert order.index("bp_invoice_raw") < order.index("bp_invoice_line_items_raw")
    assert order.index("bp_quote_raw") < order.index("bp_quote_stg")


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


def test_every_tier_table_receives_rows(built):
    rows, _ = built
    for table, table_rows in rows.items():
        assert table_rows, table


def test_raw_and_staging_agree_with_the_promoted_record(built):
    """All three tiers describe the same documents, so their counts must match."""
    rows, chains = built
    trgt = rows_for(chains)
    for raw, stg, promoted in (
        ("bp_quote_raw", "bp_quote_stg", "bp_quote_trgt"),
        ("bp_purchase_order_raw", "bp_purchase_order_stg", "bp_purchase_order_trgt"),
        ("bp_invoice_raw", "bp_invoice_stg", "bp_invoice_trgt"),
    ):
        assert len(rows[raw]) == len(rows[stg]) == len(trgt[promoted]), raw


def test_the_trigger_can_find_every_promoted_document(built):
    """trg_fn_*_trgt_outcome matches doc_pk_candidate against the _trgt key."""
    rows, chains = built
    trgt = rows_for(chains)
    for raw, promoted, key in (
        ("bp_quote_raw", "bp_quote_trgt", "quote_id"),
        ("bp_purchase_order_raw", "bp_purchase_order_trgt", "po_id"),
        ("bp_invoice_raw", "bp_invoice_trgt", "invoice_id"),
    ):
        candidates = {
            row[COLUMNS[raw].index("doc_pk_candidate")] for row in rows[raw]
        }
        promoted_ids = {
            row[TRGT_COLUMNS[promoted].index(key)] for row in trgt[promoted]
        }
        assert promoted_ids <= candidates, raw


def test_every_raw_row_names_a_source_file(built):
    rows, _ = built
    for table in ("bp_quote_raw", "bp_purchase_order_raw", "bp_invoice_raw"):
        position = COLUMNS[table].index("source_file")
        for row in rows[table]:
            assert row[position].startswith("s3://bp-testdata/")
            assert row[position].endswith(".pdf")


def test_source_file_is_unique_per_document(built):
    rows, _ = built
    seen = set()
    for table in ("bp_quote_raw", "bp_purchase_order_raw", "bp_invoice_raw"):
        position = COLUMNS[table].index("source_file")
        for row in rows[table]:
            assert row[position] not in seen
            seen.add(row[position])


def test_raw_payload_is_valid_json_describing_its_document(built):
    rows, _ = built
    columns = COLUMNS["bp_invoice_raw"]
    for row in rows["bp_invoice_raw"][:50]:
        payload = json.loads(row[columns.index("raw_payload")])
        assert payload["doc_id"] == row[columns.index("invoice_id")]
        assert payload["doc_type"] == "Invoice"
        assert payload["line_count"] > 0


def test_line_rows_reference_their_raw_parent(built):
    rows, _ = built
    for raw, lines in (
        ("bp_quote_raw", "bp_quote_line_items_raw"),
        ("bp_purchase_order_raw", "bp_po_line_items_raw"),
        ("bp_invoice_raw", "bp_invoice_line_items_raw"),
    ):
        parents = {row[COLUMNS[raw].index("raw_id")] for row in rows[raw]}
        position = COLUMNS[lines].index("raw_id")
        for row in rows[lines]:
            assert row[position] in parents, lines


def test_generated_rows_satisfy_their_required_sets(built):
    rows, _ = built
    for table, table_rows in rows.items():
        check_required(table, table_rows)


def test_a_missing_required_value_is_an_error_not_a_silent_null():
    blank = [None] * len(COLUMNS["bp_invoice_raw"])
    with pytest.raises(MissingRequiredValue, match="bp_invoice_raw.raw_id"):
        check_required("bp_invoice_raw", [blank])


def test_mapping_is_deterministic():
    chains = _chains(40)
    assert rows_for_tiers(chains) == rows_for_tiers(chains)


def test_source_file_for_uses_the_document_type_folder():
    chains = _chains(5)
    quote = chains[0].quotes[0]
    assert source_file_for(quote) == f"s3://bp-testdata/quote/{quote.doc_id}.pdf"
