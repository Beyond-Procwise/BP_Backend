import pytest
from datetime import date
from decimal import Decimal

from scripts.testdata.catalogue import build_catalogue
from scripts.testdata.documents import build_chains
from scripts.testdata.org import build_business_units, build_cost_centres
from scripts.testdata.persist import COLUMNS, rows_for
from scripts.testdata.reference import TaxonomyLeaf
from scripts.testdata.suppliers import build_suppliers

FX = {"USD": 1.0, "GBP": 0.79, "EUR": 0.92, "INR": 83.2, "AED": 3.6725}


def _chains(count: int = 40):
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


def test_every_table_has_rows():
    rows = rows_for(_chains())
    for table in COLUMNS:
        assert rows[table], table


def test_row_width_matches_the_column_list():
    rows = rows_for(_chains())
    for table, columns in COLUMNS.items():
        for row in rows[table]:
            assert len(row) == len(columns), table


def test_one_row_per_quote_and_one_per_quote_line():
    chains = _chains()
    rows = rows_for(chains)
    assert len(rows["bp_quote_trgt"]) == sum(len(c.quotes) for c in chains)
    assert len(rows["bp_quote_line_items_trgt"]) == sum(
        len(q.lines) for c in chains for q in c.quotes)


def test_line_ids_are_unique_and_derived_from_their_document():
    rows = rows_for(_chains())
    idx = COLUMNS["bp_quote_line_items_trgt"].index("quote_line_id")
    ids = [r[idx] for r in rows["bp_quote_line_items_trgt"]]
    assert len(set(ids)) == len(ids)
    assert all("-" in i for i in ids)


def test_awarded_quote_carries_its_purchase_order_reference():
    chains = _chains()
    rows = rows_for(chains)
    qid = COLUMNS["bp_quote_trgt"].index("quote_id")
    poid = COLUMNS["bp_quote_trgt"].index("po_id")
    by_quote = {r[qid]: r[poid] for r in rows["bp_quote_trgt"]}
    linked = [c for c in chains if c.purchase_order]
    assert linked, "fixture must contain at least one chain that reached a PO"
    for chain in linked:
        awarded = min(chain.quotes, key=lambda q: q.net_total)
        assert by_quote[awarded.doc_id] == chain.purchase_order.doc_id


def test_marker_fields_identify_generated_rows():
    rows = rows_for(_chains())
    idx = COLUMNS["bp_invoice_trgt"].index("created_by")
    assert {r[idx] for r in rows["bp_invoice_trgt"]} == {"testdata"}


def test_totals_survive_the_mapping():
    chains = _chains()
    rows = rows_for(chains)
    amt = COLUMNS["bp_invoice_trgt"].index("invoice_amount")
    total = sum(Decimal(str(r[amt])) for r in rows["bp_invoice_trgt"])
    expected = sum(inv.net_total for c in chains for inv in c.invoices)
    assert total == expected


def test_purchase_order_row_content_matches_its_document():
    # bp_purchase_order_trgt had only width/non-emptiness coverage, so a
    # transposition of total_amount <-> total_amount_incl_tax (both Decimals)
    # would pass silently. Check real positions against the source document.
    chains = _chains()
    rows = rows_for(chains)
    idx = COLUMNS["bp_purchase_order_trgt"]
    linked = [c for c in chains if c.purchase_order]
    assert linked, "fixture must contain at least one chain that reached a PO"
    chain = linked[0]
    po = chain.purchase_order
    awarded = min(chain.quotes, key=lambda q: q.net_total)
    # Net and gross must actually differ (VAT > 0), otherwise a swap test
    # between the two money columns would prove nothing.
    assert po.net_total != po.gross_total
    row = next(r for r in rows["bp_purchase_order_trgt"]
               if r[idx.index("po_id")] == po.doc_id)
    assert row[idx.index("po_id")] == po.doc_id
    assert row[idx.index("total_amount")] == po.net_total
    assert row[idx.index("total_amount_incl_tax")] == po.gross_total
    assert row[idx.index("quote_reference")] == awarded.doc_id


def test_po_line_row_content_matches_its_document():
    # bp_po_line_items_trgt had only width/non-emptiness coverage, so a
    # transposition of unit_price <-> line_total (both Decimals) would pass
    # silently. Use a line with quantity != 1 so the two values differ.
    chains = _chains()
    rows = rows_for(chains)
    idx = COLUMNS["bp_po_line_items_trgt"]
    candidates = [
        (c.purchase_order, line)
        for c in chains if c.purchase_order
        for line in c.purchase_order.lines
        if line.quantity != 1
    ]
    assert candidates, "fixture must contain a PO line with quantity != 1"
    po, line = candidates[0]
    assert line.unit_price != line.line_total
    row = next(r for r in rows["bp_po_line_items_trgt"]
               if r[idx.index("po_line_id")] == f"{po.doc_id}-{line.line_number}")
    assert row[idx.index("po_id")] == po.doc_id
    assert row[idx.index("unit_price")] == line.unit_price
    assert row[idx.index("line_total")] == line.line_total


def test_invoice_line_row_content_matches_its_document():
    # bp_invoice_line_items_trgt had only width/non-emptiness coverage, so a
    # transposition of unit_price <-> line_amount (both Decimals) would pass
    # silently. Use a line with quantity != 1 and an invoice that reached a
    # PO, so both the money swap and the po_id foreign key are exercised.
    chains = _chains()
    rows = rows_for(chains)
    idx = COLUMNS["bp_invoice_line_items_trgt"]
    candidates = [
        (inv, line)
        for c in chains for inv in c.invoices
        for line in inv.lines
        if line.quantity != 1 and inv.parent_doc_id is not None
    ]
    assert candidates, (
        "fixture must contain an invoice line with quantity != 1 whose "
        "invoice reached a purchase order")
    inv, line = candidates[0]
    assert line.unit_price != line.line_total
    row = next(r for r in rows["bp_invoice_line_items_trgt"]
               if r[idx.index("invoice_line_id")] == f"{inv.doc_id}-{line.line_number}")
    assert row[idx.index("invoice_id")] == inv.doc_id
    assert row[idx.index("unit_price")] == line.unit_price
    assert row[idx.index("line_amount")] == line.line_total
    assert row[idx.index("po_id")] == inv.parent_doc_id


# --- required sets and load order -------------------------------------------

def test_required_names_are_a_subset_of_columns():
    from scripts.testdata.persist import COLUMNS, REQUIRED

    for table, required in REQUIRED.items():
        unknown = [c for c in required if c not in COLUMNS[table]]
        assert not unknown, f"{table}: {unknown}"


def test_every_mapped_table_declares_a_required_set():
    from scripts.testdata.persist import COLUMNS, REQUIRED

    assert set(REQUIRED) == set(COLUMNS)


def test_load_order_covers_every_table_headers_before_lines():
    from scripts.testdata.persist import COLUMNS, LOAD_ORDER

    assert set(LOAD_ORDER) == set(COLUMNS)
    assert LOAD_ORDER.index("bp_quote_trgt") < LOAD_ORDER.index("bp_quote_line_items_trgt")
    assert LOAD_ORDER.index("bp_purchase_order_trgt") < LOAD_ORDER.index("bp_po_line_items_trgt")
    assert LOAD_ORDER.index("bp_invoice_trgt") < LOAD_ORDER.index("bp_invoice_line_items_trgt")


def test_generated_document_rows_satisfy_their_required_sets():
    from scripts.testdata.loader import _check_required
    from scripts.testdata.persist import COLUMNS, REQUIRED, rows_for

    rows = rows_for(_chains(60))
    for table, table_rows in rows.items():
        assert table_rows, table
        _check_required(table, COLUMNS[table], REQUIRED[table], table_rows)


# --- requirements -----------------------------------------------------------

def test_requirement_columns_match_the_live_schema_width():
    from scripts.testdata.persist import REQUIREMENT_COLUMNS, REQUIREMENT_REQUIRED

    assert len(REQUIREMENT_COLUMNS) == 21
    unknown = [c for c in REQUIREMENT_REQUIRED if c not in REQUIREMENT_COLUMNS]
    assert not unknown


def test_one_requirement_per_chain_satisfying_its_required_set():
    from scripts.testdata.loader import _check_required
    from scripts.testdata.persist import (
        REQUIREMENT_COLUMNS, REQUIREMENT_REQUIRED, requirement_rows,
    )

    chains = _chains(60)
    rows = requirement_rows(chains)
    assert len(rows) == len(chains)
    for row in rows:
        assert len(row) == len(REQUIREMENT_COLUMNS)
    _check_required("bp_requirement", REQUIREMENT_COLUMNS, REQUIREMENT_REQUIRED, rows)


def test_requirement_agrees_with_the_documents_beneath_it():
    from scripts.testdata.persist import REQUIREMENT_COLUMNS, requirement_rows

    chains = _chains(60)
    by_id = {c.requirement_id: c for c in chains}
    columns = REQUIREMENT_COLUMNS
    for row in requirement_rows(chains):
        chain = by_id[row[columns.index("requirement_id")]]
        awarded = min(chain.quotes, key=lambda q: q.net_total)
        assert row[columns.index("target_budget")] == awarded.net_total
        assert row[columns.index("currency")] == awarded.currency
        assert row[columns.index("category")] == awarded.lines[0].leaf_path


def test_requirement_is_raised_before_its_first_quote():
    from datetime import datetime

    from scripts.testdata.persist import REQUIREMENT_COLUMNS, requirement_rows

    chains = _chains(60)
    by_id = {c.requirement_id: c for c in chains}
    columns = REQUIREMENT_COLUMNS
    for row in requirement_rows(chains):
        chain = by_id[row[columns.index("requirement_id")]]
        earliest = min(q.doc_date for q in chain.quotes)
        raised = row[columns.index("created_at")]
        assert isinstance(raised, datetime)
        assert raised.date() < earliest


def test_requirement_status_reflects_whether_a_po_was_raised():
    from scripts.testdata.persist import REQUIREMENT_COLUMNS, requirement_rows

    chains = _chains(120)
    by_id = {c.requirement_id: c for c in chains}
    columns = REQUIREMENT_COLUMNS
    statuses = set()
    for row in requirement_rows(chains):
        chain = by_id[row[columns.index("requirement_id")]]
        status = row[columns.index("status")]
        statuses.add(status)
        assert status == (
            "handed_off" if chain.purchase_order is not None else "complete"
        )
    assert statuses == {"handed_off", "complete"}


def test_requirement_status_uses_only_values_the_schema_accepts():
    from scripts.testdata.persist import (
        REQUIREMENT_COLUMNS, REQUIREMENT_STATUSES, requirement_rows,
    )

    position = REQUIREMENT_COLUMNS.index("status")
    used = {row[position] for row in requirement_rows(_chains(120))}
    assert used <= set(REQUIREMENT_STATUSES)
    assert used == {"handed_off", "complete"}


@pytest.mark.integration
def test_declared_requirement_statuses_match_the_live_check_constraint():
    """The CHECK constraint is the authority; this fails if live changes it."""
    import re

    from scripts.testdata.db import connect
    from scripts.testdata.persist import REQUIREMENT_STATUSES

    conn = connect("bp_sqldb")
    try:
        with conn.cursor() as cur:
            cur.execute(
                "select pg_get_constraintdef(con.oid) from pg_constraint con "
                "join pg_class rel on rel.oid = con.conrelid "
                "join pg_namespace n on n.oid = rel.relnamespace "
                "where n.nspname = 'proc' and rel.relname = 'bp_requirement' "
                "and con.conname = 'bp_requirement_status_check'"
            )
            definition = cur.fetchone()[0]
    finally:
        conn.close()

    allowed = set(re.findall(r"'([a-z_]+)'::character varying", definition))
    assert set(REQUIREMENT_STATUSES) == allowed, definition
