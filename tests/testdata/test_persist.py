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
