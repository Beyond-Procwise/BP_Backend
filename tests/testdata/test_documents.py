from datetime import date
from decimal import Decimal

import pytest

from scripts.testdata.catalogue import build_catalogue
from scripts.testdata.documents import build_chains
from scripts.testdata.org import ENTITIES, build_business_units, build_cost_centres
from scripts.testdata.reference import TaxonomyLeaf
from scripts.testdata.suppliers import build_suppliers


@pytest.fixture(scope="module")
def _world():
    """Suppliers, catalogue and organisation are expensive; build them once."""
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
    units = build_business_units(42)
    centres = build_cost_centres(42, units, leaves)
    return suppliers, items, centres


@pytest.fixture(scope="module")
def chains(_world):
    suppliers, items, centres = _world
    return build_chains(42, suppliers, items, centres, count=200)


def test_builds_the_requested_number_of_chains(chains):
    assert len(chains) == 200


def test_every_chain_has_between_two_and_five_quotes(chains):
    for chain in chains:
        assert 2 <= len(chain.quotes) <= 5


def test_awarded_supplier_is_one_of_the_quoting_suppliers(chains):
    for chain in chains:
        quoting = {quote.supplier_id for quote in chain.quotes}
        assert chain.awarded_supplier_id in quoting


def test_purchase_order_when_present_matches_the_awarded_supplier(chains):
    for chain in chains:
        if chain.purchase_order is not None:
            assert chain.purchase_order.supplier_id == chain.awarded_supplier_id


def test_line_totals_sum_to_document_net_total(chains):
    for chain in chains:
        for document in [*chain.quotes, *chain.invoices]:
            summed = sum((line.line_total for line in document.lines), Decimal("0"))
            assert summed == document.net_total


def test_quantity_times_unit_price_equals_line_total(chains):
    for chain in chains:
        for document in [*chain.quotes, *chain.invoices]:
            for line in document.lines:
                if line.quantity is None or line.unit_price is None:
                    continue
                assert line.quantity * line.unit_price == line.line_total


def test_documents_carry_full_organisation_attribution(chains):
    valid_orgs = {entity.org_id for entity in ENTITIES}
    for chain in chains:
        for document in [*chain.quotes, *chain.invoices]:
            assert document.org_id in valid_orgs
            assert document.bu_id
            assert document.cc_id


def test_document_dates_fall_inside_the_window(chains):
    for chain in chains:
        for document in [*chain.quotes, *chain.invoices]:
            assert date(2023, 1, 1) <= document.doc_date <= date(2026, 7, 31)


def test_invoice_dates_never_precede_their_purchase_order(chains):
    for chain in chains:
        if chain.purchase_order is None:
            continue
        for invoice in chain.invoices:
            assert invoice.doc_date >= chain.purchase_order.doc_date


def test_document_ids_are_globally_unique(_world):
    suppliers, items, centres = _world
    seen: set[str] = set()
    for chain in build_chains(42, suppliers, items, centres, count=400):
        for document in [*chain.quotes, *chain.invoices]:
            assert document.doc_id not in seen
            seen.add(document.doc_id)
        if chain.purchase_order is not None:
            assert chain.purchase_order.doc_id not in seen
            seen.add(chain.purchase_order.doc_id)


def test_chains_are_deterministic(_world):
    suppliers, items, centres = _world
    first = build_chains(42, suppliers, items, centres, count=100)
    second = build_chains(42, suppliers, items, centres, count=100)
    assert [c.requirement_id for c in first] == [c.requirement_id for c in second]
    assert [c.awarded_supplier_id for c in first] == [
        c.awarded_supplier_id for c in second
    ]


def test_some_chains_stop_before_a_purchase_order(chains):
    """The lossy tail is what test B3 (spend with no PO) measures."""
    without_po = [chain for chain in chains if chain.purchase_order is None]
    assert without_po
    assert all(chain.invoices for chain in without_po)
