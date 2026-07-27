import json

import pytest

from scripts.testdata.catalogue import build_catalogue
from scripts.testdata.defects import (
    DEFECT_SPECS,
    plant,
    write_answer_key,
)
from scripts.testdata.documents import build_chains
from scripts.testdata.org import build_business_units, build_cost_centres
from scripts.testdata.reference import TaxonomyLeaf
from scripts.testdata.suppliers import build_suppliers

FX = {"USD": 1.0, "GBP": 0.79, "EUR": 0.92, "INR": 83.2, "AED": 3.6725}


def _build(chain_count: int):
    leaves = [
        TaxonomyLeaf(
            l1="IT & Technology", l2=f"G{i}", l3=f"C{i}", l4=f"S{i}", l5=f"Leaf{i}",
            unspsc_code=str(50000000 + i), esg_impact="Low", category_status="Active",
            spend_classification="Direct", category_risk_rating="Minimal",
            audit_frequency="Annually", policy_coverage="Full",
        )
        for i in range(60)
    ]
    suppliers = build_suppliers(42, leaves)
    items = build_catalogue(42, leaves, [s.bp_supplier_id for s in suppliers])
    units = build_business_units(42)
    centres = build_cost_centres(42, units, leaves)
    chains = build_chains(42, suppliers, items, centres, fx=FX, count=chain_count)
    return chains, centres


@pytest.fixture(scope="module")
def planted():
    chains, centres = _build(600)
    return plant(42, chains, centres)


@pytest.fixture(scope="module")
def planted_full():
    chains, centres = _build(1200)
    return plant(42, chains, centres)


def test_thirty_defect_specs_are_declared():
    assert len(DEFECT_SPECS) == 30


def test_specs_split_into_24_true_positives_and_6_negative_controls():
    kinds = [spec.kind for spec in DEFECT_SPECS]
    assert kinds.count("true_positive") == 24
    assert kinds.count("negative_control") == 6


def test_defect_refs_are_unique_and_sequential():
    refs = [spec.ref for spec in DEFECT_SPECS]
    assert len(set(refs)) == 30
    assert refs == [f"D{i:02d}" for i in range(1, 31)]


def test_planting_records_every_planted_instance(planted):
    assert planted.planted
    refs = {item.ref for item in planted.planted}
    for expected in ("D01", "D03", "D05", "D22", "D23", "D27"):
        assert expected in refs


def test_duplicate_invoices_share_a_number_with_a_different_document_id(planted):
    duplicates = [item for item in planted.planted if item.ref == "D01"]
    assert duplicates
    for item in duplicates:
        assert item.detail["original_doc_id"] != item.detail["duplicate_doc_id"]
        assert item.detail["invoice_number"]


def test_overbilling_records_a_positive_delta(planted):
    overbilled = [item for item in planted.planted if item.ref == "D03"]
    assert overbilled
    for item in overbilled:
        assert item.detail["delta_gbp"] > 0
        assert item.detail["invoice_unit_price"] > item.detail["po_unit_price"]


def test_services_lines_have_null_quantity_and_unit_price(planted):
    services = [item for item in planted.planted if item.ref == "D22"]
    assert services

    by_doc = {}
    for chain in planted.chains:
        for document in [*chain.quotes, *chain.invoices]:
            by_doc[document.doc_id] = document

    for item in services:
        document = by_doc[item.subject_id]
        nulls = [
            line for line in document.lines
            if line.quantity is None and line.unit_price is None
        ]
        assert nulls
        for line in nulls:
            assert line.line_total > 0


def test_credit_notes_are_negative(planted):
    credits = [item for item in planted.planted if item.ref == "D23"]
    assert credits
    for item in credits:
        assert item.detail["net_total"] < 0


def test_budget_overruns_exceed_the_allocation(planted):
    overruns = [item for item in planted.planted if item.ref == "D27"]
    assert overruns

    by_id = {centre.cc_id: centre for centre in planted.cost_centres}
    for item in overruns:
        centre = by_id[item.subject_id]
        assert centre.actual_spend_ytd > centre.budget_allocated_annual


def test_planting_is_deterministic():
    chains_a, centres_a = _build(600)
    chains_b, centres_b = _build(600)
    a = plant(42, chains_a, centres_a)
    b = plant(42, chains_b, centres_b)
    assert [(p.ref, p.subject_id) for p in a.planted] == [
        (p.ref, p.subject_id) for p in b.planted
    ]


def test_every_declared_defect_type_is_actually_planted(planted_full):
    planted_refs = {item.ref for item in planted_full.planted}
    missing = sorted({spec.ref for spec in DEFECT_SPECS} - planted_refs)
    assert not missing, f"declared but never planted: {missing}"


def test_negative_controls_are_all_represented(planted_full):
    negative_refs = {
        spec.ref for spec in DEFECT_SPECS if spec.kind == "negative_control"
    }
    planted_negative = {
        item.ref for item in planted_full.planted if item.kind == "negative_control"
    }
    assert planted_negative == negative_refs


def test_quantity_mismatch_invoice_exceeds_po_quantity(planted_full):
    items = [p for p in planted_full.planted if p.ref == "D04"]
    assert items
    for item in items:
        assert item.detail["invoice_quantity"] > item.detail["po_quantity"]


def test_split_pos_are_grouped_and_each_sits_below_threshold(planted_full):
    items = [p for p in planted_full.planted if p.ref == "D08"]
    assert items
    for item in items:
        sibling_ids = item.detail["sibling_doc_ids"]
        assert len(sibling_ids) >= 2
        for value in item.detail["sibling_net_totals"]:
            assert value < item.detail["threshold"]


def test_award_variance_records_the_foregone_saving(planted_full):
    items = [p for p in planted_full.planted if p.ref == "D09"]
    assert items
    for item in items:
        assert item.detail["foregone_saving_gbp"] > 0
        assert item.detail["awarded_net_total"] > item.detail["lowest_net_total"]


@pytest.fixture(scope="module")
def planted_at_scale():
    """Big enough that the largest defect targets are actually reached.

    D05 draws from the ~16% of chains that stop before a PO, so a small fixture
    exhausts the pool long before the declared target and hides an overshoot.
    """
    chains, centres = _build(3000)
    return plant(42, chains, centres)


def test_no_defect_type_overshoots_its_declared_target(planted_at_scale):
    counts: dict[str, int] = {}
    for item in planted_at_scale.planted:
        counts[item.ref] = counts.get(item.ref, 0) + 1
    overshot = {
        spec.ref: (counts[spec.ref], spec.count)
        for spec in DEFECT_SPECS
        if counts.get(spec.ref, 0) > spec.count
    }
    assert not overshot, f"planted more than declared (planted, target): {overshot}"


def test_answer_key_is_written_and_reloadable(planted, tmp_path):
    json_path = tmp_path / "answer-key.json"
    md_path = tmp_path / "answer-key.md"
    write_answer_key(planted, json_path, md_path)

    payload = json.loads(json_path.read_text())
    assert payload["total_planted"] == len(planted.planted)
    assert set(payload["by_ref"]) <= {spec.ref for spec in DEFECT_SPECS}
    assert md_path.read_text().startswith("# Test Dataset Answer Key")
