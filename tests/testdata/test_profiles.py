import pytest

from scripts.testdata import profiles
from scripts.testdata.catalogue import build_catalogue
from scripts.testdata.defects import DEFECT_SPECS, plant
from scripts.testdata.documents import build_chains
from scripts.testdata.org import build_business_units, build_cost_centres
from scripts.testdata.reference import TaxonomyLeaf
from scripts.testdata.suppliers import build_suppliers

FX = {"USD": 1.0, "GBP": 0.79, "EUR": 0.92, "INR": 83.2, "AED": 3.6725}


def _world():
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
    return suppliers, items, centres


def _plant_with(profile, count=600):
    suppliers, items, centres = _world()
    chains = build_chains(
        42, suppliers, items, centres, fx=FX, count=count,
        no_po_share=profile.no_po_share,
    )
    return plant(42, chains, centres, only=profile.plant_refs)


def test_two_profiles_are_declared():
    assert set(profiles.PROFILES) == {"test", "demo"}
    for profile in profiles.PROFILES.values():
        assert profile.purpose


def test_unknown_profile_is_rejected_by_name():
    with pytest.raises(ValueError, match="unknown profile"):
        profiles.get("production")


def test_test_profile_plants_every_defect_type():
    assert profiles.TEST.plant_refs == {spec.ref for spec in DEFECT_SPECS}
    assert profiles.TEST.plants_true_positives


def test_demo_profile_plants_no_true_positives():
    assert not profiles.DEMO.plants_true_positives
    assert profiles.DEMO.plant_refs == profiles.NEGATIVE_CONTROL_REFS


def test_demo_profile_keeps_the_legitimate_business_cases():
    """Lump-sum services and credit notes are ordinary procurement. An estate
    without them does not look real."""
    assert "D22" in profiles.DEMO.plant_refs
    assert "D23" in profiles.DEMO.plant_refs


def test_demo_estate_carries_nothing_for_a_detector_to_find():
    result = _plant_with(profiles.DEMO)
    kinds = {item.kind for item in result.planted}
    assert kinds == {"negative_control"}


def test_test_estate_carries_true_positives():
    result = _plant_with(profiles.TEST)
    kinds = {item.kind for item in result.planted}
    assert "true_positive" in kinds


def test_demo_profile_raises_a_purchase_order_for_nearly_every_chain():
    suppliers, items, centres = _world()
    demo = build_chains(42, suppliers, items, centres, fx=FX, count=600,
                        no_po_share=profiles.DEMO.no_po_share)
    without = [c for c in demo if c.purchase_order is None]
    assert len(without) / len(demo) < 0.06


def test_test_profile_keeps_a_substantial_maverick_population():
    suppliers, items, centres = _world()
    test = build_chains(42, suppliers, items, centres, fx=FX, count=600,
                        no_po_share=profiles.TEST.no_po_share)
    without = [c for c in test if c.purchase_order is None]
    assert 0.10 < len(without) / len(test) < 0.25


def test_the_demo_estate_is_markedly_cleaner_than_the_test_estate():
    demo = _plant_with(profiles.DEMO)
    test = _plant_with(profiles.TEST)
    assert len(demo.planted) < len(test.planted) / 2


def test_both_profiles_stay_deterministic():
    assert [
        (p.ref, p.subject_id) for p in _plant_with(profiles.DEMO).planted
    ] == [
        (p.ref, p.subject_id) for p in _plant_with(profiles.DEMO).planted
    ]


def test_excluded_defects_do_not_mutate_the_documents():
    """Planting must not merely go unrecorded -- the inflated prices, duplicate
    invoices and over-tolerance totals must not be in the data at all."""
    demo = _plant_with(profiles.DEMO)
    doc_ids = [
        document.doc_id
        for chain in demo.chains
        for document in [*chain.quotes, *chain.invoices]
    ]
    assert not [d for d in doc_ids if d.endswith("-DUP")]
    assert not [d for d in doc_ids if d.endswith("A")]
