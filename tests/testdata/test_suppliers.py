import re

from scripts.testdata.suppliers import (
    CROSSWALK_DDL,
    TIERS,
    build_crosswalk,
    build_suppliers,
)
from scripts.testdata.reference import TaxonomyLeaf


def _leaves(count: int = 246) -> list[TaxonomyLeaf]:
    families = [
        "IT & Technology", "Marketing & Media", "Facilities & Real Estate",
        "Professional Services", "Logistics & Supply Chain",
        "Office & Administrative Supplies",
    ]
    return [
        TaxonomyLeaf(
            l1=families[i % len(families)], l2=f"G{i}", l3=f"C{i}", l4=f"S{i}", l5=f"Leaf{i}",
            l1_id="C-2000", l2_id="C-3000", l3_id="C-4000",
            l4_id=f"C-45{i:02d}", l5_id=f"C-51{i:02d}",
            unspsc_code=str(20000000 + i), esg_impact="Low", category_status="Active",
            spend_classification="Indirect", category_risk_rating="Minimal",
            audit_frequency="Annually", policy_coverage="Partial",
        )
        for i in range(count)
    ]


def test_tiers_sum_to_five_thousand_suppliers():
    assert sum(tier.count for tier in TIERS) == 5000


def test_tier_spend_shares_sum_to_one():
    assert round(sum(tier.spend_share for tier in TIERS), 6) == 1.0


def test_builds_exactly_five_thousand_suppliers():
    assert len(build_suppliers(42, _leaves())) == 5000


def test_supplier_ids_follow_both_conventions_and_are_unique():
    suppliers = build_suppliers(42, _leaves())
    assert len({s.bp_supplier_id for s in suppliers}) == 5000
    assert len({s.uicanvas_supplier_id for s in suppliers}) == 5000
    for supplier in suppliers[:50]:
        assert supplier.bp_supplier_id.startswith("SUP-")
        assert re.fullmatch(r"SI\d{6}", supplier.uicanvas_supplier_id)


def test_primary_categories_cover_every_leaf():
    leaves = _leaves()
    suppliers = build_suppliers(42, leaves)
    assigned = {s.primary_leaf.l5 for s in suppliers}
    assert assigned == {leaf.l5 for leaf in leaves}


def test_tier_counts_are_respected():
    suppliers = build_suppliers(42, _leaves())
    counts: dict[str, int] = {}
    for supplier in suppliers:
        counts[supplier.tier] = counts.get(supplier.tier, 0) + 1
    for tier in TIERS:
        assert counts[tier.name] == tier.count


def test_every_supplier_populates_all_51_columns():
    suppliers = build_suppliers(42, _leaves())
    for supplier in suppliers[:100]:
        assert len(supplier.columns) == 51
        assert supplier.columns["supplier_name"]
        assert supplier.columns["country"]
        assert supplier.columns["default_currency"]


def test_generation_is_deterministic():
    leaves = _leaves()
    first = build_suppliers(42, leaves)
    second = build_suppliers(42, leaves)
    assert [s.bp_supplier_id for s in first] == [s.bp_supplier_id for s in second]
    assert [s.columns for s in first] == [s.columns for s in second]


def test_crosswalk_maps_every_supplier_once():
    suppliers = build_suppliers(42, _leaves())
    crosswalk = build_crosswalk(suppliers)
    assert len(crosswalk) == 5000
    assert len({row[0] for row in crosswalk}) == 5000
    assert len({row[1] for row in crosswalk}) == 5000


def test_crosswalk_ddl_uses_bp_prefix_and_index_convention():
    assert "proc.bp_supplier_id_crosswalk" in CROSSWALK_DDL
    assert "ix_bp_supplier_id_crosswalk_uicanvas_supplier_id" in CROSSWALK_DDL
