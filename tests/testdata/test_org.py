from scripts.testdata.org import (
    ENTITIES,
    GROUP_ORG_ID,
    build_business_units,
    build_cost_centres,
)
from scripts.testdata.reference import TaxonomyLeaf


def _leaves(count: int) -> list[TaxonomyLeaf]:
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


def test_there_are_six_buying_entities():
    assert len(ENTITIES) == 6
    assert GROUP_ORG_ID == "ORG-GRP"
    assert GROUP_ORG_ID not in {entity.org_id for entity in ENTITIES}


def test_entity_spend_shares_sum_to_one():
    assert round(sum(entity.spend_share for entity in ENTITIES), 6) == 1.0


def test_entity_cost_centre_counts_sum_to_500():
    assert sum(entity.cost_centre_count for entity in ENTITIES) == 500


def test_each_entity_has_a_distinct_id_and_currency_pairing():
    assert len({entity.org_id for entity in ENTITIES}) == 6
    by_id = {entity.org_id: entity for entity in ENTITIES}
    assert by_id["ORG-UK"].currency == "GBP"
    assert by_id["ORG-DE"].currency == "EUR"
    assert by_id["ORG-US"].currency == "USD"
    assert by_id["ORG-IN"].currency == "INR"
    assert by_id["ORG-AE"].currency == "AED"


def test_business_unit_tree_has_the_specified_breadth():
    units = build_business_units(42)
    assert len(units) == 400
    assert len({unit.l1 for unit in units}) == 6
    assert len({(unit.l1, unit.l2) for unit in units}) == 40
    assert len({(unit.l1, unit.l2, unit.l3) for unit in units}) == 120
    assert len({(unit.l1, unit.l2, unit.l3, unit.l4) for unit in units}) == 240


def test_every_business_unit_belongs_to_a_real_entity():
    units = build_business_units(42)
    valid = {entity.org_id for entity in ENTITIES}
    assert all(unit.org_id in valid for unit in units)


def test_business_units_are_deterministic():
    assert build_business_units(42) == build_business_units(42)


def test_cost_centres_match_entity_allocation():
    units = build_business_units(42)
    centres = build_cost_centres(42, units, _leaves(50))
    assert len(centres) == 500

    per_entity = {entity.org_id: 0 for entity in ENTITIES}
    for centre in centres:
        per_entity[centre.org_id] += 1
    for entity in ENTITIES:
        assert per_entity[entity.org_id] == entity.cost_centre_count


def test_cost_centres_have_six_levels_and_a_category_link():
    units = build_business_units(42)
    centres = build_cost_centres(42, units, _leaves(50))
    for centre in centres:
        assert len(centre.levels) == 6
        assert all(level for level in centre.levels)
        assert centre.linked_category_level_5_id


def test_cost_centre_budgets_are_positive_and_thresholds_in_range():
    units = build_business_units(42)
    centres = build_cost_centres(42, units, _leaves(50))
    for centre in centres:
        assert centre.budget_allocated_annual > 0
        assert 5000 <= centre.spend_threshold_limit <= 250000


def test_cost_centre_ids_are_unique():
    units = build_business_units(42)
    centres = build_cost_centres(42, units, _leaves(50))
    assert len({centre.cc_id for centre in centres}) == 500


def test_cost_centre_links_the_real_category_level_5_id():
    """Plan 1 put the UNSPSC code in this column. It wants the L5 id."""
    units = build_business_units(42)
    leaves = _leaves(50)
    centres = build_cost_centres(42, units, leaves)
    valid = {leaf.l5_id for leaf in leaves}
    for centre in centres:
        assert centre.linked_category_level_5_id in valid
