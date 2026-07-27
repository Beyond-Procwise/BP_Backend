from datetime import date
from decimal import Decimal

from scripts.testdata.catalogue import build_catalogue, price_on
from scripts.testdata.reference import TaxonomyLeaf


def _leaves(count: int = 246) -> list[TaxonomyLeaf]:
    return [
        TaxonomyLeaf(
            l1="IT & Technology", l2=f"G{i}", l3=f"C{i}", l4=f"S{i}", l5=f"Leaf{i}",
            l1_id="C-2000", l2_id="C-3000", l3_id="C-4000",
            l4_id=f"C-45{i:02d}", l5_id=f"C-51{i:02d}",
            unspsc_code=str(30000000 + i), esg_impact="Low", category_status="Active",
            spend_classification="Direct", category_risk_rating="Minimal",
            audit_frequency="Annually", policy_coverage="Full",
        )
        for i in range(count)
    ]


def _supplier_ids(count: int = 500) -> list[str]:
    return [f"SUP-Supplier{i}" for i in range(count)]


def test_builds_exactly_five_thousand_items():
    assert len(build_catalogue(42, _leaves(), _supplier_ids())) == 5000


def test_every_item_maps_to_a_real_leaf():
    leaves = _leaves()
    valid = {leaf.l5 for leaf in leaves}
    for item in build_catalogue(42, leaves, _supplier_ids()):
        assert item.leaf.l5 in valid


def test_every_leaf_receives_at_least_one_item():
    leaves = _leaves()
    items = build_catalogue(42, leaves, _supplier_ids())
    assert {item.leaf.l5 for item in items} == {leaf.l5 for leaf in leaves}


def test_item_ids_are_unique():
    items = build_catalogue(42, _leaves(), _supplier_ids())
    assert len({item.item_id for item in items}) == 5000


def test_base_prices_are_positive():
    for item in build_catalogue(42, _leaves(), _supplier_ids()):
        assert item.base_price > 0


def test_catalogue_is_deterministic():
    leaves, suppliers = _leaves(), _supplier_ids()
    first = build_catalogue(42, leaves, suppliers)
    second = build_catalogue(42, leaves, suppliers)
    assert [(i.item_id, i.base_price) for i in first] == [
        (i.item_id, i.base_price) for i in second
    ]


def test_price_on_returns_two_decimal_places():
    item = build_catalogue(42, _leaves(), _supplier_ids())[0]
    value = price_on(item, date(2024, 6, 1), seed=42)
    assert isinstance(value, Decimal)
    assert value == value.quantize(Decimal("0.01"))


def test_price_on_is_deterministic_for_a_given_date():
    item = build_catalogue(42, _leaves(), _supplier_ids())[0]
    assert price_on(item, date(2024, 6, 1), seed=42) == price_on(
        item, date(2024, 6, 1), seed=42
    )


def test_price_drifts_upward_over_three_years():
    item = build_catalogue(42, _leaves(), _supplier_ids())[0]
    early = price_on(item, date(2023, 1, 1), seed=42)
    late = price_on(item, date(2026, 1, 1), seed=42)
    assert late > early


def test_descriptions_are_unique_across_the_catalogue():
    """Two items sharing description+uom+currency would pool as one product in
    the benchmark engine, averaging unrelated prices together."""
    items = build_catalogue(42, _leaves(), _supplier_ids())
    keys = [(i.description, i.unit_of_measure, i.currency) for i in items]
    assert len(set(keys)) == len(keys)


def test_description_still_names_its_leaf():
    items = build_catalogue(42, _leaves(), _supplier_ids())
    for item in items[:100]:
        assert item.leaf.l5 in item.description


def test_descriptions_carry_no_part_code():
    """A part code stapled to the end reads as synthetic on a demo screen."""
    for item in build_catalogue(42, _leaves(), _supplier_ids())[:200]:
        assert item.item_id not in item.description
        assert "ITM" not in item.description


def test_units_suit_what_is_being_bought():
    """A subscription is not sold by the tonne."""
    services = {"Service Package", "Subscription", "Retainer", "Installation"}
    goods_units = {"each", "box", "pack", "case", "metre", "tonne"}
    for item in build_catalogue(42, _leaves(), _supplier_ids()):
        noun = item.description.rsplit(" ", 1)[-1]
        if any(item.description.endswith(s) for s in services):
            assert item.unit_of_measure not in goods_units, item.description
