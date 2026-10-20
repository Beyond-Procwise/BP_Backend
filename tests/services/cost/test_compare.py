"""Comparing options: a component counts only if every option has a value for it."""
import pytest

from src.services.cost.compare import compare_landed
from src.services.cost.landed_cost import landed_cost

LINES = [{"unit_price": 20.0, "quantity": 1000, "currency": "GBP"}]


def option(name, price=20.0, **kw):
    base = dict(lines=[{"unit_price": price, "quantity": 1000, "currency": "GBP"}],
                buyer_currency="GBP", cross_border=True, incoterm="FOB Shanghai",
                components=[{"key": "freight", "amount": 1200.0}],
                duty_rate=0.04, customs_basis="CIF", default_insurance_rate=0.003,
                vat_recoverable=True)
    base.update(kw)
    return {"name": name, "landed": landed_cost(**base)}


def test_a_component_one_option_lacks_is_left_out_for_everyone():
    a = option("A", price=20.0, components=[{"key": "freight", "amount": 1200.0},
                                           {"key": "brokerage", "amount": 150.0}])
    b = option("B", price=19.0, components=[{"key": "freight", "amount": 1500.0}])  # no brokerage
    out = compare_landed([a, b])
    assert "brokerage" not in out["comparable_components"]
    by = {o["name"]: o for o in out["options"]}
    assert [e["name"] for e in by["A"]["excluded"]] == ["brokerage"]
    # goods + freight + insurance + duty only, for both
    assert by["A"]["comparable_total"] == pytest.approx(
        a["landed"]["landed_total"] - 150.0)
    assert by["B"]["comparable_total"] == pytest.approx(b["landed"]["landed_total"])
    assert out["partial"] is True


def test_the_cheaper_option_changes_when_a_hidden_cost_is_counted_for_one_side_only():
    # A looks cheaper on its own total only because it states less (no freight stated)
    a = option("A", price=19.0, components=[])
    b = option("B", price=20.0, components=[{"key": "freight", "amount": 100.0}])
    assert a["landed"]["landed_total"] < b["landed"]["landed_total"]
    out = compare_landed([a, b])
    assert "freight" not in out["comparable_components"]   # A has no freight value
    assert out["partial"] is True


def test_identical_coverage_with_nothing_missing_is_not_partial():
    a = option("A", price=20.0, incoterm="DDP", components=[])
    b = option("B", price=21.0, incoterm="DDP", components=[])
    out = compare_landed([a, b])
    assert out["partial"] is False
    by = {o["name"]: o for o in out["options"]}
    assert by["A"]["comparable_total"] == pytest.approx(a["landed"]["landed_total"])
    assert out["lowest"] == ["A"]


def test_a_component_missing_for_every_option_makes_the_comparison_partial():
    out = compare_landed([option("A"), option("B")])
    assert out["partial"] is True          # destination charges missing on both
    assert "terminal_handling" in out["missing_for_all"]


def test_included_costs_count_as_a_value_of_zero():
    a = option("A", incoterm="DDP", components=[])
    b = option("B", incoterm="FOB Shanghai")
    out = compare_landed([a, b])
    # freight: A includes it (0), B states it (1200) -> comparable, and B is dearer for it
    assert "freight" in out["comparable_components"]


def test_an_option_that_cannot_be_priced_is_reported_and_not_ranked():
    a = option("A")
    out = compare_landed([a, {"name": "B", "landed": None}])
    assert out["unassessed"] == ["B"]
    assert [o["name"] for o in out["options"]] == ["A"]


def test_ties_are_all_named_as_lowest():
    out = compare_landed([option("A"), option("B")])
    assert sorted(out["lowest"]) == ["A", "B"]


def test_fewer_than_two_options_is_not_a_comparison():
    with pytest.raises(ValueError, match="two"):
        compare_landed([option("A")])


def test_options_in_different_currencies_are_refused():
    b = option("B", buyer_currency="USD",
               lines=[{"unit_price": 20.0, "quantity": 1000, "currency": "USD"}])
    with pytest.raises(ValueError, match="currency"):
        compare_landed([option("A"), b])
