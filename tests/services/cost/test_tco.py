"""TCO: present value of everything over the horizon, timing applied only here."""
import pytest

from src.services.cost.landed_cost import landed_cost
from src.services.cost.tco import present_value, tco

LANDED = landed_cost(
    lines=[{"unit_price": 20.0, "quantity": 1000, "currency": "GBP"}], buyer_currency="GBP",
    cross_border=True, incoterm="FOB Shanghai",
    components=[{"key": "freight", "amount": 1200.0}, {"key": "brokerage", "amount": 150.0}],
    duty_rate=0.04, customs_basis="CIF", default_insurance_rate=0.003, vat_recoverable=True)
TERMS = {"net_days": 60, "discount_pct": 0.02, "discount_days": 10}
LIFE = [{"key": "maintenance_support", "amount": 500.0, "year": y} for y in (1, 2, 3)] + [
    {"key": "disposal", "amount": 300.0, "year": 3}]


def run(**over):
    args = dict(landed=LANDED, cost_of_capital=0.08, horizon_years=3, payment_terms=TERMS,
                ap_can_take_discount=True, lifetime_items=LIFE)
    return tco(**{**args, **over})


def test_present_value():
    assert present_value(100.0, 365, 0.08) == pytest.approx(100 / 1.08)
    assert present_value(100.0, 0, 0.08) == 100.0


def test_the_specification_case():
    r = run()
    assert r["discount_choice"] == "discount"
    goods = next(c for c in r["components"] if c["name"] == "goods")
    assert goods["pv"] == pytest.approx(19558.72, abs=0.005)
    # the other landed components fall at day 0 because nobody stated their timing
    others = sum(c["pv"] for c in r["components"] if c["name"] != "goods" and c["pv"] is not None)
    assert others == pytest.approx(2270.76, abs=0.005)
    pvs = {(c["name"], c["pay_day"]): c["pv"] for c in r["lifetime_items"]}
    assert sum(c["pv"] for c in r["lifetime_items"] if c["name"] == "maintenance_support") \
        == pytest.approx(1288.55, abs=0.005)
    assert next(c["pv"] for c in r["lifetime_items"] if c["name"] == "disposal") \
        == pytest.approx(238.15, abs=0.005)
    # full precision: 23,356.17 (the specification's .18 adds up already-rounded parts)
    assert r["tco"] == pytest.approx(23356.17, abs=0.005)
    assert r["tco_status"] == "incomplete" == r["landed_status"]
    assert r["missing"] == LANDED["missing"]


def test_ap_that_cannot_take_the_discount_pays_net():
    r = run(ap_can_take_discount=False)
    assert r["discount_choice"] == "net"
    goods = next(c for c in r["components"] if c["name"] == "goods")
    assert goods["pv"] == pytest.approx(19748.57, abs=0.005)
    assert goods["pay_day"] == 60


def test_an_unstated_ap_capability_is_not_assumed_and_is_flagged():
    r = run(ap_can_take_discount=None)
    assert r["discount_choice"] == "net"
    assert any("ap_can_take_discount" in w for w in r["warnings"])


def test_the_discount_is_never_combined_with_full_term_timing():
    r = run()
    goods = next(c for c in r["components"] if c["name"] == "goods")
    assert goods["pay_day"] == 10
    assert goods["pv"] == pytest.approx(present_value(20000 * 0.98, 10, 0.08))


def test_a_discount_worth_less_than_net_is_not_taken():
    # a 0.01% discount for paying 60 days early costs more in capital than it saves
    r = run(payment_terms={"net_days": 60, "discount_pct": 0.0001, "discount_days": 1})
    assert r["discount_choice"] == "net"


def test_no_payment_terms_means_day_zero_not_a_financing_benefit():
    r = run(payment_terms=None, lifetime_items=[])
    goods = next(c for c in r["components"] if c["name"] == "goods")
    assert goods["pv"] == 20000.0


def test_an_inflow_reduces_the_total():
    base = run()["tco"]
    r = run(lifetime_items=LIFE + [{"key": "residual_value", "amount": 1000.0, "year": 3,
                                    "direction": "inflow"}])
    assert r["tco"] == pytest.approx(base - present_value(1000.0, 3 * 365, 0.08))


def test_items_beyond_the_horizon_are_left_out_and_named():
    r = run(lifetime_items=LIFE + [{"key": "switching_cost", "amount": 900.0, "year": 4}])
    assert r["beyond_horizon"] == ["switching_cost"]
    assert r["tco"] == pytest.approx(run()["tco"])


def test_an_item_on_the_horizon_boundary_counts():
    assert run(horizon_years=3)["tco"] == pytest.approx(
        run(horizon_years=3, lifetime_items=LIFE)["tco"])
    shorter = run(horizon_years=2)
    assert shorter["beyond_horizon"] == ["maintenance_support", "disposal"]  # both year-3 items


def test_every_unsupplied_ownership_item_is_listed_not_zeroed():
    r = run()
    assert "installation" in r["excluded_lifetime"]
    assert "maintenance_support" not in r["excluded_lifetime"]
    assert "disposal" not in r["excluded_lifetime"]


def test_an_item_with_no_timing_is_not_guessed():
    r = run(lifetime_items=[{"key": "energy_utilities", "amount": 100.0}])
    assert r["lifetime_items"] == []
    assert "energy_utilities" in r["excluded_lifetime"]


def test_a_lifetime_item_in_another_currency_is_refused_not_mixed():
    with pytest.raises(ValueError, match="convert"):
        run(lifetime_items=[{"key": "disposal", "amount": 300.0, "year": 3, "currency": "USD"}])


def test_an_unknown_lifetime_key_is_reported():
    r = run(lifetime_items=[{"key": "mystery", "amount": 1.0, "year": 1}])
    assert r["unrecognised_lifetime_items"] == ["mystery (not on the ownership checklist)"]
    assert r["tco"] == pytest.approx(run(lifetime_items=[])["tco"])


def test_missing_landed_components_stay_out_of_the_tco():
    r = run(lifetime_items=[])
    missing = [c for c in r["components"] if c["status"] == "missing"]
    assert missing and all(c["pv"] is None for c in missing)


def test_a_zero_cost_of_capital_is_just_the_sum():
    r = run(cost_of_capital=0.0, payment_terms=None)
    assert r["tco"] == pytest.approx(LANDED["landed_total"] + 3 * 500 + 300)
