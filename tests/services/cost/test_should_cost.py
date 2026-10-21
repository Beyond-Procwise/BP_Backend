import pytest

from src.services.cost import should_cost as sc

BOM = [{"material_id": "PP", "net_weight": 0.5, "price_per_kg": 2.0, "scrap_rate": 0.2,
        "scrap_recovery_price_pct": 0.5}]
OP = {"process": "injection_moulding", "cycle_time_s": 36, "cavities": 2, "oee": 0.8, "setup_time_h": 2,
      "operators_per_machine": 1, "machine_rate": 80.0, "labour_rate": 20.0}
KW = dict(bom=BOM, routing=[OP], batch_size=1000, overhead_pct=0.10, sga_pct=0.05, margin_pct=0.08,
          logistics_per_unit=0.05, currency="GBP")


def run(**over):
    return sc.should_cost(**{**KW, **over})


def test_hand_worked_unit_cost_by_element():
    out = run()
    e = out["per_unit"]
    assert e["material"] == pytest.approx(1.125)              # 0.625 gross kg x 2 less 0.125 kg scrap at 50%
    assert e["conversion"] == pytest.approx(0.66)             # 0.00625 h x 80 + 2 h x 80 / 1000
    assert e["labour"] == pytest.approx(0.125)                # 0.00625 h x 20
    assert e["overhead"] == pytest.approx(0.0785)             # 10% of conversion + labour
    assert e["sga"] == pytest.approx(0.099425)                # 5% of manufacturing cost 1.9885
    assert e["logistics"] == pytest.approx(0.05)
    assert e["margin"] == pytest.approx(0.08 * (0.66 + 0.125 + 0.0785 + 0.099425))   # value-add only
    assert out["unit_cost"] == pytest.approx(sum(e.values()))
    assert out["label"] == "modelled estimate"


def test_margin_on_total_cost_marks_up_material_too():
    a, b = run(margin_base="value_add_only"), run(margin_base="total_cost")
    assert b["per_unit"]["margin"] > a["per_unit"]["margin"]
    assert b["per_unit"]["margin"] == pytest.approx(0.08 * (1.125 + 0.66 + 0.125 + 0.0785 + 0.099425 + 0.05))


def test_annual_cost_uses_annual_volume():
    out = run(annual_volume=100000)
    assert out["annual"]["total"] == pytest.approx(out["unit_cost"] * 100000)


def test_machine_rate_is_derived_when_not_given():
    op = {k: v for k, v in OP.items() if k != "machine_rate"}
    op["machine"] = {"capex": 100000, "life_years": 10, "interest_rate": 0.06, "floor_m2": 20, "cost_per_m2": 100,
                     "kw": 50, "load_factor": 0.5, "energy_price": 0.2, "available_hours": 4000, "maintenance_pct": 0.03}
    rate = (100000 / 10 + 100000 * 0.06 / 2 + 20 * 100 + 50 * 0.5 * 0.2 * 4000 + 100000 * 0.03) / 4000
    out = run(routing=[op])
    assert out["operations"][0]["machine_rate"] == pytest.approx(rate)
    assert any("derived" in a for a in out["assumptions"])


@pytest.mark.parametrize("bad,match", [
    ({"cavities": 0}, "cavities"), ({"oee": 1.2}, "oee"), ({"oee": 0}, "oee"), ({"cycle_time_s": -1}, "cycle"),
])
def test_impossible_operation_inputs_are_refused(bad, match):
    with pytest.raises(sc.ShouldCostError, match=match):
        run(routing=[{**OP, **bad}])


def test_scrap_of_one_or_more_is_refused_and_zero_is_fine():
    with pytest.raises(sc.ShouldCostError, match="scrap"):
        run(bom=[{**BOM[0], "scrap_rate": 1.0}])
    out = run(bom=[{**BOM[0], "scrap_rate": 0.0}])
    assert out["per_unit"]["material"] == pytest.approx(1.0)


def test_missing_labour_rate_is_a_named_error_not_a_zero():
    with pytest.raises(sc.ShouldCostError, match="labour_rate"):
        run(routing=[{k: v for k, v in OP.items() if k != "labour_rate"}])


def test_bought_in_part_is_priced_not_modelled():
    out = run(bom=BOM + [{"material_id": "screw", "bought_in_price": 0.04}])
    assert out["per_unit"]["material"] == pytest.approx(1.125 + 0.04)


def test_batch_size_one_prototype_carries_the_whole_setup():
    assert run(batch_size=1)["per_unit"]["conversion"] == pytest.approx(0.5 + 160.0)


def test_cost_is_monotonic_in_cycle_time_scrap_and_rate():
    base = run()["unit_cost"]
    assert run(routing=[{**OP, "cycle_time_s": 40}])["unit_cost"] > base
    assert run(bom=[{**BOM[0], "scrap_rate": 0.3}])["unit_cost"] > base
    assert run(routing=[{**OP, "machine_rate": 90.0}])["unit_cost"] > base


def test_tornado_is_sorted_by_swing_and_names_its_drivers():
    t = run(sensitivity_pct=10)["sensitivity"]["tornado"]
    swings = [x["swing"] for x in t]
    assert swings == sorted(swings, reverse=True)
    assert {"material_price", "cycle_time", "machine_rate", "labour_rate"} <= {x["driver"] for x in t}
    mp = next(x for x in t if x["driver"] == "material_price")
    assert mp["low"] < run()["unit_cost"] < mp["high"] and mp["elasticity"] > 0


def test_monte_carlo_is_reproducible_ordered_and_needs_a_seed():
    d = {"material_price": {"min": 0.9, "mode": 1.0, "max": 1.3}, "cycle_time": {"min": 0.95, "mode": 1.0, "max": 1.15}}
    a = run(monte_carlo_n=500, seed=3, distributions=d)["sensitivity"]["monte_carlo"]
    b = run(monte_carlo_n=500, seed=3, distributions=d)["sensitivity"]["monte_carlo"]
    assert a == b and a["p10"] <= a["p50"] <= a["p90"]
    with pytest.raises(sc.ShouldCostError, match="seed"):
        run(monte_carlo_n=500, distributions=d)


def test_a_degenerate_distribution_is_allowed():
    d = {"material_price": {"min": 1.0, "mode": 1.0, "max": 1.0}}
    a = run(monte_carlo_n=50, seed=1, distributions=d)["sensitivity"]["monte_carlo"]
    assert a["p10"] == pytest.approx(a["p90"])


def test_quote_comparison_flags_elements_over_ten_percent_above_should_cost():
    q = {"material": 1.125 * 1.2, "conversion": 0.66, "total": 3.0}
    out = run(quote_breakdown=q)["quote_gap"]
    assert out["elements"]["material"]["flag"] is True and out["elements"]["conversion"]["flag"] is False
    assert out["total"]["gap"] == pytest.approx(3.0 - run()["unit_cost"])


def test_every_rate_used_is_listed_with_its_source():
    out = run()
    assert {"machine_rate", "labour_rate"} <= {a["name"] for a in out["rates_used"]}
    assert all(a["source"] for a in out["rates_used"])
