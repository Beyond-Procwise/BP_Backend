from src.services.cost import compare as C

A = {"name": "A", "lines": [{"unit_price": 32000, "quantity": 1, "currency": "GBP"}], "incoterm": "DAP Leeds",
     "cross_border": False, "payment_terms": {"net_days": 30},
     "lifetime_items": [{"key": "maintenance", "amount": 1500, "year": y} for y in (1, 2, 3, 4, 5)]}
B = {"name": "B", "lines": [{"unit_price": 28000, "quantity": 1, "currency": "GBP"}], "incoterm": "DAP Leeds",
     "cross_border": False, "payment_terms": {"net_days": 30},
     "lifetime_items": [{"key": "maintenance", "amount": 2600, "year": y} for y in (1, 2, 3, 4, 5)]}


def _fake_run(landed_by_name, tco_by_name):
    calls = []

    def run(name, inputs):
        calls.append((name, inputs))
        if name == "landed_cost.total_cost":
            opt = inputs["lines"][0]["unit_price"]
            return landed_by_name[opt]
        if name == "tco.total_cost":
            return tco_by_name[inputs["landed"]["landed_total"]]
        if name == "deal.pct_change":
            return {"assessed": True, "value": round((inputs["new"] - inputs["base"]) / inputs["base"] * 100, 2)}
        raise AssertionError(name)
    run.calls = calls
    return run


def test_refuses_bad_inputs_before_running_anything():
    assert "at least 2" in C.compare([A], 5, 0.12)["note"]
    assert "lifetime item" in C.compare([A, {**B, "lifetime_items": [{"amount": 1}]}], 5, 0.12)["note"]
    assert "cost_of_capital" in C.compare([A, B], 5, 12)["note"]
    assert "horizon_years" in C.compare([A, B], 0, 0.12)["note"]
    assert "no lines" in C.compare([A, {"name": "B"}], 5, 0.12)["note"]
    assert "both called" in C.compare([A, {**B, "name": "A"}], 5, 0.12)["note"]


def test_runs_landed_then_tco_per_option_and_ranks_by_present_value():
    landed = {32000: {"assessed": True, "calculation": "landed_cost.total_cost",
                      "value": {"landed_total": 32000.0, "landed_status": "complete", "missing": [], "not_stated": [], "estimated": []}},
              28000: {"assessed": True, "calculation": "landed_cost.total_cost",
                      "value": {"landed_total": 28000.0, "landed_status": "complete", "missing": [], "not_stated": [], "estimated": []}}}
    tco = {32000.0: {"assessed": True, "calculation": "tco.total_cost", "value": {"tco": 37400.0, "tco_status": "complete", "discount_choice": "net", "excluded_lifetime": ["disposal"], "warnings": []}},
           28000.0: {"assessed": True, "calculation": "tco.total_cost", "value": {"tco": 37370.0, "tco_status": "complete", "discount_choice": "net", "excluded_lifetime": ["disposal"], "warnings": []}}}
    run = _fake_run(landed, tco)
    out = C.compare([A, B], 5, 0.12, run=run)
    assert out["found"] and out["ranking"] == ["B", "A"] and out["cheapest"] == "B" and out["lower_bounds"] == []
    assert out["gaps_pct"] == {"A": 0.08}
    names = [c[0] for c in run.calls]
    assert names == ["landed_cost.total_cost", "tco.total_cost", "landed_cost.total_cost", "tco.total_cost", "deal.pct_change"]
    tco_call = run.calls[1][1]
    assert tco_call["landed"]["landed_total"] == 32000.0 and tco_call["cost_of_capital"] == 0.12 and tco_call["horizon_years"] == 5.0
    assert tco_call["lifetime_items"] == A["lifetime_items"] and tco_call["payment_terms"] == {"net_days": 30}
    assert "nothing is estimated" in out["note"]


def test_a_component_missing_for_every_option_is_said_not_to_change_the_ranking():
    landed = {32000: {"assessed": True, "value": {"landed_total": 32000.0, "landed_status": "incomplete", "missing": ["import_duty"], "not_stated": [], "estimated": []}},
              28000: {"assessed": True, "value": {"landed_total": 28000.0, "landed_status": "incomplete", "missing": ["import_duty", "freight"], "not_stated": [], "estimated": []}}}
    tco = {32000.0: {"assessed": True, "value": {"tco": 37400.0, "tco_status": "incomplete"}},
           28000.0: {"assessed": True, "value": {"tco": 37370.0, "tco_status": "incomplete"}}}
    out = C.compare([A, B], 5, 0.12, run=_fake_run(landed, tco))
    assert out["lower_bounds"] == ["B", "A"] and out["missing_for_all"] == ["import_duty"]
    assert "lower bound" in out["note"] and "do not change the ranking" in out["note"]


def test_an_option_the_calculation_refuses_is_reported_with_its_reason_and_the_rest_still_rank():
    landed = {32000: {"assessed": False, "why": "lines need a currency", "expects": [{"name": "lines"}]},
              28000: {"assessed": True, "value": {"landed_total": 28000.0, "landed_status": "complete", "missing": [], "not_stated": [], "estimated": []}}}
    tco = {28000.0: {"assessed": True, "value": {"tco": 37370.0, "tco_status": "complete"}}}
    out = C.compare([A, B], 5, 0.12, run=_fake_run(landed, tco))
    bad = out["options"][0]
    assert bad["assessed"] is False and bad["stage"] == "landed_cost" and "currency" in bad["why"] and bad["expects"]
    assert out["unassessed"] == ["A"] and out["ranking"] == ["B"] and out["gaps_pct"] == {} and "no ranking" in out["note"]


def test_the_real_calculations_cost_two_domestic_delivered_quotes():
    """End to end through the registry: the dearer-to-buy option is cheaper to own over five years."""
    out = C.compare([A, B], 5, 0.12)
    assert out["found"] and out["unassessed"] == [], out
    by = {o["name"]: o for o in out["options"]}
    assert by["A"]["landed_total"] == 32000.0 and by["B"]["landed_total"] == 28000.0
    assert by["A"]["tco"] > 32000 and by["B"]["tco"] > 28000
    assert by["A"]["tco"] < by["B"]["tco"], (by["A"]["tco"], by["B"]["tco"])
    assert out["ranking"] == ["A", "B"] and out["gaps_pct"]["B"] > 0
    assert by["A"]["calculations"] == ["landed_cost.total_cost", "tco.total_cost"]
