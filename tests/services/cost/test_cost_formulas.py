"""The two formulas through the registry: the entry points agents and reports actually use."""
import pytest

from src.services.calculation_tools import _formulas, list_calculations, run_calculation

evaluate = _formulas().evaluate
REGISTRY = _formulas().REGISTRY

LANDED_IN = {
    "lines": [{"unit_price": 20.0, "quantity": 1000, "currency": "GBP"}],
    "buyer_currency": "GBP", "cross_border": True, "incoterm": "FOB Shanghai",
    "evaluation_date": "2026-10-20",
    "components": [{"key": "freight", "amount": 1200.0}, {"key": "brokerage", "amount": 150.0}],
    "duty_rate": 0.04, "customs_basis": "CIF", "default_insurance_rate": 0.003,
    "vat_recoverable": True,
}


def test_both_formulas_are_registered_with_their_audit_names():
    assert "landed_cost.total_cost" in REGISTRY
    assert "tco.total_cost" in REGISTRY


def test_a_model_can_discover_them():
    names = {c["name"] for c in list_calculations()["calculations"]} \
        if "calculations" in list_calculations() else set()
    assert {"landed_cost.total_cost", "tco.total_cost"} <= names


def test_landed_cost_runs_through_run_calculation_with_json_style_inputs():
    out = run_calculation("landed_cost.total_cost", LANDED_IN)
    assert out["assessed"] is True, out
    value = out["value"] if "value" in out else out["result"]
    assert value["landed_total"] == pytest.approx(22270.76, abs=0.005)
    assert value["landed_status"] == "incomplete"
    assert value["evaluation_date"] == "2026-10-20"


def test_tco_chains_from_the_landed_result():
    landed = evaluate("landed_cost.total_cost", LANDED_IN, audit=False).value
    res = evaluate("tco.total_cost", {
        "landed": landed, "cost_of_capital": 0.08, "horizon_years": 3,
        "payment_terms": {"net_days": 60, "discount_pct": 0.02, "discount_days": 10},
        "ap_can_take_discount": True,
        "lifetime_items": [{"key": "maintenance_support", "amount": 500.0, "year": y}
                           for y in (1, 2, 3)] + [{"key": "disposal", "amount": 300.0, "year": 3}],
    }, audit=False)
    assert res.value["tco"] == pytest.approx(23356.17, abs=0.005)


def test_unpriceable_goods_is_a_refusal_not_a_zero():
    bad = {**LANDED_IN, "lines": [{"unit_price": 20.0, "quantity": 10, "currency": "NZD"}]}
    res = evaluate("landed_cost.total_cost", bad, audit=False)
    assert res.unassessed


def test_a_missing_required_input_is_refused_not_defaulted():
    res = evaluate("landed_cost.total_cost", {k: v for k, v in LANDED_IN.items()
                                              if k != "cross_border"}, audit=False)
    assert res.unassessed


def test_an_out_of_range_rate_is_refused():
    res = evaluate("landed_cost.total_cost", {**LANDED_IN, "duty_rate": 4.0}, audit=False)
    assert res.unassessed


def test_should_cost_is_registered_and_runs_through_run_calculation():
    assert "cost.should_cost" in REGISTRY
    out = run_calculation("cost.should_cost", {
        "bom": [{"material_id": "PP", "net_weight": 0.5, "price_per_kg": 2.0, "scrap_rate": 0.2,
                 "scrap_recovery_price_pct": 0.5}],
        "routing": [{"process": "injection_moulding", "cycle_time_s": 36, "cavities": 2, "oee": 0.8,
                     "setup_time_h": 2, "operators_per_machine": 1, "machine_rate": 80.0, "labour_rate": 20.0}],
        "batch_size": 1000, "overhead_pct": 0.1, "sga_pct": 0.05, "margin_pct": 0.08, "logistics_per_unit": 0.05})
    assert out["assessed"] is True, out
    v = out["value"] if "value" in out else out["result"]
    assert v["unit_cost"] == pytest.approx(2.214959, abs=1e-6) and v["label"] == "modelled estimate"


def test_should_cost_declines_with_the_reason():
    out = run_calculation("cost.should_cost", {
        "bom": [{"material_id": "PP", "net_weight": 0.5, "price_per_kg": 2.0}],
        "routing": [{"process": "x", "cycle_time_s": 36, "cavities": 0, "oee": 0.8, "operators_per_machine": 1,
                     "machine_rate": 80.0, "labour_rate": 20.0}],
        "batch_size": 1000, "overhead_pct": 0.1, "sga_pct": 0.05, "margin_pct": 0.08})
    assert out["assessed"] is False and "cavities" in out["why"], out
