"""Step 3b: the figures a model passes INTO a tool. Assumptions (rates, horizons, thresholds) must
come from the request, an earlier tool result or the skill's settings; the model may not supply them."""
from datetime import date
from types import SimpleNamespace

import pytest

from orchestration import argument_check as ac

TASK = ("Compare two bids. Quality weight 0.6, price weight 0.4. Acme price 100,000 GBP over a 3 year term. "
        "Our cost of capital is 10 percent and the rate is 0.85 GBP per EUR.")


def call(name, result, ok=True):
    return SimpleNamespace(name=name, ok=ok, result=result, arguments={}, error=None)


def found(args, task=TASK, calls=(), config=None, today=date(2026, 10, 21)):
    return ac.untraced_assumptions(args, task=task, calls=list(calls), skill_config=config, today=today)


def test_an_assumption_the_model_made_up_is_found_with_its_path():
    got = found({"name": "tco.total_cost", "inputs": {"cost_of_capital": 0.08, "horizon_years": 7}})
    assert [(p, v) for p, v in got] == [("inputs.cost_of_capital", 0.08), ("inputs.horizon_years", 7)]


@pytest.mark.parametrize("value", [0.1, 10, "10", 0.10])
def test_a_rate_the_user_gave_as_ten_percent_traces_as_a_ratio_or_a_percentage(value):
    assert found({"inputs": {"cost_of_capital": value}}) == []


def test_k_and_m_suffixes_and_number_words_in_the_request_are_understood():
    t = "The reference price is 120k, the horizon is five years and the rate is 1.2m per unit."
    assert found({"inputs": {"price_reference": 120000, "horizon_years": 5}}, task=t) == []


def test_a_rate_from_an_earlier_successful_tool_result_is_a_source():
    prior = call("get_policy", {"details": {"rules": {"cost_of_capital_apr": 0.12}}})
    assert found({"inputs": {"cost_of_capital": 0.12}}, calls=[prior]) == []


def test_what_a_failed_call_returned_is_not_a_source():
    prior = call("get_policy", {"error": "cost_of_capital 0.12"}, ok=False)
    assert found({"inputs": {"cost_of_capital": 0.12}}, calls=[prior]) != []


def test_a_figure_in_the_skills_own_settings_is_a_source():
    args = {"inputs": {"horizon_years": 7}}
    assert found(args) != []
    assert found(args, config={"defaults": {"horizon_years": 7}}) == []


def test_an_exchange_rate_table_is_checked_leaf_by_leaf():
    got = found({"inputs": {"fx_rates": {"EUR": 0.85, "USD": 1.27}}})
    assert [p for p, _ in got] == ["inputs.fx_rates.USD"]


def test_amounts_scores_and_weights_are_not_assumptions_and_are_not_checked_here():
    assert found({"inputs": {"criteria": [{"weight": "0.99"}], "scores": [{"score": 4.7}], "unit_price": 123.45,
                             "total_commitment": 49000}}) == []


def test_booleans_text_and_none_are_ignored():
    assert found({"inputs": {"cost_of_capital": None, "vat_recoverable": True, "duty_rate": "unknown"}}) == []


def test_a_numeric_string_is_checked_like_a_number():
    assert [p for p, _ in found({"inputs": {"cost_of_capital": "0.08"}})] == ["inputs.cost_of_capital"]


def test_todays_date_is_not_needed_for_a_rate_but_a_year_is_a_known_source():
    assert found({"inputs": {"horizon_years": 2026}}, today=date(2026, 10, 21)) == []


def test_the_refusal_names_the_figures_and_tells_the_model_not_to_assume():
    r = ac.refusal("run_calculation", [("inputs.cost_of_capital", 0.08), ("inputs.horizon_years", 7)])
    assert r["result"] == "blocked" and r["reasonCode"] == "figure_not_sourced"
    assert "inputs.cost_of_capital = 0.08" in r["reason"] and "inputs.horizon_years = 7" in r["reason"]
    assert "ask the user" in r["reason"].lower() and "run_calculation was not run" in r["reason"]


def test_the_checker_passes_the_call_when_nothing_is_invented_and_refuses_when_something_is():
    check = ac.make_checker(TASK, None, today=date(2026, 10, 21))
    result = SimpleNamespace(calls=[])
    assert check("run_calculation", {"inputs": {"cost_of_capital": 0.10}}, result) is None
    r = check("run_calculation", {"inputs": {"cost_of_capital": 0.08}}, result)
    assert r and r["reasonCode"] == "figure_not_sourced"


def test_it_reads_what_the_run_has_gathered_so_far_when_it_checks():
    check = ac.make_checker("x", None, today=date(2026, 10, 21))
    empty, gathered = SimpleNamespace(calls=[]), SimpleNamespace(calls=[call("get_policy", {"r": {"cost_of_capital": 0.12}})])
    assert check("run_calculation", {"inputs": {"cost_of_capital": 0.12}}, empty) is not None
    assert check("run_calculation", {"inputs": {"cost_of_capital": 0.12}}, gathered) is None


def test_every_name_on_the_assumption_list_is_a_real_input_somewhere():
    from services import calculation_tools
    terms = set()
    for spec in calculation_tools._formulas().REGISTRY.values():
        terms |= {t.name for t in spec.contract.inputs}
    terms |= {"fx_rate_to_policy_currency"}  # check_policy
    stale = sorted(ac.ASSUMPTION_ARGS - terms)
    assert stale == [], f"on the list but not an input of any calculation or tool: {stale}"


def test_checking_changes_nothing_it_was_given():
    import copy
    args = {"inputs": {"cost_of_capital": 0.08, "fx_rates": {"EUR": 0.85}}}
    before = copy.deepcopy(args)
    found(args)
    assert args == before


CATALOGUE = {"calculations": [{"name": "landed_cost.total_cost",
                               "inputs": [{"name": "duty_rate", "range": "[0, 1]", "description": "policy rate on 110% of goods"}]}]}


@pytest.mark.parametrize("tool", ["list_calculations", "list_governance", "list_skills", "use_skill", "get_prompt"])
def test_a_catalogue_or_playbook_describes_inputs_it_does_not_supply_values(tool):
    prior = call(tool, CATALOGUE)
    assert [p for p, _ in found({"inputs": {"duty_rate": 0, "default_insurance_rate": 0.01}}, calls=[prior])] == [
        "inputs.duty_rate", "inputs.default_insurance_rate"]


def test_a_governed_policy_a_calculation_and_the_corpus_do_supply_values():
    for tool in ("get_policy", "run_calculation", "get_corpus_facts", "check_policy"):
        assert found({"inputs": {"cost_of_capital": 0.12}}, calls=[call(tool, {"v": {"cost_of_capital": 0.12}})]) == []
