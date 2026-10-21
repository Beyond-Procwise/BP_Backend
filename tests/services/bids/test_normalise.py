import math

import pytest

from src.services.bids import normalise as nb

MASTER = [{"line_id": "L1", "description": "Steel bracket 40mm", "uom": "EA", "qty": 1000},
          {"line_id": "L2", "description": "Steel bracket 60mm", "uom": "EA", "qty": 500}]
A = {"supplier_id": "A", "currency": "EUR", "incoterm": "FCA", "payment_days": 30,
     "lines": [{"line_id": "L1", "unit_price": 10.0, "uom": "EA"}, {"line_id": "L2", "unit_price": 12.0, "uom": "EA"}],
     "adjustments": [{"type": "freight", "amount": 300.0, "currency": "GBP", "effect": "add",
                      "note": "FCA to DAP, quoted by our forwarder"}]}
B = {"supplier_id": "B", "currency": "GBP", "incoterm": "DAP", "payment_days": 90,
     "lines": [{"line_id": "L1", "unit_price": 9.5, "uom": "EA"}, {"line_id": "L2", "unit_price": 11.0, "uom": "EA"}]}
KW = dict(line_master=MASTER, bids=[A, B], base_currency="GBP", fx_rates={"EUR": 0.85, "GBP": 1.0},
          fx_type="ECB reference", fx_date="2026-10-01", target_incoterm="DAP", target_payment_days=60,
          cost_of_capital=0.10)


def run(**over):
    return nb.normalise_bids(**{**KW, **over})


def cell(out, line, supplier):
    return out["matrix"][line][supplier]


def pay(days_bid, target=60, r=0.10):
    return (1 + r) ** ((target - days_bid) / 365)


def test_hand_worked_cell_traces_from_raw_price_to_normalised_price():
    c = cell(run(), "L1", "A")
    assert c["raw_unit_price"] == 10.0 and c["raw_currency"] == "EUR"
    assert c["fx_rate"] == 0.85 and c["fx_date"] == "2026-10-01"
    assert c["normalised_unit_price"] == pytest.approx(10.0 * 0.85 * pay(30), abs=1e-6)
    assert c["extended_value"] == pytest.approx(round(10.0 * 0.85 * pay(30) * 1000, 2), abs=0.005)
    assert [a["type"] for a in c["adjustments"]] == ["payment_terms"]


def test_longer_payment_terms_lower_the_normalised_price_and_shorter_raise_it():
    out = run()
    assert cell(out, "L1", "B")["normalised_unit_price"] == pytest.approx(9.5 * pay(90), abs=1e-6)
    assert cell(out, "L1", "B")["normalised_unit_price"] < 9.5 < 9.5 * pay(30)


def test_bid_level_adjustment_is_shown_separately_and_added_to_the_total():
    out = run()
    a = out["suppliers"]["A"]
    lines = sum(cell(out, l, "A")["extended_value"] for l in ("L1", "L2"))
    assert a["bid_adjustments"] == [{"type": "freight", "amount": 300.0, "currency": "GBP", "effect": "add",
                                     "note": "FCA to DAP, quoted by our forwarder", "amount_base": 300.0}]
    assert a["total"] == pytest.approx(lines + 300.0, abs=0.01)


def test_differing_incoterm_with_no_priced_adjustment_is_flagged_not_assumed():
    a = {k: v for k, v in A.items() if k != "adjustments"}
    out = run(bids=[a, B])
    assert any(e["category"] == "incoterm" and e["severity"] == "unpriced" for e in out["exceptions"]["A"])
    assert out["suppliers"]["A"]["terms_not_normalised"] == ["incoterm"]


def test_payment_terms_are_not_adjusted_without_a_cost_of_capital_and_say_so():
    out = run(cost_of_capital=None)
    c = cell(out, "L1", "B")
    assert c["normalised_unit_price"] == pytest.approx(9.5)
    assert "payment_terms" in out["suppliers"]["B"]["terms_not_normalised"]


def test_missing_fx_rate_makes_the_line_non_comparable_never_one_to_one():
    out = run(fx_rates={"GBP": 1.0})
    c = cell(out, "L1", "A")
    assert c["comparable"] is False and c["normalised_unit_price"] is None and "no_fx_rate" in c["flags"]


def test_price_per_thousand_is_converted_to_per_each():
    b = {**B, "lines": [{"line_id": "L1", "unit_price": 9500.0, "price_unit": 1000, "uom": "EA"}, B["lines"][1]]}
    assert cell(run(bids=[A, b]), "L1", "B")["normalised_unit_price"] == pytest.approx(9.5 * pay(90), abs=1e-6)


def test_pack_size_conversion_is_explicit():
    b = {**B, "lines": [{"line_id": "L1", "unit_price": 114.0, "uom": "BOX", "unit": "box of 12"}, B["lines"][1]]}
    out = run(bids=[A, b], uom_conversions=[{"line_id": "L1", "from_uom": "BOX", "to_uom": "EA", "factor": 12}])
    c = cell(out, "L1", "B")
    assert c["normalised_unit_price"] == pytest.approx(114.0 / 12 * pay(90), abs=1e-6) and c["uom_factor"] == 12


def test_unconvertible_uom_is_non_comparable():
    b = {**B, "lines": [{"line_id": "L1", "unit_price": 9.5, "uom": "KG"}, B["lines"][1]]}
    c = cell(run(bids=[A, b]), "L1", "B")
    assert c["comparable"] is False and "uom_unconvertible" in c["flags"]


def test_price_tiers_are_evaluated_at_the_evaluated_volume_and_a_break_uses_the_tier():
    tiers = [{"min_qty": 1, "unit_price": 10.0}, {"min_qty": 1000, "unit_price": 9.0}]
    b = {**B, "lines": [{"line_id": "L1", "unit_price": 10.0, "price_tiers": tiers, "uom": "EA"}, B["lines"][1]]}
    c = cell(run(bids=[A, b]), "L1", "B")
    assert c["tier_used"] == {"min_qty": 1000, "unit_price": 9.0}
    assert c["normalised_unit_price"] == pytest.approx(9.0 * pay(90), abs=1e-6)


def test_a_line_matched_by_description_carries_its_confidence_and_a_poor_match_is_refused():
    b = {**B, "lines": [{"description": "Steel bracket 40 mm", "unit_price": 9.5, "uom": "EA"}, B["lines"][1]]}
    c = cell(run(bids=[A, b]), "L1", "B")
    assert c["match"]["method"] == "description" and c["match"]["confidence"] >= 0.85
    worse = {**B, "lines": [{"description": "Garden hose reel", "unit_price": 9.5, "uom": "EA"}, B["lines"][1]]}
    out = run(bids=[A, worse])
    assert out["matrix"]["L1"].get("B") is None
    assert out["suppliers"]["B"]["unmatched_lines"][0]["description"] == "Garden hose reel"


def test_missing_line_is_flagged_by_default_and_completeness_is_reported():
    b = {**B, "lines": [B["lines"][0]]}
    out = run(bids=[A, b])
    assert out["suppliers"]["B"]["completeness_pct"] == 50.0 and out["suppliers"]["B"]["missing_lines"] == ["L2"]
    assert out["suppliers"]["B"]["total"] is not None and "flag_only" == out["missing_line_policy"]


@pytest.mark.parametrize("policy", ["fill_with_highest_bid", "fill_with_average_bid"])
def test_filled_lines_are_labelled_assumptions(policy):
    b = {**B, "lines": [B["lines"][0]]}
    out = run(bids=[A, b], missing_line_policy=policy)
    c = cell(out, "L2", "B")
    assert c["filled"] is True and any("filled" in a for a in out["assumptions"])
    other = cell(out, "L2", "A")["normalised_unit_price"]
    assert c["normalised_unit_price"] == pytest.approx(other)  # only one other bid, so highest == average == it


def test_exclude_supplier_from_total_leaves_no_total_for_an_incomplete_bid():
    b = {**B, "lines": [B["lines"][0]]}
    assert run(bids=[A, b], missing_line_policy="exclude_supplier_from_total")["suppliers"]["B"]["total"] is None


def test_fill_with_last_paid_needs_the_prices():
    b = {**B, "lines": [B["lines"][0]]}
    with pytest.raises(nb.NormaliseError, match="last_paid"):
        run(bids=[A, b], missing_line_policy="fill_with_last_paid_price")
    out = run(bids=[A, b], missing_line_policy="fill_with_last_paid_price", last_paid_prices={"L2": 11.5})
    assert cell(out, "L2", "B")["raw_unit_price"] == 11.5


def test_alternates_never_enter_the_base_total_and_show_their_delta():
    b = {**B, "lines": B["lines"] + [{"line_id": "L1", "unit_price": 8.0, "uom": "EA", "alternate_flag": True}]}
    out = run(bids=[A, b])
    base = run(bids=[A, B])
    assert out["suppliers"]["B"]["total"] == base["suppliers"]["B"]["total"]
    alt = out["alternates"]["B"][0]
    assert alt["line_id"] == "L1" and alt["delta_vs_compliant"] == pytest.approx(8.0 * pay(90) - 9.5 * pay(90), abs=1e-6)


def test_exceptions_are_extracted_with_severity_and_a_priced_one_is_loaded():
    b = {**B, "exceptions": [{"text": "valid 7 days", "category": "validity", "severity": "unpriced"},
                             {"text": "excludes packaging", "category": "scope", "severity": "priced",
                              "cost_loading": 120.0, "currency": "GBP"}]}
    out = run(bids=[A, b])
    assert {e["severity"] for e in out["exceptions"]["B"]} == {"unpriced", "priced"}
    assert any(a["type"] == "exception_loading" and a["amount_base"] == 120.0 for a in out["suppliers"]["B"]["bid_adjustments"])


def test_a_line_named_in_an_unpriced_exception_is_non_comparable():
    b = {**B, "exceptions": [{"text": "subject to metal surcharge", "category": "price", "severity": "unpriced",
                              "applies_to": ["L1"]}]}
    c = cell(run(bids=[A, b]), "L1", "B")
    assert c["comparable"] is False and "unpriced_exception" in c["flags"]


def test_bid_validity_before_the_award_date_is_warned():
    b = {**B, "validity_date": "2026-11-01"}
    out = run(bids=[A, b], award_date="2026-12-01")
    assert any("validity" in w and "B" in w for w in out["warnings"])


def test_a_rebate_line_is_allowed_and_flagged():
    b = {**B, "lines": [B["lines"][0], {"line_id": "L2", "unit_price": -1.0, "uom": "EA"}]}
    assert "rebate" in cell(run(bids=[A, b]), "L2", "B")["flags"]


def test_lowest_per_line_and_cherry_pick_against_best_single_supplier():
    out = run()
    assert set(out["lowest_per_line"]) == {"L1", "L2"}
    assert out["cherry_pick_total"] <= out["best_single_supplier"]["total"] + 1e-9


def test_sealed_bids_cannot_be_compared_before_opening():
    with pytest.raises(nb.NormaliseError, match="opening"):
        run(sealed=True, opened_at="2026-10-05T12:00:00Z", as_of="2026-10-05T11:00:00Z")
    run(sealed=True, opened_at="2026-10-05T12:00:00Z", as_of="2026-10-05T12:00:01Z")


def test_inputs_are_never_modified_and_one_bid_per_supplier():
    import copy
    before = copy.deepcopy(KW)
    run()
    assert KW == before
    with pytest.raises(nb.NormaliseError, match="twice"):
        run(bids=[A, A])


def test_a_non_decimal_currency_like_jpy_converts_without_rounding_the_rate():
    jpy = {**B, "supplier_id": "J", "currency": "JPY", "payment_days": 60, "incoterm": "DAP",
           "lines": [{"line_id": "L1", "unit_price": 1500, "uom": "EA"}, {"line_id": "L2", "unit_price": 1700, "uom": "EA"}]}
    out = run(bids=[A, jpy], fx_rates={"EUR": 0.85, "JPY": 0.0053})
    assert cell(out, "L1", "J")["normalised_unit_price"] == pytest.approx(1500 * 0.0053, abs=1e-9)


def test_every_adjustment_has_a_source_and_assumptions_are_listed():
    out = run()
    for l in ("L1", "L2"):
        for s in ("A", "B"):
            for a in cell(out, l, s)["adjustments"]:
                assert a["type"] and a["basis"]
    assert isinstance(out["assumptions"], list)


@pytest.mark.parametrize("bad", [0, -0.85, "n/a"])
def test_a_zero_negative_or_junk_rate_is_a_missing_rate_not_one_to_one(bad):
    c = cell(run(fx_rates={"EUR": bad, "GBP": 1.0}), "L1", "A")
    assert c["comparable"] is False and "no_fx_rate" in c["flags"]
