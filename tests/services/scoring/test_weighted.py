import pytest

from src.services.scoring import weighted as w

CRITERIA = [
    {"criterion_id": "quality", "parent_id": None, "weight": "0.6", "type": "scored"},
    {"criterion_id": "price", "parent_id": None, "weight": "0.4", "type": "price", "direction": "lower_better"},
    {"criterion_id": "security", "parent_id": None, "weight": "0", "type": "gate_pass_fail"},
]
SCORES = [
    {"supplier_id": "A", "criterion_id": "quality", "evaluator_id": "e1", "score": 4},
    {"supplier_id": "B", "criterion_id": "quality", "evaluator_id": "e1", "score": 3},
    {"supplier_id": "A", "criterion_id": "security", "evaluator_id": "e1", "score": 1},
    {"supplier_id": "B", "criterion_id": "security", "evaluator_id": "e1", "score": 1},
]
GATE_A = {"supplier_id": "A", "criterion_id": "security", "evaluator_id": "e1", "score": 1}
METRICS = [{"supplier_id": "A", "criterion_id": "price", "value": 100},
           {"supplier_id": "B", "criterion_id": "price", "value": 120}]


def run(**kw):
    args = dict(criteria=CRITERIA, scale_min=0, scale_max=5, scores=SCORES, metrics=METRICS,
                normalisation="ratio_to_best")
    args.update(kw)
    return w.weighted_scoring(**args)


def total(out, s):
    return next(r["total"] for r in out["ranking"] if r["supplier_id"] == s)


def test_hand_checked_totals_and_ranks():
    out = run()
    assert total(out, "A") == pytest.approx(0.88, abs=1e-9)
    assert total(out, "B") == pytest.approx(0.6 * 0.6 + 0.4 * (100 / 120), abs=1e-9)
    assert [r["supplier_id"] for r in out["ranking"]] == ["A", "B"] and out["ranking"][0]["rank"] == 1


def test_contributions_add_up_to_the_total():
    r = run()["ranking"][0]
    assert sum(r["contributions"].values()) == pytest.approx(r["total"], abs=1e-9)


def test_a_gate_fail_removes_the_supplier_with_the_reason():
    sc = [dict(s) for s in SCORES]; sc[3]["score"] = 0
    out = run(scores=sc)
    assert [r["supplier_id"] for r in out["ranking"]] == ["A"]
    assert out["excluded"] == [{"supplier_id": "B", "reason": "failed gate security"}]


@pytest.mark.parametrize("weights", [("0.6", "0.399"), ("0.7", "0.4"), ("0.5", "0.5001")])
def test_sibling_weights_must_sum_to_one(weights):
    crit = [dict(CRITERIA[0], weight=weights[0]), dict(CRITERIA[1], weight=weights[1]), CRITERIA[2]]
    with pytest.raises(w.ScoringError, match="sum to 1"):
        run(criteria=crit)


def test_weights_are_parsed_as_decimals_so_point_one_plus_point_two_works():
    crit = [dict(CRITERIA[0], weight="0.1"), dict(CRITERIA[1], weight="0.9"), CRITERIA[2]]
    run(criteria=crit)


def test_nested_global_weight_is_the_product_down_the_tree():
    crit = [
        {"criterion_id": "tech", "parent_id": None, "weight": "0.5", "type": "group"},
        {"criterion_id": "t1", "parent_id": "tech", "weight": "0.5", "type": "scored"},
        {"criterion_id": "t2", "parent_id": "tech", "weight": "0.5", "type": "scored"},
        {"criterion_id": "price", "parent_id": None, "weight": "0.5", "type": "price", "direction": "lower_better"},
    ]
    sc = [{"supplier_id": s, "criterion_id": c, "evaluator_id": "e", "score": 5} for s in "AB" for c in ("t1", "t2")]
    out = run(criteria=crit, scores=sc)
    assert out["global_weights"] == {"t1": pytest.approx(0.25), "t2": pytest.approx(0.25), "price": pytest.approx(0.5)}


def test_consensus_scores_override_individual_aggregates_but_both_are_returned():
    sc = SCORES + [{"supplier_id": "A", "criterion_id": "quality", "evaluator_id": "e2", "score": 0}]
    mean = run(scores=sc, aggregation="mean")
    cons = run(scores=sc, consensus_scores=[{"supplier_id": "A", "criterion_id": "quality", "score": 5}])
    assert total(mean, "A") == pytest.approx(0.6 * 0.4 + 0.4)
    assert total(cons, "A") == pytest.approx(0.6 * 1.0 + 0.4)
    assert cons["aggregated_scores"]["A"]["quality"] == {"individual": 2.0, "consensus": 5.0, "used": 5.0}


@pytest.mark.parametrize("agg,expected", [("mean", 3.0), ("median", 3.0), ("trimmed_mean", 3.0)])
def test_aggregations(agg, expected):
    sc = [{"supplier_id": "A", "criterion_id": "quality", "evaluator_id": f"e{i}", "score": v}
          for i, v in enumerate([1, 2, 3, 4, 5])]
    out = run(scores=sc + [GATE_A] + [s for s in SCORES if s["supplier_id"] == "B"], aggregation=agg)
    assert out["aggregated_scores"]["A"]["quality"]["used"] == expected


def test_trimmed_mean_drops_high_and_low_only_with_five_or_more_evaluators():
    sc = [{"supplier_id": "A", "criterion_id": "quality", "evaluator_id": f"e{i}", "score": v}
          for i, v in enumerate([0, 3, 3, 3, 5])]
    out = run(scores=sc + [GATE_A] + [s for s in SCORES if s["supplier_id"] == "B"], aggregation="trimmed_mean")
    assert out["aggregated_scores"]["A"]["quality"]["used"] == 3.0
    few = run(scores=[dict(s) for s in SCORES], aggregation="trimmed_mean")
    assert few["aggregated_scores"]["A"]["quality"]["used"] == 4.0


def test_min_max_with_all_equal_gives_one_for_all_and_warns():
    m = [{"supplier_id": s, "criterion_id": "price", "value": 100} for s in "AB"]
    out = run(metrics=m, normalisation="min_max")
    assert any("equal" in x for x in out["warnings"])
    assert total(out, "A") == pytest.approx(0.6 * 0.8 + 0.4)


def test_min_max_lower_better_and_a_warning_about_rank_reversal():
    out = run(normalisation="min_max")
    assert total(out, "A") == pytest.approx(0.6 * 0.8 + 0.4 * 1.0)
    assert total(out, "B") == pytest.approx(0.6 * 0.6 + 0.4 * 0.0)
    assert any("reverse" in x for x in out["warnings"])


def test_price_interpolation_reference_scores_zero():
    out = run(normalisation="price_interpolation", price_reference=120)
    assert total(out, "B") == pytest.approx(0.6 * 0.6)
    assert total(out, "A") == pytest.approx(0.6 * 0.8 + 0.4)


def test_rank_based_normalisation():
    out = run(normalisation="rank_based")
    assert total(out, "A") == pytest.approx(0.6 * 0.8 + 0.4 * 1.0)
    assert total(out, "B") == pytest.approx(0.6 * 0.6 + 0.4 * 0.0)


def test_non_positive_price_cannot_be_ratioed():
    m = [{"supplier_id": "A", "criterion_id": "price", "value": 0}, METRICS[1]]
    with pytest.raises(w.ScoringError, match="price"):
        run(metrics=m)


def test_a_score_outside_the_scale_is_refused():
    sc = [dict(SCORES[0], score=9)] + SCORES[1:]
    with pytest.raises(w.ScoringError, match="scale"):
        run(scores=sc)


def test_a_missing_score_is_refused_not_treated_as_zero():
    with pytest.raises(w.ScoringError, match="no score"):
        run(scores=[s for s in SCORES if not (s["supplier_id"] == "B" and s["criterion_id"] == "quality")])


def test_equal_totals_to_six_places_are_a_tie_with_shared_rank():
    sc = [dict(s) for s in SCORES]; sc[1]["score"] = 4
    m = [{"supplier_id": s, "criterion_id": "price", "value": 100} for s in "AB"]
    out = run(scores=sc, metrics=m)
    assert [r["rank"] for r in out["ranking"]] == [1, 1] and all(r["tied"] for r in out["ranking"])


def test_supplier_order_does_not_change_the_result():
    a = run()
    b = run(scores=list(reversed(SCORES)), metrics=list(reversed(METRICS)))
    assert [(r["supplier_id"], r["total"]) for r in a["ranking"]] == [(r["supplier_id"], r["total"]) for r in b["ranking"]]


def test_raising_a_score_never_lowers_a_total():
    base = total(run(), "B")
    sc = [dict(s) for s in SCORES]; sc[1]["score"] = 4
    assert total(run(scores=sc), "B") >= base


def test_evaluator_who_scores_every_supplier_the_same_is_flagged():
    sc = []
    for e in ("e1", "e2", "e3"):
        for s, v in (("A", 4), ("B", 2)):
            sc.append({"supplier_id": s, "criterion_id": "quality", "evaluator_id": e, "score": v if e != "e3" else 3})
        for s in "AB":
            sc.append({"supplier_id": s, "criterion_id": "security", "evaluator_id": e, "score": 1})
    out = run(scores=sc)
    assert any(f["evaluator_id"] == "e3" and f["flag"] == "identical_scores" for f in out["evaluator_flags"])


def test_criterion_with_a_wide_spread_is_flagged_for_moderation():
    sc = [dict(s) for s in SCORES] + [{"supplier_id": "A", "criterion_id": "quality", "evaluator_id": "e2", "score": 1}]
    out = run(scores=sc)
    assert any(f["criterion_id"] == "quality" and f["flag"] == "score_range_over_40_percent" for f in out["criterion_flags"])


def test_one_at_a_time_sensitivity_reports_rank_flips_and_break_even():
    out = run(sensitivity_steps_pp=[5, 10, 20])
    steps = out["sensitivity"]["one_at_a_time"]
    assert {s["criterion_id"] for s in steps} == {"quality", "price"}
    be = {b["criterion_id"]: b for b in out["sensitivity"]["break_even"]}
    # A leads on both criteria here, so no weight swaps the top two
    assert be["quality"]["weight_at_swap"] is None
    sc = [dict(s) for s in SCORES]; sc[1]["score"] = 5; sc[0]["score"] = 1
    flip = run(scores=sc)
    be = {b["criterion_id"]: b for b in flip["sensitivity"]["break_even"]}
    assert be["quality"]["weight_at_swap"] is not None and 0 < be["quality"]["weight_at_swap"] < 1


def test_break_even_is_where_the_two_totals_meet():
    sc = [dict(s) for s in SCORES]; sc[1]["score"] = 5; sc[0]["score"] = 1
    out = run(scores=sc)
    b = next(x for x in out["sensitivity"]["break_even"] if x["criterion_id"] == "quality")
    q = b["weight_at_swap"]
    # quality normalised: A 0.2, B 1.0; price normalised: A 1.0, B 0.8333; weights q and 1-q
    assert q * 0.2 + (1 - q) * 1.0 == pytest.approx(q * 1.0 + (1 - q) * (100 / 120), abs=1e-6)


def test_monte_carlo_is_reproducible_by_seed_and_needs_one():
    a = run(monte_carlo_n=300, seed=7)["sensitivity"]["monte_carlo"]
    b = run(monte_carlo_n=300, seed=7)["sensitivity"]["monte_carlo"]
    c = run(monte_carlo_n=300, seed=8)["sensitivity"]["monte_carlo"]
    assert a == b and a != c
    assert abs(sum(a["probability_rank_1"].values()) - 1.0) < 1e-9
    with pytest.raises(w.ScoringError, match="seed"):
        run(monte_carlo_n=300)


def test_what_is_not_computed_is_said():
    assert "krippendorff_alpha" in run()["not_computed"]


def test_single_supplier_is_ranked_first_without_error():
    out = run(scores=[s for s in SCORES if s["supplier_id"] == "A"], metrics=[METRICS[0]])
    assert out["ranking"][0]["rank"] == 1


@pytest.mark.parametrize("bad,needle", [
    ({"criteria": [{"name": "quality", "weight": 1}]}, "criterion_id"),
    ({"scores": [{"supplier": "A", "criterion": "quality", "score": 4}]}, "supplier_id"),
    ({"metrics": [{"supplier_id": "A", "criterion_id": "price", "price": 100}]}, "value"),
    ({"criteria": [dict(CRITERIA[0], type="rating"), CRITERIA[1], CRITERIA[2]]}, "type"),
])
def test_rows_in_the_wrong_shape_say_which_field_is_missing(bad, needle):
    with pytest.raises(w.ScoringError) as e:
        run(**bad)
    assert needle in str(e.value) and "expected" in str(e.value)
