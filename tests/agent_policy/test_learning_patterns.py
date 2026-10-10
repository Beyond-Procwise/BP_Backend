"""Which parts of a condition can be loosened and what the evidence supports. No database.

How a threshold is judged (fix round 1, ruling F1 restated). The decisions on the loosening side of the
limit are ordered by value, nearest the limit first, and cut into bands of `min_decisions` (30). Equal
values are never split; a tail shorter than a band is not evidence. Every band must pass every guardrail
on its own, and the run of supported bands stops at the first one that fails. With the default Wilson
floor of 0.85 a band of exactly 30 needs 30 of 30 (29 of 30 is 0.8333).
"""
import math
from datetime import datetime, timedelta, timezone

import pytest

from services.agent_policy import learning_patterns as LP
from services.agent_policy import learning_stats as LS
from services.agent_policy.enforcement import MASK
from services.agent_policy.settings import DEFAULTS

CFG = DEFAULTS["learning"]
T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
COND = {"all": [{"field": "tool.name", "op": "in", "value": ["refund.issue", "credit.issue"]},
                {"field": "args.amount", "op": "gt", "value": 500}]}
GTE = {"field": "args.amount", "op": "gte", "value": 500}
LTE = {"field": "args.margin", "op": "lte", "value": 10}


def decided(values, approved=True, field="args.amount", reversed_at=()):
    """One decision per value, 60 s each. Time does NOT rise with the value: decision i is asked on
    slot (7 * i) % 30 of 30 slots spread over 40 days, by person i % 3. Any 30 decisions in a row
    therefore cover all 30 slots (40 days) and all 3 people, so a band of 30 neighbouring values
    meets the days and approvers guardrails by itself, as the whole does."""
    out = []
    for i, v in enumerate(values):
        asked = T0 + timedelta(days=40 * ((7 * i) % 30) / 29)
        ok = approved if isinstance(approved, bool) else approved[i]
        out.append(LS.Decided(firing_id=i, values={field: v, "tool.name": "refund.issue"}, approved=ok,
                              decided_by=f"p{i % 3}", asked_at=asked, decided_at=asked + timedelta(seconds=60),
                              reversed=i in reversed_at))
    return out


def test_the_helper_gives_every_run_of_thirty_the_full_spread():
    for start in (0, 1, 17):
        s = LS.stats(decided(list(range(100)))[start:start + 30])
        assert (s["days"], s["approvers"]) == (40, 3)


# ---- which leaves ---------------------------------------------------------------------------

def test_eligible_leaves_are_thresholds_and_lists_every_match_must_satisfy():
    cond = {"all": [COND["all"][0], COND["all"][1],
                    {"field": "customer.country", "op": "in", "value": ["FR", "DE"]},
                    {"any": [{"field": "args.qty", "op": "gt", "value": 5}]},
                    {"not": {"field": "args.x", "op": "lt", "value": 1}},
                    {"field": "args.flag", "op": "eq", "value": True},
                    {"field": "args.one", "op": "in", "value": ["only"]}]}
    assert [(l.kind, l.field) for l in LP.eligible_leaves(cond)] == [
        ("threshold", "args.amount"), ("list", "customer.country")]


def test_a_single_leaf_condition_is_eligible():
    assert [l.field for l in LP.eligible_leaves(GTE)] == ["args.amount"]


def test_a_boolean_text_or_non_finite_threshold_is_not_a_threshold():
    for value in (True, "high", float("nan"), float("inf")):
        assert LP.eligible_leaves({"field": "a", "op": "gt", "value": value}) == []


def test_a_list_needs_two_distinct_values():
    assert LP.eligible_leaves({"field": "c", "op": "in", "value": ["FR", "FR"]}) == []
    assert LP.eligible_leaves({"field": "c", "op": "in", "value": [1, 1.0]}) == [], "1 and 1.0 match alike"
    assert len(LP.eligible_leaves({"field": "c", "op": "in", "value": [1, "1"]})) == 1


def test_an_all_that_is_not_a_list_is_read_as_nothing():
    assert LP.eligible_leaves({"all": None}) == []
    assert LP.propose({"all": None}, decided([600] * 30), CFG) == []


def test_two_leaves_on_the_same_field_and_operator_are_not_eligible():
    cond = {"all": [{"field": "args.amount", "op": "gt", "value": 500},
                    {"field": "args.amount", "op": "gt", "value": 1000}]}
    assert LP.eligible_leaves(cond) == []
    assert LP.propose(cond, decided(list(range(1001, 1100))), CFG) == []
    for frm in (500, 1000):
        p = {"kind": "threshold", "field": "args.amount", "op": "gt", "from": frm, "to": 2000}
        assert not LP.targets(p, cond) and not LP.applied(p, cond)
    two = {"all": [{"field": "c", "op": "in", "value": ["FR", "DE"]}, {"field": "c", "op": "in", "value": ["DE", "IT"]}]}
    q = {"kind": "list", "field": "c", "op": "in", "remove": "FR"}
    assert LP.eligible_leaves(two) == [] and not LP.targets(q, two) and not LP.applied(q, two)


# ---- thresholds: the reviewer's five data sets ----------------------------------------------

def test_set_1_a_single_far_outlier_is_not_evidence():
    # 100 approvals at 510.1 … 520.0 (one each) plus one at 50,000.
    # Bands of 30: 510.1–513.0, 513.1–516.0, 516.1–519.0. Left over: 10 values + the outlier = 11 < 30,
    # not a band. Edge 519.0. Rounded toward the limit it is 510, which releases nothing, so 519 stands.
    values = [round(510 + 0.1 * i, 1) for i in range(1, 101)] + [50000]
    [p] = LP.propose(COND, decided(values), CFG)
    assert (p["to"], p["bands"], p["stats"]["decisions"]) == (519, 3, 90)


def test_set_2_a_handful_far_away_with_rejections_is_not_reached():
    # 510 … 609 (one each) then 900 … 1000 (five, two rejected).
    # Bands 510–539, 540–569, 570–599 pass; left over 10 + 5 = 15, not a band. Edge 599.
    # Rounded: 590, releasing 510 … 590 = 81 approvals, which passes. So 590.
    values = list(range(510, 610)) + [900, 925, 950, 975, 1000]
    approved = [True] * 100 + [True, False, True, False, True]
    [p] = LP.propose(COND, decided(values, approved), CFG)
    assert (p["to"], p["bands"], p["stats"]["decisions"], p["stats"]["approved"]) == (590, 3, 81, 81)


def test_set_3_the_run_of_bands_ends_at_the_first_band_that_fails():
    # 510 … 709 (200 approvals) then 2000 … 2900 (ten, five rejected).
    # Six bands pass (510 … 689). The seventh is 690 … 709 (20) + the ten = 25 of 30: fails.
    # The total would look fine (205 of 210, Wilson 0.945), which is exactly the trap. Edge 689 → 680 (171).
    values = list(range(510, 710)) + list(range(2000, 3000, 100))
    approved = [True] * 200 + [i % 2 == 0 for i in range(10)]
    [p] = LP.propose(COND, decided(values, approved), CFG)
    assert (p["to"], p["bands"], p["stats"]["decisions"]) == (680, 6, 171)


@pytest.mark.parametrize("rejected", [(0, 2, 4, 6, 8), (1, 3, 5, 7, 9), (0, 1, 2, 3, 4), (5, 6, 7, 8, 9)])
def test_set_4_rejections_just_past_the_approvals_are_never_released(rejected):
    # 510 … 609 (100 approvals) then 610, 620, … 700 (ten, five rejected, in any layout).
    # Bands 510–539, 540–569, 570–599 pass; left over 600 … 609 + the ten = 20, not a band.
    # Edge 599 → 590 (81 approvals). Never 610 or beyond.
    values = list(range(510, 610)) + list(range(610, 710, 10))
    approved = [True] * 100 + [i not in rejected for i in range(10)]
    [p] = LP.propose(COND, decided(values, approved), CFG)
    assert (p["to"], p["bands"], p["stats"]["decisions"]) == (590, 3, 81)


def test_set_5_one_decision_beyond_the_band_does_not_move_the_limit():
    # 30 approvals at 760 are one band (edge 760); the one at 1000 is a tail of 1.
    [p] = LP.propose(COND, decided([760] * 30 + [1000]), CFG)
    assert (p["from"], p["to"], p["bands"], p["stats"]["decisions"]) == (500, 760, 1, 30)


# ---- thresholds: ties, rejections, reversals -------------------------------------------------

def test_equal_values_are_never_split_between_bands():
    # 50 decisions all at 760, 8 rejected. Cutting at 30 would make a clean band of approvals and
    # propose 760, releasing the rejections too. The band is all 50: 42 of 50 fails. No proposal.
    approved = [True] * 30 + [i >= 8 for i in range(20)]
    assert LP.propose(COND, decided([760] * 50, approved), CFG) == []


def test_a_tie_extends_its_band_rather_than_starting_the_next_one():
    # 50 decisions at 760 (one of the last 20 rejected), then 770 … 799 (30 approvals).
    # Band 1 is all 50 at 760: 49 of 50, Wilson 0.895, passes. Band 2 is 770 … 799: 30 of 30.
    # Edge 799 → 790 releases 50 + 21 = 71 with 70 approved: passes. So 790 on 2 bands.
    # (Cut at 30 regardless of ties, band 2 would be the other 20 at 760 + 770 … 779 = 29 of 30, which
    # fails, and the answer would be 760: the same value judged twice, differently.)
    values = [760] * 50 + list(range(770, 800))
    approved = [True] * 40 + [False] + [True] * 39
    assert LS.wilson_lower(49, 50) >= 0.85
    [p] = LP.propose(COND, decided(values, approved), CFG)
    assert (p["to"], p["bands"], p["stats"]["decisions"], p["stats"]["approved"]) == (790, 2, 71, 70)


def test_a_rejection_in_the_range_stops_the_suggestion_below_it():
    # 40 approvals at 510, 515 … 705 and one rejection at 650.5.
    # In order, the first 30 decisions are 510 … 650 (29 approvals) and the rejection: 29 of 30, Wilson
    # 0.8333, fails. The first band fails, so nothing is supported, though 40 of 41 overall is 0.874.
    values = sorted([510 + 5 * i for i in range(40)] + [650.5])
    approved = [v != 650.5 for v in values]
    assert LP.propose(COND, decided(values, approved), CFG) == []


def test_a_rejection_in_the_first_band_means_no_proposal_whatever_lies_above():
    # 300 approvals at 511 … 810 and one rejection at 520.5 (301 decisions, Wilson overall 0.98).
    values = sorted(list(range(511, 811)) + [520.5])
    assert LP.propose(COND, decided(values, [v != 520.5 for v in values]), CFG) == []


def test_a_rejection_in_the_second_band_stops_at_the_first_bands_edge():
    # 70 approvals at 510, 515 … 855 and one rejection at 700.5.
    # Band 1: 510 … 655, 30 of 30. Band 2: 660 … 700, the rejection, 705 … 800: 29 of 30, fails.
    # Edge 655. Rounded 650 would release 29 decisions (too few), so 655 stands.
    values = sorted([510 + 5 * i for i in range(70)] + [700.5])
    [p] = LP.propose(COND, decided(values, [v != 700.5 for v in values]), CFG)
    assert (p["to"], p["bands"], p["stats"]["decisions"]) == (655, 1, 30)


def test_good_bands_beyond_a_failed_band_do_not_count():
    # 510 … 629 (120 decisions): band 2 (540 … 569) has two rejections (28 of 30 fails); bands 3 and 4
    # are clean and the total is 118 of 120 (Wilson 0.941). Only band 1 is supported: edge 539 → 530
    # releases 21 (too few) → 539.
    values = list(range(510, 630))
    [p] = LP.propose(COND, decided(values, [v not in (545, 555) for v in values]), CFG)
    assert (p["to"], p["bands"], p["stats"]["decisions"]) == (539, 1, 30)


def test_a_reversal_inside_the_second_band_stops_at_the_first_bands_edge():
    # 90 approvals at 510, 515 … 955; the one at 710 (index 40, in band 2: 660 … 805) was later reversed.
    values = [510 + 5 * i for i in range(90)]
    [p] = LP.propose(COND, decided(values, reversed_at=(40,)), CFG)
    assert (p["to"], p["bands"], p["stats"]["reversed"]) == (655, 1, 0)


def test_too_little_evidence_proposes_nothing():
    assert LP.propose(COND, decided([600] * 29), CFG) == []


def test_a_band_must_meet_the_days_guardrail_on_its_own():
    # 60 approvals whose value rises with time over 40 days: the whole spans 40 days, but each band of
    # 30 spans under 20. Neither band is evidence on its own.
    ds = [LS.Decided(firing_id=i, values={"args.amount": 510 + i}, approved=True, decided_by=f"p{i % 3}",
                     asked_at=T0 + timedelta(days=40 * i / 59),
                     decided_at=T0 + timedelta(days=40 * i / 59, seconds=60)) for i in range(60)]
    assert LS.failed_guardrails(LS.stats(ds), CFG) == []
    assert LP.propose(COND, ds, CFG) == []


# ---- thresholds: rounding, edges, direction --------------------------------------------------

def test_raise_the_limit_to_what_people_approved_rounded_inside_the_evidence():
    # 510, 515 … 800 (59 values) then 837.42. Band 1: 510 … 655. Band 2: 660 … 800 (29) + 837.42.
    # Edge 837.42 → 830, which releases the other 59, all approved.
    values = [510 + 5 * i for i in range(59)] + [837.42]
    [p] = LP.propose(COND, decided(values), CFG)
    assert (p["kind"], p["field"], p["op"], p["from"], p["to"]) == ("threshold", "args.amount", "gt", 500, 830)
    assert p["changeKey"] == "threshold:args.amount:gt:500"
    assert (p["stats"]["decisions"], p["bands"]) == (59, 2) and "outerStats" not in p


def test_never_beyond_the_evidence_when_rounding_would_drop_below_the_guardrails():
    values = [round(801 + i * 0.1, 1) for i in range(30)]          # one band, 801.0 … 803.9
    [p] = LP.propose(COND, decided(values), CFG)
    assert p["to"] == 803.9, "800 would leave no evidence; the furthest decided value is used"


def test_masked_missing_and_non_finite_values_are_not_evidence():
    junk = [MASK, None, "NaN", "Infinity", "-inf", float("nan"), float("inf"), True, "abc", [600]]
    assert LP.propose(COND, decided(junk * 5), CFG) == []
    [p] = LP.propose(COND, decided(junk + [760] * 30), CFG)
    assert (p["to"], p["stats"]["decisions"]) == (760, 30)
    assert all(math.isfinite(float(d.values["args.amount"])) for d in LP.group_for(p, decided(junk + [760] * 30)))


def test_text_numbers_are_read_as_the_evaluator_reads_them():
    [p] = LP.propose(COND, decided([str(600 + i) for i in range(30)]), CFG)
    assert p["to"] == 629, "rounding to 620 would leave 21 decisions, so the furthest decided value stands"


def test_decisions_exactly_at_a_gt_limit_are_never_evidence():
    # Under `gt 500` a request for exactly 500 is not gated; 15 such rows must not pad the band.
    # 15 at 500 + 20 at 600 would be 35 if counted; only 20 are evidence: too few.
    assert LP.propose(COND, decided([500] * 15 + [600] * 20), CFG) == []
    lt = {"field": "args.margin", "op": "lt", "value": 10}
    assert LP.propose(lt, decided([10] * 15 + [9] * 20, field="args.margin"), CFG) == []


def test_gte_releases_everything_below_the_new_limit_and_not_the_limit_itself():
    # 500, 505 … 795 (60 approvals). Bands 500 … 645 and 650 … 795. Edge 795; `gte 795` releases
    # 500 … 790 (59). Rounded 790 releases 500 … 785 (58), still passing. So 790 with 58.
    [p] = LP.propose(GTE, decided([500 + 5 * i for i in range(60)]), CFG)
    assert (p["op"], p["from"], p["to"], p["bands"], p["stats"]["decisions"]) == ("gte", 500, 790, 2, 58)


def test_gte_never_proposes_a_limit_that_releases_only_a_few():
    # Reviewer's data: 5 at 600 and 35 at exactly 830. One band (5 + 25, extended to all the ties: 40).
    # `gte 830` would release only the 5 at 600. Step back: no band left. No proposal.
    assert LP.propose(GTE, decided([600] * 5 + [830] * 35), CFG) == []


def test_gte_steps_back_a_band_when_the_decisions_at_the_edge_carried_it():
    # Band 1: 600 … 629 (30 approved). Band 2: 630 … 659 (30 approved).
    # Band 3: 660 … 679 (20 REJECTED), 680 … 688 (9 approved), then 200 approvals all at 700; ties are
    # not split, so it is one band of 229 with 209 approved (Wilson 0.869): it passes.
    # But `gte 700` releases only what is below 700: 60 + 20 rejected + 9 = 69 of 89 approved: fails.
    # Step back to band 2: edge 659, `gte 659` releases 600 … 658 (59 approved): passes.
    # Rounded 650 releases 600 … 649 (50 approved): passes. So 650, on 2 bands.
    values = list(range(600, 660)) + list(range(660, 689)) + [700] * 200
    approved = [True] * 60 + [False] * 20 + [True] * 209
    assert LS.wilson_lower(209, 229) >= 0.85 > LS.wilson_lower(69, 89)
    [p] = LP.propose(GTE, decided(values, approved), CFG)
    assert (p["to"], p["bands"], p["stats"]["decisions"], p["stats"]["approved"]) == (650, 2, 50, 50)


def test_a_lower_bound_is_loosened_downward():
    # `lt 10`; 9.5, 9.4 … 6.6 (30 approvals): one band, edge 6.6, already two figures.
    cond = {"field": "args.margin", "op": "lt", "value": 10}
    [p] = LP.propose(cond, decided([round(9.5 - 0.1 * i, 1) for i in range(30)], field="args.margin"), CFG)
    assert (p["op"], p["from"], p["to"], p["stats"]["decisions"]) == ("lt", 10, 6.6, 30)


def test_lte_releases_everything_above_the_new_limit_and_not_the_limit_itself():
    # `lte 10`; 10.0, 9.9 … 4.1 (60 approvals). Bands 10.0 … 7.1 and 7.0 … 4.1. Edge 4.1;
    # `lte 4.1` releases 4.2 … 10.0 (59). 4.1 is already two figures.
    values = [round(10 - 0.1 * i, 1) for i in range(60)]
    [p] = LP.propose(LTE, decided(values, field="args.margin"), CFG)
    assert (p["op"], p["from"], p["to"], p["bands"], p["stats"]["decisions"]) == ("lte", 10, 4.1, 2, 59)


def test_lte_never_proposes_a_limit_that_releases_only_a_few():
    # Reviewer's data: 5 at 9 and 35 at exactly 5. `lte 5` would release only the 5 at 9.
    assert LP.propose(LTE, decided([9] * 5 + [5] * 35, field="args.margin"), CFG) == []


def test_a_negative_limit_rounds_toward_the_limit():
    # `gt -100`; -99.5, -98.5 … -40.5 (60 approvals). Bands -99.5 … -70.5 and -69.5 … -40.5. Edge -40.5.
    # Toward the limit (downward) at two figures is -41, releasing -99.5 … -41.5 (59). So -41, not -40.
    cond = {"field": "args.delta", "op": "gt", "value": -100}
    [p] = LP.propose(cond, decided([-99.5 + i for i in range(60)], field="args.delta"), CFG)
    assert (p["from"], p["to"], p["stats"]["decisions"]) == (-100, -41, 59)


# ---- lists ----------------------------------------------------------------------------------

def test_a_list_value_people_always_approve():
    cond = {"all": [{"field": "customer.country", "op": "in", "value": ["FR", "DE"]}]}
    ds = decided(["FR"] * 30, field="customer.country") + decided(["DE"] * 5, field="customer.country")
    [p] = LP.propose(cond, ds, CFG)
    assert (p["kind"], p["remove"], p["changeKey"]) == ("list", "FR", "list:customer.country:FR")


def test_list_evidence_matches_the_way_the_evaluator_matches():
    # The evaluator uses `actual in expected`: 1.0 matches 1, "1" does not.
    cond = {"field": "args.tier", "op": "in", "value": [1, 2]}
    # 40 approvals recorded as 1 and 10 rejections recorded as 1.0 are one group: 40 of 50 fails.
    ds = decided([1] * 40 + [1.0] * 10, [True] * 40 + [False] * 10, field="args.tier")
    assert LP.propose(cond, ds, CFG) == []
    p = {"kind": "list", "field": "args.tier", "op": "in", "remove": 1}
    assert len(LP.group_for(p, ds)) == 50
    # 40 approvals recorded as the text "1" never matched this list, so they are not evidence for it
    assert LP.propose(cond, decided(["1"] * 40, field="args.tier"), CFG) == []
    # a boolean is not the number 1, and nothing equals a list or None
    assert LP.propose(cond, decided([True] * 40, field="args.tier"), CFG) == []
    assert not LP._same(None, None) and not LP._same([1], [1]) and not LP._same(True, 1) and LP._same(1.0, 1)
    assert LP.targets(p, cond) and not LP.targets({**p, "remove": "1"}, cond)
    assert LP.applied(p, {"field": "args.tier", "op": "in", "value": ["1", 2]})
    assert not LP.applied(p, {"field": "args.tier", "op": "in", "value": [1.0, 2]})


# ---- targets, applied, group_for -------------------------------------------------------------

def test_targets_and_applied():
    p = {"kind": "threshold", "field": "args.amount", "op": "gt", "from": 500, "to": 830}
    looser = {"all": [COND["all"][0], {"field": "args.amount", "op": "gt", "value": 700}]}
    tighter = {"all": [COND["all"][0], {"field": "args.amount", "op": "gt", "value": 400}]}
    gone = {"all": [COND["all"][0]]}
    assert LP.targets(p, COND) and not LP.targets(p, looser)
    assert LP.applied(p, looser) and not LP.applied(p, COND) and not LP.applied(p, tighter)
    assert not LP.applied(p, gone), "dropping the limit makes the policy match MORE, which is not a loosening"
    q = {"kind": "list", "field": "c", "op": "in", "remove": "FR"}
    has = {"field": "c", "op": "in", "value": ["FR", "DE"]}
    assert LP.targets(q, has) and not LP.applied(q, has)
    assert LP.applied(q, {"field": "c", "op": "in", "value": ["DE"]})


@pytest.mark.parametrize("op,frm,to,expect", [
    ("gt", 500, 600, [550, 600]),        # 500 < v <= 600
    ("gte", 500, 600, [500, 550]),       # 500 <= v < 600
    ("lt", 600, 500, [500, 550]),        # 500 <= v < 600
    ("lte", 600, 500, [550, 600]),       # 500 < v <= 600
])
def test_group_for_is_exactly_what_the_change_releases(op, frm, to, expect):
    p = {"kind": "threshold", "field": "args.amount", "op": op, "from": frm, "to": to}
    got = [d.values["args.amount"] for d in LP.group_for(p, decided([499, 500, 550, 600, 601, "NaN"]))]
    assert got == expect
