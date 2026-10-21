import copy

import pytest

from src.services.policy_check import evaluate as ev

CONFIG = {
    "thresholds": {"split_window_days": 30, "split_near_threshold_pct": 10, "split_min_orders": 3},
    "governed_policies": {"approval_limit": "approval_threshold_policy"},
    "ungoverned_rules": ["competitive_bid_thresholds", "waiver_rules"],
    "rule_policy_keywords": {"competitive_bid_thresholds": ["bid", "tender"], "waiver_rules": ["waiver"]},
}
APPROVAL = {"slug": "approval_threshold_policy", "policy_name": "ApprovalThresholdPolicy",
            "details": {"rules": {"currency": "GBP", "on_above": "escalate", "on_at_or_below": "approve",
                                  "default_threshold_gbp": 10000},
                        "required_role": "Approver", "applies_to": ["spend.approve"]}}


def gov(slugs=("approval_threshold_policy",), approval=APPROVAL, available=True):
    return {"available": available, "slugs": list(slugs), "approval_policy": approval}


def run(request, **kw):
    g = kw.pop("gov", gov())
    return ev.evaluate(request, g, CONFIG, **kw)


def outcome(v, rule):
    return next(r["outcome"] for r in v["rules_evaluated"] if r["rule"] == rule)


def test_at_or_below_the_limit_needs_no_approval():
    v = run({"total_commitment": 9999.99, "currency": "GBP"})
    assert v["verdict"] == "COMPLIANT" and v["required_approvals"] == []
    assert v["policy"]["threshold"] == 10000 and v["policy"]["slug"] == "approval_threshold_policy"


def test_exactly_at_the_limit_is_not_above_it():
    assert run({"total_commitment": 10000, "currency": "GBP"})["verdict"] == "COMPLIANT"


def test_above_the_limit_needs_the_policys_role():
    v = run({"total_commitment": 25000, "currency": "GBP"})
    assert v["verdict"] == "COMPLIANT_WITH_CONDITIONS"
    assert v["required_approvals"] == [{"role": "Approver", "reason_rule": "approval threshold policy"}]


def test_the_numbers_come_from_the_policy_not_the_request():
    other = copy.deepcopy(APPROVAL); other["details"]["rules"]["default_threshold_gbp"] = 50000
    v = run({"total_commitment": 25000, "currency": "GBP"}, gov=gov(approval=other))
    assert v["verdict"] == "COMPLIANT" and v["policy"]["threshold"] == 50000


def test_the_policy_fingerprint_changes_when_the_policy_does():
    other = copy.deepcopy(APPROVAL); other["details"]["rules"]["default_threshold_gbp"] = 50000
    a = run({"total_commitment": 1, "currency": "GBP"})["policy"]["fingerprint"]
    b = run({"total_commitment": 1, "currency": "GBP"}, gov=gov(approval=other))["policy"]["fingerprint"]
    assert a != b and len(a) == 12


@pytest.mark.parametrize("request_", [{}, {"currency": "GBP"}, {"total_commitment": None, "currency": "GBP"},
                                      {"total_commitment": "lots", "currency": "GBP"}, {"total_commitment": -5, "currency": "GBP"}])
def test_missing_or_bad_amount_is_insufficient_information(request_):
    v = run(request_)
    assert v["verdict"] == "INSUFFICIENT_INFO" and "total_commitment" in v["needed"]


def test_an_annual_figure_for_a_multi_year_term_is_not_a_total():
    v = run({"annual_value": 4000, "term_months": 36, "currency": "GBP"})
    assert v["verdict"] == "INSUFFICIENT_INFO" and "total_commitment" in v["needed"]


def test_other_currency_without_a_supplied_rate_cannot_be_compared():
    v = run({"total_commitment": 30000, "currency": "USD"})
    assert v["verdict"] == "INSUFFICIENT_INFO" and "fx_rate_to_policy_currency" in v["needed"]


def test_a_supplied_rate_is_used_and_labelled_as_supplied():
    v = run({"total_commitment": 14000, "currency": "USD", "fx_rate_to_policy_currency": 0.8})
    assert v["verdict"] == "COMPLIANT_WITH_CONDITIONS" and v["facts_used"]["amount_in_policy_currency"] == 11200
    assert any("rate" in a and "supplied" in a for a in v["assumptions"])


def test_an_unreadable_policy_store_fails_closed():
    v = run({"total_commitment": 100, "currency": "GBP"}, gov=gov(available=False, approval=None))
    assert v["verdict"] == "INSUFFICIENT_INFO" and "could not be read" in v["summary"]


def test_a_missing_approval_policy_is_insufficient_not_compliant():
    v = run({"total_commitment": 100, "currency": "GBP"}, gov=gov(approval=None))
    assert v["verdict"] == "INSUFFICIENT_INFO"


def test_rules_with_no_governed_policy_are_listed_as_not_assessed_not_passed():
    v = run({"total_commitment": 5000, "currency": "GBP"})
    assert outcome(v, "competitive bid thresholds") == "not_governed"
    assert set(v["not_assessed"]) == {"competitive bid thresholds", "waiver rules"}
    assert v["verdict"] == "COMPLIANT" and "not assessed" in v["summary"]


def test_a_rule_becomes_assessable_only_when_a_matching_policy_exists():
    v = run({"total_commitment": 5000, "currency": "GBP"}, gov=gov(slugs=["approval_threshold_policy", "tender_policy"]))
    assert outcome(v, "competitive bid thresholds") == "governed_but_not_evaluated"
    assert "competitive bid thresholds" not in v["not_assessed"] and "competitive bid thresholds" in v["needs_review"]


def test_self_approval_is_blocking():
    v = run({"total_commitment": 500, "currency": "GBP", "requester": "u1", "approver": "u1"})
    assert v["verdict"] == "NON_COMPLIANT" and v["violations"][0]["severity"] == "blocking"
    assert v["violations"][0]["remediation"]


def test_planned_orders_each_under_the_limit_but_over_it_together_raise_a_warning():
    orders = [{"amount": 6000, "source": "planned"}] * 4
    v = run({"total_commitment": 24000, "currency": "GBP", "orders": orders})
    assert v["split_signal"]["fired"] is True and v["verdict"] == "COMPLIANT_WITH_CONDITIONS"
    assert v["violations"][0]["severity"] == "warning"
    assert any("combined" in a for a in v["required_actions"])


def test_a_split_that_includes_orders_already_placed_is_blocking():
    orders = [{"amount": 6000, "source": "history"}, {"amount": 6000, "source": "planned"}]
    v = run({"total_commitment": 12000, "currency": "GBP", "orders": orders })
    assert v["verdict"] == "NON_COMPLIANT" and v["violations"][0]["severity"] == "blocking"


def test_three_orders_just_under_the_limit_fire_the_near_threshold_signal():
    orders = [{"amount": 9500, "source": "planned"}] * 3
    v = run({"total_commitment": 9500, "currency": "GBP", "orders": orders})
    assert v["split_signal"]["fired"] is True


def test_with_no_related_orders_the_split_check_says_it_could_not_be_made():
    v = run({"total_commitment": 5000, "currency": "GBP"})
    assert v["split_signal"]["performed"] is False and v["split_signal"]["fired"] is False
    assert "orders" in v["split_signal"]["reason_not_performed"]


def test_a_claim_that_history_was_checked_is_not_believed_without_orders():
    v = run({"total_commitment": 5000, "currency": "GBP" })
    assert v["split_signal"]["performed"] is False


def test_the_split_check_states_that_it_covers_only_the_orders_listed():
    orders = [{"amount": 3000, "source": "planned"}, {"amount": 2000, "source": "planned"}]
    v = run({"total_commitment": 5000, "currency": "GBP", "orders": orders})
    assert v["split_signal"]["performed"] is True and any("only the orders listed" in a for a in v["assumptions"])


def test_a_waiver_id_is_never_applied():
    v = run({"total_commitment": 25000, "currency": "GBP", "waiver_id": "W-1"})
    assert v["waivers_applied"] == [] and any("waiver" in a for a in v["assumptions"])


def test_the_reply_text_has_no_underscore_labels_and_states_the_numbers():
    v = run({"total_commitment": 25000, "currency": "GBP"})
    import re
    assert not re.search(r"\b[A-Z]+_[A-Z_]+\b", v["reply_text"])
    assert "25,000" in v["reply_text"] and "10,000" in v["reply_text"] and "Approver" in v["reply_text"]


def test_same_facts_same_policy_same_verdict():
    r = {"total_commitment": 25000, "currency": "GBP"}
    a, b = run(r, now="2026-10-21T00:00:00Z"), run(r, now="2026-10-21T00:00:00Z")
    assert a == b


def test_one_large_order_among_small_ones_does_not_hide_the_split():
    orders = [{"amount": 25000, "source": "planned"}] + [{"amount": 6000, "source": "planned"}] * 4
    v = run({"total_commitment": 49000, "currency": "GBP", "orders": orders })
    assert v["split_signal"]["fired"] is True and v["split_signal"]["combined"] == 24000
    assert v["verdict"] == "COMPLIANT_WITH_CONDITIONS"


def test_junk_placeholder_values_are_ignored_not_trusted():
    v = run({"total_commitment": 5000, "currency": "GBP", "requester": "unknown", "approver": "unknown",
             "waiver_id": "", "fx_rate_to_policy_currency": 1})
    assert v["verdict"] == "COMPLIANT" and v["violations"] == []


def test_a_single_order_is_never_a_split():
    v = run({"total_commitment": 9000, "currency": "GBP", "orders": [{"amount": 9000, "source": "planned"}]})
    assert v["split_signal"]["fired"] is False
