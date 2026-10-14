"""'Draft this change' (brief §6.4): an unsaved form, built by code. No database."""
import copy

import pytest

from services.agent_policy import conditions, learning_draft as LD
from tests.agent_policy.fixtures import FORM_EXAMPLE, SETTINGS

P = {"kind": "threshold", "field": "args.amount", "op": "gt", "from": 500, "to": 830,
     "changeKey": "threshold:args.amount:gt:500",
     "stats": {"decisions": 33, "approved": 33, "wilsonLower": 0.8957, "approvers": 3, "days": 40,
               "edited": 0, "reversed": 0, "medianSeconds": 60.0}}


def confirmed():
    f = copy.deepcopy(FORM_EXAMPLE)
    f["checked"] = {"by": "u1", "at": "2026-10-01T00:00:00+00:00"}
    return f


def test_plain_words_and_evidence():
    assert LD.plain(P, FORM_EXAMPLE) == "Raise the limit on Refund amount from 500 USD to 830 USD?"
    assert LD.evidence(P, FORM_EXAMPLE) == ("33 of 33 requests between 500 USD and 830 USD were approved, "
                                            "by 3 different people over 40 days.")


def test_the_draft_changes_the_condition_the_wording_and_clears_the_confirmation():
    before = confirmed()
    out = LD.build(before, P)
    f = out["form"]
    assert f["hidden"]["condition"]["all"][1] == {"field": "args.amount", "op": "gt", "value": 830}
    assert f["situation"] == "The agent is about to issue a refund or credit above $830."
    assert f["messageForAgent"] == "Refunds over $830 need Finance approval."
    assert f["checked"] is None
    assert out["changeNote"].startswith("Suggested by learning: Raise the limit on Refund amount from 500 USD to 830 USD?")
    assert "33 of 33" in out["changeNote"]
    assert out["checkWording"] == ["messageForPerson"], "it mentions no 500, so a person must read it"
    assert out["changedWording"] == ["situation", "messageForAgent"], "what code rewrote, to be read too"
    assert before == confirmed(), "build never edits the form it was given"


def test_the_examples_sit_on_the_new_boundary_and_are_computed_by_code():
    f = LD.build(confirmed(), P)["form"]
    amounts = [e["input"]["args.amount"] for e in f["examples"]]
    assert {829, 830, 831} <= set(amounts) and not {499, 500, 501} & set(amounts)
    assert 3 <= len(f["examples"]) <= 6 and not any(e["flipped"] for e in f["examples"])
    rows = conditions.reviewer_view(f, SETTINGS)
    by_amount = {r["input"]["args.amount"]: r["computed"] for r in rows if r["input"]["tool.name"] == "refund.issue"}
    assert (by_amount[829], by_amount[830], by_amount[831]) == ("none", "none", "approve")
    assert all(r["agent_expected"] == r["computed"] for r in rows)


def test_decimal_limits_step_by_a_hundredth():
    p = {**P, "to": 803.9}
    amounts = {e["input"]["args.amount"] for e in LD.build(confirmed(), p)["form"]["examples"]}
    assert {803.89, 803.9, 803.91} <= amounts


def test_wording_with_no_single_match_is_left_alone_and_flagged():
    f = confirmed()
    f["situation"] = "Refunds above the limit need approval."
    out = LD.build(f, P)
    assert out["form"]["situation"] == f["situation"] and "situation" in out["checkWording"]
    f["situation"] = "Above $500, or 500 in credits."
    assert "situation" in LD.build(f, P)["checkWording"]


def test_a_number_with_a_thousands_comma_is_replaced_in_the_same_style():
    f = confirmed()
    f["hidden"]["condition"]["all"][1]["value"] = 10000
    f["situation"] = "The agent is about to issue a refund above $10,000."
    out = LD.build(f, {**P, "from": 10000, "to": 12500})
    assert out["form"]["situation"] == "The agent is about to issue a refund above $12,500."


def test_a_list_value_is_removed():
    f = confirmed()
    f["hidden"]["condition"] = {"all": [{"field": "tool.name", "op": "in", "value": ["refund.issue"]},
                                        {"field": "customer.country", "op": "in", "value": ["FR", "DE"]}]}
    f["hidden"]["inputs"].append({"name": "Customer country", "field": "customer.country", "type": "string",
                                  "from": "action", "showApprover": True, "sensitive": False})
    p = {"kind": "list", "field": "customer.country", "op": "in", "remove": "FR",
         "changeKey": "list:customer.country:FR", "stats": P["stats"]}
    assert LD.plain(p, f) == "Stop asking for approval when Customer country is FR?"
    out = LD.build(f, p)
    assert out["form"]["hidden"]["condition"]["all"][1]["value"] == ["DE"]
    assert out["form"]["checked"] is None and "situation" in out["checkWording"]
    assert out["changedWording"] == [] and out["checkWording"] == list(LD.WORDING)


def test_a_form_that_has_moved_on_gives_no_draft():
    f = confirmed()
    f["hidden"]["condition"]["all"][1]["value"] = 700
    assert LD.build(f, P) is None


@pytest.mark.parametrize("text", [
    "Order A500 refunds need approval.",          # part of a reference
    "Refunds of 500% of price need approval.",    # a percentage, not the amount
    "Refunds under plan 500-B need approval.",    # part of a code
    "See section 4.500 for refunds.",             # a section number
    "Refunds on 500/2 terms need approval.",
])
def test_a_number_that_is_not_the_limit_is_not_rewritten(text):
    f = confirmed()
    f["situation"] = text
    out = LD.build(f, P)
    assert out["form"]["situation"] == text
    assert "situation" in out["checkWording"] and "situation" not in out["changedWording"]


def test_a_date_is_not_mistaken_for_a_limit_of_ten():
    f = confirmed()
    f["hidden"]["condition"]["all"][1]["value"] = 10
    f["situation"] = "From 2026-10-01 over the limit"
    out = LD.build(f, {**P, "from": 10, "to": 20})
    assert out["form"]["situation"] == "From 2026-10-01 over the limit" and "situation" in out["checkWording"]


def test_a_comma_after_the_limit_is_punctuation_and_is_kept():
    f = confirmed()
    f["situation"] = "Above $500, refunds need approval."
    out = LD.build(f, P)
    assert out["form"]["situation"] == "Above $830, refunds need approval."
    assert "situation" in out["changedWording"]


def test_a_number_followed_by_a_unit_word_is_rewritten_and_listed_for_reading():
    # "500 days" cannot be told from "500 dollars" by code: it is rewritten, and it is in changedWording
    # so the screen tells the person to read it (the confirmation is cleared either way).
    f = confirmed()
    f["messageForPerson"] = "Refunds over 500 need sign-off within 500 days."     # two: left alone
    f["situation"] = "Refunds need sign-off within 500 days."
    out = LD.build(f, P)
    assert out["form"]["situation"] == "Refunds need sign-off within 830 days."
    assert out["changedWording"] == ["situation", "messageForAgent"] and out["checkWording"] == ["messageForPerson"]


def test_changed_and_check_wording_together_cover_all_three_for_a_threshold():
    for situation in ("Above $500.", "Above the limit.", None, 500):
        f = confirmed()
        f["situation"] = situation
        out = LD.build(f, P)
        assert sorted(out["changedWording"] + out["checkWording"]) == sorted(LD.WORDING)
        assert not set(out["changedWording"]) & set(out["checkWording"])


def test_numbers_are_never_shown_as_raw_floats():
    noisy = 803.9000000000001                       # what 801 + 29 * 0.1 style arithmetic leaves behind
    f = confirmed()
    f["hidden"]["condition"]["all"][1]["value"] = 0.1 + 0.2          # 0.30000000000000004
    p = {**P, "from": 0.1 + 0.2, "to": noisy}
    f["situation"] = "Above $0.3 a person decides."
    out = LD.build(f, p)
    assert LD.plain(p, f) == "Raise the limit on Refund amount from 0.3 USD to 803.9 USD?"
    assert LD.evidence(p, f).startswith("33 of 33 requests between 0.3 USD and 803.9 USD were approved")
    assert out["form"]["situation"] == "Above $803.9 a person decides."
    assert "00000" not in out["changeNote"] and "803.9 USD" in out["changeNote"]


def test_one_day_is_singular():
    p = {**P, "stats": {**P["stats"], "days": 1}}
    assert LD.evidence(p, FORM_EXAMPLE).endswith("by 3 different people over 1 day.")


def test_a_list_value_is_removed_the_way_the_evaluator_matches_it():
    f = confirmed()
    f["hidden"]["condition"] = {"field": "args.tier", "op": "in", "value": ["1", 1.0, 2]}
    p = {"kind": "list", "field": "args.tier", "op": "in", "remove": 1, "changeKey": "list:args.tier:1",
         "stats": P["stats"]}
    assert LD.build(f, p)["form"]["hidden"]["condition"]["value"] == ["1", 2], "1.0 is 1; the text is not"


def test_no_draft_that_would_leave_a_list_empty():
    p = {"kind": "list", "field": "c", "op": "in", "remove": "FR", "changeKey": "list:c:FR", "stats": P["stats"]}
    for value in (["FR"], ["FR", "FR"]):
        f = confirmed()
        f["hidden"]["condition"] = {"field": "c", "op": "in", "value": value}
        assert LD.build(f, p) is None


def test_a_condition_or_example_that_cannot_be_read_gives_no_draft_and_never_raises():
    f = confirmed()
    f["hidden"]["condition"]["all"].append({"field": "args.x", "op": "roughly", "value": 1})
    assert LD.build(f, P) is None
    f = confirmed()
    f["examples"][0]["input"]["args.amount"] = "a lot"        # a refund.issue example, so the amount is read
    assert LD.build(f, P) is None
    assert LD.build({"hidden": {"condition": {"all": None}}}, P) is None
