"""'Draft this change' (brief §6.4): an unsaved form, built by code. No database."""
import copy

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


def test_a_form_that_has_moved_on_gives_no_draft():
    f = confirmed()
    f["hidden"]["condition"]["all"][1]["value"] = 700
    assert LD.build(f, P) is None
