"""Stable-ID matching and revision decisions (pure)."""
import copy

from services.agent_policy import matching
from tests.agent_policy.fixtures import FORM_EXAMPLE


def _form(ref="1.1", outcome="approve", threshold=500, excerpt=None, **extra):
    f = copy.deepcopy(FORM_EXAMPLE)
    f["outcome"] = outcome
    f["source"] = dict(f["source"], reference=ref,
                       excerpt=excerpt or f"Refunds above {threshold} dollars need approval from the Finance Manager.")
    f["hidden"]["condition"] = {"all": [{"field": "tool.name", "op": "in", "value": ["refund.issue"]},
                                        {"field": "args.amount", "op": "gt", "value": threshold}]}
    f.update(extra)
    return f


def _existing(key, form, split=None):
    return {"policyKey": key, "reference": form["source"]["reference"],
            "split": split if split is not None else matching.split_key(form), "form": form}


def test_split_key_separates_the_tiers_of_one_clause():
    assert matching.split_key(_form(threshold=500)) == "approve|gt:500"
    assert matching.split_key(_form(outcome="block", threshold=10000)) == "block|gt:10000"
    assert matching.split_key(_form(threshold=500.0)) == "approve|gt:500"   # 500 and 500.0 are one key
    both = _form()
    both["hidden"]["condition"] = {"any": [{"field": "args.amount", "op": "lt", "value": 10},
                                           {"field": "args.amount", "op": "gte", "value": 2.5}]}
    assert matching.split_key(both) == "approve|gte:2.5,lt:10"


def test_exact_match_on_reference_and_split_is_unchanged():
    old = _form()
    out = matching.match([_existing("GEN-0001", old)], [copy.deepcopy(old)])
    assert out == [{"policyKey": "GEN-0001", "decision": "unchanged"}]


def test_tiers_match_their_own_policy_even_out_of_order():
    a, b = _form(threshold=500), _form(outcome="block", threshold=10000)
    existing = [_existing("GEN-0001", a), _existing("GEN-0002", b)]
    out = matching.match(existing, [copy.deepcopy(b), copy.deepcopy(a)])
    assert out == [{"policyKey": "GEN-0002", "decision": "unchanged"},
                   {"policyKey": "GEN-0001", "decision": "unchanged"}]


def test_same_reference_new_threshold_is_changed():
    out = matching.match([_existing("GEN-0001", _form(threshold=500))], [_form(threshold=750)])
    assert out == [{"policyKey": "GEN-0001", "decision": "changed"}]


def test_same_reference_two_candidates_with_the_same_outcome_is_not_guessed():
    existing = [_existing("GEN-0001", _form(threshold=500)), _existing("GEN-0002", _form(threshold=900))]
    out = matching.match(existing, [_form(threshold=750, excerpt="Something else entirely about the refunds desk.")])
    assert out[0] == {"policyKey": None, "decision": "new"}
    assert sorted(o["policyKey"] for o in out[1:]) == ["GEN-0001", "GEN-0002"]
    assert all(o["decision"] == "proposed_retire" for o in out[1:])


def test_renumbered_clause_matches_on_its_excerpt():
    old = _form(ref="1.1")
    moved = copy.deepcopy(old)
    moved["source"]["reference"] = "3.4"
    out = matching.match([_existing("GEN-0001", old)], [moved])
    assert out == [{"policyKey": "GEN-0001", "decision": "unchanged"}]


def test_a_different_outcome_never_matches_by_excerpt():
    old = _form(ref="1.1")
    other = _form(ref="3.4", outcome="block")
    out = matching.match([_existing("GEN-0001", old)], [other])
    assert out == [{"policyKey": None, "decision": "new"},
                   {"policyKey": "GEN-0001", "decision": "proposed_retire"}]


def test_exact_match_is_not_stolen_by_an_earlier_loose_match():
    # The first proposed form would match GEN-0001 by reference+outcome; the second matches it exactly.
    old = _form(threshold=500)
    loose, exact = _form(threshold=600), copy.deepcopy(old)
    out = matching.match([_existing("GEN-0001", old)], [loose, exact])
    assert out == [{"policyKey": None, "decision": "new"}, {"policyKey": "GEN-0001", "decision": "unchanged"}]


def test_substance_ignores_what_the_agent_expected_the_confirmation_note_owner_and_dates():
    old = _form()
    new = copy.deepcopy(old)
    new["examples"][0]["agentExpected"] = "block"
    new["checked"] = {"by": "x", "at": "2026-10-08T00:00:00Z"}
    new["changeNote"] = "anything"
    new["owner"] = "Someone else"
    new["effectiveFrom"], new["reviewBy"] = "2027-01-01", "2028-01-01"
    new["examples"].reverse()                                         # order of examples is not substance
    new["hidden"]["condition"]["all"].reverse()                       # nor the order of an all-list
    new["hidden"]["condition"]["all"][0]["value"] = 500.0             # nor 500 vs 500.0
    assert matching.substantively_equal(old, new)


def test_substance_sees_every_compared_field():
    old = _form()
    changes = [
        lambda f: f.update(situation="Something else."),
        lambda f: f.update(outcome="block"),
        lambda f: f["hidden"]["condition"]["all"][1].update(value=501),
        lambda f: f.update(deciders=["CFO"]),
        lambda f: f.update(notify=["Finance team"]),
        lambda f: f["source"].update(excerpt="A different excerpt for this policy."),
        lambda f: f["hidden"]["inputs"].append({"name": "Currency", "field": "args.currency"}),
        lambda f: f["hidden"].update(checkpoint="message.send.before"),
        lambda f: f["examples"][0]["input"].update({"args.amount": 777}),
    ]
    for change in changes:
        new = copy.deepcopy(old)
        change(new)
        assert not matching.substantively_equal(old, new), change
