"""The history reader's masking and who may see what (no database)."""
from types import SimpleNamespace

import pytest

from services.agent_policy import conflict_history as CH
from services.agent_policy.enforcement import MASK

MAPPING = {"Owner A": {"groups": [], "emails": ["oa@x.test"]}, "Approver B": {"groups": ["G-B"], "emails": []}}


def _p(email="", groups=()):
    return SimpleNamespace(subject=f"sub-{email}", email=email, claims={"cognito:groups": list(groups)})


def _entry(reason="Paying the 900 refund is fine.", example=None):
    return {"caseId": "pc_9", "kind": "live", "isOpen": False, "raisedAt": "2026-10-09T10:00:00+00:00",
            "policies": [{"id": "TST-0001", "version": 1}, {"id": "TST-0002", "version": 1}],
            "example": example if example is not None else {"tool.name": "t", "args.amount": 900},
            "decision": {"option": "approve", "scope": "this_action", "decidedBy": {"kind": "person", "name": "sub-b"},
                         "decidedAt": "2026-10-09T11:00:00+00:00", "reason": reason},
            "citedCases": [], "proposal": None,
            "_owners": ["Owner A"], "_deciders": ["Approver B"], "_args": {"amount": 900}}


@pytest.mark.parametrize("principal,admin,expect", [
    (_p("oa@x.test"), False, True),                 # linked to an owner
    (_p("b@x.test", ["G-B"]), False, True),          # linked to a decider
    (_p("admin@x.test"), True, True),                # the Admin role
    (_p("stranger@x.test"), False, False),
    (None, False, False),
])
def test_who_may_see_full_values(principal, admin, expect):
    assert CH.may_see(CH.Viewer(principal, admin, MAPPING), _entry()) is expect


def test_a_stranger_reads_the_reason_masked_never_dropped():
    [e] = CH.shown([_entry()], sensitive={"args.amount"}, unmasked_for=lambda _e: False)
    assert e["decision"]["reason"] == f"Paying the {MASK} refund is fine."
    assert e["example"] == {"tool.name": "t", "args.amount": MASK}
    assert not any(k.startswith("_") for k in e)


def test_an_eligible_reader_sees_everything_and_the_input_is_never_changed():
    entry = _entry()
    [e] = CH.shown([entry], sensitive={"args.amount"}, unmasked_for=lambda _e: True)
    assert e["decision"]["reason"] == "Paying the 900 refund is fine." and e["example"]["args.amount"] == 900
    [masked] = CH.shown([entry], sensitive={"args.amount"}, unmasked_for=lambda _e: False)
    assert entry["decision"]["reason"] == "Paying the 900 refund is fine.", "shown() never edits its input"
    assert masked["decision"]["reason"] != entry["decision"]["reason"]


def test_masking_is_by_whole_token():
    entry = _entry(reason="Approved in 2900 cases; 900 is fine")
    [e] = CH.shown([entry], sensitive={"args.amount"}, unmasked_for=lambda _e: False)
    assert e["decision"]["reason"] == f"Approved in 2900 cases; {MASK} is fine"


def test_an_open_case_and_a_missing_reason_pass_through():
    entry = {**_entry(), "isOpen": True, "decision": None}
    [e] = CH.shown([entry], sensitive={"args.amount"}, unmasked_for=lambda _e: False)
    assert e["decision"] is None
    entry = _entry(reason=None)
    [e] = CH.shown([entry], sensitive={"args.amount"}, unmasked_for=lambda _e: False)
    assert e["decision"]["reason"] is None


def test_raw_needs_exactly_one_key():
    with pytest.raises(ValueError):
        CH.raw(object())
    with pytest.raises(ValueError):
        CH.raw(object(), policy_key="TST-0001", pair_key="TST-0001|TST-0002")


# ---------------------------------------------------------------------------- CSV
@pytest.mark.parametrize("v", ["=1+1", "+1", "-1", "@SUM(A1)", "\t=1", "\r=1", "  =1+1", "\tfoo", "\rbar"])
def test_csv_cell_neutralises_formulas_like_the_ui(v):
    assert CH.csv_cell(v).strip('"').startswith("'")


def test_csv_cell_quotes_commas_and_quotes_and_blanks_none():
    assert CH.csv_cell('a,"b"') == '"a,""b"""'
    assert CH.csv_cell(None) == '""'


def test_to_csv_has_the_header_and_one_row_per_case():
    open_case = {**_entry(), "caseId": "pc_10", "kind": "policy", "isOpen": True, "decision": None}
    precedent = {**_entry(), "caseId": "pc_11", "citedCases": ["pc_9", "pc_8"],
                 "decision": {"option": "reject", "scope": "this_action",
                              "decidedBy": {"kind": "precedent", "name": "system:precedent"},
                              "decidedAt": "2026-10-09T12:00:00+00:00", "reason": "=cmd|' /C calc'!A0"}}
    text = CH.to_csv([precedent, open_case, _entry()])
    lines = text.split("\r\n")
    assert text.endswith("\r\n") and lines[-1] == ""
    assert lines[0] == '"Case","Kind","Raised","Policies","Decided by","Name","Decision","Scope","Decided at","Reason","Cited cases"'
    assert lines[1] == ('"pc_11","During an action","2026-10-09T10:00:00+00:00","TST-0001 v1; TST-0002 v1",'
                        '"Precedent","system:precedent","Reject","this_action","2026-10-09T12:00:00+00:00",'
                        '"\'=cmd|\' /C calc\'!A0","pc_9 pc_8"')
    assert lines[2].startswith('"pc_10","Between policies",') and '"Waiting for a decision"' in lines[2]
    assert '"Person","sub-b","Approve"' in lines[3]


def test_decision_words_use_the_screen_labels():
    assert CH.decision_words("keep_both:FIN-0001") == "Keep both: FIN-0001 takes priority"
    assert CH.decision_words("moot") == "Closed: policy retired"
    assert CH.decision_words(None) == ""


def test_a_case_closed_by_the_system_says_so_in_the_csv():
    """Final review I2: a system-closed case (decidedBy.kind "system") is never a blank "Decided by"."""
    system = {**_entry(), "decision": {"option": "reject", "scope": "this_action",
                                       "decidedBy": {"kind": "system", "name": "system:group"},
                                       "decidedAt": "2026-10-09T12:00:00+00:00", "reason": None}}
    assert CH.DECIDED_BY_WORDS["system"] == "By the system"
    assert '"By the system","system:group","Reject"' in CH.to_csv([system]).split("\r\n")[1]


# ---------------------------------------------------------------------------- the value range (Task 12)
import json  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

AT = datetime(2026, 10, 9, 12, 0, tzinfo=timezone.utc)
RANGE = {"pct": 20.0, "fields": {"args.amount": {"value": 1000, "max": 900, "limit": 1080.0}}}


def _precedent_row(value_range=..., cited=5):
    facts = {"policies": [{"id": "TST-0001", "owner": "Owner A"}], "action": {"args": {"amount": 1000}}}
    if value_range is not ...:
        facts["valueRange"] = value_range
    evidence = [{"kind": "overlap", "example": {"args.amount": 1000}}]
    row = (12, "live", False, {"TST-0001": 1, "TST-0002": 1}, AT, "approve", "system:precedent", AT,
           json.dumps(facts), json.dumps(evidence))
    act = ("approve", "this_action", "system:precedent", AT,
           "Decided the same way (approve) 5 times before by people: pc_1, pc_2.",
           {"kind": "precedent", "name": "system:precedent"},
           [{"kind": "precedent", "caseId": f"pc_{i}"} for i in range(1, cited + 1)])
    return CH._entry(row, "pc_12", act)


def test_a_precedent_entry_carries_its_value_range_and_says_so_in_its_reason():
    e = _precedent_row(RANGE)
    assert e["valueRange"] == RANGE
    assert e["decision"]["reason"] == ("On precedent: decided the same way 5 times; "
                                       "within 20% of the largest earlier value")


def test_a_precedent_entry_with_no_range_check_says_so():
    e = _precedent_row({"pct": None, "fields": {}}, cited=3)
    assert e["decision"]["reason"] == "On precedent: decided the same way 3 times; no value range applied"


def test_a_precedent_entry_from_before_the_range_has_none():
    e = _precedent_row()
    assert e["valueRange"] is None
    assert e["decision"]["reason"] == "On precedent: decided the same way 5 times"


def test_a_person_entry_has_no_value_range_and_keeps_its_reason():
    row = (13, "live", False, {"TST-0001": 1}, AT, "approve", "sub-b", AT, "{}", "[]")
    act = ("approve", "this_action", "sub-b", AT, "Fine.", {"kind": "person"}, [])
    e = CH._entry(row, "pc_13", act)
    assert "valueRange" not in e and e["decision"]["reason"] == "Fine."


def test_the_value_range_is_masked_for_a_reader_who_may_not_see_it():
    entry = _precedent_row(RANGE)
    [hidden] = CH.shown([entry], sensitive={"args.amount"}, unmasked_for=lambda _e: False)
    assert hidden["valueRange"] == {"pct": 20.0, "fields": {"args.amount": {"value": MASK, "max": MASK, "limit": MASK}}}
    [full] = CH.shown([entry], sensitive={"args.amount"}, unmasked_for=lambda _e: True)
    assert full["valueRange"] == RANGE
    assert entry["valueRange"] == RANGE, "shown() never edits its input"
    [other] = CH.shown([entry], sensitive={"args.other"}, unmasked_for=lambda _e: False)
    assert other["valueRange"] == RANGE, "only sensitive fields are masked"


def test_the_csv_reason_of_a_precedent_row_names_the_range():
    [e] = CH.shown([_precedent_row(RANGE)], sensitive={"args.amount"}, unmasked_for=lambda _e: False)
    row = CH.to_csv([e]).split("\r\n")[1]
    assert ('"Precedent","system:precedent","Approve","this_action","2026-10-09T12:00:00+00:00",'
            '"On precedent: decided the same way 5 times; within 20% of the largest earlier value",'
            '"pc_1 pc_2 pc_3 pc_4 pc_5"') in row
