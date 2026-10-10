"""The thresholds endpoint answers 422, never 500, for a number it cannot read (review of B1-B3).

Two faults: a JSON integer too large for a float raised OverflowError out of the range check; and
the authorize audit recorded the raw body BEFORE it was judged, so a body holding NaN or Infinity
failed the audit write (jsonb refuses the token) and the Admin got a 500. No database here; the
real audit write is test_learning_settings_unreadable_live's.
"""
import json
import math

import pytest

from api.routers import agent_policies as R
from services.agent_policy import settings as S
from tests.agent_policy.test_learning_settings import GOOD, URL, _hdr, client, state  # noqa: F401 - fixtures

HUGE = 10 ** 400                                    # a whole number no float can hold
WHOLE_MESSAGE = "Enter a whole number from 10 to 1000."
NUMBER_MESSAGE = "Enter a number from 0.5 to 0.99."


@pytest.mark.parametrize("value", [HUGE, -HUGE, 10 ** 309, 2 ** 5000])
def test_a_whole_number_too_large_for_a_float_is_an_ordinary_problem(value):
    assert S.learning_problems({**GOOD, "min_decisions": value}) == [{"field": "min_decisions", "message": WHOLE_MESSAGE}]
    # the one threshold that is not a whole number takes integers too: the same answer there
    assert S.learning_problems({**GOOD, "wilson_lower": value}) == [{"field": "wilson_lower", "message": NUMBER_MESSAGE}]
    assert S._in_range("min_decisions", value) is False and S._in_range("wilson_lower", value) is False


def test_the_ranges_are_what_they_were():
    assert S.learning_problems(GOOD) == []
    assert S._in_range("wilson_lower", 0.5) and S._in_range("wilson_lower", 0.99) and not S._in_range("wilson_lower", 1)
    assert S._in_range("min_decisions", 10) and S._in_range("min_decisions", 1000)
    for bad in (9, 1001, 30.0, True, "30", None, float("nan"), float("inf")):
        assert not S._in_range("min_decisions", bad), repr(bad)
    for bad in (float("nan"), float("inf"), float("-inf"), True, "0.9", None):
        assert not S._in_range("wilson_lower", bad), repr(bad)


def test_a_huge_whole_number_through_the_app_is_422_beside_its_field(client, state):
    r = client.put(URL, content=json.dumps({**GOOD, "min_decisions": HUGE}),
                   headers={**_hdr(), "Content-Type": "application/json"})
    assert r.status_code == 422, r.text
    assert r.json() == {"problems": [{"field": "min_decisions", "message": WHOLE_MESSAGE}]}
    assert state["saves"] == []


# ------------------------------------------------------------------ what the authorize audit keeps of the body
@pytest.mark.parametrize("value,kept", [
    (40, 40), (0.9, 0.9), (-3, -3), (0, 0), ("forty", "forty"), ("", ""), ("x" * 100, "x" * 100),
    (True, True), (False, False),                                   # JSON booleans are whole numbers to Python
    (float("nan"), "nan"), (float("inf"), "inf"), (float("-inf"), "-inf"),
    (None, "None"), ([1, 2], "[1, 2]"), ({"a": float("nan")}, "{'a': nan}"),
    ("x" * 101, repr("x" * 101)[:100]),
    (HUGE, repr(HUGE)[:100]),
    ([float("nan")] * 200, repr([float("nan")] * 200)[:100]),
])
def test_only_json_safe_scalars_are_recorded_everything_else_as_its_capped_repr(value, kept):
    got = R._audit_scalar(value)
    assert got == kept and type(got) is type(kept)
    assert len(repr(got)) <= 102
    json.loads(json.dumps(got, allow_nan=False))                    # what a jsonb column accepts


@pytest.mark.parametrize("token,shown", [("NaN", "nan"), ("Infinity", "inf"), ("-Infinity", "-inf")])
def test_a_body_holding_a_token_jsonb_refuses_is_judged_and_audited_not_crashed(client, state, monkeypatch, token,
                                                                                shown):
    def strict(**kw):                       # as the real audit column: it refuses what is not JSON
        json.dumps(kw["details"], allow_nan=False)
        state["audits"].append(kw)
    monkeypatch.setattr(R.agent_actions, "record_action_or_fail", strict)
    body = json.dumps(GOOD).replace('"wilson_lower": 0.9', f'"wilson_lower": {token}')
    assert token in body
    r = client.put(URL, content=body, headers={**_hdr(), "Content-Type": "application/json"})
    assert r.status_code == 422, r.text
    assert r.json() == {"problems": [{"field": "wilson_lower", "message": NUMBER_MESSAGE}]}
    first, second = state["audits"]
    assert (first["phase"], first["status"]) == ("authorize", "allowed")
    assert first["details"]["requested"] == {**GOOD, "wilson_lower": shown}
    assert (second["phase"], second["status"], second["details"]["fields"]) == ("act", "refused", ["wilson_lower"])
    assert state["saves"] == []


def test_a_good_body_is_recorded_as_it_was_sent(client, state):
    r = client.put(URL, json=GOOD, headers=_hdr())
    assert r.status_code == 200, r.text
    assert state["audits"][0]["details"]["requested"] == GOOD
    assert all(not isinstance(v, str) and math.isfinite(v) for v in state["audits"][0]["details"]["requested"].values())
