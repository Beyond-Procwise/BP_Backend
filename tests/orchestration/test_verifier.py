"""The verify step: nothing the model says becomes the answer until it has been checked."""
import json
from types import SimpleNamespace

import pytest

from orchestration import verifier as vf


def call(name, result, ok=True, **args):
    return SimpleNamespace(name=name, ok=ok, result=result, arguments=args, error=None)


CHECK = call("check_policy", {"verdict": "COMPLIANT_WITH_CONDITIONS",
                              "reply_text": "Compliant with conditions.\nTotal commitment 25,000 GBP against the limit of 10,000 GBP."})
AUTH = {"authoritative_reply_tool": "check_policy"}


def verify(answer, calls=(), task="", config=None, retried=False):
    return vf.verify(answer, list(calls), task, skill_config=config, retried=retried)


def test_an_authoritative_tool_reply_is_shown_verbatim_instead_of_the_models_wording():
    v = verify("It is not allowed at all, 99,999 GBP.", [CHECK], config=AUTH)
    assert v.action == "replace" and v.answer == CHECK.result["reply_text"]


def test_the_last_authoritative_call_wins():
    later = call("check_policy", {"reply_text": "Second."})
    assert verify("x", [CHECK, later], config=AUTH).answer == "Second."


def test_a_failed_or_empty_authoritative_call_is_not_used():
    assert verify("No numbers here.", [call("check_policy", None, ok=False)], config=AUTH).action == "keep"
    assert verify("No numbers here.", [call("check_policy", {"reply_text": "  "})], config=AUTH).action == "keep"


def test_without_the_skill_saying_so_a_tool_reply_does_not_replace_the_answer():
    assert verify("A full memo with no figures.", [CHECK], config={}).action == "keep"
    assert verify("A full memo with no figures.", [CHECK], config=None).action == "keep"


def test_figures_found_in_the_task_or_a_tool_result_are_traced():
    calls = [call("get_policy", {"details": {"rules": {"default_threshold_gbp": 10000}}})]
    v = verify("The limit is 10,000 GBP and you asked about 25,000 GBP.", calls, task="a PO for 25,000 GBP")
    assert v.action == "keep" and v.untraced == []


def test_a_figure_from_nowhere_makes_the_model_try_again_once_naming_it():
    v = verify("The threshold is 40,000 GBP.", [call("get_policy", {"rules": {"limit": 10000}})], task="x")
    assert v.action == "retry" and v.untraced == ["40,000"]
    assert "40,000" in v.retry_message and "tool" in v.retry_message.lower()


def test_after_the_retry_the_answer_stands_but_the_figures_are_called_unverified():
    v = verify("The threshold is 40,000 GBP.", [], task="x", retried=True)
    assert v.action == "caution" and v.untraced == ["40,000"]
    assert v.answer.startswith("The threshold is 40,000 GBP.") and "not from a tool result" in v.answer
    assert "40,000" in v.answer.split("not from a tool result")[-1]


def test_numbers_in_the_skills_own_settings_are_traced():
    v = verify("Checks older than 30 days are stale.", [], task="x", config={"thresholds": {"screening_max_age_days": 30}})
    assert v.action == "keep"


def test_numbers_in_a_skills_worked_example_are_not_traced_because_they_are_not_a_source():
    v = verify("That comes to 410,000 GBP.", [], task="x", config={"thresholds": {"a": 30}})
    assert v.action == "retry"


def test_an_answer_with_no_figures_is_kept():
    assert verify("Nothing numeric here.", [], task="x").action == "keep"


def test_a_figure_the_model_made_up_inside_a_tool_argument_is_not_a_source():
    v = verify("The total is 49,000 GBP.", [call("check_policy", None, ok=False, total_commitment=49000)], task="x")
    assert v.action == "retry"


def test_a_figure_a_tool_returned_is_a_source_even_when_nested():
    v = verify("The total is 49,000 GBP.", [call("t", {"facts_used": {"total": 49000}})], task="x")
    assert v.action == "keep"


def test_the_verdict_is_data_it_never_modifies_what_it_was_given():
    calls = [CHECK]
    before = json.dumps(CHECK.result)
    verify("x", calls, config=AUTH)
    assert json.dumps(CHECK.result) == before


def test_todays_date_is_a_source_so_the_model_can_say_when_it_checked():
    from datetime import date
    ans = "Checked on 21 Oct 2026."
    assert vf.verify(ans, [], "x", today=date(2026, 10, 21)).action == "keep"
    assert vf.verify(ans, [], "x", today=date(2025, 1, 1)).action == "retry"


def test_a_year_the_model_invents_is_still_caught():
    from datetime import date
    v = vf.verify("Policy version 2024-03-15.", [], "x", today=date(2026, 10, 21))
    assert v.action == "retry" and "2024" in v.untraced


def test_what_a_failed_call_returned_is_not_a_source():
    failed = call("get_policy", {"error": "the limit is 40,000"}, ok=False)
    assert verify("The limit is 40,000 GBP.", [failed], task="x").action == "retry"


def test_a_failed_authoritative_call_is_never_shown_even_if_it_carries_text():
    failed = call("check_policy", {"reply_text": "Approved by everyone."}, ok=False)
    v = verify("No figures.", [failed], config=AUTH)
    assert v.action == "keep" and v.answer == "No figures."
