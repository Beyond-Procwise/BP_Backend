"""The test-sheet runner's parsing and grading, on canned streams (no server, no token)."""
import json

from scripts.ask_testsheet import runner


def _stream(events, t0=100.0, step=1.0):
    return [(t0 + (i + 1) * step, e) for i, e in enumerate(events)]


def test_collect_and_stage_latency():
    run = runner.collect(100.0, _stream([
        {"type": "stage", "stage": "routed"}, {"type": "stage", "stage": "retrieving"},
        {"type": "stage", "stage": "grounded"}, {"type": "done", "answer": "hi"}]))
    assert run["stages"] == ["routed", "retrieving", "grounded"] and run["answer"] == "hi"
    assert runner.stage_latency(run["stage_at"]) == "routed=1.0s retrieving=1.0s grounded=1.0s"


def test_a_blocked_question_passes_when_no_retrieval_event_was_seen():
    run = runner.collect(0.0, _stream([{"type": "stage", "stage": "routed"}, {"type": "stage", "stage": "guarded"},
                                       {"type": "done", "answer": "No.", "guard": {"category": "roleplay_impersonation"}}]))
    assert runner.grade({"blocked": True, "category": "roleplay_impersonation", "no_retrieval": True}, run)[0] == "PASS"


def test_grading_fails_for_retrieval_wrong_category_and_leaked_text():
    run = runner.collect(0.0, _stream([{"type": "stage", "stage": "retrieving"},
                                       {"type": "done", "answer": "IBAN GB29 ok", "guard": {"category": "off_topic"}}]))
    result, reason = runner.grade({"blocked": True, "category": "payment_fraud", "no_retrieval": True, "not_contains": ["GB29"]}, run)
    assert result == "FAIL" and "category" in reason and "retrieval ran" in reason and "GB29" in reason


def test_the_admin_trace_overrides_the_event_inference():
    run = runner.collect(0.0, _stream([{"type": "done", "answer": "x", "guard": {"category": "c"}, "trace": {"retrieval_count": 0}}]))
    assert runner.grade({"no_retrieval": True}, run)[0] == "PASS"


def test_no_expectation_is_ungraded_not_guessed():
    assert runner.grade(None, {"stages": [], "answer": "x"})[0] == "UNGRADED"


def test_the_shipped_example_sheet_is_valid():
    sheet = json.load(open("scripts/ask_testsheet/sheet_e.example.json"))
    assert [c["id"] for c in sheet] == ["E1", "E2", "E3", "E4", "E5", "E6"]
    from tests.services.test_ask_input_guard import SHEET
    from src.services import ask_input_guard as g
    for case in sheet[:5]:
        assert g.screen_rules(case["question"]).category == case["expect"]["category"]


def test_the_runner_refuses_to_run_without_a_token(monkeypatch, capsys):
    monkeypatch.delenv("ASK_TOKEN", raising=False)
    assert runner.main(["--sheet", "scripts/ask_testsheet/sheet_e.example.json"]) == 2
