"""What the repair model is told. Measured live 2026-10-09 on 8 failing drafts: internal codes ('ungrounded_figure: 43.10')
fixed 1; plain instructions fixed 6. Asked to 'add a deadline' with none given, the model invented one ('by the end of the
week'), which no check can catch, so a missing deadline is repaired only with the deadline we hold, or not at all."""

from src.services.draft_assurance import repair as RP


def _v(kind, detail):
    return {"kind": kind, "detail": detail, "severity": "fail"}


def test_each_problem_becomes_a_plain_instruction_naming_the_text_never_a_code():
    lines = RP.instructions([_v("ungrounded_figure", "43.10"), _v("unresolved_placeholder", "[name]"),
                             _v("ungrounded_date", "15 November 2026"), _v("internal_figure_leaked", "walkaway_price"),
                             _v("forbidden_content", "liability_admission: We accept full liability")])
    text = "\n".join(lines)
    assert "43.10" in text and "[name]" in text and "15 November 2026" in text and "We accept full liability" in text
    assert "ungrounded_figure" not in text and "unresolved_placeholder" not in text and "walkaway_price" not in text
    assert "Never fill it in" in text and "Do not replace it with another number" in text


def test_a_missing_deadline_is_repaired_with_the_deadline_we_hold():
    (line,) = RP.instructions([_v("missing_required_element", "deadline")], deadline="30 October 2026")
    assert "30 October 2026" in line


def test_a_missing_deadline_we_do_not_hold_is_left_for_a_person():
    assert RP.instructions([_v("missing_required_element", "deadline")], deadline=None) == []


def test_a_missing_ask_may_be_added_without_new_figures():
    (line,) = RP.instructions([_v("missing_required_element", "explicit_ask")])
    assert "question" in line and "no new" in line


def test_an_unknown_problem_is_passed_on_as_text():
    (line,) = RP.instructions([_v("something_new", "detail here")])
    assert "detail here" in line
