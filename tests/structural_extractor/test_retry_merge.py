from src.services.structural_extractor.retry.merge import merge_attempt_into_state
from src.services.structural_extractor.retry.state import AttemptOutput, RetryState
from src.services.structural_extractor.types import ExtractedValue


def _ev(v, attempt=1):
    return ExtractedValue(value=v, provenance="extracted", source="structural", attempt=attempt)


def test_merge_adds_new_fields():
    state = RetryState(
        doc=None, doc_type="Invoice",
        target_fields={"a", "b"}, unresolved={"a", "b"},
    )
    att = AttemptOutput(
        attempt=1, source="structural", extracted={"a": _ev("X")},
        line_items=None, validation_failures=[], residual_unresolved=["b"],
        latency_ms=0,
    )
    merge_attempt_into_state(state, att)
    assert "a" in state.accepted_header
    assert state.accepted_header["a"].value == "X"
    assert "b" in state.unresolved
    assert "a" not in state.unresolved


def test_merge_does_not_overwrite_existing():
    state = RetryState(
        doc=None, doc_type="Invoice",
        target_fields={"a"}, unresolved=set(),
    )
    state.accepted_header["a"] = _ev("EARLY", attempt=1)
    att = AttemptOutput(
        attempt=2, source="nlu_ner", extracted={"a": _ev("LATER", attempt=2)},
        line_items=None, validation_failures=[], residual_unresolved=[],
        latency_ms=0,
    )
    merge_attempt_into_state(state, att)
    # Earlier value kept (identical-math case)
    assert state.accepted_header["a"].value == "EARLY"


def test_merge_appends_attempt_to_history():
    state = RetryState(
        doc=None, doc_type="Invoice", target_fields={"a"}, unresolved={"a"},
    )
    att = AttemptOutput(
        attempt=1, source="structural", extracted={"a": _ev("X")},
        line_items=None, validation_failures=[], residual_unresolved=[],
        latency_ms=10,
    )
    merge_attempt_into_state(state, att)
    assert len(state.attempts) == 1
    assert state.attempts[0].source == "structural"


def test_merge_line_items_replaces_when_more_items():
    state = RetryState(
        doc=None, doc_type="Invoice", target_fields=set(), unresolved=set(),
    )
    att1 = AttemptOutput(
        attempt=1, source="structural", extracted={},
        line_items=[{"x": _ev(1)}], validation_failures=[],
        residual_unresolved=[], latency_ms=0,
    )
    merge_attempt_into_state(state, att1)
    assert state.accepted_line_items is not None
    assert len(state.accepted_line_items) == 1

    att2 = AttemptOutput(
        attempt=2, source="nlu_ner", extracted={},
        line_items=[{"x": _ev(1)}, {"x": _ev(2)}], validation_failures=[],
        residual_unresolved=[], latency_ms=0,
    )
    merge_attempt_into_state(state, att2)
    assert len(state.accepted_line_items) == 2


def test_merge_line_items_keeps_when_fewer_items():
    state = RetryState(
        doc=None, doc_type="Invoice", target_fields=set(), unresolved=set(),
    )
    att1 = AttemptOutput(
        attempt=1, source="structural", extracted={},
        line_items=[{"x": _ev(1)}, {"x": _ev(2)}], validation_failures=[],
        residual_unresolved=[], latency_ms=0,
    )
    merge_attempt_into_state(state, att1)
    att2 = AttemptOutput(
        attempt=2, source="nlu_ner", extracted={},
        line_items=[{"x": _ev(9)}], validation_failures=[],
        residual_unresolved=[], latency_ms=0,
    )
    merge_attempt_into_state(state, att2)
    # The larger earlier list should be kept
    assert len(state.accepted_line_items) == 2
    assert state.accepted_line_items[0]["x"].value == 1
