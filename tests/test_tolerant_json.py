"""Tests for the tolerant JSON parser."""
from __future__ import annotations

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.services.tolerant_json import parse_tolerant  # noqa: E402


def test_clean_json_passes_through_with_recovered_false():
    res = parse_tolerant('{"a": 1, "b": [1, 2, 3]}')
    assert res.data == {"a": 1, "b": [1, 2, 3]}
    assert res.recovered is False
    assert res.completeness == 1.0


def test_markdown_fences_are_stripped():
    text = "```json\n{\"x\": 42}\n```"
    res = parse_tolerant(text)
    assert res.data == {"x": 42}


def test_assistant_turn_marker_is_stripped():
    text = ']{"\\n"}{"\\n"}assistant\n{"a": 1}'
    res = parse_tolerant(text)
    assert res.data == {"a": 1}


def test_truncated_object_mid_string_recovers():
    text = '{"name": "Acme", "items": [{"desc": "Widget", "qty"'
    res = parse_tolerant(text)
    # Either the partial object salvages OR returns None — but it must not crash.
    assert res.recovered is True


def test_truncated_after_full_first_item_recovers_first_item():
    text = (
        '{"line_items": ['
        '{"desc": "A", "qty": 1, "price": 10}, '
        '{"desc": "B", "qty": 2, "pric'
    )
    res = parse_tolerant(text)
    assert res.recovered is True
    assert res.data is not None
    items = res.data.get("line_items", [])
    # The clean first item must survive
    assert any(it.get("desc") == "A" for it in items)


def test_truncated_number_is_trimmed():
    """LLM cuts off at `"tax_amount": 202.` — the parser must trim
    the partial number and close the structure."""
    text = (
        '{"line_items": [{"desc": "X", "qty": 1, '
        '"line_amount": 100, "tax_amount": 202.'
    )
    res = parse_tolerant(text)
    assert res.recovered is True
    assert res.data is not None
    items = res.data.get("line_items", [])
    # First (and only) item should have the clean fields
    assert items[0]["desc"] == "X"
    assert items[0]["qty"] == 1
    assert items[0]["line_amount"] == 100


def test_empty_string_returns_none_completeness_zero():
    res = parse_tolerant("")
    assert res.data is None
    assert res.completeness == 0.0


def test_no_json_present_returns_none():
    res = parse_tolerant("This is just prose with no braces at all.")
    assert res.data is None


def test_prompt_echo_before_json_is_skipped():
    """Real production case: the response contains echoed prompt
    metadata before the actual JSON starts."""
    text = (
        "Terms: Net 30\n"
        "[FOOTER]\nPERRY LTD\n"
        "[EXTRA] Extract all fields\n"
        '{"header": {"id": "100"}, "line_items": []}'
    )
    res = parse_tolerant(text)
    assert res.data == {"header": {"id": "100"}, "line_items": []}


def test_partial_object_keeps_complete_keys():
    text = (
        '{"a": 1, "b": "hello", "c": [1, 2, 3], "d":'
    )
    res = parse_tolerant(text)
    assert res.recovered is True
    if res.data is not None:
        # 'a', 'b', 'c' should all survive; 'd' is incomplete and may not
        assert res.data.get("a") == 1
        assert res.data.get("b") == "hello"


def test_extracts_first_header_object_from_repeated_responses():
    """If the model loops and emits multiple {header:..., line_items:...}
    blocks, take only the FIRST one — it's the one that matches our doc."""
    inner = '{"header": {"id": "%s"}, "line_items": []}'
    text = "Garbage prefix\n" + inner % "FIRST" + "\n\nMore garbage\n" + inner % "SECOND"
    res = parse_tolerant(text)
    assert res.data is not None
    assert res.data["header"]["id"] == "FIRST"


def test_finds_inner_header_when_wrapped_in_fake_conversation():
    """Real failure mode: fine-tuned model leaks
    ``{"user": ..., "assistant": '{"header": ..., "line_items": [...]}'}``
    where the actual extraction JSON is nested as a single-quoted string.
    The wrapper isn't valid JSON, but the inner object IS — the parser
    should locate ``{"header"`` and recover from there."""
    inner_json = (
        '{"header": {"invoice_id": "INV148769", "supplier_name": "Perry Ltd"}, '
        '"line_items": [{"item_description": "Widget", "quantity": 5}, '
        '{"item_description": "Gadget", "quantity": 3}]}'
    )
    text = (
        'Extract ALL fields present in the document.]'
        '{"user": "Re-extract this Invoice", '
        f"\"assistant\": '{inner_json}'"
        '}'
    )
    res = parse_tolerant(text)
    assert res.data is not None, "parser must recover the inner header object"
    header = res.data.get("header", {})
    assert header.get("invoice_id") == "INV148769"
    assert header.get("supplier_name") == "Perry Ltd"
    lines = res.data.get("line_items", [])
    assert len(lines) == 2
    assert lines[0]["item_description"] == "Widget"
    assert "found_inner_header_object" in res.recovery_ops
