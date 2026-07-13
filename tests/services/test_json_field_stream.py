"""JsonFieldStreamer must emit prose, never JSON syntax, at any chunk boundary."""

import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

import pytest

from services.json_field_stream import JsonFieldStreamer


def _drain(chunks, field="answer"):
    streamer = JsonFieldStreamer(field)
    return "".join(streamer.feed(c) for c in chunks)


def test_extracts_the_field_from_a_single_chunk():
    doc = '{"answer": "Total spend is 101120.", "follow_ups": ["a"]}'
    assert _drain([doc]) == "Total spend is 101120."


def test_never_emits_json_syntax():
    doc = '{"answer": "hello", "follow_ups": []}'
    out = _drain(list(doc))  # one character at a time — the worst case
    assert out == "hello"
    for token in ('{', '}', '"answer"', 'follow_ups', '[', ']'):
        assert token not in out


@pytest.mark.parametrize("size", [1, 2, 3, 5, 7, 13, 64])
def test_chunk_boundaries_never_corrupt_the_output(size):
    """The key, the colon, and escapes must survive being split anywhere."""
    doc = '{"answer": "Line one.\\nLine two: \\"quoted\\" and \\\\ backslash.", "follow_ups": []}'
    chunks = [doc[i : i + size] for i in range(0, len(doc), size)]
    assert _drain(chunks) == 'Line one.\nLine two: "quoted" and \\ backslash.'


def test_stops_at_the_closing_quote_and_ignores_later_fields():
    doc = '{"answer": "done", "follow_ups": ["not part of the answer"]}'
    streamer = JsonFieldStreamer("answer")
    out = "".join(streamer.feed(c) for c in doc)
    assert out == "done"
    assert streamer.complete
    assert "not part of the answer" not in out


def test_unicode_escapes_are_decoded():
    doc = '{"answer": "caf\\u00e9 \\u2014 done", "follow_ups": []}'
    assert _drain([doc]) == "café — done"


def test_unicode_escape_split_across_chunks():
    doc = '{"answer": "caf\\u00e9", "follow_ups": []}'
    # Split right through the middle of the é sequence.
    assert _drain([doc[:18], doc[18:]]) == "café"


def test_leading_keys_before_the_target_are_skipped():
    doc = '{"follow_ups": ["x"], "answer": "the real answer"}'
    assert _drain([doc]) == "the real answer"


def test_a_null_value_yields_nothing_rather_than_garbage():
    streamer = JsonFieldStreamer("answer")
    out = "".join(streamer.feed(c) for c in '{"answer": null}')
    assert out == ""


def test_matches_json_loads_for_a_realistic_answer():
    """The decoded stream must equal what json.loads would have produced."""
    payload = {
        "answer": (
            "Total spend:\n"
            '- £101,120.00 (GBP)\n'
            '- $73,839.00 (USD)\n\n'
            'The supplier "PeopleFirst HR Solutions Ltd" accounts for £100,000.00.\n'
            "Path: C:\\reports\\spend.csv"
        ),
        "follow_ups": ["Which suppliers?", "Show the trend"],
    }
    doc = json.dumps(payload)
    chunks = [doc[i : i + 11] for i in range(0, len(doc), 11)]
    assert _drain(chunks) == payload["answer"]
