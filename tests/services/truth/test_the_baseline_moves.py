"""The spec called this the test that matters, and it was never written.

    "the test that matters, over the real 311: the baseline must move. If
     excluding unverifiable fields from the numerator does not change the score,
     the labeller is not doing anything and the run should be treated as failed."

The evidence for it was a manual run and a prose result document. Nothing pinned
it, so nothing would have noticed a future change quietly restoring
self-agreement scoring. These tests pin the property itself, on constructed data,
so they run in milliseconds and need no corpus.
"""
import json

from src.services.truth.baseline import score


def _write(tmp_path, rows):
    p = tmp_path / "labelled.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    return str(p)


def _record(doc_type, outcomes):
    return {
        "doc_type": doc_type,
        "fields": {f"f{i}": {"value": i, "outcome": o, "rule": "x"}
                   for i, o in enumerate(outcomes)},
    }


def test_making_a_field_unverifiable_never_raises_accuracy_above_the_truth():
    """The one way to game this: reclassify what you got wrong as uncheckable.

    Accuracy may rise -- that is arithmetic, the wrong answer left the
    denominator -- but coverage MUST fall to pay for it. A change that raises
    accuracy while holding coverage is the failure this whole design exists to
    prevent.
    """
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as honest_dir, \
            tempfile.TemporaryDirectory() as gamed_dir:
        honest_result = score(_write(
            Path(honest_dir), [_record("Invoice", ["verified"] * 7 + ["unsupported"] * 3)]))
        gamed_result = score(_write(
            Path(gamed_dir), [_record("Invoice", ["verified"] * 7 + ["unverifiable"] * 3)]))

    assert honest_result["accuracy"] == 0.7
    assert gamed_result["accuracy"] == 1.0, "hiding the failures does raise accuracy"
    assert gamed_result["coverage"] < honest_result["coverage"], (
        "and coverage must fall to pay for it -- otherwise the score is gameable "
        "with no visible cost"
    )


def test_a_wholly_uncheckable_corpus_reports_no_accuracy_rather_than_perfect(tmp_path):
    r = score(_write(tmp_path, [_record("Invoice", ["unverifiable"] * 20)]))
    assert r["accuracy"] is None
    assert r["coverage"] == 0.0


def test_unverifiable_fields_are_absent_from_the_numerator(tmp_path):
    """Directly: adding uncheckable fields to a record cannot change accuracy."""
    before = score(_write(tmp_path, [_record("Invoice", ["verified", "unsupported"])]))
    after = score(_write(tmp_path, [
        _record("Invoice", ["verified", "unsupported"] + ["unverifiable"] * 50)]))
    assert before["accuracy"] == after["accuracy"] == 0.5
    assert after["coverage"] < before["coverage"]
