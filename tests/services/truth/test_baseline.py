import json

import pytest

from src.services.truth.baseline import format_report, score


def _labelled(tmp_path, rows):
    p = tmp_path / "labelled.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    return str(p)


def _row(doc_type, outcomes):
    fields = {f"f{i}": {"value": i, "outcome": o, "rule": "x"}
              for i, o in enumerate(outcomes)}
    counts = {"verified": outcomes.count("verified"),
              "unsupported": outcomes.count("unsupported"),
              "unverifiable": outcomes.count("unverifiable")}
    return {"pk": 1, "doc_type": doc_type, "fields": fields, "counts": counts}


def test_accuracy_counts_only_verifiable_fields(tmp_path):
    rows = [_row("Invoice", ["verified", "verified", "unsupported", "unverifiable"])]
    r = score(_labelled(tmp_path, rows))
    assert r["accuracy"] == pytest.approx(2 / 3)
    assert r["coverage"] == pytest.approx(3 / 4)


def test_unverifiable_fields_never_count_as_correct(tmp_path):
    # The whole point: a set that cannot be checked does not score 1.0.
    rows = [_row("Invoice", ["unverifiable"] * 10)]
    r = score(_labelled(tmp_path, rows))
    assert r["accuracy"] is None, "no verifiable field means no accuracy, not perfect accuracy"
    assert r["coverage"] == 0.0


def test_the_report_always_shows_coverage_next_to_accuracy(tmp_path):
    rows = [_row("Invoice", ["verified", "unverifiable"])]
    text = format_report(score(_labelled(tmp_path, rows)))
    assert "accuracy" in text.lower() and "coverage" in text.lower()


def test_results_are_broken_down_by_document_type(tmp_path):
    rows = [_row("Invoice", ["verified"]), _row("Quote", ["unsupported"])]
    r = score(_labelled(tmp_path, rows))
    assert r["by_doc_type"]["Invoice"]["accuracy"] == 1.0
    assert r["by_doc_type"]["Quote"]["accuracy"] == 0.0
