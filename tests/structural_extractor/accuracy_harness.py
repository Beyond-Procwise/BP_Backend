"""Accuracy harness for recursive extraction improvement.

Runs `extract()` on every fixture document, compares field-by-field to
`ground_truth.yaml`, and prints a per-doc / per-field report. Exit code 0
if accuracy is 100%, 1 otherwise.

Usage:
    .venv/bin/python tests/structural_extractor/accuracy_harness.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

# Make `src.services.X` importable
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml
from src.services.structural_extractor import extract
from src.services.structural_extractor.llm_fallback import _call_llm as _real_call_llm  # noqa
from src.services.structural_extractor import llm_fallback

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "docs"
GROUND_TRUTH = Path(__file__).parent / "fixtures" / "ground_truth.yaml"


def _get(ev, default=None):
    """Extract .value from an ExtractedValue or return default."""
    if ev is None:
        return default
    return getattr(ev, "value", ev)


def _match(actual: Any, expected: Any) -> bool:
    if expected is None:
        return actual is None or actual == "" or actual == {}
    if actual is None:
        return False
    if isinstance(expected, float) or isinstance(actual, float):
        try:
            return abs(float(actual) - float(expected)) < 0.01
        except (TypeError, ValueError):
            return False
    if isinstance(expected, str) and isinstance(actual, str):
        return actual.strip().lower() == expected.strip().lower()
    return actual == expected


def _contains_match(actual: Any, expected: Any) -> bool:
    """Softer match for descriptions: expected substring in actual."""
    if expected is None:
        return actual is None
    if not isinstance(expected, str) or not isinstance(actual, str):
        return _match(actual, expected)
    return expected.strip().lower() in actual.strip().lower()


def score_document(name: str, spec: dict, use_real_llm: bool = False) -> dict:
    """Score one document. Returns per-field match status."""
    file_path = FIXTURE_DIR / spec["file"]
    if not file_path.exists():
        return {"error": f"fixture missing: {spec['file']}"}

    # Stub LLM for speed unless explicitly asked to use the real one
    if not use_real_llm:
        llm_fallback._call_llm = lambda prompt: "{}"

    try:
        result = extract(file_path.read_bytes(), spec["file"], spec["doc_type"])
    except Exception as exc:
        return {"error": f"extract() raised: {exc}"}

    actual_header = {k: _get(v) for k, v in result.header.items()}
    actual_lines = [{k: _get(v) for k, v in item.items()} for item in result.line_items]

    report: dict[str, Any] = {"header": {}, "line_items": [], "unresolved_fields": result.unresolved_fields}

    # Header field comparison
    for field, expected in spec.get("header", {}).items():
        actual = actual_header.get(field)
        ok = _match(actual, expected)
        report["header"][field] = {
            "expected": expected,
            "actual": actual,
            "match": ok,
        }

    # Line items
    expected_lines = spec.get("line_items", [])
    for i, e_line in enumerate(expected_lines):
        a_line = actual_lines[i] if i < len(actual_lines) else {}
        line_report: dict[str, Any] = {}
        for field, expected in e_line.items():
            actual = a_line.get(field)
            if field == "item_description":
                ok = _contains_match(actual, expected)
            else:
                ok = _match(actual, expected)
            line_report[field] = {"expected": expected, "actual": actual, "match": ok}
        report["line_items"].append(line_report)
    # Extra lines: mark as extras
    if len(actual_lines) > len(expected_lines):
        for extra in actual_lines[len(expected_lines):]:
            report["line_items"].append({"_extra": {"actual": extra, "match": False}})
    return report


def aggregate(all_reports: dict[str, dict]) -> dict:
    """Summary stats across all docs."""
    total_fields = 0
    matched_fields = 0
    per_doc: dict[str, tuple[int, int]] = {}
    for doc_name, rep in all_reports.items():
        if "error" in rep:
            per_doc[doc_name] = (0, 0)
            continue
        doc_total = 0
        doc_matched = 0
        for field, entry in rep.get("header", {}).items():
            doc_total += 1
            if entry["match"]:
                doc_matched += 1
        for line in rep.get("line_items", []):
            for field, entry in line.items():
                if field == "_extra":
                    doc_total += 1
                    continue
                doc_total += 1
                if entry["match"]:
                    doc_matched += 1
        total_fields += doc_total
        matched_fields += doc_matched
        per_doc[doc_name] = (doc_matched, doc_total)
    return {
        "total_fields": total_fields,
        "matched_fields": matched_fields,
        "accuracy_pct": (matched_fields / total_fields * 100) if total_fields else 0.0,
        "per_doc": per_doc,
    }


def main(use_real_llm: bool = False):
    with open(GROUND_TRUTH) as f:
        gt = yaml.safe_load(f)

    reports: dict[str, dict] = {}
    for doc_name, spec in gt.items():
        reports[doc_name] = score_document(doc_name, spec, use_real_llm=use_real_llm)

    # Print detailed report
    print("=" * 72)
    print("Per-document field-level report")
    print("=" * 72)
    for doc_name, rep in reports.items():
        if "error" in rep:
            print(f"\n{doc_name}: ERROR — {rep['error']}")
            continue
        print(f"\n{doc_name}:")
        for field, entry in rep.get("header", {}).items():
            mark = "✓" if entry["match"] else "✗"
            if entry["match"]:
                print(f"  {mark} {field}: {entry['actual']}")
            else:
                print(f"  {mark} {field}: expected={entry['expected']!r} actual={entry['actual']!r}")
        if rep.get("line_items"):
            for idx, line in enumerate(rep["line_items"], 1):
                if "_extra" in line:
                    print(f"  ✗ EXTRA line item: {line['_extra']['actual']}")
                    continue
                for field, entry in line.items():
                    mark = "✓" if entry["match"] else "✗"
                    if entry["match"]:
                        print(f"  {mark} line[{idx}].{field}: {entry['actual']}")
                    else:
                        print(f"  {mark} line[{idx}].{field}: expected={entry['expected']!r} actual={entry['actual']!r}")
        if rep.get("unresolved_fields"):
            print(f"  [unresolved: {rep['unresolved_fields']}]")

    # Summary
    agg = aggregate(reports)
    print("\n" + "=" * 72)
    print(f"Overall accuracy: {agg['matched_fields']}/{agg['total_fields']} "
          f"= {agg['accuracy_pct']:.1f}%")
    for doc, (m, t) in agg["per_doc"].items():
        print(f"  {doc}: {m}/{t}" + (f" ({m/t*100:.0f}%)" if t else ""))
    print("=" * 72)

    return 0 if agg["accuracy_pct"] == 100.0 else 1


if __name__ == "__main__":
    use_real = "--real-llm" in sys.argv
    sys.exit(main(use_real_llm=use_real))
