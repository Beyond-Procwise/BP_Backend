"""Extraction V3 Accuracy Dashboard.

Reads all records persisted by the V3 pipeline (via bp_extraction_provenance_v3)
and computes per-field accuracy metrics against the ground-truth expected.json
fixtures in tests/extraction_v3/fixtures/.

Usage:
    .venv/bin/python3 scripts/extraction_v3_accuracy_audit.py

Outputs a table of per-field accuracy stats to stdout.
"""
from __future__ import annotations

import json
import sys
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.db import get_conn

FIXTURES_DIR = Path(__file__).parent.parent / "tests/extraction_v3/fixtures"


def _load_ground_truth() -> dict[str, dict]:
    """Load all expected.json fixtures, keyed by invoice_id or po_id."""
    gt: dict[str, dict] = {}
    for ej in FIXTURES_DIR.rglob("*.expected.json"):
        data = json.loads(ej.read_text())
        header = data.get("header", {})
        pk = header.get("invoice_id") or header.get("po_id")
        if pk:
            gt[pk] = data
    return gt


def _load_db_records() -> list[dict]:
    """Load all extraction provenance records from the DB."""
    records = []
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT doc_pk, doc_type, field_path, value, model, final_confidence, extracted_at
            FROM proc.bp_extraction_provenance_v3
            ORDER BY doc_pk, field_path
        """)
        rows = cur.fetchall()
        cols = [d.name for d in cur.description]
        for row in rows:
            records.append(dict(zip(cols, row)))
    return records


def _group_by_doc(records: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in records:
        out.setdefault(r["doc_pk"], []).append(r)
    return out


def _coerce(v) -> str | None:
    if v is None:
        return None
    if isinstance(v, Decimal):
        v = float(v)
    return str(v).strip().lower()


def _match(actual_str: str | None, expected_val) -> bool:
    if expected_val is None:
        return actual_str is None or actual_str in ("none", "null", "")
    if actual_str is None:
        return False
    expected_str = _coerce(expected_val)
    if expected_str is None:
        return actual_str in ("none", "null", "")
    # Numeric tolerance
    try:
        af = float(actual_str)
        ef = float(expected_str)
        return abs(af - ef) <= 0.05
    except (ValueError, TypeError):
        pass
    return actual_str == expected_str


def main():
    print("=== Extraction V3 Accuracy Dashboard ===\n")

    gt = _load_ground_truth()
    records = _load_db_records()
    by_doc = _group_by_doc(records)

    if not by_doc:
        print("No extraction records found in bp_extraction_provenance_v3.")
        return

    print(f"Found {len(by_doc)} extracted documents, {len(gt)} ground-truth fixtures.\n")

    # Per-document accuracy
    doc_results = []
    for doc_pk, fields in by_doc.items():
        if doc_pk not in gt:
            print(f"  [WARN] {doc_pk}: no ground-truth fixture, skipping accuracy check")
            continue

        expected = gt[doc_pk]
        expected_header = expected.get("header", {})
        doc_type = fields[0]["doc_type"]

        actual_by_field: dict[str, str] = {}
        for f in fields:
            fp = f["field_path"]
            if "[" not in fp:  # header fields only (not line_items[N].*)
                actual_by_field[fp] = f["value"]

        correct = 0
        total = 0
        mismatches = []
        for field_name, expected_val in expected_header.items():
            if field_name == "supplier_name":
                continue  # resolves to supplier_id; skip for now
            total += 1
            actual_str = _coerce(actual_by_field.get(field_name))
            if _match(actual_str, expected_val):
                correct += 1
            else:
                mismatches.append((field_name, expected_val, actual_str))

        accuracy = correct / total * 100 if total else 0.0
        doc_results.append({
            "doc_pk": doc_pk,
            "doc_type": doc_type,
            "correct": correct,
            "total": total,
            "accuracy": accuracy,
            "mismatches": mismatches,
        })

    # Print results
    print(f"{'Doc PK':<20} {'Type':<15} {'Accuracy':>10}  {'Correct/Total':>14}  Mismatches")
    print("-" * 90)
    for r in sorted(doc_results, key=lambda x: x["accuracy"], reverse=True):
        ct = f"{r['correct']}/{r['total']}"
        print(f"{r['doc_pk']:<20} {r['doc_type']:<15} {r['accuracy']:>9.1f}%  {ct:>14}", end="")
        if r["mismatches"]:
            mm_str = ", ".join(f"{f}={a!r}(exp {e!r})" for f, e, a in r["mismatches"])
            print(f"  [{mm_str}]", end="")
        print()

    # Aggregate stats
    if doc_results:
        total_correct = sum(r["correct"] for r in doc_results)
        total_fields = sum(r["total"] for r in doc_results)
        overall = total_correct / total_fields * 100 if total_fields else 0.0
        print()
        print(f"Overall accuracy: {total_correct}/{total_fields} = {overall:.1f}%")

        # Per-field breakdown
        field_stats: dict[str, dict[str, int]] = {}
        for r in doc_results:
            expected_header = gt[r["doc_pk"]].get("header", {})
            actual_by_field_raw: dict[str, str] = {}
            for f in by_doc[r["doc_pk"]]:
                if "[" not in f["field_path"]:
                    actual_by_field_raw[f["field_path"]] = f["value"]
            for field_name, expected_val in expected_header.items():
                if field_name == "supplier_name":
                    continue
                fs = field_stats.setdefault(field_name, {"correct": 0, "total": 0})
                fs["total"] += 1
                actual_str = _coerce(actual_by_field_raw.get(field_name))
                if _match(actual_str, expected_val):
                    fs["correct"] += 1

        print("\nPer-field accuracy:")
        for fname, fs in sorted(field_stats.items()):
            pct = fs["correct"] / fs["total"] * 100 if fs["total"] else 0.0
            bar = "█" * int(pct / 10) + "░" * (10 - int(pct / 10))
            print(f"  {fname:<30} {bar}  {pct:5.1f}%  ({fs['correct']}/{fs['total']})")


if __name__ == "__main__":
    main()
