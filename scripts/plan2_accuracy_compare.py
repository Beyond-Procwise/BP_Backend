#!/usr/bin/env python3
"""Plan 2 Accuracy Comparison Script.

Compares per-field extraction rates between a baseline JSONL and a new-run JSONL.
Also computes overall fill rates, hallucination counts, and residual rates.

Usage:
    .venv/bin/python scripts/plan2_accuracy_compare.py \
        --baseline artifacts/log_monitor/v3_iter14_invoices.jsonl \
        --new      artifacts/log_monitor/v3_plan2_invoices.jsonl \
        --doc-type invoice
"""
from __future__ import annotations
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

# Header fields of interest for invoices
INVOICE_HEADER_FIELDS = [
    "invoice_id", "supplier_name", "buyer_id", "po_id",
    "invoice_date", "due_date", "payment_terms",
    "currency", "invoice_amount", "tax_amount", "tax_percent",
    "invoice_total_incl_tax", "country", "ship_to_country",
    "requested_by",
]
PO_HEADER_FIELDS = [
    "po_id", "supplier_name", "buyer_id",
    "order_date", "delivery_date", "payment_terms",
    "currency", "po_total", "tax_percent",
    "country", "ship_to_country", "requested_by",
]


def load_jsonl(path: Path) -> list[dict]:
    results = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            results.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return results


def compute_stats(records: list[dict], header_fields: list[str]) -> dict:
    """Compute per-field fill rates, model breakdown, hallucination counts."""
    total = len(records)
    error_count = sum(1 for r in records if "error" in r)
    ok_records = [r for r in records if "error" not in r]

    # Per-field: how many docs have this field committed
    field_filled: Counter = Counter()
    field_model: dict[str, Counter] = defaultdict(Counter)
    halluc_count = 0
    residual_fields: Counter = Counter()
    avg_commit_fields = 0
    has_line_items = 0

    for rec in ok_records:
        committed = rec.get("committed", [])
        committed_fields = set()
        for cf in committed:
            f = cf["field"]
            if "[" not in f:  # header only
                committed_fields.add(f)
                field_filled[f] += 1
                field_model[f][cf.get("model", "?")] += 1
        avg_commit_fields += len([c for c in committed if "[" not in c["field"]])
        has_line_items += 1 if any("[" in c["field"] for c in committed) else 0
        halluc_count += len(rec.get("hallucinations", []))
        for r in rec.get("residuals", []):
            residual_fields[r["field"]] += 1

    n_ok = len(ok_records)
    return {
        "total": total,
        "errors": error_count,
        "ok": n_ok,
        "hallucinations": halluc_count,
        "avg_header_fields": round(avg_commit_fields / n_ok, 2) if n_ok else 0,
        "has_line_items_pct": round(has_line_items / n_ok * 100, 1) if n_ok else 0,
        "field_fill_pct": {
            f: round(field_filled.get(f, 0) / n_ok * 100, 1) if n_ok else 0
            for f in header_fields
        },
        "field_model": {f: dict(c) for f, c in field_model.items()},
        "residual_fields": dict(residual_fields.most_common(10)),
    }


def print_comparison(baseline_stats: dict, new_stats: dict, header_fields: list[str], label: str):
    print(f"\n{'='*80}")
    print(f"  {label} — Field Fill Rate Comparison")
    print(f"{'='*80}")
    print(f"  Baseline docs: {baseline_stats['ok']} ok / {baseline_stats['errors']} errors")
    print(f"  New-run docs:  {new_stats['ok']} ok / {new_stats['errors']} errors")
    print(f"  Baseline hallucinations: {baseline_stats['hallucinations']}")
    print(f"  New-run hallucinations:  {new_stats['hallucinations']}")
    print(f"  Baseline avg header fields/doc: {baseline_stats['avg_header_fields']}")
    print(f"  New-run avg header fields/doc:  {new_stats['avg_header_fields']}")
    print(f"  Baseline line-items present:  {baseline_stats['has_line_items_pct']}%")
    print(f"  New-run line-items present:   {new_stats['has_line_items_pct']}%")
    print()

    b_fill = baseline_stats["field_fill_pct"]
    n_fill = new_stats["field_fill_pct"]

    print(f"  {'Field':<30} {'Baseline':>10} {'New Run':>10} {'Delta':>8}  {'Top Model (new)':}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*8}  {'-'*20}")

    for field in header_fields:
        b_pct = b_fill.get(field, 0.0)
        n_pct = n_fill.get(field, 0.0)
        delta = n_pct - b_pct
        delta_str = f"{delta:+.1f}%" if delta != 0 else "  ---"
        flag = " <-- REGRESS" if delta < -5 else (" <-- IMPROVE" if delta > 5 else "")

        # Top model for new run
        new_models = new_stats["field_model"].get(field, {})
        top_model = max(new_models, key=new_models.get) if new_models else "-"

        bar_b = "█" * int(b_pct / 10)
        bar_n = "█" * int(n_pct / 10)
        print(f"  {field:<30} {b_pct:>9.1f}% {n_pct:>9.1f}% {delta_str:>8}  {top_model}{flag}")

    # Overall fill rate (average across all fields)
    b_avg = sum(b_fill.values()) / len(header_fields) if header_fields else 0
    n_avg = sum(n_fill.values()) / len(header_fields) if header_fields else 0
    delta_avg = n_avg - b_avg
    print(f"\n  {'OVERALL AVG':<30} {b_avg:>9.1f}% {n_avg:>9.1f}% {delta_avg:>+7.1f}%")

    # Residual breakdown for new run
    if new_stats["residual_fields"]:
        print(f"\n  Top residual fields (new run):")
        for f, cnt in list(new_stats["residual_fields"].items())[:8]:
            print(f"    {f:<35} {cnt} docs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--doc-type", default="invoice", choices=["invoice", "purchase_order"])
    args = ap.parse_args()

    header_fields = INVOICE_HEADER_FIELDS if args.doc_type == "invoice" else PO_HEADER_FIELDS

    baseline_records = load_jsonl(Path(args.baseline))
    new_records = load_jsonl(Path(args.new))

    baseline_stats = compute_stats(baseline_records, header_fields)
    new_stats = compute_stats(new_records, header_fields)

    label = f"Invoice Extraction (v3 iter14 vs plan2)" if args.doc_type == "invoice" else f"PO Extraction"
    print_comparison(baseline_stats, new_stats, header_fields, label)


if __name__ == "__main__":
    main()
