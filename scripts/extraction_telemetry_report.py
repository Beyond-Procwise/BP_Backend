"""Analysis report over proc.bp_extraction_telemetry.

Summarises how well extraction is doing and where the gaps are, so the data can
drive improvement: outcome mix, gap/discrepancy distribution, USD-computation
coverage, and the vendor/layout patterns with the most issues.

Read-only. Run: .venv/bin/python scripts/extraction_telemetry_report.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.services.db import get_conn  # noqa: E402


def main() -> int:
    with get_conn() as conn, conn.cursor() as cur:
        # Latest telemetry row per process_monitor_id (most recent processing).
        cur.execute("""
            select distinct on (process_monitor_id)
                   doc_type, status, completeness_status, confidence,
                   line_items, n_discrepancies, discrepancy_types, currency,
                   converted_amount_usd, vendor_hint, file_path
              from proc.bp_extraction_telemetry
             order by process_monitor_id, captured_at desc
        """)
        cols = [d.name for d in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]

    if not rows:
        print("No telemetry captured yet. Run: "
              ".venv/bin/python -m src.services.extraction_telemetry.telemetry_service --once")
        return 0

    n = len(rows)
    print(f"=== Extraction Telemetry Report ({n} documents) ===\n")

    # 1. Outcome by doc_type x completeness_status
    print("Completeness by doc_type:")
    by = Counter((r["doc_type"], r["completeness_status"]) for r in rows)
    for (dt, cs), c in sorted(by.items()):
        print(f"  {dt:16s} {str(cs):18s} {c}")

    # 2. Clean vs flagged
    clean = sum(1 for r in rows if r["completeness_status"] == "complete" and r["n_discrepancies"] == 0)
    print(f"\nClean (complete, 0 discrepancies): {clean}/{n} = {clean/n*100:.1f}%")

    # 3. Discrepancy / gap distribution
    dtypes: Counter = Counter()
    for r in rows:
        try:
            for k, v in json.loads(r["discrepancy_types"] or "{}").items():
                dtypes[k] += int(v)
        except (ValueError, TypeError):
            pass
    print("\nGaps / discrepancies by type (across all docs):")
    if dtypes:
        for k, c in dtypes.most_common():
            print(f"  {k:24s} {c}")
    else:
        print("  (none)")

    # 4. USD computation coverage
    usd_ok = sum(1 for r in rows if r["converted_amount_usd"] is not None)
    print(f"\nUSD computed (converted_amount_usd present): {usd_ok}/{n} = {usd_ok/n*100:.1f}%")
    missing_usd = [r for r in rows if r["converted_amount_usd"] is None and r["completeness_status"] not in (None, "no_stg_row")]
    for r in missing_usd[:10]:
        print(f"  [no USD] {r['doc_type']} {r['file_path']}")

    # 5. Vendor/layout patterns with the most issues
    print("\nVendor patterns with gaps (issues per vendor):")
    vissues: Counter = Counter()
    vtotal: Counter = Counter()
    for r in rows:
        v = r["vendor_hint"] or "(unknown)"
        vtotal[v] += 1
        if r["completeness_status"] not in ("complete", None) or r["n_discrepancies"]:
            vissues[v] += 1
    for v, c in vissues.most_common(10):
        print(f"  {v:24s} {c}/{vtotal[v]} docs with gaps")
    if not vissues:
        print("  (no vendor shows gaps)")

    # 6. Docs needing attention (explicit list)
    attention = [r for r in rows
                 if r["completeness_status"] not in ("complete", None) or r["n_discrepancies"]]
    print(f"\nDocuments needing attention: {len(attention)}")
    for r in attention:
        types = ",".join(json.loads(r["discrepancy_types"] or "{}").keys())
        print(f"  [{r['completeness_status']}] {r['doc_type']} {Path(r['file_path']).name}"
              f"  discrep={r['n_discrepancies']}({types})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
