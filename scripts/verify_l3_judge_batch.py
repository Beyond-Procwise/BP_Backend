"""Larger-sample verification of the L3 judge wiring.

Pull all rows that previously hit `discrepancy` from the most recent batch,
re-dispatch them through the renovation pipeline (which now invokes the
grounded L3 judge on missing required fields), and tally:

  - promotion rate before vs after
  - per-doc missing_required fields before vs after
  - total fields lifted

This re-dispatches each source_file ONCE. It writes new raw rows; the old
discrepancy rows are left in place for audit.
"""
from __future__ import annotations

import os
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import logging
logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s %(message)s")
# Keep dispatch chatter visible; mute deep stack noise.
logging.getLogger("src.services.extraction.dispatch").setLevel(logging.INFO)
logging.getLogger("src.services.extraction.judge_gate").setLevel(logging.INFO)
logging.getLogger("src.services.extraction_v3.judge.grounded_last_resort").setLevel(logging.WARNING)

from src.services.db import get_conn  # noqa: E402
from src.services.extraction.dispatch import dispatch_document  # noqa: E402


DOC_TYPES = ("invoice", "purchase_order", "quote")
RAW_TABLE = {
    "invoice": "proc.bp_invoice_raw",
    "purchase_order": "proc.bp_purchase_order_raw",
    "quote": "proc.bp_quote_raw",
}


def _fetch_stuck_docs(conn, limit_per_type: int = 6) -> list[tuple[str, str]]:
    """Return (doc_type, source_file) for the most recent stuck docs."""
    out: list[tuple[str, str]] = []
    for dt in DOC_TYPES:
        table = RAW_TABLE[dt]
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT DISTINCT ON (source_file) source_file
                FROM {table}
                WHERE promotion_status = 'discrepancy'
                  AND source_file IS NOT NULL
                ORDER BY source_file, extracted_at DESC
                LIMIT %s
                """,
                (limit_per_type,),
            )
            for r in cur.fetchall():
                out.append((dt, r[0]))
    return out


def _previously_missing(conn, doc_type: str, source_file: str) -> list[str]:
    table = RAW_TABLE[doc_type]
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT d.field_name
            FROM proc.bp_extraction_discrepancy d
            JOIN {table} r ON r.raw_id = d.raw_id
            WHERE d.doc_type = %s
              AND r.source_file = %s
              AND d.issue_type = 'missing_required'
              AND d.blocks_promotion = true
            ORDER BY d.created_at DESC NULLS LAST
            """,
            (doc_type, source_file),
        )
        return sorted({r[0] for r in cur.fetchall()})


def main() -> int:
    print(f"L3 judge backend: {os.environ.get('EXTRACTION_V3_JUDGE_MODEL', 'qwen (default)')}")
    with get_conn() as conn:
        docs = _fetch_stuck_docs(conn, limit_per_type=6)
        print(f"sample size: {len(docs)} stuck docs ({Counter(d for d, _ in docs)})")

        # Capture before-state for each
        before: dict[tuple[str, str], list[str]] = {}
        for dt, sf in docs:
            before[(dt, sf)] = _previously_missing(conn, dt, sf)

    results = []
    t0 = time.time()
    for i, (dt, sf) in enumerate(docs, 1):
        print(f"\n[{i}/{len(docs)}] {dt} :: {sf}")
        print(f"  before missing: {before[(dt, sf)]}")
        try:
            r = dispatch_document(process_monitor_id=None, file_path=sf, doc_type=dt)
        except Exception as exc:
            print(f"  ERROR: {exc!r}")
            results.append({"doc_type": dt, "file": sf, "error": str(exc),
                            "before_missing": before[(dt, sf)],
                            "new_status": "error", "still_missing": None})
            continue
        print(f"  -> status={r['status']} doc_pk={r['doc_pk']} missing_required={r['missing_required']}")
        results.append({
            "doc_type": dt, "file": sf,
            "before_missing": before[(dt, sf)],
            "new_status": r["status"],
            "still_missing": r["missing_required"],
            "lifted": sorted(set(before[(dt, sf)]) - set(r["missing_required"])),
        })
    dt_total = time.time() - t0
    print(f"\nElapsed: {dt_total:.1f}s for {len(docs)} docs ({dt_total/max(1,len(docs)):.1f}s/doc)")

    promoted = sum(1 for x in results if x["new_status"] == "promoted")
    partial_lift = sum(1 for x in results if x.get("lifted"))
    fields_lifted_total = sum(len(x.get("lifted") or []) for x in results)
    print(f"\n===== AGGREGATE =====")
    print(f"  docs:                   {len(results)}")
    print(f"  newly promoted:         {promoted} (was 0)")
    print(f"  docs with ANY lift:     {partial_lift}")
    print(f"  total fields lifted:    {fields_lifted_total}")
    print(f"  remaining-missing histogram:")
    rem = Counter()
    for x in results:
        for f in (x.get("still_missing") or []):
            rem[(x["doc_type"], f)] += 1
    for k, v in rem.most_common():
        print(f"    {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
