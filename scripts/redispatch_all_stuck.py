"""Re-dispatch every stuck doc through the renovated pipeline.

For each raw row with promotion_status='discrepancy', call dispatch_document
again. Skips files already dispatched after timestamp `since` so reruns are
idempotent within a single batch.

Writes a JSON line per attempt to /tmp/redispatch_log.jsonl for downstream
accuracy auditing.
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import logging
logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logging.getLogger("src.services.extraction.dispatch").setLevel(logging.INFO)
logging.getLogger("src.services.extraction.judge_gate").setLevel(logging.INFO)
logging.getLogger("src.services.extraction_v3.judge.grounded_last_resort").setLevel(logging.WARNING)

from src.services.db import get_conn  # noqa: E402
from src.services.extraction.dispatch import dispatch_document  # noqa: E402


RAW_TABLE = {
    "invoice": "proc.bp_invoice_raw",
    "purchase_order": "proc.bp_purchase_order_raw",
    "quote": "proc.bp_quote_raw",
}

LOG_PATH = Path("/tmp/redispatch_log.jsonl")


def _fetch_all_stuck(conn) -> list[tuple[str, str, int]]:
    """Return (doc_type, source_file, prev_raw_id) for every stuck doc.
    De-duplicates by source_file, keeping the latest raw_id per file."""
    out = []
    seen = set()
    for dt, table in RAW_TABLE.items():
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT DISTINCT ON (source_file) source_file, raw_id
                FROM {table}
                WHERE promotion_status = 'discrepancy'
                  AND source_file IS NOT NULL
                ORDER BY source_file, extracted_at DESC
                """,
            )
            for sf, rid in cur.fetchall():
                key = (dt, sf)
                if key in seen:
                    continue
                seen.add(key)
                out.append((dt, sf, rid))
    return out


def main() -> int:
    start = datetime.now(timezone.utc)
    print(f"=== redispatch start {start.isoformat()} ===")
    with get_conn() as conn:
        docs = _fetch_all_stuck(conn)
    print(f"stuck docs to retry: {len(docs)} ({Counter(d for d, _, _ in docs)})")

    LOG_PATH.write_text("")  # truncate

    results = []
    t0 = time.time()
    for i, (dt, sf, prev_raw_id) in enumerate(docs, 1):
        print(f"\n[{i:>2}/{len(docs)}] {dt} :: prev_raw_id={prev_raw_id} :: {sf}")
        rec = {"doc_type": dt, "source_file": sf, "prev_raw_id": prev_raw_id}
        try:
            r = dispatch_document(process_monitor_id=None, file_path=sf, doc_type=dt)
            rec.update({
                "new_raw_id": r["raw_id"],
                "new_status": r["status"],
                "doc_pk": r["doc_pk"],
                "n_fields": r["n_fields"],
                "missing_required": r["missing_required"],
            })
            print(f"  -> status={r['status']} doc_pk={r['doc_pk']} missing={r['missing_required']}")
        except Exception as exc:
            rec.update({"error": str(exc), "new_status": "error"})
            print(f"  ERROR: {exc!r}")
        results.append(rec)
        with LOG_PATH.open("a") as f:
            f.write(json.dumps(rec, default=str) + "\n")

    elapsed = time.time() - t0
    promoted = sum(1 for x in results if x.get("new_status") == "promoted")
    discrep = sum(1 for x in results if x.get("new_status") == "discrepancy")
    err = sum(1 for x in results if x.get("new_status") == "error")
    print(f"\n===== REDISPATCH AGGREGATE =====")
    print(f"  elapsed:        {elapsed:.1f}s ({elapsed/max(1, len(results)):.1f}s/doc)")
    print(f"  promoted:       {promoted} / {len(results)}  ({100*promoted/max(1,len(results)):.0f}%)")
    print(f"  discrepancy:    {discrep}")
    print(f"  error:          {err}")
    print(f"  log: {LOG_PATH}")
    print(f"\n  remaining missing-required histogram:")
    rem = Counter()
    for x in results:
        for f in (x.get("missing_required") or []):
            rem[(x["doc_type"], f)] += 1
    for k, v in rem.most_common():
        print(f"    {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
