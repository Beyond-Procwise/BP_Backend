"""Backfill every uploaded document through the context_layer pipeline.

Reads every distinct file_path from proc.process_monitor (and any _raw
table rows that aren't currently represented), runs each one through
extraction.dispatch.dispatch_document. The new pipeline is idempotent —
ON CONFLICT DO UPDATE in promotion.promote() means re-processing a doc
that's already in _stg just refreshes it.

Use after a pipeline change to bring _stg back into lock-step with the
authoritative source-of-truth (the uploaded PDFs).
"""
from __future__ import annotations

import json
import logging
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logging.getLogger("src.services.extraction.dispatch").setLevel(logging.INFO)
logging.getLogger("src.services.extraction.context_layer").setLevel(logging.INFO)
logging.getLogger("src.services.extraction.promotion").setLevel(logging.INFO)
logging.getLogger("src.services.extraction.judge_gate").setLevel(logging.WARNING)
logging.getLogger("src.services.extraction_v3.judge.qwen_vl").setLevel(logging.ERROR)

from src.services.db import get_conn  # noqa: E402
from src.services.extraction.dispatch import dispatch_document  # noqa: E402


CATEGORY_TO_DOCTYPE = {
    "invoice": "invoice",
    "Invoice": "invoice",
    "purchase_order": "purchase_order",
    "PurchaseOrder": "purchase_order",
    "po": "purchase_order",
    "PO": "purchase_order",
    "quote": "quote",
    "Quote": "quote",
    "contract": "contract",
    "Contract": "contract",
}


def _all_unique_files() -> list[tuple[str, str]]:
    """Return distinct (doc_type, file_path) across process_monitor + _raw.

    process_monitor is the upload manifest — every file the user
    submitted lives there. The _raw tables hold whatever the pipeline
    has written. We dedup on file_path within each doc_type so a doc
    that lives in both places is dispatched once.
    """
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str]] = []
    with get_conn() as conn:
        cur = conn.cursor()
        # process_monitor (the authoritative upload list)
        cur.execute(
            """
            SELECT DISTINCT category, file_path
              FROM proc.process_monitor
             WHERE file_path IS NOT NULL
               AND category IS NOT NULL
            """,
        )
        for cat, fp in cur.fetchall():
            dt = CATEGORY_TO_DOCTYPE.get(cat)
            if dt is None:
                continue
            key = (dt, fp)
            if key in seen:
                continue
            seen.add(key)
            out.append(key)
        # _raw tables (catch any pending / no_pk / discrepancy rows whose
        # source_file is NOT in process_monitor, e.g. from a bulk import).
        for dt, table in (
            ("invoice", "proc.bp_invoice_raw"),
            ("purchase_order", "proc.bp_purchase_order_raw"),
            ("quote", "proc.bp_quote_raw"),
        ):
            cur.execute(
                f"SELECT DISTINCT source_file FROM {table} "
                f"WHERE source_file IS NOT NULL",
            )
            for (fp,) in cur.fetchall():
                key = (dt, fp)
                if key in seen:
                    continue
                seen.add(key)
                out.append(key)
    return out


def main() -> int:
    start = datetime.now(timezone.utc)
    print(f"=== backfill start {start.isoformat()} ===")
    files = _all_unique_files()
    by_type = Counter(dt for dt, _ in files)
    print(f"files to dispatch: {len(files)}  by_type={dict(by_type)}")
    print()

    log_path = Path("/tmp/backfill_log.jsonl")
    log_path.write_text("")

    results: list[dict] = []
    t0 = time.time()
    for i, (dt, fp) in enumerate(files, 1):
        print(f"[{i:>3}/{len(files)}] {dt:<14}  {fp}")
        rec: dict = {
            "i": i,
            "doc_type": dt,
            "source_file": fp,
            "start": datetime.now(timezone.utc).isoformat(),
        }
        try:
            result = dispatch_document(
                process_monitor_id=None, file_path=fp, doc_type=dt,
            )
            rec["status"] = result.get("status")
            rec["doc_pk"] = result.get("doc_pk")
            rec["missing_required"] = result.get("missing_required", [])
            rec["n_discrepancies"] = result.get("n_discrepancies")
            rec["n_fields"] = result.get("n_fields")
        except Exception as exc:  # noqa: BLE001
            rec["status"] = "exception"
            rec["error"] = str(exc)[:300]
            print(f"  ! exception: {exc!s}"[:200])
        rec["elapsed_s"] = round(time.time() - t0, 1)
        results.append(rec)
        with log_path.open("a") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")
        # Inline progress beacon every 10 docs
        if i % 10 == 0:
            promoted = sum(1 for r in results if r.get("status") == "promoted")
            discrepancy = sum(1 for r in results if r.get("status") == "discrepancy")
            failed = sum(
                1 for r in results
                if r.get("status") in ("exception", "no_pk", "pending")
            )
            print(
                f"   ...progress: promoted={promoted} "
                f"discrepancy={discrepancy} failed={failed}"
            )

    elapsed = time.time() - t0
    promoted = sum(1 for r in results if r.get("status") == "promoted")
    discrepancy = sum(1 for r in results if r.get("status") == "discrepancy")
    no_pk = sum(1 for r in results if r.get("status") == "no_pk")
    pending = sum(1 for r in results if r.get("status") == "pending")
    exception = sum(1 for r in results if r.get("status") == "exception")

    print()
    print("===== BACKFILL AGGREGATE =====")
    print(f"  elapsed:        {elapsed:.1f}s ({elapsed/max(len(files),1):.1f}s/doc)")
    print(f"  total:          {len(files)}")
    print(f"  promoted:       {promoted}")
    print(f"  discrepancy:    {discrepancy}")
    print(f"  no_pk:          {no_pk}")
    print(f"  pending:        {pending}")
    print(f"  exception:      {exception}")
    print(f"  log: {log_path}")

    if discrepancy or pending or exception or no_pk:
        print()
        print("--- DOCS NOT PROMOTED ---")
        for r in results:
            if r.get("status") != "promoted":
                missing = r.get("missing_required") or []
                print(
                    f"  [{r['doc_type']}] {r['source_file']}: "
                    f"status={r.get('status')} missing={missing} "
                    f"err={r.get('error','')[:120]}"
                )

    return 0 if (discrepancy + pending + exception + no_pk) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
