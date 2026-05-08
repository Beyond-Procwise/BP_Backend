#!/usr/bin/env python3
"""Live smoke test for extraction_v3 against the proc.process_monitor queue.

Pulls the most recent N invoices and runs PipelineV3 on each WITHOUT
persisting. Reports:
  - extraction success / failure rate
  - per-doc judge call count + latency
  - any committed value whose evidence_text is NOT a substring of parsed.full_text
    (P0 hallucination per project iteration mandate)
  - residuals_count distribution

Usage:
  .venv/bin/python scripts/extraction_v3_live_smoke.py --n 20
  .venv/bin/python scripts/extraction_v3_live_smoke.py --n 50 --json > artifacts/log_monitor/extraction_v3_smoke.json

DOES NOT WRITE to proc.bp_invoice / _line_items / _provenance — uses
the pipeline's run() but skips persist(). Safe to run against live DB.
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

# Ensure the src/ tree is importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.WARNING)  # suppress chatty model logs
log = logging.getLogger("extraction_v3_smoke")
log.setLevel(logging.INFO)


def fetch_recent_invoices(n: int) -> list[dict]:
    """Recent N completed/extracting invoices from proc.process_monitor."""
    from src.services.db import get_conn  # imported here so module loads without DB

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, file_path, category
            FROM proc.process_monitor
            WHERE category = 'invoice'
              AND status IN ('Completed', 'Extracting', 'Extracted')
              AND file_path IS NOT NULL
            ORDER BY id DESC
            LIMIT %s
            """,
            (n,),
        )
        rows = cur.fetchall()
    return [{"record_id": r[0], "file_path": r[1], "category": r[2]} for r in rows]


def _resolve_path(file_path: str) -> Path | None:
    """Try to resolve a DB file_path (may be a relative or S3-style key) to a local Path.

    The live DB stores S3 keys such as:
        documents/invoice/TECHWORLD INV-005-30 for PO405867.pdf

    This function tries several common base directories and also strips the
    leading 'documents/' prefix segment when the full key doesn't match.
    Returns None when no local copy is found (S3-only case).
    """
    candidate = Path(file_path)
    if candidate.is_absolute() and candidate.exists():
        return candidate

    # Bases to search under when path is relative
    search_bases = [
        ROOT / "documents",
        ROOT,
        Path.home() / "Downloads",
    ]

    for base in search_bases:
        full = base / file_path
        if full.exists():
            return full

        # Also try stripping a leading "documents/" segment
        trimmed = file_path
        for prefix in ("documents/invoice/", "documents/", "invoice/"):
            if file_path.startswith(prefix):
                trimmed = file_path[len(prefix):]
                break
        if trimmed != file_path:
            full2 = base / trimmed
            if full2.exists():
                return full2

    return None


def smoke_one(pipeline: Any, doc: dict) -> dict[str, Any]:
    """Run the pipeline against one document. NEVER persist. Return summary dict."""
    from src.services.extraction_v3.parsers.router import parse as parse_doc

    record_id = doc["record_id"]
    file_path = doc["file_path"]
    out: dict[str, Any] = {"record_id": record_id, "file_path": file_path}

    t0 = time.time()
    try:
        resolved = _resolve_path(file_path)
        if resolved is None:
            return {
                **out,
                "error": f"file not found (likely S3-only): {file_path}",
                "latency_s": round(time.time() - t0, 2),
            }

        result = pipeline.run(resolved, "invoice")
        parsed = parse_doc(resolved)

        hallucinations = [
            cf.field_path
            for cf in result.committed
            if cf.evidence_text and cf.evidence_text not in parsed.full_text
        ]

        out.update({
            "doc_pk": result.doc_pk,
            "committed_count": len(result.committed),
            "residuals_count": len(result.residuals),
            "judge_calls": result.judge_calls,
            "hallucinations": hallucinations,
            "latency_s": round(time.time() - t0, 2),
        })
    except Exception as exc:
        log.exception("pipeline failed for record %s", record_id)
        out.update({"error": str(exc), "latency_s": round(time.time() - t0, 2)})
    return out


def summarize(results: list[dict]) -> dict[str, Any]:
    total = len(results)
    errored = sum(1 for r in results if "error" in r)
    succeeded = total - errored
    hallucinated = sum(1 for r in results if r.get("hallucinations"))
    avg_judge = sum(r.get("judge_calls", 0) for r in results) / max(total, 1)
    avg_latency = sum(r.get("latency_s", 0.0) for r in results) / max(total, 1)
    latencies = sorted(r.get("latency_s", 0.0) for r in results)
    p95_latency = latencies[max(0, int(0.95 * total) - 1)] if latencies else 0.0
    avg_residuals = (
        sum(r.get("residuals_count", 0) for r in results) / max(succeeded, 1)
    )
    return {
        "total": total,
        "succeeded": succeeded,
        "errored": errored,
        "DOCS_WITH_HALLUCINATIONS": hallucinated,  # capitalized so it stands out
        "avg_judge_calls": round(avg_judge, 2),
        "avg_latency_s": round(avg_latency, 2),
        "p95_latency_s": round(p95_latency, 2),
        "avg_residuals": round(avg_residuals, 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Read-only smoke test: run PipelineV3 against recent invoices."
    )
    ap.add_argument(
        "--n", type=int, default=20, help="Number of recent invoices to smoke (default: 20)"
    )
    ap.add_argument(
        "--json", action="store_true", help="Output full JSON results (one object)"
    )
    args = ap.parse_args()

    log.info("Fetching recent %d invoices from proc.process_monitor", args.n)
    try:
        docs = fetch_recent_invoices(args.n)
    except Exception as exc:
        log.error("Failed to query process_monitor: %s", exc)
        sys.exit(2)

    log.info("Got %d invoices", len(docs))
    if not docs:
        msg = "No invoice rows found in proc.process_monitor (status Completed/Extracting/Extracted)."
        if args.json:
            print(json.dumps({"summary": {"total": 0}, "results": [], "note": msg}))
        else:
            print(msg)
        sys.exit(0)

    # Import here so module is importable without a running DB / ML stack
    from src.services.extraction_v3.pipeline import PipelineV3
    pipeline = PipelineV3()

    results: list[dict] = []
    for i, doc in enumerate(docs, 1):
        log.info("[%d/%d] %s", i, len(docs), doc["file_path"])
        results.append(smoke_one(pipeline, doc))

    summary = summarize(results)

    if args.json:
        print(json.dumps({"summary": summary, "results": results}, indent=2, default=str))
    else:
        print("\n=== EXTRACTION V3 LIVE SMOKE SUMMARY ===")
        for k, v in summary.items():
            print(f"  {k:35s} {v}")
        print()
        print("Per-doc results:")
        for r in results:
            if "error" in r:
                print(f"  [{r['record_id']}] ERROR: {r['error']}")
            else:
                halluc_flag = " *HALLUCINATIONS!" if r.get("hallucinations") else ""
                print(
                    f"  [{r['record_id']}] commit={r['committed_count']:3d} "
                    f"resid={r['residuals_count']:3d} "
                    f"judge={r['judge_calls']:2d} "
                    f"lat={r['latency_s']:6.2f}s"
                    f"{halluc_flag}"
                )
                if r.get("hallucinations"):
                    for field in r["hallucinations"]:
                        print(f"              hallucinated field: {field}")
        if summary["DOCS_WITH_HALLUCINATIONS"] > 0:
            print(
                f"\n*** P0: {summary['DOCS_WITH_HALLUCINATIONS']} doc(s) had hallucinated "
                "values committed (evidence_text not substring of parsed.full_text) ***"
            )
            sys.exit(1)


if __name__ == "__main__":
    main()
