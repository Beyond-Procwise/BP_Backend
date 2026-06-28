#!/usr/bin/env python
"""Read-only extraction benchmark for the GPU-upgrade speed work (Workstream A).

Runs the v4 hybrid engine (`_run_hybrid_v4`) over a fixed sample of real
documents pulled from S3. This is the SAME extraction path used in production
(NuExtract + LLM-fill safety net) but WITHOUT persistence or DB writes — it
never mutates live data.

For each document it records:
  - elapsed wall-clock seconds
  - a stable signature (sha1) of the extracted header + line-item shape

The signature lets the before/after comparison prove that the speed changes
did NOT alter extraction output (the Workstream-A accuracy gate is
"identical extraction, faster").

Usage:
    python scripts/gpu_upgrade/bench_extraction.py --label before
    python scripts/gpu_upgrade/bench_extraction.py --label after

Manifest: artifacts/gpu_upgrade/sample_docs.txt  (one "s3_key | category" per line)
Output:   artifacts/gpu_upgrade/bench_<label>.json
Cache:    <scratch>/bench_docs/  (downloaded once, reused)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

# --- load .env so S3 + Ollama config is present -----------------------------
ROOT = Path(__file__).resolve().parents[2]
ENV = ROOT / ".env"
if ENV.exists():
    for line in ENV.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

MANIFEST = ROOT / "artifacts" / "gpu_upgrade" / "sample_docs.txt"
OUT_DIR = ROOT / "artifacts" / "gpu_upgrade"
CACHE = Path(os.environ.get("BENCH_DOC_CACHE", "/tmp/claude-1001/-home-muthu-PycharmProjects-BP-Backend/d811c8c2-8e51-48bb-a257-6de72c8cdf61/scratchpad/bench_docs"))


def _download(key: str) -> Path:
    """Download an S3 object to the local cache (once); return the local path."""
    import boto3

    CACHE.mkdir(parents=True, exist_ok=True)
    safe = key.replace("/", "__")
    dest = CACHE / safe
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    bucket = os.environ["S3_BUCKET_NAME"]
    boto3.client("s3", region_name="eu-west-1").download_file(bucket, key, str(dest))
    return dest


def _signature(result) -> str:
    """Stable sha1 over the extraction output, ignoring volatile fields."""
    try:
        if hasattr(result, "to_dict"):
            d = result.to_dict()
        elif hasattr(result, "__dict__"):
            d = dict(vars(result))
        else:
            d = dict(result)
    except Exception:
        d = {"repr": repr(result)}

    volatile = {"elapsed", "elapsed_s", "timestamp", "created_at", "duration",
                "processing_time", "_cleanup", "source_file", "doc_pk", "raw_id"}
    cleaned = {k: v for k, v in d.items() if k not in volatile}
    blob = json.dumps(cleaned, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="before | after | <tag>")
    ap.add_argument("--limit", type=int, default=0, help="cap number of docs (0=all)")
    args = ap.parse_args()

    if not MANIFEST.exists():
        print(f"manifest missing: {MANIFEST}", file=sys.stderr)
        return 2

    items = []
    for line in MANIFEST.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, cat = line.partition("|")
        items.append((key.strip(), cat.strip()))
    if args.limit:
        items = items[: args.limit]

    from src.services.extraction_v3.dispatch import _run_hybrid_v4

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    batch_t0 = time.time()
    for key, cat in items:
        try:
            local = _download(key)
        except Exception as exc:
            print(f"  download FAIL {key}: {exc}", file=sys.stderr)
            records.append({"key": key, "category": cat, "error": f"download:{exc}"})
            continue
        t0 = time.time()
        try:
            result = _run_hybrid_v4(str(local), cat)
            elapsed = time.time() - t0
            sig = _signature(result)
            rec = {"key": key, "category": cat, "elapsed_s": round(elapsed, 2),
                   "signature": sig}
            print(f"  {elapsed:7.2f}s  {sig[:12]}  {key}")
        except Exception as exc:
            elapsed = time.time() - t0
            rec = {"key": key, "category": cat, "elapsed_s": round(elapsed, 2),
                   "error": f"extract:{type(exc).__name__}:{exc}"}
            print(f"  {elapsed:7.2f}s  ERROR        {key}: {exc}", file=sys.stderr)
        records.append(rec)

    batch_total = round(time.time() - batch_t0, 2)
    ok = [r for r in records if "elapsed_s" in r and "error" not in r]
    summary = {
        "label": args.label,
        "doc_count": len(items),
        "ok_count": len(ok),
        "batch_total_s": batch_total,
        "mean_doc_s": round(sum(r["elapsed_s"] for r in ok) / len(ok), 2) if ok else None,
        "ollama_max_concurrent": os.environ.get("OLLAMA_MAX_CONCURRENT", "default"),
        "records": records,
    }
    out = OUT_DIR / f"bench_{args.label}.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nbatch_total={batch_total}s  mean={summary['mean_doc_s']}s  ok={len(ok)}/{len(items)}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
