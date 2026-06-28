#!/usr/bin/env python
"""Read-only THROUGHPUT benchmark for the LIVE renovation extraction pipeline.

Production runs `services/extraction/dispatch.dispatch_document` (renovation,
EXTRACTION_RENOVATION_ENABLED=1). Its per-doc latency is dominated by the
context_layer AgentNick calls, which queue at the Ollama daemon's
OLLAMA_NUM_PARALLEL. This bench measures how fast a BATCH of documents
completes when run concurrently — the throughput lever the GPU-upgrade
throttle changes target (doc-workers 4->8, NUM_PARALLEL 2->8).

It calls the REAL dispatch_document so the timing reflects live extraction
code exactly, but monkeypatches every DB side-effect (record_action,
persistence.*, promotion.promote) to a no-op IN THIS PROCESS ONLY — nothing is
written to bp_sqldb and production is untouched.

The throttle changes do NOT alter per-doc extraction logic, so output is
identical by construction; we still capture a coarse result signature
(doc_pk + field/line counts + status) as an integrity check.

Usage:
    # before: old throttle — point at prod daemon (NUM_PARALLEL=2), 4 workers, MAX_CONCURRENT=2
    OLLAMA_BASE_URL=http://127.0.0.1:11434 OLLAMA_MAX_CONCURRENT=2 \
        python -u scripts/gpu_upgrade/bench_renovation.py --label before --workers 4
    # after: lifted throttle — point at NUM_PARALLEL=8 daemon, 8 workers, MAX_CONCURRENT=8
    OLLAMA_BASE_URL=http://127.0.0.1:11435 OLLAMA_MAX_CONCURRENT=8 \
        python -u scripts/gpu_upgrade/bench_renovation.py --label after --workers 8
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENV = ROOT / ".env"
if ENV.exists():
    for line in ENV.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        # Do NOT let .env override an explicitly-exported throttle/endpoint.
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

MANIFEST = ROOT / "artifacts" / "gpu_upgrade" / "sample_docs.txt"
OUT_DIR = ROOT / "artifacts" / "gpu_upgrade"
CACHE = Path(os.environ.get("BENCH_DOC_CACHE", "/tmp/claude-1001/-home-muthu-PycharmProjects-BP-Backend/d811c8c2-8e51-48bb-a257-6de72c8cdf61/scratchpad/bench_docs"))


def _patch_db_sinks() -> None:
    """No-op every DB write so dispatch runs extraction but persists nothing."""
    from src.services.extraction import dispatch as d
    from src.services.extraction import persistence as p

    def _noop(*a, **k):
        return None

    # Schema YAML load runs a DB schema-consistency check; skip it (the schema
    # itself loads from the YAML file, no DB needed for extraction).
    from src.services.extraction_v3.yaml_schema import loader as _loader
    _loader._verify_db_consistency = _noop

    d.record_action = _noop
    p.write_raw = lambda *a, **k: 0
    p.write_line_items_raw = _noop
    p.write_discrepancies = _noop
    p.write_provenance = _noop
    p.update_promotion_status = _noop
    # promote() -> not-ok keeps final_status='pending' and skips the
    # _read_persisted_confidence DB read.
    from src.services.extraction import promotion as prom
    prom.promote = lambda *a, **k: {"ok": False, "reason": "bench-noop"}
    # dispatch imported promotion as a name; repoint it too.
    d.promotion.promote = prom.promote


def _download(key: str) -> Path:
    import boto3

    CACHE.mkdir(parents=True, exist_ok=True)
    dest = CACHE / key.replace("/", "__")
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    boto3.client("s3", region_name="eu-west-1").download_file(
        os.environ["S3_BUCKET_NAME"], key, str(dest))
    return dest


def _sig(result: dict) -> str:
    keep = {k: result.get(k) for k in
            ("doc_pk", "n_fields", "line_items", "completeness_status")}
    return hashlib.sha1(json.dumps(keep, sort_keys=True, default=str).encode()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    items = []
    for line in MANIFEST.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, cat = line.partition("|")
        items.append((key.strip(), cat.strip()))

    _patch_db_sinks()
    from src.services.extraction.dispatch import dispatch_document

    # Pre-download all docs (not timed).
    local = {}
    for key, cat in items:
        try:
            local[key] = _download(key)
        except Exception as exc:
            print(f"  download FAIL {key}: {exc}", file=sys.stderr)

    def _run_one(item):
        key, cat = item
        lp = local.get(key)
        if lp is None:
            return {"key": key, "error": "no-local"}
        t0 = time.time()
        try:
            res = dispatch_document(process_monitor_id=None, file_path=str(lp), doc_type=cat)
            el = time.time() - t0
            print(f"  {el:7.2f}s  {_sig(res)[:12]}  {res.get('status'):9} {key}", flush=True)
            return {"key": key, "category": cat, "elapsed_s": round(el, 2),
                    "status": res.get("status"), "signature": _sig(res),
                    "n_fields": res.get("n_fields"), "line_items": res.get("line_items")}
        except Exception as exc:
            el = time.time() - t0
            print(f"  {el:7.2f}s  ERROR  {key}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            return {"key": key, "category": cat, "elapsed_s": round(el, 2),
                    "error": f"{type(exc).__name__}:{exc}"}

    # Warm up models on ONE doc (untimed) so the timed batch reflects warm state.
    print(f"warmup ({items[0][0]}) ...", flush=True)
    _run_one(items[0])

    # Timed concurrent batch — all docs through a {workers}-wide pool.
    print(f"\ntimed batch: {len(items)} docs, {args.workers} workers, "
          f"OLLAMA_BASE_URL={os.environ.get('OLLAMA_BASE_URL')} "
          f"MAX_CONCURRENT={os.environ.get('OLLAMA_MAX_CONCURRENT')}", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        records = list(ex.map(_run_one, items))
    batch_total = round(time.time() - t0, 2)

    ok = [r for r in records if "elapsed_s" in r and "error" not in r]
    summary = {
        "label": args.label,
        "workers": args.workers,
        "ollama_base_url": os.environ.get("OLLAMA_BASE_URL"),
        "ollama_max_concurrent": os.environ.get("OLLAMA_MAX_CONCURRENT"),
        "doc_count": len(items),
        "ok_count": len(ok),
        "batch_total_s": batch_total,
        "mean_doc_s": round(sum(r["elapsed_s"] for r in ok) / len(ok), 2) if ok else None,
        "sum_doc_s": round(sum(r["elapsed_s"] for r in ok), 2) if ok else None,
        "records": records,
    }
    out = OUT_DIR / f"renov_{args.label}.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nbatch_total={batch_total}s  mean_doc={summary['mean_doc_s']}s  "
          f"sum_doc={summary['sum_doc_s']}s  ok={len(ok)}/{len(items)}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
