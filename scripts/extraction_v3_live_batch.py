#!/usr/bin/env python3
"""Live batch extraction harness.

Iterates over real documents in a directory, runs PipelineV3 (with fast-failing
Ollama so the live procwise service's contention doesn't block us), and writes
one JSONL line per document to an append-mode output file. Resumable: re-runs
skip already-processed docs.

Usage:
  OLLAMA_TIMEOUT=10 .venv/bin/python scripts/extraction_v3_live_batch.py \
      --dir "/home/muthu/Downloads/OneDrive_1_6-20-2025/Invoice Data" \
      --doc-type invoice \
      --max 10 \
      --out artifacts/log_monitor/v3_live_invoices.jsonl
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OLLAMA_TIMEOUT", "10")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
logging.basicConfig(level=logging.WARNING)
for noisy in ("docling", "rapidocr", "RapidOCR", "paddleocr", "transformers"):
    logging.getLogger(noisy).setLevel(logging.ERROR)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.services.extraction_v3.pipeline import PipelineV3
from src.services.extraction_v3.parsers.router import parse as parse_doc


def list_docs(directory: Path, max_docs: int | None) -> list[Path]:
    extensions = {".pdf", ".docx", ".png", ".jpg", ".jpeg"}
    docs = sorted(
        p for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions and not p.name.startswith(".")
    )
    if max_docs:
        docs = docs[:max_docs]
    return docs


def already_processed(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    seen = set()
    for line in out_path.read_text().splitlines():
        try:
            d = json.loads(line)
            seen.add(d.get("source_path", ""))
        except json.JSONDecodeError:
            continue
    return seen


def run_one(pipeline: PipelineV3, path: Path, doc_type: str) -> dict:
    out: dict = {"source_path": str(path), "doc_type": doc_type, "name": path.name}
    t0 = time.time()
    try:
        parsed = parse_doc(path)
        out["parser_backend"] = parsed.parser_backend
        out["parser_confidence"] = round(parsed.parser_confidence, 3)
        out["full_text_len"] = len(parsed.full_text)

        result = pipeline.run(path, doc_type)

        # Anti-hallucination check
        hallucinations = [
            {"field": cf.field_path, "value": cf.value, "evidence": cf.evidence_text}
            for cf in result.committed
            if cf.evidence_text not in parsed.full_text
        ]

        out["committed"] = [
            {"field": cf.field_path, "value": cf.value, "evidence": cf.evidence_text,
             "model": cf.model, "conf": round(cf.final_confidence, 3)}
            for cf in result.committed
        ]
        out["residuals"] = [
            {"field": r.field_path, "reason": r.reason} for r in result.residuals
        ]
        out["doc_pk"] = result.doc_pk
        out["judge_calls"] = result.judge_calls
        out["hallucinations"] = hallucinations
        out["latency_s"] = round(time.time() - t0, 2)
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
        out["latency_s"] = round(time.time() - t0, 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="Directory of documents")
    ap.add_argument("--doc-type", required=True,
                    choices=["invoice", "purchase_order", "quote", "contract"])
    ap.add_argument("--max", type=int, default=None,
                    help="Max docs to process (default: all)")
    ap.add_argument("--out", required=True, help="Output JSONL path (append mode)")
    args = ap.parse_args()

    directory = Path(args.dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    docs = list_docs(directory, args.max)
    seen = already_processed(out_path)
    pending = [d for d in docs if str(d) not in seen]
    print(f"directory: {directory}")
    print(f"docs found: {len(docs)} | already processed: {len(seen)} | pending: {len(pending)}")

    if not pending:
        print("Nothing to do.")
        return

    print("Initializing PipelineV3...")
    pipeline = PipelineV3()
    print(f"Ready. Processing {len(pending)} docs (writing to {out_path})...\n")

    fp = open(out_path, "a")
    try:
        for i, p in enumerate(pending, 1):
            print(f"[{i}/{len(pending)}] {p.name}", flush=True)
            result = run_one(pipeline, p, args.doc_type)

            # Brief stdout summary
            if "error" in result:
                print(f"  ERROR: {result['error']} ({result['latency_s']}s)")
            else:
                halluc = len(result.get("hallucinations") or [])
                print(f"  pk={result.get('doc_pk')!r:30s} commit={len(result['committed']):2d} "
                      f"resid={len(result['residuals']):2d} judge={result['judge_calls']:2d} "
                      f"halluc={halluc} parser={result['parser_backend']} "
                      f"({result['latency_s']}s)")
                if halluc > 0:
                    for h in result["hallucinations"]:
                        print(f"    *** HALLUC: {h['field']} = {h['value']!r}")

            fp.write(json.dumps(result, default=str) + "\n")
            fp.flush()
    finally:
        fp.close()

    print(f"\nDone. Output at: {out_path}")


if __name__ == "__main__":
    main()
