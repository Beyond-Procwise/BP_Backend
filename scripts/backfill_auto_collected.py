"""Bootstrap src/data/training/auto_collected_examples.jsonl with the
gold-standard extractions sitting in `_stg` right now.

Why this exists: process_monitor_watcher._collect_training_example is the
mechanism that should accumulate verified extractions, but it never
recorded any of today's renovation-pipeline runs (compound effect of a
wrong pk_map and a missing _source_text field on dispatch's result).
The DB has the verified outputs and the source PDFs are still in S3, so
we reparse each doc once to recover its full_text and emit a clean
training example per (doc_type, doc_pk).

Idempotent: existing (doc_type, pk) entries with non-empty source_text
are skipped; entries with empty source_text are *upgraded* with the
fresh source text.

Runs CPU-only (skips Qwen judge load) for speed and to avoid contending
with the procwise live extractor's GPU memory.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
# Keep parser CPU-only — avoids fighting the live extractor for GPU memory.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

from src.services.db import get_conn
from src.services.extraction.parser import parse as parse_doc

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("backfill_auto_collected")

OUTPUT = ROOT / "src" / "data" / "training" / "auto_collected_examples.jsonl"

_STG_TABLES = {
    "invoice": ("proc.bp_invoice_stg", "invoice_id",
                "proc.bp_invoice_line_items_stg", "invoice_id"),
    "purchase_order": ("proc.bp_purchase_order_stg", "po_id",
                       "proc.bp_po_line_items_stg", "po_id"),
    "quote": ("proc.bp_quote_stg", "quote_id",
              "proc.bp_quote_line_items_stg", "quote_id"),
}

_CATEGORY_TO_DOC_TYPE = {
    "invoice": "invoice",
    "po": "purchase_order",
    "purchase_order": "purchase_order",
    "quote": "quote",
}

_AUDIT_SKIP = {
    "created_date", "created_by", "last_modified_date", "last_modified_by",
    "confidence_score", "needs_review", "ai_flag_required",
}


def _doc_index() -> list[tuple[str, str, str, float]]:
    """Return [(doc_type, doc_pk, file_path, confidence)] for everything
    in _stg that we can match back to a process_monitor file_path."""
    out: list[tuple[str, str, str, float]] = []
    with get_conn() as conn:
        cur = conn.cursor()
        for category, doc_type in _CATEGORY_TO_DOC_TYPE.items():
            if doc_type not in _STG_TABLES:
                continue
            hdr_tbl, pk_col, _, _ = _STG_TABLES[doc_type]
            cur.execute(
                f"SELECT {pk_col}, confidence_score FROM {hdr_tbl}",
            )
            for pk, conf in cur.fetchall():
                # Best-effort file_path lookup. Two strategies:
                #  1. ILIKE %pk% — works when the filename contains the pk
                #  2. ILIKE %<bare>% (alnum-only pk) — handles "INV-2025-056" → "INV2025056"
                cur.execute(
                    "SELECT file_path FROM proc.process_monitor "
                    "WHERE category = %s AND file_path ILIKE %s "
                    "ORDER BY id DESC LIMIT 1",
                    (category, f"%{pk}%"),
                )
                row = cur.fetchone()
                if not row:
                    bare = "".join(ch for ch in pk if ch.isalnum())
                    if bare and bare != pk:
                        cur.execute(
                            "SELECT file_path FROM proc.process_monitor "
                            "WHERE category = %s AND file_path ILIKE %s "
                            "ORDER BY id DESC LIMIT 1",
                            (category, f"%{bare}%"),
                        )
                        row = cur.fetchone()
                if not row:
                    log.warning("no file_path match for %s/%s", doc_type, pk)
                    continue
                out.append((doc_type, pk, row[0], float(conf) if conf is not None else 0.0))
    return out


def _existing_examples() -> dict[tuple[str, str], dict]:
    """Index existing auto_collected entries by (doc_type, pk)."""
    out: dict[tuple[str, str], dict] = {}
    if not OUTPUT.exists():
        return out
    with open(OUTPUT) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (ex.get("doc_type", ""), ex.get("pk", ""))
            out[key] = ex
    return out


def _header_and_lines(doc_type: str, pk: str) -> tuple[dict, list[dict]] | None:
    """Read the verified _stg row + its line items as JSON."""
    hdr_tbl, pk_col, ln_tbl, ln_fk = _STG_TABLES[doc_type]
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(f"SELECT row_to_json(t) FROM {hdr_tbl} t WHERE {pk_col} = %s", (pk,))
        row = cur.fetchone()
        if not row:
            return None
        header = {k: v for k, v in row[0].items() if k not in _AUDIT_SKIP and v is not None}
        # We don't ORDER BY here — line ordering already matches the
        # promote() loop's insertion order, and json columns aren't
        # orderable in Postgres without an explicit cast.
        cur.execute(f"SELECT row_to_json(t) FROM {ln_tbl} t WHERE {ln_fk} = %s", (pk,))
        lines = [
            {k: v for k, v in r[0].items() if k not in _AUDIT_SKIP and v is not None}
            for r in cur.fetchall()
        ]
    return header, lines


def _parse_full_text(file_path: str) -> str:
    """Re-parse the document to recover its full_text. We cap at 6000
    chars to match the watcher's collector contract."""
    parsed = parse_doc(file_path)
    return (parsed.full_text or "")[:6000]


def main() -> int:
    docs = _doc_index()
    log.info("found %d candidate _stg rows for backfill", len(docs))
    existing = _existing_examples()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    new_entries: list[dict] = []
    upgraded = 0
    skipped = 0

    # Rewrite the file with upgrades + appended new entries — cleaner than
    # an in-place patch because dedup-by-line is fragile and we want the
    # output deterministic by sort order.
    seen: set[tuple[str, str]] = set()
    output_entries: list[dict] = []

    # First emit existing entries (preserved or upgraded).
    for key, ex in existing.items():
        if not ex.get("source_text") and any(
            key == (dt, pk) for dt, pk, _, _ in docs
        ):
            # Upgrade in-place: add source_text + refresh extracted from DB.
            doc_type, pk = key
            fp = next(fp for dt, p, fp, _ in docs if (dt, p) == key)
            try:
                full_text = _parse_full_text(fp)
            except Exception as exc:  # noqa: BLE001
                log.warning("[%s/%s] reparse failed: %s — keeping original entry",
                            doc_type, pk, exc)
                output_entries.append(ex)
                seen.add(key)
                continue
            hl = _header_and_lines(doc_type, pk)
            if not hl:
                output_entries.append(ex)
                seen.add(key)
                continue
            header, lines = hl
            ex = dict(ex)
            ex["source_text"] = full_text
            ex["extracted"] = {"header": header, "line_items": lines}
            ex["timestamp"] = datetime.now(timezone.utc).isoformat()
            ex["upgraded_at"] = ex["timestamp"]
            upgraded += 1
        output_entries.append(ex)
        seen.add(key)

    # Then append new entries from docs that aren't in existing.
    for doc_type, pk, fp, conf in docs:
        if (doc_type, pk) in seen:
            skipped += 1
            continue
        try:
            full_text = _parse_full_text(fp)
        except Exception as exc:  # noqa: BLE001
            log.warning("[%s/%s] reparse failed: %s — skipping", doc_type, pk, exc)
            continue
        hl = _header_and_lines(doc_type, pk)
        if not hl:
            log.warning("[%s/%s] _stg row vanished mid-flight — skipping", doc_type, pk)
            continue
        header, lines = hl
        ex = {
            "doc_type": doc_type,
            "pk": pk,
            "file_path": fp,
            "confidence": conf / 100.0 if conf else 0.0,
            "source_text": full_text,
            "extracted": {"header": header, "line_items": lines},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        output_entries.append(ex)
        new_entries.append(ex)

    log.info("output: %d total entries (existing kept=%d, upgraded=%d, new=%d, "
             "already-present skipped=%d)",
             len(output_entries), len(existing) - upgraded, upgraded,
             len(new_entries), skipped)

    with open(OUTPUT, "w") as f:
        for ex in output_entries:
            f.write(json.dumps(ex, default=str) + "\n")
    log.info("wrote %s", OUTPUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
