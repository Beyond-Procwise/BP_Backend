"""One-shot re-dispatch for documents with missing or partial line items.

Locates every _stg row whose line_items are absent or numerically empty,
finds the matching process_monitor file_path, and re-runs
extraction.dispatch_document on each. Idempotent: the renovation
promote() does DELETE-then-INSERT for line items keyed on doc_pk, so
re-running is safe and replaces the stale rows.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.services.db import get_conn
from src.services.extraction.dispatch import dispatch_document

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("redispatch_affected_docs")


_DOC_TYPE_TO_CATEGORY = {
    "invoice": "invoice",
    "purchase_order": "po",
    "quote": "quote",
}

_DOC_TYPE_TO_PK_COLS = {
    "invoice": ("proc.bp_invoice_stg", "invoice_id", "proc.bp_invoice_line_items_stg",
                "invoice_id", "line_amount"),
    "purchase_order": ("proc.bp_purchase_order_stg", "po_id", "proc.bp_po_line_items_stg",
                       "po_id", "line_total"),
    "quote": ("proc.bp_quote_stg", "quote_id", "proc.bp_quote_line_items_stg",
              "quote_id", "line_total"),
}


def _bad_doc_pks() -> dict[str, list[str]]:
    """Return {doc_type: [doc_pk, ...]} of all docs whose lines need a refresh."""
    out: dict[str, list[str]] = {}
    with get_conn() as conn:
        cur = conn.cursor()
        for dt, (hdr, pk_col, ln, ln_pk_col, amount_col) in _DOC_TYPE_TO_PK_COLS.items():
            cur.execute(f"""
                SELECT pk FROM (
                    SELECT {pk_col} pk FROM {hdr}
                    WHERE NOT EXISTS (
                        SELECT 1 FROM {ln} WHERE {ln}.{ln_pk_col} = {hdr}.{pk_col}
                    )
                    UNION
                    SELECT {ln_pk_col} pk FROM {ln}
                    WHERE item_description IS NULL OR item_description=''
                       OR quantity IS NULL OR unit_price IS NULL OR {amount_col} IS NULL
                ) t ORDER BY pk
            """)
            out[dt] = [r[0] for r in cur.fetchall()]
    return out


def _file_path_for_doc(doc_type: str, doc_pk: str) -> str | None:
    """Find the most recent process_monitor.file_path containing this PK.

    Matches the PK as a substring (case-insensitive). For docs like
    `INV-2025-055`, the filename contains `INV-2025-055` verbatim.
    """
    category = _DOC_TYPE_TO_CATEGORY[doc_type]
    with get_conn() as conn:
        cur = conn.cursor()
        # Two-tier match: prefer ILIKE on '%<pk>%' as a single token;
        # fall back to stripping non-alnum from the pk and trying again.
        cur.execute(
            "SELECT file_path FROM proc.process_monitor "
            "WHERE category=%s AND file_path ILIKE %s "
            "ORDER BY id DESC LIMIT 1",
            (category, f"%{doc_pk}%"),
        )
        row = cur.fetchone()
        if row:
            return row[0]
        bare = "".join(ch for ch in doc_pk if ch.isalnum())
        if bare and bare != doc_pk:
            cur.execute(
                "SELECT file_path FROM proc.process_monitor "
                "WHERE category=%s AND file_path ILIKE %s "
                "ORDER BY id DESC LIMIT 1",
                (category, f"%{bare}%"),
            )
            row = cur.fetchone()
            if row:
                return row[0]
    return None


def _process_monitor_id(file_path: str) -> int | None:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT id FROM proc.process_monitor WHERE file_path=%s "
            "ORDER BY id DESC LIMIT 1",
            (file_path,),
        )
        row = cur.fetchone()
        return int(row[0]) if row else None


def main() -> int:
    bad = _bad_doc_pks()
    total = sum(len(v) for v in bad.values())
    log.info("found %d affected docs across %d types", total, len(bad))
    ok, fail = 0, 0
    for dt, pks in bad.items():
        for pk in pks:
            fp = _file_path_for_doc(dt, pk)
            if not fp:
                log.warning("[%s/%s] no file_path in process_monitor — skipping", dt, pk)
                fail += 1
                continue
            pmid = _process_monitor_id(fp)
            try:
                result = dispatch_document(
                    process_monitor_id=pmid,
                    file_path=fp,
                    doc_type=dt,
                )
                log.info("[%s/%s] -> %s (lines=%d, missing=%s)",
                         dt, pk, result.get("status"),
                         result.get("line_items", 0),
                         result.get("missing_required") or "-")
                if result.get("status") in ("promoted", "pending"):
                    ok += 1
                else:
                    fail += 1
            except Exception as exc:  # noqa: BLE001
                log.exception("[%s/%s] dispatch failed: %s", dt, pk, exc)
                fail += 1
    log.info("Done. ok=%d fail=%d", ok, fail)
    return 0 if fail == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
