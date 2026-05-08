"""Seed vendor templates by downloading already-processed production PDFs.

Looks at proc.process_monitor for vendors that aren't covered by the
local-fixture seeder (NEXASPARK, RUBILOGY, etc.), downloads the
corresponding PDFs via DirectExtractionService (which handles the S3
read), parses them, and writes a template into
proc.bp_extraction_template using the filename-derived canonical
supplier name.

Run: python scripts/seed_templates_from_production.py
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

# Force CUDA env consistent with the running service
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from agents.base_agent import AgentNick  # noqa: E402
from services.db import get_conn  # noqa: E402
from services.direct_extraction_service import DirectExtractionService  # noqa: E402
from src.services.extraction_v2.fingerprint import compute_fingerprint  # noqa: E402
from src.services.extraction_v2.template_store import LineItemHints  # noqa: E402
from src.services.extraction_v2.template_store_pg import (  # noqa: E402
    PostgresTemplateStore,
)
from src.services.structural_extractor.parsing import parse  # noqa: E402


# Vendors we want to seed from production. The canonical supplier name
# comes from the filename ("NEXASPARK INV..." → "Nexaspark"). doc_type
# is inferred from the file_path prefix ("invoice/" → Invoice etc.).
TARGET_VENDORS = {
    # match prefix → canonical_supplier_name
    "NEXASPARK": "Nexaspark",
    "RUBILOGY":  "Rubilogy",
}

# Default line-item column hints — same as the most common invoice layout
# in the seed fixtures. The next successful extraction will refine these.
DEFAULT_LINE_ITEM_COLUMNS = {
    "Description": "item_description",
    "Quantity":    "quantity",
    "Unit Price":  "unit_price",
    "Amount":      "line_amount",
}


def _doc_type_from_path(file_path: str) -> str:
    """invoice/foo.pdf → Invoice ; po/foo.pdf → Purchase_Order ; quote/ → Quote."""
    parts = file_path.lower().split("/")
    if "invoice" in parts:
        return "Invoice"
    if "po" in parts or "purchase_order" in parts:
        return "Purchase_Order"
    if "quote" in parts or "quotes" in parts:
        return "Quote"
    return "Invoice"  # default


def _vendor_for(file_path: str) -> tuple[str, str] | None:
    base = os.path.basename(file_path).upper()
    for prefix, canonical in TARGET_VENDORS.items():
        if base.startswith(prefix) or f" {prefix} " in f" {base} ":
            return prefix, canonical
    return None


def main() -> int:
    # Step 1 — discover one representative file_path per target vendor
    discovered: dict[str, str] = {}  # canonical_name → file_path
    with get_conn() as conn:
        with conn.cursor() as cur:
            for prefix, canonical in TARGET_VENDORS.items():
                cur.execute(
                    """SELECT file_path FROM proc.process_monitor
                        WHERE file_path ILIKE %s
                        ORDER BY id DESC LIMIT 1""",
                    (f"%{prefix}%",),
                )
                row = cur.fetchone()
                if row:
                    discovered[canonical] = row[0]
                    print(f"  [found] {canonical} → {row[0]}")
                else:
                    print(f"  [miss]  {canonical}: no process_monitor row")

    if not discovered:
        print("No target vendors found in process_monitor — nothing to seed.")
        return 1

    # Step 2 — download bytes, parse, fingerprint, seed
    nick = AgentNick()
    svc = DirectExtractionService(nick)
    store = PostgresTemplateStore(get_conn)

    seeded = 0
    print()
    for canonical, file_path in discovered.items():
        try:
            _text, file_bytes = svc._get_document_text(file_path)
            if not file_bytes:
                print(f"  [skip] {canonical}: no bytes returned for {file_path}")
                continue
            doc = parse(file_bytes, os.path.basename(file_path))
            fp = compute_fingerprint(doc)
            doc_type = _doc_type_from_path(file_path)

            # Write supplier_name as a correction (correction_count > 0
            # protects from auto-learn overwrite).
            store.record_correction(
                fingerprint=fp,
                field="supplier_name",
                value=canonical,
                confidence=0.99,
                doc_type=doc_type,
                vendor_name=canonical,
            )
            # Attach default line-item hints — refined by future
            # successful extractions.
            store.record_line_item_hints(
                fp,
                LineItemHints(
                    header_anchors=list(DEFAULT_LINE_ITEM_COLUMNS.keys()),
                    column_map=DEFAULT_LINE_ITEM_COLUMNS,
                    expected_min_rows=1,
                ),
                doc_type=doc_type, vendor_name=canonical,
            )
            print(f"  [seeded] {canonical} ({doc_type}) "
                  f"fp={fp[:8]} from {file_path}")
            seeded += 1
        except Exception as exc:
            print(f"  [error] {canonical}: {exc}")

    # Step 3 — verification
    print(f"\n=== verification ===")
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT vendor_name, doc_type, correction_count
                     FROM proc.bp_extraction_template
                    WHERE vendor_name = ANY(%s)
                    ORDER BY vendor_name""",
                (list(TARGET_VENDORS.values()),),
            )
            for row in cur.fetchall():
                print(f"  [OK] {row[0]:<14} {row[1]:<16} corrections={row[2]}")

    print(f"\n=== summary ===")
    print(f"  seeded:  {seeded}")
    return 0 if seeded == len(discovered) else 2


if __name__ == "__main__":
    sys.exit(main())
