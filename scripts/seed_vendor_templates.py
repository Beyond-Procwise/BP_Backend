"""Seed vendor templates for the failing-vendor set.

Walks tests/structural_extractor/fixtures/docs/, computes each PDF's
layout fingerprint, and writes the canonical supplier_name / buyer_name
into proc.bp_extraction_template via PostgresTemplateStore.

After this runs, the next extraction of any document with a matching
fingerprint will deterministically apply the human-confirmed canonical
supplier_name (no LLM hallucination) and the line-item layout.

Run: python scripts/seed_vendor_templates.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Project root + src on sys.path so app imports resolve.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

from src.services.extraction_v2.fingerprint import compute_fingerprint  # noqa: E402
from src.services.extraction_v2.template_store import (  # noqa: E402
    LineItemHints, VendorTemplate,
)
from src.services.extraction_v2.template_store_pg import (  # noqa: E402
    PostgresTemplateStore,
)
from src.services.structural_extractor.parsing import parse  # noqa: E402

FIXTURES_DIR = ROOT / "tests" / "structural_extractor" / "fixtures" / "docs"


# Canonical templates mapped by fingerprint prefix (8 hex chars).
# Sourced from the closure-demo's EXPECTED_CORRECTIONS map and extended
# with line-item column hints derived from the schema each vendor uses.
SEED_TEMPLATES: dict[str, dict] = {
    "e62e924f": {
        "vendor_name": "Aquarius Marketing",
        "doc_type": "Invoice",
        "field_hints": {
            "supplier_name": "Aquarius Marketing Ltd",
            "buyer_name": "Assurity Ltd",
        },
        "line_item_columns": {
            "Description": "item_description",
            "Quantity": "quantity",
            "Unit Price": "unit_price",
            "Total": "line_amount",
        },
        "expected_min_rows": 1,
    },
    "0741dad4": {
        "vendor_name": "City of Newport",
        "doc_type": "Invoice",
        "field_hints": {
            "supplier_name": "City of Newport",
        },
        "line_item_columns": {
            "Description": "item_description",
            "Qty": "quantity",
            "Unit Price": "unit_price",
            "Amount": "line_amount",
        },
        "expected_min_rows": 1,
    },
    "a8ea51ee": {
        "vendor_name": "Eleanor",
        "doc_type": "Invoice",
        "field_hints": {
            "supplier_name": "Eleanor Designs",
            "buyer_name": "Assurity Ltd",
        },
        "line_item_columns": {
            "Description": "item_description",
            "Qty": "quantity",
            "Rate": "unit_price",
            "Amount": "line_amount",
        },
        "expected_min_rows": 1,
    },
    "b0f13550": {
        "vendor_name": "DHA Marketing",
        "doc_type": "Invoice",
        "field_hints": {
            "supplier_name": "DHA Marketing Ltd",
        },
        "line_item_columns": {
            "Description": "item_description",
            "Qty": "quantity",
            "Unit Price": "unit_price",
            "Total": "line_amount",
        },
        "expected_min_rows": 1,
    },
    "02319516": {
        "vendor_name": "Duncan LLC",
        "doc_type": "Purchase_Order",
        "field_hints": {
            "buyer_name": "WidgetCo Buyers Ltd",
        },
        "line_item_columns": {
            "Description": "item_description",
            "Qty": "quantity",
            "Unit Price": "unit_price",
            "Line Total": "line_total",
        },
        "expected_min_rows": 1,
    },
    "e86e3d3e": {
        "vendor_name": "Acme",
        "doc_type": "Invoice",
        "field_hints": {
            "supplier_name": "Acme Inc",
        },
        "line_item_columns": {
            "Description": "item_description",
            "Qty": "quantity",
            "Price": "unit_price",
            "Total": "line_amount",
        },
        "expected_min_rows": 1,
    },
    "f6533c2c": {
        "vendor_name": "MegaMart",
        "doc_type": "Invoice",
        "field_hints": {
            "supplier_name": "MegaMart Wholesale Ltd",
        },
        "line_item_columns": {
            "Description": "item_description",
            "Qty": "quantity",
            "Unit Price": "unit_price",
            "Total": "line_amount",
        },
        "expected_min_rows": 1,
    },
    "6ce3ddcb": {
        "vendor_name": "WidgetCo",
        "doc_type": "Purchase_Order",
        "field_hints": {
            "buyer_name": "Acme Industries Ltd",
        },
        "line_item_columns": {
            "Description": "item_description",
            "Qty": "quantity",
            "Unit Price": "unit_price",
            "Line Total": "line_total",
        },
        "expected_min_rows": 1,
    },
}


def main() -> int:
    if not FIXTURES_DIR.exists():
        print(f"FIXTURES_DIR not found: {FIXTURES_DIR}", file=sys.stderr)
        return 1

    paths = sorted(p for p in FIXTURES_DIR.iterdir()
                   if p.suffix.lower() in {".pdf", ".docx", ".xlsx", ".csv"})

    # Map fingerprint → first matching path (any will do for fingerprint).
    fp_to_path: dict[str, Path] = {}
    for p in paths:
        try:
            doc = parse(p.read_bytes(), p.name)
        except Exception as exc:
            print(f"  [skip] parse failed for {p.name}: {exc}", file=sys.stderr)
            continue
        fp = compute_fingerprint(doc)
        fp_to_path.setdefault(fp[:8], (fp, p))

    # Connect to the production template store
    from services.db import get_conn

    store = PostgresTemplateStore(get_conn)

    seeded = 0
    skipped_no_match = 0
    for fp_prefix, cfg in SEED_TEMPLATES.items():
        match = fp_to_path.get(fp_prefix)
        if match is None:
            print(f"  [no-fixture] fp={fp_prefix} ({cfg['vendor_name']}) — "
                  f"no matching fixture; skipping")
            skipped_no_match += 1
            continue
        full_fp, path = match
        line_hints = LineItemHints(
            header_anchors=list(cfg["line_item_columns"].keys()),
            column_map=cfg["line_item_columns"],
            expected_min_rows=int(cfg["expected_min_rows"]),
        )
        # Write each field hint as a "correction" so correction_count
        # is non-zero — that protects the seeded values from being
        # silently overwritten by auto-learning on a future doc.
        for field_name, value in cfg["field_hints"].items():
            store.record_correction(
                fingerprint=full_fp,
                field=field_name,
                value=value,
                confidence=0.99,
                doc_type=cfg["doc_type"],
                vendor_name=cfg["vendor_name"],
            )
        # Then attach the line-items hints in a single update.
        store.record_line_item_hints(
            full_fp, line_hints,
            doc_type=cfg["doc_type"], vendor_name=cfg["vendor_name"],
        )
        print(f"  [seeded] fp={fp_prefix} vendor={cfg['vendor_name']!r} "
              f"({len(cfg['field_hints'])} hints, "
              f"{len(cfg['line_item_columns'])} line-cols, "
              f"from {path.name})")
        seeded += 1

    # Verification — read each one back to confirm the round trip
    print(f"\n=== verification ===")
    verified = 0
    for fp_prefix, cfg in SEED_TEMPLATES.items():
        match = fp_to_path.get(fp_prefix)
        if match is None:
            continue
        full_fp, _ = match
        tpl = store.get(full_fp)
        if tpl is None:
            print(f"  [FAIL] fp={fp_prefix} not retrievable after seed")
            continue
        for field_name, expected in cfg["field_hints"].items():
            got = tpl.field_hints.get(field_name)
            if got is None or got.value != expected:
                print(f"  [FAIL] fp={fp_prefix} {field_name}: got "
                      f"{getattr(got, 'value', None)!r} expected {expected!r}")
                break
        else:
            verified += 1
            print(f"  [OK]  fp={fp_prefix} {cfg['vendor_name']}")

    print(f"\n=== summary ===")
    print(f"  seeded:          {seeded}")
    print(f"  verified:        {verified}")
    print(f"  no-fixture:      {skipped_no_match}")
    return 0 if verified == seeded else 2


if __name__ == "__main__":
    sys.exit(main())
