"""Live end-to-end demonstration: template store closes the gap.

Two-pass run over every fixture in tests/structural_extractor/fixtures/docs:

    Pass 1 (cold):   no templates yet — measures pure-algorithm coverage.
    Bootstrap:       once per unique fingerprint, register the corrections
                     a human reviewer would have entered (drawn from the
                     EXPECTED dict defined here — these are the ground-
                     truth supplier/buyer names).
    Pass 2 (warm):   re-run against the same set with the templates in
                     place. Reports the lift.

Run: python tests/extraction_v2_live_closure.py
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))
os.chdir(project_root)

from src.services.extraction_v2.fingerprint import compute_fingerprint
from src.services.extraction_v2.pipeline import ExtractionPipelineV2
from src.services.extraction_v2.template_store import InMemoryTemplateStore
from src.services.extraction_v2.types import IsoDate
from src.services.structural_extractor.parsing import parse


FIXTURES_DIR = (
    project_root / "tests" / "structural_extractor" / "fixtures" / "docs"
)


# Per-fingerprint corrections that a human onboarder would record once
# for each new vendor layout. Keys are fingerprint prefixes (8 hex chars)
# discovered by the cold pass; values are the (vendor_name, hints) the
# human would enter via /vendors/onboard.
EXPECTED_CORRECTIONS: dict[str, dict] = {
    "e62e924f": {
        "vendor_name": "Aquarius Marketing",
        "hints": {
            "supplier_name": "Aquarius Marketing Ltd",
            "buyer_name": "Assurity Ltd",
        },
    },
    "0741dad4": {
        "vendor_name": "City of Newport",
        "hints": {
            "supplier_name": "City of Newport",
            "invoice_id": None,  # already commits via structural+filename
        },
    },
    "a8ea51ee": {
        "vendor_name": "Eleanor",
        "hints": {
            "supplier_name": "Eleanor Designs",
            "buyer_name": "Assurity Ltd",
        },
    },
    "b0f13550": {
        "vendor_name": "DHA Marketing",
        "hints": {
            "invoice_id": "DHA-2025-143",
            "supplier_name": "DHA Marketing Ltd",
        },
    },
    "02319516": {
        "vendor_name": "Duncan LLC",
        "hints": {
            "buyer_name": "WidgetCo Buyers Ltd",
            "expected_delivery_date": IsoDate("2024-01-31"),
        },
    },
    "e86e3d3e": {
        "vendor_name": "Acme",
        "hints": {
            "invoice_id": "ACME-INV-001",
            "supplier_name": "Acme Inc",
        },
    },
    "f6533c2c": {
        "vendor_name": "MegaMart",
        "hints": {
            "invoice_id": "MEGA-2024-001",
            "supplier_name": "MegaMart Wholesale Ltd",
        },
    },
    "6ce3ddcb": {
        "vendor_name": "WidgetCo",
        "hints": {
            "buyer_name": "Acme Industries Ltd",
            "expected_delivery_date": IsoDate("2024-07-15"),
        },
    },
    # d32484d2 = line_items_bulk.csv — genuinely no header data; cannot
    # be onboarded. The pipeline correctly abstains on all fields.
}


def _classify_filename(name: str) -> str:
    n = name.upper()
    if "PO" in n and "INV" not in n:
        return "Purchase_Order"
    if "QUT" in n or "QUOTE" in n or "QTE" in n:
        return "Quote"
    return "Invoice"


def run_pass(label: str, store, paths: list[Path]) -> dict:
    pipeline = ExtractionPipelineV2(template_store=store)
    counts = {"committed": 0, "residual": 0, "abstained_fp": defaultdict(int)}
    fp_seen: dict[str, list[str]] = defaultdict(list)
    print(f"\n{'='*72}")
    print(f" {label}")
    print(f"{'='*72}")
    for p in paths:
        try:
            doc = parse(p.read_bytes(), p.name)
        except Exception as e:
            print(f"  [PARSE ERR] {p.name}: {e}")
            continue
        doc_type = _classify_filename(p.name)
        result = pipeline.extract(doc, doc_type)
        counts["committed"] += len(result.committed)
        counts["residual"] += len(result.residuals)
        fp_seen[result.fingerprint[:8]].append(p.name)
        print(f"  [{doc_type}] {p.name}: "
              f"committed={len(result.committed)} "
              f"residual={len(result.residuals)} "
              f"fp={result.fingerprint[:8]} "
              f"template={'Y' if result.template_used else 'N'}")
    counts["fp_seen"] = dict(fp_seen)
    return counts


def main():
    paths = sorted(p for p in FIXTURES_DIR.iterdir()
                   if p.suffix.lower() in {".pdf", ".docx", ".xlsx", ".csv"})

    store = InMemoryTemplateStore()

    # Pass 1 — cold
    cold = run_pass("PASS 1: COLD (no templates)", store, paths)

    # Bootstrap: register human corrections once per fingerprint
    print(f"\n{'-'*72}")
    print(" BOOTSTRAP (simulating human reviewer corrections)")
    print(f"{'-'*72}")
    for fp_prefix, fps in cold["fp_seen"].items():
        cfg = EXPECTED_CORRECTIONS.get(fp_prefix)
        if cfg is None:
            print(f"  fp={fp_prefix}: no correction registered "
                  f"({len(fps)} doc{'s' if len(fps)>1 else ''})")
            continue
        for field, value in cfg["hints"].items():
            if value is None:
                continue
            full_fp = next((p for p in fps if p), None)
            full_fp_hash = next(iter(
                fp for fp in [_full_fingerprint(paths, fp_prefix)]
                if fp is not None
            ), fp_prefix)
            store.record_correction(
                fingerprint=full_fp_hash,
                field=field,
                value=value,
                confidence=0.95,
                doc_type="Invoice",
                vendor_name=cfg["vendor_name"],
            )
        print(f"  fp={fp_prefix} → vendor={cfg['vendor_name']!r}, "
              f"{len(cfg['hints'])} hint(s)")

    # Pass 2 — warm
    warm = run_pass("PASS 2: WARM (templates loaded)", store, paths)

    # Lift report
    print(f"\n{'='*72}")
    print(" LIFT")
    print(f"{'='*72}")
    cold_total = cold["committed"] + cold["residual"]
    warm_total = warm["committed"] + warm["residual"]
    cold_pct = 100 * cold["committed"] / max(1, cold_total)
    warm_pct = 100 * warm["committed"] / max(1, warm_total)
    print(f"  Cold:  {cold['committed']}/{cold_total} ({cold_pct:.1f}%)")
    print(f"  Warm:  {warm['committed']}/{warm_total} ({warm_pct:.1f}%)")
    print(f"  Lift:  +{warm['committed'] - cold['committed']} fields, "
          f"{warm_pct - cold_pct:+.1f}pp")


def _full_fingerprint(paths: list[Path], prefix: str) -> str | None:
    """Reverse-lookup full fingerprint hex from its 8-char prefix."""
    for p in paths:
        try:
            doc = parse(p.read_bytes(), p.name)
        except Exception:
            continue
        fp = compute_fingerprint(doc)
        if fp.startswith(prefix):
            return fp
    return None


if __name__ == "__main__":
    main()
