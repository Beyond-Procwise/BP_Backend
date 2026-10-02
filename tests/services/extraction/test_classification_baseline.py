"""Today's classification outcome for every document whose parsed text is stored.

The contract-structures work (specs/2026-10-02-contract-structures-design.md)
adds two structures and a stand-down rule. Its first promise is that no document
which classifies correctly today classifies differently afterwards. That is a
measurement, not an assertion, so the outcome is captured here as a fixture and
compared on every run.

103 raw rows under documents/ carry a parser snapshot, covering 53 distinct
documents (50 agreed, 3 declared_only); the same document has several raw rows
from repeated extractions, and the newest row per source_file is the one read.
Probe rows outside documents/ are excluded (see _stored_documents).

Live-only. Run with:
    set -a && . ./.env && set +a
    CUDA_VISIBLE_DEVICES="" PROCWISE_TEST_LIVE_DB=1 ./venv/bin/python -m pytest \
        tests/services/extraction/test_classification_baseline.py -v

To re-capture after an INTENDED change, and only then:
    CUDA_VISIBLE_DEVICES="" PROCWISE_TEST_LIVE_DB=1 ./venv/bin/python \
        tests/services/extraction/test_classification_baseline.py --recapture
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

_LIVE = os.environ.get("PROCWISE_TEST_LIVE_DB", "").strip().lower() in ("1", "true", "yes", "on")
pytestmark = pytest.mark.skipif(not _LIVE, reason="needs PROCWISE_TEST_LIVE_DB=1")

BASELINE = Path(__file__).resolve().parents[2] / "fixtures" / "contract_structures" / "classification_baseline.json"

#: The uploader's declared concept, derived from the S3 key's zone folder. The
#: live path declares it from proc.process_monitor.category; this reproduces the
#: same answer for a stored document without needing the monitor row.
_ZONE_TO_CONCEPT = (
    ("/quote/", "doctype.quote"),
    ("/invoice/", "doctype.invoice"),
    ("/po/", "doctype.order"),
    ("/purchase", "doctype.order"),
    ("/contract", "doctype.contract_unspecified"),
)

_RAW_TABLES = ("bp_quote_raw", "bp_invoice_raw", "bp_purchase_order_raw", "bp_contract_raw")


def declared_concept_for(source_file: str) -> str | None:
    low = (source_file or "").lower()
    for seg, code in _ZONE_TO_CONCEPT:
        if seg in low:
            return code
    return None


def _stored_documents() -> dict[str, str]:
    """source_file -> full_text, taking the NEWEST raw row per document."""
    from src.services.db import get_conn

    out: dict[str, str] = {}
    with get_conn() as conn:
        cur = conn.cursor()
        for table in _RAW_TABLES:
            cur.execute(
                f"""SELECT source_file, parser_snapshot
                      FROM proc.{table}
                     WHERE parser_snapshot IS NOT NULL
                     ORDER BY raw_id ASC"""
            )
            for source_file, snapshot in cur.fetchall():
                snap = snapshot if isinstance(snapshot, dict) else json.loads(snapshot)
                text = (snap or {}).get("full_text") or ""
                # Corpus documents are S3 keys of the form documents/<zone>/<file>;
                # an absolute path or a scratchpad path is a probe, not a document.
                # A genuine document stored under a different prefix would be
                # excluded here, which is safe: it shows up immediately as a
                # coverage-count change rather than silently.
                if not str(source_file).startswith("documents/"):
                    continue
                if text.strip():
                    out[source_file] = text   # ORDER BY ASC: the last write wins
    return out


def _resolve_stored_documents() -> dict[str, dict]:
    from src.services.extraction.type_resolver import resolve_document_type

    result = {}
    for source_file, text in _stored_documents().items():
        r = resolve_document_type(
            declared_concept=declared_concept_for(source_file), full_text=text,
        )
        result[source_file] = {
            "agreement": r.agreement,
            "evidence_concept": r.evidence_concept,
            "status": r.status,
            "candidates": list(r.candidates),
        }
    return result


def test_the_baseline_covers_every_stored_document():
    """A baseline that silently lost documents would pass while checking nothing."""
    assert BASELINE.exists(), f"baseline fixture missing: {BASELINE}"
    recorded = json.loads(BASELINE.read_text())
    live = _stored_documents()
    assert len(recorded) == len(live), (
        f"baseline holds {len(recorded)} documents, the database has {len(live)}. "
        "A document was added or removed; re-capture deliberately."
    )


def test_no_stored_document_classifies_differently_than_the_baseline():
    recorded = json.loads(BASELINE.read_text())
    live = _resolve_stored_documents()
    drift = []
    for source_file, want in sorted(recorded.items()):
        got = live.get(source_file)
        if got is None:
            drift.append(f"{source_file}: GONE from the database")
        elif got != want:
            drift.append(f"{source_file}:\n      baseline={want}\n      now     ={got}")
    assert not drift, (
        f"{len(drift)} of {len(recorded)} documents classify differently:\n  "
        + "\n  ".join(drift)
    )


if __name__ == "__main__":
    if "--recapture" in sys.argv:
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        data = _resolve_stored_documents()
        BASELINE.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
        print(f"captured {len(data)} documents to {BASELINE}")
