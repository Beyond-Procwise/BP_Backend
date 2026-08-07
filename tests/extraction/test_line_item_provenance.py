"""Line-item values must carry provenance, not just header fields.

The live renovation pipeline wrote provenance only for header fields. Every
line-item value — the unit prices and quantities that all the commercial
analysis is built on — reached the _trgt tables with no record of where it came
from. Measured 2026-08-07: bp_testdb (the database the backend actually runs
against, per DB_NAME in .env) held 722 provenance rows and NOT ONE of them was
a line_items path.

The cause is mechanical rather than deliberate. build_header_record() skips
every ``line_items[`` candidate, and write_provenance() is fed its output. The
registry stores line-field metadata under ``line_items.<field>`` while
candidates carry ``line_items[<idx>].<field>``, so a naive lookup raises
KeyError — which is very likely why they were excluded in the first place.

Consequence: the Phase 1b fact model requires line provenance by construction
and fails closed without it, so it produces zero facts on the live database for
every document, past and future.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.services.extraction import persistence  # noqa: E402
from src.services.extraction.pattern_registry import get_registry  # noqa: E402
from src.services.extraction.types import Candidate, Span  # noqa: E402


def _c(field: str, value: str, conf: float = 0.9) -> Candidate:
    return Candidate(
        field=field, value=value,
        span=Span(page=0, bbox=(10.0, 20.0, 30.0, 40.0), text=value),
        source="table", pattern_name=None, confidence=conf,
    )


CANDIDATES = [
    _c("invoice_id", "INV-1"),
    _c("currency", "GBP"),
    _c("line_items[0].unit_price", "86.94"),
    _c("line_items[0].quantity", "2"),
    _c("line_items[0].item_description", "Laptop"),
    _c("line_items[1].unit_price", "12.50"),
]


def test_header_record_still_excludes_line_items():
    """Pin the existing behaviour so the fix cannot blur the two record types."""
    registry = get_registry("invoice")
    _cols, picked, _errs = persistence.build_header_record(CANDIDATES, registry)
    assert not any(f.startswith("line_items[") for f in picked)


def test_line_candidates_are_pickable_for_provenance():
    """The gap: there must be a way to select the line candidates that were
    actually bound to a column, mirroring build_header_record's pick."""
    registry = get_registry("invoice")
    picked = persistence.pick_line_candidates(CANDIDATES, registry)
    assert "line_items[0].unit_price" in picked
    assert "line_items[0].quantity" in picked
    assert "line_items[1].unit_price" in picked
    assert picked["line_items[0].unit_price"].value == "86.94"


def test_the_highest_confidence_line_candidate_wins():
    registry = get_registry("invoice")
    cands = [
        _c("line_items[0].unit_price", "86.94", conf=0.4),
        _c("line_items[0].unit_price", "99.99", conf=0.95),
    ]
    picked = persistence.pick_line_candidates(cands, registry)
    assert picked["line_items[0].unit_price"].value == "99.99"


def test_a_line_field_no_schema_declares_is_not_picked():
    """Provenance must never reference a field the schema does not define —
    that is how a shadow vocabulary starts."""
    registry = get_registry("invoice")
    picked = persistence.pick_line_candidates(
        [_c("line_items[0].not_a_real_field", "x")], registry)
    assert picked == {}


def test_provenance_rows_are_built_for_line_items():
    """The end-to-end shape: write_provenance must emit rows whose field_path
    is the indexed line path, with the evidence span intact."""
    registry = get_registry("invoice")
    _cols, header_picked, _errs = persistence.build_header_record(CANDIDATES, registry)
    line_picked = persistence.pick_line_candidates(CANDIDATES, registry)

    rows = persistence.build_provenance_rows(
        doc_type="invoice", doc_pk="INV-1", pipeline_version="test",
        picked={**header_picked, **line_picked}, registry=registry,
    )

    paths = {r[2] for r in rows}
    assert "invoice_id" in paths, "header provenance must still be written"
    assert "line_items[0].unit_price" in paths, "line provenance must now be written"
    assert "line_items[1].unit_price" in paths

    row = next(r for r in rows if r[2] == "line_items[0].unit_price")
    assert row[3] == "86.94"          # value
    assert row[4] == 0                # page
    assert row[5:9] == (10.0, 20.0, 30.0, 40.0)  # bbox
    assert row[9] == "86.94"          # evidence_text


def test_the_indexed_path_matches_what_the_fact_assembler_looks_up():
    """The assembler resolves provenance by (doc_type, doc_pk, field_path) with
    a 0-based index. If this writer emitted a different shape the two halves
    would never meet, and the failure would be silent."""
    registry = get_registry("invoice")
    picked = persistence.pick_line_candidates(CANDIDATES, registry)
    rows = persistence.build_provenance_rows(
        doc_type="invoice", doc_pk="INV-1", pipeline_version="test",
        picked=picked, registry=registry,
    )
    import re
    for r in rows:
        assert re.fullmatch(r"line_items\[\d+\]\.\w+", r[2]), r[2]
