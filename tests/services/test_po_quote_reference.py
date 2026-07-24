"""The quote->PO award link, as it is actually carried by the documents.

Every PO in the corpus names the bid it was raised against ONCE, in its header:

    Reference  Against BAFO quote SDP-Q-44120 · RFQ PROC-2024-RFQ-FRT-005

That is the grain the documents use. The legacy carrier
``bp_po_line_items_stg.quote_number`` is per-LINE and is 0/316 filled across
the whole database, so award detection never fired on real data. These tests
pin the header carrier end to end: schema pattern -> stored column -> consumer.
"""
import re

import pytest
import yaml

from src.services import deal_clustering as dc
from src.services.version_collapse import base_reference

SCHEMA_PATH = "extraction_schemas/purchase_order.yaml"

# Header text exactly as the parser emits it (from
# bp_purchase_order_raw.parser_snapshot.full_text on the live batch — note the
# label carries no colon once docling flattens the header table, and the RFQ ref
# wraps onto its own line).
LIVE_HEADERS = {
    "PO-2024-0091": ("PO number PO-2024-0091\n\nDate 15 Apr 2024\n\n"
                     "Reference Against BAFO quote SDP-Q-44120 · RFQ\n\n"
                     "PROC-2024-RFQ-FRT-005\n", "SDP-Q-44120"),
    "PO-2024-0114": ("PO number:  PO-2024-0114\nDate:  2 May 2024\n"
                     "Reference:  Against BAFO quote MCG/2024/PS/0847 (V3)  ·  "
                     "RFQ PROC-2024-RFQ-PRS-002\n", "MCG/2024/PS/0847"),
    "PO-2024-0128": ("Reference:  Against BAFO quote SYN-MSA-3320  ·  "
                     "RFQ PROC-2024-RFQ-MAN-007\n", "SYN-MSA-3320"),
    "PO-2024-0145": ("Reference:  Against BAFO quote ORB-Q-6612 (V3)  ·  "
                     "RFQ PROC-2024-RFQ-SAA-011\n", "ORB-Q-6612"),
    # A works package raised against a TENDER, not a quote — and no matching
    # bid was uploaded, so it must stay orphaned downstream.
    "PO-2024-0163": ("Reference:  Against BAFO tender CBC-T-2231  ·  "
                     "RFQ PROC-2024-RFQ-WRK-014\n", "CBC-T-2231"),
}


def _schema_field(name):
    with open(SCHEMA_PATH) as fh:
        schema = yaml.safe_load(fh)
    return next((f for f in schema["fields"] if f["name"] == name), None)


def _apply(patterns, text):
    """Mirror pattern_extractor.run_pattern_extractor's anchor->window->search."""
    for p in sorted(patterns, key=lambda q: -q["prior_confidence"]):
        for am in re.compile(p["anchor"]).finditer(text):
            window = text[am.end():am.end() + p["max_span_after_anchor_chars"]]
            vm = re.compile(p["value"]).search(window)
            if vm:
                return vm.group(1).strip()
    return None


# --- the schema must ASK for the field ----------------------------------

def test_purchase_order_schema_declares_quote_reference():
    f = _schema_field("quote_reference")
    assert f is not None, "purchase_order.yaml never asks for the cited quote ref"
    assert f["db_column"] == "quote_reference"
    assert f["required"] is False   # POs raised without a quote are legitimate


@pytest.mark.parametrize("po_id", sorted(LIVE_HEADERS))
def test_pattern_recovers_cited_reference_from_live_header(po_id):
    text, expected = LIVE_HEADERS[po_id]
    got = _apply(_schema_field("quote_reference")["patterns"], text)
    assert got is not None, f"{po_id}: no quote reference extracted"
    # Stored verbatim (a "(V3)" suffix in the doc is kept), collapsed by consumers.
    assert base_reference(got) == expected


def test_pattern_does_not_fire_without_a_cited_quote():
    """A PO that names no quote must yield NULL, not a guess from the RFQ ref."""
    text = ("PO number:  PO-2024-0999\nDate:  1 Jun 2024\n"
            "Reference:  RFQ PROC-2024-RFQ-GEN-001\n")
    assert _apply(_schema_field("quote_reference")["patterns"], text) is None


# --- the consumer must READ the header carrier --------------------------

def test_explicit_award_matches_po_header_quote_reference():
    bid = {"quote_id": "SDP-Q-44120 (V3 (BAFO))", "base_reference": "SDP-Q-44120"}
    pos = [{"po_id": "PO-2024-0163", "quote_reference": "CBC-T-2231"},
           {"po_id": "PO-2024-0091", "quote_reference": "SDP-Q-44120"}]
    assert dc._explicit_award(bid, pos, {}) == "PO-2024-0091"


def test_explicit_award_collapses_version_suffix_on_the_po_side():
    """PO-2024-0145 cites "ORB-Q-6612 (V3)"; the bid collapsed to "ORB-Q-6612".
    Without collapsing the PO side too, _norm_ref gives orbq6612v3 != orbq6612."""
    bid = {"quote_id": "ORB-Q-6612", "base_reference": "ORB-Q-6612"}
    pos = [{"po_id": "PO-2024-0145", "quote_reference": "ORB-Q-6612 (V3)"}]
    assert dc._explicit_award(bid, pos, {}) == "PO-2024-0145"


def test_explicit_award_still_honours_legacy_line_level_quote_number():
    """The per-line carrier keeps working for data that has it."""
    bid = {"quote_id": "SYN-MSA-3320", "base_reference": "SYN-MSA-3320"}
    pos = [{"po_id": "PO-2024-0128"}]
    po_lines = {"PO-2024-0128": [{"quote_number": "SYN-MSA-3320"}]}
    assert dc._explicit_award(bid, pos, po_lines) == "PO-2024-0128"


def test_explicit_award_returns_none_when_no_po_cites_the_bid():
    bid = {"quote_id": "FSM-Q-9912", "base_reference": "FSM-Q-9912"}
    pos = [{"po_id": "PO-2024-0163", "quote_reference": "CBC-T-2231"}]
    assert dc._explicit_award(bid, pos, {}) is None


def test_header_carrier_takes_precedence_over_a_conflicting_line_carrier():
    """Header is the grain the document states the award at; if the two
    disagree the header wins, and the result must be deterministic."""
    bid = {"quote_id": "SDP-Q-44120", "base_reference": "SDP-Q-44120"}
    pos = [{"po_id": "PO-HEADER", "quote_reference": "SDP-Q-44120"},
           {"po_id": "PO-LINE"}]
    po_lines = {"PO-LINE": [{"quote_number": "SDP-Q-44120"}]}
    assert dc._explicit_award(bid, pos, po_lines) == "PO-HEADER"
