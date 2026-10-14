"""The goods receipt as a document type, and the price fields it must not have.

Two subjects live here because they are the same promise read at two levels: the
vocabulary says a goods receipt is a transaction document that sits under an
order, and the extraction schema says the thing we read off one carries
quantities and nothing priced. Either one failing alone would let a fabricated
price onto the document that proves delivery.
"""
from __future__ import annotations

import pytest

from src.services.concepts.seed import DOCUMENT_TYPES


def test_the_vocabulary_knows_a_goods_receipt():
    dt = DOCUMENT_TYPES["doctype.goods_receipt"]
    assert dt.pipeline_doc_type == "goods_receipt"
    assert dt.default_parent_type == "doctype.order"
    assert dt.role == "role.transaction"


@pytest.mark.parametrize("alias", ["goods receipt", "grn", "delivery note",
                                   "despatch note", "proof of delivery", "packing slip"])
def test_the_names_a_supplier_actually_uses_resolve(alias):
    assert alias in DOCUMENT_TYPES["doctype.goods_receipt"].aliases


# --------------------------------------------------------------------------
# Review Focus #4: a delivery note that restates the order's prices.
#
# The brief imagined an `extract_goods_receipt(document) -> dict`. No such
# function exists or should: extraction is data-driven through
# dispatch_document(doc_type=...) + the YAML registry, and inventing a
# doc-type-specific entry point would be the one special case the design
# exists to avoid. So these test the two pieces of PRODUCTION code that
# actually decide whether a price can reach a goods-receipt record:
#
#   * the registry built from extraction_schemas/goods_receipt.yaml -- a field
#     that is not declared cannot be committed, because
#     persistence.build_header_record only writes fields with a db_column;
#   * persistence.build_line_items, which silently drops a line field the
#     schema does not know. That silence is the enforcement, so it is asserted
#     rather than trusted.
# --------------------------------------------------------------------------
import re

from src.services.extraction.pattern_registry import get_registry
from src.services.extraction.persistence import build_header_record, build_line_items
from src.services.extraction.types import Candidate, Span

_PRICED = re.compile(r"(price|amount|total|value|cost|currency|tax)", re.I)


def _cand(field: str, value: str, conf: float = 0.9) -> Candidate:
    return Candidate(
        field=field, value=value, confidence=conf,
        span=Span(page=1, bbox=(0.0, 0.0, 1.0, 1.0), text=value),
        source="regex", pattern_name="test",
    )


def test_the_goods_receipt_schema_declares_no_priced_field():
    """A column is an invitation and so is a schema field. There is neither."""
    schema = get_registry("goods_receipt").schema
    named = [f.name for f in schema.fields] + [f.db_column for f in schema.fields]
    if schema.line_items:
        named += [f.name for f in schema.line_items.fields]
        named += [f.db_column for f in schema.line_items.fields]
    offenders = sorted({n for n in named if n and _PRICED.search(n)})
    assert offenders == [], f"priced fields on a goods receipt: {offenders}"


def test_a_delivery_note_restating_po_prices_populates_no_price_field():
    """Real delivery notes restate the order's prices for the driver's
    paperwork, and the extractor will find them. The receipt record must not
    keep them: this document proves delivery, and a price on it would be read
    as corroboration of a value it never witnessed."""
    registry = get_registry("goods_receipt")
    lines = build_line_items([
        _cand("line_items[0].description", "Widget A"),
        _cand("line_items[0].quantity_received", "10"),
        _cand("line_items[0].unit_of_measure", "each"),
        # What the note also prints, and what must not survive:
        _cand("line_items[0].unit_price", "12.50"),
        _cand("line_items[0].line_amount", "125.00"),
        _cand("line_items[0].currency", "GBP"),
        _cand("line_items[0].total_amount_incl_tax", "150.00"),
    ], registry)

    assert len(lines) == 1
    flat = repr(lines).lower()
    for forbidden in ("12.50", "125.00", "150.00", "unit_price", "line_amount",
                      "currency", "gbp"):
        assert forbidden not in flat, f"{forbidden!r} reached the receipt line"
    assert lines[0]["quantity_received"] == 10


def test_the_header_keeps_the_receipt_facts_and_has_nowhere_to_put_a_price():
    """The same promise one level up: the declared header fields bind, and
    `currency` -- the one a delivery note most often carries -- is not a field
    this doc type has, so build_header_record would raise rather than quietly
    invent a column for it."""
    registry = get_registry("goods_receipt")
    columns, _picked, errors = build_header_record([
        _cand("grn_id", "GRN-5521"),
        _cand("po_id", "4500018832"),
        _cand("supplier_name", "Northwind Trading Ltd"),
    ], registry)
    assert errors == []
    assert columns["grn_id"] == "GRN-5521"
    assert columns["po_id"] == "4500018832"
    assert not any(_PRICED.search(c) for c in columns)

    with pytest.raises(KeyError):
        build_header_record([_cand("currency", "GBP")], registry)


# --------------------------------------------------------------------------
# What the LIVE run of 2026-10-05 found, on a real delivery note against
# PO000645. Both defects let the table extractor do the wrong thing; neither
# was reachable by the fixture tests above, because both are about reading a
# real table rather than a candidate list.
# --------------------------------------------------------------------------
from src.services.extraction.engineered.table_extractor import (
    _find_header, extract_line_items,
)


class _Cell:
    def __init__(self, col_index, text):
        self.col_index, self.text = col_index, text
        self.bbox = (0.0, 0.0, 1.0, 1.0)


class _Table:
    def __init__(self, rows, header_row_index=0, page=1):
        self.rows = [[_Cell(i, t) for i, t in enumerate(r)] for r in rows]
        self.header_row_index, self.page = header_row_index, page


class _Page:
    def __init__(self, tables):
        self.tables = tables


class _Parsed:
    def __init__(self, pages):
        self.pages = pages


#: Exactly the table docling read off the live delivery note. Kept verbatim,
#: merged header cell and all, because a tidied version does not reproduce
#: either defect.
_LIVE_ROWS = [
    ["Item No", "Description", "Qty Delivered", "Qty Rejected Unit", "Unit Price", "Line Total"],
    ["1", "Standard Compliance Subscription ITM004425 5", "0", "each", "GBP 13.59", "GBP 67.95"],
    ["2", "Heavy-Duty Compliance Subscription ITM004436 1", "0", "tonne", "GBP 1.15", "GBP 1.15"],
    ["3", "Heavy-Duty A4/A3 Installation ITM001058", "8 0", "pack", "GBP 2.70", "GBP 21.60"],
]


#: A delivery note whose columns do not overlap -- what a supplier's own
#: template produces, and what the regenerated live document produces after the
#: first attempt's columns ran together.
_WELL_FORMED_ROWS = [
    ["Item No", "Description", "Qty Delivered", "Qty Rejected", "Unit",
     "Unit Price", "Line Total"],
    ["1", "Standard Compliance Subscription ITM004425", "5", "0", "each",
     "GBP 13.59", "GBP 67.95"],
    ["2", "Heavy-Duty Compliance Subscription ITM004436", "1", "0", "tonne",
     "GBP 1.15", "GBP 1.15"],
    ["3", "Heavy-Duty A4/A3 Installation ITM001058", "8", "0", "pack",
     "GBP 2.70", "GBP 21.60"],
]


def _lines_from(rows):
    registry = get_registry("goods_receipt")
    cands = extract_line_items(_Parsed([_Page([_Table(rows)])]), registry.schema)
    by_line: dict[int, dict] = {}
    for c in cands:
        m = re.match(r"line_items\[(\d+)\]\.(\w+)$", c.field)
        by_line.setdefault(int(m.group(1)), {})[m.group(2)] = c.value
    return by_line


def test_a_real_delivery_note_table_yields_line_items():
    """The live run read the header correctly and emitted ZERO lines.

    `extract_line_items` requires a field literally named `item_description`
    before it will accept a row as a line item (its summary-row guard). The
    goods-receipt schema called it `description`, so every row was discarded in
    silence -- the receipt reached _trgt with a header, a PO link and a deal,
    and no lines, which the match then reads as "no receipt" and passes. That
    is the worst possible failure for this feature: a control that reports
    nothing wrong because it looked at nothing.
    """
    by_line = _lines_from(_WELL_FORMED_ROWS)
    assert len(by_line) == 3, f"expected three lines, got {by_line}"
    assert by_line[0]["item_description"].startswith("Standard Compliance Subscription")
    assert by_line[0]["quantity_received"] == "5"
    assert by_line[0]["unit_of_measure"] == "each"
    assert by_line[2]["quantity_received"] == "8"
    assert by_line[2]["unit_of_measure"] == "pack"


def test_rows_are_still_recovered_from_a_note_whose_columns_run_together():
    """The live document's own columns overlapped, so docling merged two
    headers into "Qty Rejected Unit" and the delivered quantity bled into the
    description. Three rows are still recognised as line items -- the point of
    this test -- but the unit lands in the wrong field, which is a document
    problem and not a code one. It is here so the next reader knows this table
    shape produces lines whose quantities cannot be trusted, rather than
    discovering it from a wrong match result.
    """
    by_line = _lines_from(_LIVE_ROWS)
    assert len(by_line) == 3
    assert "unit_of_measure" not in by_line[0]


def test_a_price_column_cannot_be_read_as_a_unit_of_measure():
    """The live run mapped the 'Unit Price' column onto unit_of_measure.

    `unit_of_measure` carries the canonical label "Unit", "Unit" is a substring
    of "Unit Price", and the goods-receipt schema has no price field for the
    longest-match rule to prefer -- so 'GBP 13.59' was about to be stored as
    the unit a line was delivered in. A doc type with no priced field must
    refuse a money-named header outright, which is the same rule as the absent
    column and the absent schema field, applied one layer further out.
    """
    registry = get_registry("goods_receipt")
    header = _Table(_LIVE_ROWS).rows[0]
    _idx, mapping = _find_header(_Table(_LIVE_ROWS), registry.schema.line_items.fields)
    priced_cols = [i for i, cell in enumerate(header)
                   if _PRICED.search(cell.text or "")]
    assert priced_cols, "the live header had a Unit Price and a Line Total column"
    for col in priced_cols:
        assert col not in mapping, (
            f"column {col!r} ({header[col].text!r}) mapped to {mapping.get(col)!r}")

    cands = extract_line_items(_Parsed([_Page([_Table(_LIVE_ROWS)])]), registry.schema)
    flat = repr([c.value for c in cands]).lower()
    for forbidden in ("13.59", "67.95", "gbp"):
        assert forbidden not in flat, f"{forbidden!r} reached a receipt line"


def test_an_invoice_still_reads_its_price_columns():
    """The refusal above is scoped to a doc type with NO priced field. An
    invoice has them, so nothing about its reading may change."""
    registry = get_registry("invoice")
    rows = [["Description", "Qty", "Unit Price", "Amount"],
            ["Widget A", "10", "12.50", "125.00"]]
    _idx, mapping = _find_header(_Table(rows), registry.schema.line_items.fields)
    assert mapping.get(2) == "unit_price"
    assert mapping.get(3) == "line_amount"


# --------------------------------------------------------------------------
# The header fields the live run read WRONG.
# --------------------------------------------------------------------------
from src.services.extraction.pattern_extractor import run_pattern_extractor


class _LiveParsed:
    """Only what run_pattern_extractor reads: the text and a page to anchor on."""

    def __init__(self, text):
        self.full_text = text
        self.pages = [type("P", (), {"page_number": 1, "text": text, "tables": [],
                                     "blocks": []})()]
        self.source_path = "test.pdf"


#: The first five lines of the live delivery note, verbatim from
#: proc.bp_goods_receipt_raw.parser_snapshot (raw_id 100, 2026-10-05).
_LIVE_HEADER_TEXT = (
    "## DELIVERY NOTE\n\n"
    "Delivery Note No: DN-000645\n\n"
    "Against PO PO000645\n\n"
    "Date Received: 18 March 2024\n\n"
    "Supplier: SUP-WindroseSupplies7\n\n"
    "Consignment No: CON-884215\n\n"
    "Received by: J. Okafor\n\n"
    "Signature: ____________________\n"
)


def _header(text):
    out = {}
    for c in run_pattern_extractor(_LiveParsed(text), "goods_receipt"):
        if not c.field.startswith("line_items["):
            out.setdefault(c.field, []).append((c.confidence, c.value))
    return {k: max(v)[1] for k, v in out.items()}


def test_the_delivery_note_number_keeps_its_prefix():
    """The live run stored `000645`, not `DN-000645`.

    The value class was `[A-Z]{0,4}[0-9]...`, which cannot span the separator
    in "DN-000645": it matched "DN", needed a digit, found "-", and the engine
    advanced to the bare digits. A receipt whose id is the PO's digits is
    indistinguishable from the order it cites, and two notes against two orders
    ending in the same digits would collide on the _stg primary key.
    """
    assert _header(_LIVE_HEADER_TEXT)["grn_id"] == "DN-000645"


def test_the_delivery_note_number_is_still_read_without_a_prefix():
    """A note numbered 5521 with no letters must not regress."""
    assert _header("Delivery Note No: 5521\nAgainst PO 4500018832\n")["grn_id"] == "5521"


def test_who_signed_for_the_goods_stops_at_the_end_of_the_line():
    """The live run stored "J. Okafor\\n\\nSignature" as the person who received
    the goods. The name class allowed the following line to be read as a
    further forename."""
    assert _header(_LIVE_HEADER_TEXT)["received_by"] == "J. Okafor"


def test_the_date_the_goods_arrived_is_read():
    """The live run read no receipt_date at all, so the receipt reached _trgt
    with no record of WHEN anything arrived -- which is most of what a delivery
    note is for."""
    assert _header(_LIVE_HEADER_TEXT)["receipt_date"] == "18 March 2024"


def test_the_purchase_order_is_read_from_the_against_line():
    assert _header(_LIVE_HEADER_TEXT)["po_id"] == "PO000645"


def test_the_consignment_reference_is_read():
    assert _header(_LIVE_HEADER_TEXT)["carrier_ref"] == "CON-884215"
