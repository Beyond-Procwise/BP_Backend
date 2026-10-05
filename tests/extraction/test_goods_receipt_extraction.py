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
