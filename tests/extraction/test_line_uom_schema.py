"""unit_of_measure must be declared on invoice and PO line items, and a table
header cell reading 'UOM' or 'Unit' must map to it — without stealing the
'Unit Price' column, which _header_to_field resolves by longest-label-wins.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.services.extraction.engineered.table_extractor import _header_to_field  # noqa: E402
from src.services.extraction_v3.yaml_schema.loader import load_doc_schema  # noqa: E402

DOC_TYPES = ["invoice", "purchase_order", "quote"]


@pytest.mark.parametrize("doc_type", DOC_TYPES)
def test_line_items_declare_unit_of_measure(doc_type):
    schema = load_doc_schema(doc_type)
    by_name = {f.name: f for f in schema.line_items.fields}
    assert "unit_of_measure" in by_name, f"{doc_type} line items must declare unit_of_measure"
    f = by_name["unit_of_measure"]
    assert f.db_column == "unit_of_measure"
    assert f.type == "string"
    assert f.required is False, "a new field must never block promotion"


@pytest.mark.parametrize("doc_type", DOC_TYPES)
@pytest.mark.parametrize("header", ["UOM", "Unit", "Unit of Measure", "U/M", "Measure"])
def test_uom_headers_map_to_unit_of_measure(doc_type, header):
    schema = load_doc_schema(doc_type)
    assert _header_to_field(header, schema.line_items.fields) == "unit_of_measure"


@pytest.mark.parametrize("doc_type", DOC_TYPES)
@pytest.mark.parametrize("header", ["Unit Price", "Unit Cost", "Price per Unit"])
def test_uom_does_not_steal_the_unit_price_column(doc_type, header):
    schema = load_doc_schema(doc_type)
    assert _header_to_field(header, schema.line_items.fields) == "unit_price"
