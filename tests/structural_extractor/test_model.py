import dataclasses

import pytest

from src.services.structural_extractor.parsing.model import (
    BBox, CellRef, ColumnRef, NodeRef, Token, Region, Table, ParsedDocument
)


def test_bbox_frozen():
    b = BBox(page=1, x0=0.0, y0=0.0, x1=10.0, y1=10.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        b.x0 = 1.0


def test_cellref_requires_sheet_row_col():
    c = CellRef(sheet="Sheet1", row=1, col=2)
    assert c.sheet == "Sheet1" and c.row == 1 and c.col == 2 and c.merged_range is None


def test_columnref_defaults():
    c = ColumnRef(row=0, col=0, column_name="Invoice No")
    assert c.row == 0 and c.col == 0 and c.column_name == "Invoice No"


def test_noderef_paragraph_vs_table():
    p = NodeRef(kind="paragraph", paragraph_index=3)
    t = NodeRef(kind="table_cell", table_index=0, row=1, col=2)
    assert p.kind == "paragraph" and t.kind == "table_cell"


def test_token_has_anchor():
    t = Token(text="Invoice", anchor=BBox(1, 0, 0, 50, 20), order=0)
    assert t.text == "Invoice"
    assert isinstance(t.anchor, BBox)


def test_parsed_document_shape():
    d = ParsedDocument(
        source_format="pdf", filename="x.pdf", tokens=[], regions=[],
        tables=[], pages_or_sheets=1, full_text="", raw_bytes=b"",
    )
    assert d.source_format == "pdf"
