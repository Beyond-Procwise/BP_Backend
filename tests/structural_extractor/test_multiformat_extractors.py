"""Multi-format extraction tests for DOCX / XLSX / CSV fixtures.

Exercises the structured-anchor branches added in iteration 3:
- DOCX: invoice_acme.docx (ACME letterhead + 3-item table + Payment Terms).
- XLSX: invoice_megamart.xlsx + po_widgetco.xlsx (header-block + line-item block).
- CSV:  line_items_bulk.csv (line-items-only export with labeled columns).
"""
from pathlib import Path

import pytest

from src.services.structural_extractor.extractors.amounts import extract_amounts
from src.services.structural_extractor.extractors.line_items import extract_line_items
from src.services.structural_extractor.extractors.parties import extract_parties
from src.services.structural_extractor.extractors.payment_terms import extract_payment_terms
from src.services.structural_extractor.parsing import parse

FIX = Path(__file__).parent / "fixtures" / "docs"


def _doc(name: str):
    p = FIX / name
    if not p.exists():
        pytest.skip(f"{name} fixture missing")
    return parse(p.read_bytes(), name)


# --- DOCX ---


def test_docx_line_items_three_rows():
    doc = _doc("invoice_acme.docx")
    items = extract_line_items(doc, "Invoice")
    assert len(items) == 3
    assert items[0]["item_description"].value == "Technical Consulting"
    assert items[0]["quantity"].value == 20
    assert abs(items[0]["unit_price"].value - 150.0) < 0.01
    assert abs(items[0]["line_total"].value - 3000.0) < 0.01


def test_docx_parties_supplier_and_buyer():
    doc = _doc("invoice_acme.docx")
    out = extract_parties(doc, "Invoice")
    assert out["supplier_id"].value == "ACME PROFESSIONAL SERVICES LTD"
    # Buyer value may include continuation text; assert the key ORG portion.
    assert "Assurity Ltd" in out["buyer_id"].value


def test_docx_payment_terms_stops_at_currency_section():
    doc = _doc("invoice_acme.docx")
    out = extract_payment_terms(doc, "Invoice")
    assert out["payment_terms"].value == "Net 30"


# --- XLSX: MegaMart invoice ---


def test_xlsx_invoice_line_items():
    doc = _doc("invoice_megamart.xlsx")
    items = extract_line_items(doc, "Invoice")
    assert len(items) == 2
    assert items[1]["item_description"].value == "Standing Desk"
    assert items[1]["quantity"].value == 5
    assert abs(items[1]["line_total"].value - 1700.0) < 0.01


def test_xlsx_invoice_parties():
    doc = _doc("invoice_megamart.xlsx")
    out = extract_parties(doc, "Invoice")
    assert out["supplier_id"].value == "MEGAMART SUPPLIES INC"
    assert out["buyer_id"].value == "Assurity Ltd"


def test_xlsx_invoice_amounts_and_percent():
    doc = _doc("invoice_megamart.xlsx")
    out = extract_amounts(doc, "Invoice")
    # Arithmetic fit: 2300 + 230 = 2530
    assert abs(out["invoice_amount"].value - 2300.0) < 0.01
    assert abs(out["tax_amount"].value - 230.0) < 0.01
    assert abs(out["invoice_total_incl_tax"].value - 2530.0) < 0.01
    assert abs(out["tax_percent"].value - 10.0) < 0.1
    assert out["currency"].value == "USD"


def test_xlsx_invoice_payment_terms_adjacent_cell():
    doc = _doc("invoice_megamart.xlsx")
    out = extract_payment_terms(doc, "Invoice")
    assert out["payment_terms"].value == "Net 14"


# --- XLSX: WidgetCo PO ---


def test_xlsx_po_parties_supplier_name():
    doc = _doc("po_widgetco.xlsx")
    out = extract_parties(doc, "Purchase_Order")
    # Schema emits both supplier_name and supplier_id.
    assert out["supplier_name"].value == "WidgetCo Ltd"


def test_xlsx_po_line_items():
    doc = _doc("po_widgetco.xlsx")
    items = extract_line_items(doc, "Purchase_Order")
    assert len(items) == 2
    descriptions = {it["item_description"].value for it in items}
    assert descriptions == {"Widget Type A", "Widget Type B"}


# --- CSV ---


def test_csv_line_items_three_rows():
    doc = _doc("line_items_bulk.csv")
    items = extract_line_items(doc, "Purchase_Order")
    assert len(items) == 3
    assert items[0]["item_description"].value == "Server Rack"
    assert items[0]["quantity"].value == 2
    assert abs(items[0]["unit_price"].value - 850.0) < 0.01
    assert abs(items[0]["line_total"].value - 1700.0) < 0.01
    assert items[2]["item_description"].value == "Cable Bundle"
    assert abs(items[2]["unit_price"].value - 45.5) < 0.01


def test_csv_column_name_mapping_is_case_insensitive():
    """CSV header names are normalized via substring match — ensure
    uppercase/titlecase variants still map to the same roles.
    """
    from io import BytesIO  # noqa: F401 — for scope

    from src.services.structural_extractor.parsing.csv_parser import parse_csv

    content = (
        b"Description,Quantity,Unit Price,Line Total\n"
        b"Widget,3,10.00,30.00\n"
    )
    doc = parse_csv(content, "t.csv")
    items = extract_line_items(doc, "Purchase_Order")
    assert len(items) == 1
    assert items[0]["item_description"].value == "Widget"
    assert items[0]["quantity"].value == 3
    assert abs(items[0]["unit_price"].value - 10.0) < 0.01
    assert abs(items[0]["line_total"].value - 30.0) < 0.01
