"""Line-item column mapping: specific labels win; spreadsheet headers found."""
import os

import pytest

from src.services.extraction.engineered.table_extractor import _header_to_field, extract_line_items
from src.services.extraction.pattern_registry import get_registry

LF = get_registry("quote").schema.line_items.fields
QDIR = "/home/muthu/Downloads/OneDrive_1_6-20-2025/quotes"


def test_specific_label_wins_over_generic():
    # "Unit" (unit_of_measure) must NOT steal the "Unit Price" column.
    assert _header_to_field("Unit Price (£)", LF) == "unit_price"
    assert _header_to_field("Unit Price", LF) == "unit_price"
    assert _header_to_field("Unit", LF) == "unit_of_measure"
    assert _header_to_field("Qty", LF) == "quantity"
    assert _header_to_field("Description", LF) == "item_description"


@pytest.mark.skipif(not os.path.exists(f"{QDIR}/quote_scenario_1.xlsx"), reason="sample not present")
def test_xlsx_line_items_now_extracted():
    from src.services.extraction.parser import parse as P
    doc = P(f"{QDIR}/quote_scenario_1.xlsx")
    li = extract_line_items(doc, get_registry("quote").schema)
    assert li, "spreadsheet line items should now be extracted (was 0)"
    assert any(c.field.endswith(".unit_price") for c in li), "unit price mapped correctly"
    descs = [c.value for c in li if c.field.endswith(".item_description")]
    assert any("Herman Miller" in d for d in descs), f"expected real product rows, got {descs[:3]}"
    # the price must NOT be mis-filed under unit_of_measure
    assert not any(c.field.endswith(".unit_of_measure") and c.value.replace(",", "").isdigit()
                   for c in li), "numeric price mis-mapped to unit_of_measure"


@pytest.mark.skipif(not os.path.exists(f"{QDIR}/QUOTE_WSG100024_watermark_split tables.docx"),
                    reason="sample not present")
def test_docx_unit_price_not_in_unit_of_measure():
    from src.services.extraction.parser import parse as P
    doc = P(f"{QDIR}/QUOTE_WSG100024_watermark_split tables.docx")
    li = extract_line_items(doc, get_registry("quote").schema)
    uom = [c.value for c in li if c.field.endswith(".unit_of_measure")]
    # unit_of_measure should not be holding a bare price number (e.g. "760")
    assert not any(v.replace(",", "").replace(".", "").isdigit() for v in uom), \
        f"unit_of_measure holds a numeric price: {uom[:5]}"


# --- Professional-services layouts (Meridian, Vantage set, 2026-10-08) -----------------------
# The deterministic step missed the staffing table ("Role / Grade", "Days"), took the "Detail"
# column as the description of the provisions table, and kept "Fixed-fee subtotal" rows. Its sum
# never reconciled, so the AI fallback replaced it -- and that fallback dropped Overtime and
# Expenses on V2 and V3 while keeping them on V1.

from src.services.extraction_v3.schemas.parsed_document import Cell, Page, ParsedDocument, Table


def _doc(*tables):
    def tbl(rows):
        return Table(page=0, bbox=(0, 0, 1, 1), header_row_index=0, rows=[
            [Cell(page=0, bbox=(0, 0, 1, 1), text=t, row_index=r, col_index=c) for c, t in enumerate(row)]
            for r, row in enumerate(rows)])
    return ParsedDocument(source_path="x.docx", file_format="docx", full_text="", parser_backend="docling",
                          parser_confidence=1.0,
                          pages=[Page(index=0, width=1, height=1, rotation=0, regions=[], tokens=[],
                                      tables=[tbl(t) for t in tables])])


def _lines(doc):
    rows = {}
    for c in extract_line_items(doc, get_registry("quote").schema):
        i = c.field.split("]")[0]
        rows.setdefault(i, {})[c.field.split(".")[-1]] = c.value
    return list(rows.values())


STAFF = [["Role / Grade", "IR35", "Days", "Day rate (£)", "Gross (£)", "Disc.", "Net (£)"],
         ["Programme Director", "Outside IR35", "175", "£1,650", "£288,750", "5%", "£274,312"],
         ["Consultant", "Outside IR35", "665", "£820", "£545,300", "5%", "£518,035"],
         ["Staffing subtotal"] * 6 + ["£792,347"]]
FIXED = [["Item description", "Basis", "Quantity", "Fee (£)", "Disc.", "Net (£)"],
         ["Programme mobilisation & governance setup", "Fixed fee", "1 package", "£80,000", "6%", "£75,200"],
         ["Fixed-fee subtotal"] * 5 + ["£75,200"]]
PROV = [["Provision", "Detail", "Amount (£)"],
        ["Overtime / out-of-hours", "Standard 1.5×; weekend 2×.", "£24,000"],
        ["Expenses (travel, accom., subsistence)", "Hard cap of £75,000 introduced for the first time.", "£72,000"],
        ["Provisions subtotal", "Provisions subtotal", "£96,000"]]


def test_a_staffing_table_is_read_role_days_rate_net():
    lines = _lines(_doc(STAFF))
    assert [l["item_description"] for l in lines] == ["Programme Director", "Consultant"]
    assert lines[0]["quantity"] == "175"
    assert lines[0]["unit_price"] == "£1,650"
    assert lines[0]["line_amount"] == "£274,312"


def test_the_named_column_is_the_description_not_the_detail_beside_it():
    lines = _lines(_doc(PROV))
    assert [l["item_description"] for l in lines] == [
        "Overtime / out-of-hours", "Expenses (travel, accom., subsistence)"]
    assert [l["line_amount"] for l in lines] == ["£24,000", "£72,000"]


def test_section_subtotal_rows_are_not_lines():
    lines = _lines(_doc(STAFF, FIXED, PROV))
    descs = [l["item_description"] for l in lines]
    assert not any("subtotal" in d.lower() for d in descs), descs
    assert len(lines) == 5


def test_a_real_item_that_mentions_total_is_still_a_line():
    lines = _lines(_doc([["Description", "Qty", "Amount"],
                         ["Total cost of ownership review", "1", "£5,000"],
                         ["Totals reconciliation service", "2", "£800"]]))
    assert len(lines) == 2


def test_header_mapping_for_the_new_labels():
    assert _header_to_field("Role / Grade", LF) == "item_description"
    assert _header_to_field("Days", LF) == "quantity"
    assert _header_to_field("Provision", LF) == "item_description"
