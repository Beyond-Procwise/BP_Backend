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
