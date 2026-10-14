"""Commercial terms are captured as the document states them (2026-10-08).

The freight quotes' "Commercial terms" table carries the fuel surcharge -- Condor "fixed 4.0%",
Swift "DTI-linked, capped at 3.0%", Meridian Freight "4.2%". Read as line items it became an
amountless "line"; it is a term of the quote, so it is captured as one: each row's label and
value, verbatim.
"""
from src.services.extraction.engineered.terms_extractor import terms_from_markdown, term_key

FREIGHT = """
## Commercial terms

| Commercial terms           | Commercial terms        |
|----------------------------|-------------------------|
| Payment terms              | 21 days from invoice    |
| Lead time / service        | Next-day standard       |
| Validity                   | 30 days from quote date |
| Fuel surcharge -fixed 4.0% | variable                |

## Scope &amp; assumptions
"""

MERIDIAN = """
| Payment terms   | 30 days from invoice   |
|-----------------|------------------------|
| Invoicing       | Monthly in arrears     |
| Price validity  | 90 days from issue     |
"""

LINES = """
| Description   | Qty | Unit price | Amount  |
|---------------|-----|------------|---------|
| Rack install  | 2   | 100        | 200     |
"""

ADDRESS = """
| Quotation to Assurity Ltd | Quote ref: MCG/2024/PS/0847 |
|---------------------------|-----------------------------|
"""


def test_a_commercial_terms_table_is_captured_row_by_row_verbatim():
    assert terms_from_markdown(FREIGHT) == [
        {"label": "Payment terms", "value": "21 days from invoice"},
        {"label": "Lead time / service", "value": "Next-day standard"},
        {"label": "Validity", "value": "30 days from quote date"},
        {"label": "Fuel surcharge -fixed 4.0%", "value": "variable"},
    ]


def test_a_terms_table_with_no_title_row_is_captured_from_its_labels():
    labels = [t["label"] for t in terms_from_markdown(MERIDIAN)]
    assert labels == ["Payment terms", "Invoicing", "Price validity"]


def test_line_item_and_address_tables_are_not_terms():
    assert terms_from_markdown(LINES) == []
    assert terms_from_markdown(ADDRESS) == []


def test_terms_line_up_across_suppliers_on_the_term_not_its_value():
    assert term_key("Fuel surcharge -fixed 4.0%") == term_key("Fuel surcharge -4.2%") \
        == term_key("Fuel surcharge -DTI-linked, capped at 3.0%") == "fuel surcharge"
    assert term_key("Payment terms") == "payment terms"


def test_an_untitled_block_keeps_only_its_term_rows():
    block = """
| PO reference   | PO-2024-0114          |
|----------------|-----------------------|
| Payment terms  | 30 days from invoice  |
"""
    assert terms_from_markdown(block) == [{"label": "Payment terms", "value": "30 days from invoice"}]


def test_a_new_upload_reads_its_terms_from_the_parsed_tables():
    import os
    import pytest
    f = "/home/muthu/Downloads/ProcureIQ_Demo_Pack/quotes/01_Freight/Condor_Logistics_UK_Ltd_V1.pdf"
    if not os.path.exists(f):
        pytest.skip("sample not present")
    from src.services.extraction.parser import parse
    from src.services.extraction.engineered.terms_extractor import terms_from_parsed
    terms = {term_key(t["label"]): t for t in terms_from_parsed(parse(f))}
    assert "fuel surcharge" in terms and "4.0%" in terms["fuel surcharge"]["label"]
    assert terms["payment terms"]["value"] == "21 days from invoice"
