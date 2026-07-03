# tests/extraction/test_table_extractor_header_and_heading.py
"""Deterministic line-item recovery for two real-world layout misses:

1. A markdown table whose amount column is headed "Total Cost" (AQUARIUS
   invoices) — the column must map to line_amount, not be dropped.
2. A services quote where the priced item is a `## heading` separated from
   the "SERVICES" column header by linearised column-header tokens
   ("TOTAL RATE", "QTY") and whose amount only appears at SUBTOTAL
   (DESIGN HOUSE AGENCY quote DHA-2025-102).

Both must be recovered WITHOUT the LLM context_layer (which is best-effort
and returned nothing for these docs in production).
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.services.extraction.engineered.table_extractor import extract_line_items  # noqa: E402
from src.services.extraction.pattern_registry import get_registry  # noqa: E402


def _parsed(full_text: str) -> SimpleNamespace:
    # No structural tables -> exercises the markdown-table + text fallbacks,
    # exactly as docling leaves these paragraph-layout PDFs.
    return SimpleNamespace(full_text=full_text, pages=[])


def _by_suffix(cands, suffix):
    return [c for c in cands if c.field.endswith(suffix)]


# --- real parser_snapshot full_text from the live DB --------------------------

AQUARIUS_TEXT = (
    "<!-- image -->\n\n## BILLED TO:\n\n"
    "Assurity Ltd +44-7955-405-495610 Redkiln Way, Horsham, West Sussex RH13 5QH, United Kingdom\n\n"
    "| Item                                     | Total Cost   |\n"
    "|------------------------------------------|--------------|\n"
    "| Marketing and Brand Development Services | £5,000       |\n\n"
    "Subtotal\n\n£5,000\n\nTax (20%)\n\n£1,000\n\nTotal\n\n£6,000\n"
)

DHA_QUOTE_TEXT = (
    "<!-- image -->\n\n## CUSTOMER QUOTE\n\n## DESIGN HOUSE AGENCY\n\n"
    "BILL TO: ASSURITY LTD 10 REDKILN WAY, HORSHAM WEST SUSSEX, RH13 5QH\n\n"
    "PAYABLE TO: DESIGN HOUSE AGENCY 85 BROOK STREET, MANCHESTER, M1 7HY\n\n"
    "SERVICES\n\nTOTAL RATE\n\nQTY\n\n## Social Media Management\n\n"
    "- Instragram\n- Facebook\n- LinkedIn\n\nO\n\n"
    "SUBTOTAL\n\n£6,000\n\nVAT 20%\n\n£1,200\n\n## TOTAL\n\n"
    "## PAYMENT SCHEDULE:\n\n- MONTHLY INVOICES: $2,000/MONTH\n- PAYMENT DUE: NET 14\n\n£7,200\n"
)


def test_total_cost_header_maps_to_line_amount():
    """AQUARIUS: the '£5,000' under a 'Total Cost' column must become line_amount."""
    schema = get_registry("invoice").schema
    cands = extract_line_items(_parsed(AQUARIUS_TEXT), schema)
    descs = _by_suffix(cands, "item_description")
    amts = _by_suffix(cands, "line_amount")
    assert descs, "no description captured"
    assert descs[0].value == "Marketing and Brand Development Services"
    assert amts, "line_amount not captured from the 'Total Cost' column"
    assert "5000" in str(amts[0].value).replace(",", "").replace("£", "")


def test_dha_services_heading_recovered_as_line_item():
    """DHA quote: the '## Social Media Management' service line must be recovered
    with its £6,000 subtotal amount, despite the intervening column-header tokens."""
    schema = get_registry("quote").schema
    cands = extract_line_items(_parsed(DHA_QUOTE_TEXT), schema)
    descs = _by_suffix(cands, "item_description")
    assert descs, "no line item recovered (missing_line_items)"
    assert any("Social Media Management" in c.value for c in descs), (
        f"service heading not recovered; got {[c.value for c in descs]}"
    )
    amts = [c for c in cands if c.field.endswith("line_amount") or c.field.endswith("line_total")]
    assert amts, "no amount recovered for the service line"
    assert "6000" in str(amts[0].value).replace(",", "").replace("£", "")


def test_no_summary_rows_leak_as_line_items():
    """Neither recovery may emit SUBTOTAL / VAT / TOTAL rows as line items."""
    for doc_type, text in (("invoice", AQUARIUS_TEXT), ("quote", DHA_QUOTE_TEXT)):
        schema = get_registry(doc_type).schema
        cands = extract_line_items(_parsed(text), schema)
        for c in _by_suffix(cands, "item_description"):
            low = c.value.strip().lower().rstrip(":")
            assert low not in {"subtotal", "vat", "total", "vat 20%", "tax"}, (
                f"summary row leaked as line item: {c.value!r}"
            )


# Q-005-41 (TechWorld quote): a markdown table whose amount column is headed
# "MONTHLY COST", with the subtotal/tax/total block bleeding into trailing rows
# (payment-terms prose in the description column).
Q005_TEXT = (
    "<!-- image -->\n\n## n TECHWORLD\n\n"
    "Bill To: Assurity Ltd 10 Redkiln Way Horsham West Sussex RH13 5QH\n\n"
    "## QUOTE\n\nQuote No: Q-005-41\n\n"
    "| PRODUCT/SERVICE                | MONTHLY COST   |\n"
    "|--------------------------------|----------------|\n"
    "| GENERAL IT CONSULTANT          | £2,000         |\n"
    "| SOFTWARE IMPLEMENTATION        | £3,500         |\n"
    "| TRAINING & WORKSHOPS           | £1,250         |\n"
    "| Payment must be made within 30 | £6750          |\n"
    "| days of receiving the invoice. | £675           |\n"
    "|                                | £1215          |\n"
    "|                                | £7290          |\n"
)


def test_monthly_cost_header_maps_to_line_amount():
    """Q-005-41: the 'MONTHLY COST' column must map to line_amount for quotes."""
    schema = get_registry("quote").schema
    cands = extract_line_items(_parsed(Q005_TEXT), schema)
    by_desc = {}
    for c in cands:
        if c.field.endswith("item_description"):
            idx = c.field.split("[")[1].split("]")[0]
            by_desc[idx] = c.value
    amt_by_idx = {}
    for c in cands:
        if c.field.endswith("line_amount"):
            idx = c.field.split("[")[1].split("]")[0]
            amt_by_idx[idx] = str(c.value).replace(",", "").replace("£", "")
    # the three real services must each carry their amount
    want = {"GENERAL IT CONSULTANT": "2000",
            "SOFTWARE IMPLEMENTATION": "3500",
            "TRAINING & WORKSHOPS": "1250"}
    for idx, desc in by_desc.items():
        if desc in want:
            assert amt_by_idx.get(idx) and want[desc] in amt_by_idx[idx], (
                f"{desc!r} missing amount {want[desc]} (got {amt_by_idx.get(idx)!r})"
            )
    assert set(want).issubset(set(by_desc.values())), "real service lines not all captured"


def test_summary_block_not_over_captured_as_line_items():
    """Q-005-41: the payment-terms / subtotal-total block must NOT appear as line
    items — subtotal closure (2000+3500+1250=6750) trims the trailing rows."""
    schema = get_registry("quote").schema
    cands = extract_line_items(_parsed(Q005_TEXT), schema)
    descs = [c.value for c in cands if c.field.endswith("item_description")]
    assert not any("Payment must be made" in d for d in descs), (
        f"payment-terms prose over-captured as line item: {descs}"
    )
    assert not any("days of receiving" in d for d in descs), (
        f"payment-terms continuation over-captured: {descs}"
    )
    # exactly the 3 real services remain
    assert len([d for d in descs]) == 3, f"expected 3 clean line items, got {descs}"


def test_aquarius_pdf_end_to_end_captures_line_amount():
    """End-to-end through the real PDF parser: the AQUARIUS invoice's
    'Total Cost' £5,000 must surface as a line_amount via the live engineered
    path (proves parse + synonym fix work together, not just on cached text)."""
    import pytest

    fixture = Path(__file__).resolve().parents[1] / (
        "structural_extractor/fixtures/docs/AQUARIUS INV-25-050 for PO508084 .pdf"
    )
    if not fixture.exists():
        pytest.skip("AQUARIUS fixture missing")
    try:
        from src.services.extraction.parser import parse as parse_document
    except Exception as exc:  # pragma: no cover - parser backend optional
        pytest.skip(f"PDF parser unavailable: {exc}")

    doc = parse_document(str(fixture))
    cands = extract_line_items(doc, get_registry("invoice").schema)
    amts = _by_suffix(cands, "line_amount")
    assert amts, "line_amount not captured end-to-end from the 'Total Cost' column"
    assert "5000" in str(amts[0].value).replace(",", "").replace("£", "")


def test_description_containing_header_word_not_dropped():
    """Regression guard for the heading recovery: a real description that happens
    to contain a column-header word ('Rate') must NOT be skipped as a header."""
    text = (
        "SERVICES\n\nTOTAL RATE\n\nQTY\n\n"
        "## Rate Card Design and Brand Guidelines\n\n"
        "SUBTOTAL\n\n£3,200\n"
    )
    schema = get_registry("quote").schema
    cands = extract_line_items(_parsed(text), schema)
    descs = _by_suffix(cands, "item_description")
    assert any("Rate Card Design" in c.value for c in descs), (
        f"description with header-like word was dropped; got {[c.value for c in descs]}"
    )
