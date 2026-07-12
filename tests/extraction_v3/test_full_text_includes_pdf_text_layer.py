"""Regression test for the docling truncated-table bug.

Docling's markdown export clips the final column of some PDF tables. On
`Invoice_INV618706.pdf` the TOTAL column header comes back as "T" and the line
total "£1169.58" as "£1", while the SUB TOTAL / TAX / GRAND TOTAL values vanish
entirely:

    | ITEM DESCRIPTION | QTY         | PRICE   | T   |
    | Acer TravelMate  | 2           | £584.79 | £1  |
    |                  | SUB TOTAL   | SUB TOTAL   |  |
    |                  | TAX (20%)   |             |  |
    |                  | GRAND TOTAL | GRAND TOTAL |  |

`parsed.full_text` is the corpus the grounding guard checks values against, so
any figure missing from it is rejected as a hallucination — even when the VLM
read it correctly off the page. That is what silently turned a £1,169.58 invoice
into £584.79 (the unit price): the true net/tax/total were all thrown away, and
the amount was then re-derived from unit_price, ignoring quantity.

The PDF has a perfectly good native text layer carrying all three figures, so
full_text must include it.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.services.extraction_v3.parsers.router import parse as parse_document

FIXTURE = Path(__file__).parent / "fixtures" / "INV618706_qty2.pdf"


@pytest.mark.skipif(not FIXTURE.exists(), reason="fixture PDF not available")
def test_full_text_contains_totals_docling_truncates():
    """full_text must carry every money figure printed on the invoice.

    Guards the whole chain: if these are absent, the grounding guard rejects the
    correct values and the invoice is understated by the line quantity.
    """
    parsed = parse_document(str(FIXTURE))
    full_text = parsed.full_text or ""

    # The unit price survives docling's export today; the rest are the regression.
    for figure in ("584.79", "1169.58", "233.92", "1403.50"):
        assert figure in full_text, (
            f"{figure!r} is printed on the invoice but missing from parsed.full_text. "
            "The grounding guard will reject it as a hallucination and the amount "
            "will be silently re-derived from the unit price."
        )
