"""Label-anchored locator: find the value by looking for known labels
(e.g., "Invoice No:", "PO Number") and reading the adjacent token.

This is the most universally-useful strategy — works on any document
that uses field labels, regardless of layout. Vocabulary is externalized
so adding a new vendor's label variant is a data change, not code.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional

from src.services.extraction_v2.locator.base import AnchorRef, Locator, LocatorOutput
from src.services.structural_extractor.parsing.model import BBox, ParsedDocument


__all__ = ["LabelAnchoredLocator", "LABEL_VOCAB"]


# Externalized label vocabulary per field.
# Adding a new label variant for a vendor: append to the list.
LABEL_VOCAB: dict[str, list[str]] = {
    "invoice_id": [
        "invoice no", "invoice number", "invoice #", "inv no", "inv #",
        "invoice", "bill no", "bill number", "bill #",
    ],
    "po_id": [
        "po number", "po no", "po #", "purchase order", "purchase order no",
        "purchase order #", "order no", "order number", "order #",
        "p.o. no", "p.o. number",
    ],
    "quote_id": [
        "quote no", "quote number", "quote #", "qte no", "qte number",
        "quotation no", "quotation #", "ref",
    ],
    "invoice_date": [
        "invoice date", "date of issue", "date issued", "bill date", "date",
    ],
    "due_date": [
        "due date", "payment due", "pay by", "payment due by", "due by",
    ],
    "order_date": [
        "po date", "order date", "date ordered", "issue date",
    ],
    "expected_delivery_date": [
        "expected delivery", "delivery date", "ship by", "required by",
        "needed by",
    ],
    "quote_date": [
        "quote date", "quotation date", "date",
    ],
    "validity_date": [
        "valid until", "valid through", "expires", "expiry date",
    ],
    "supplier_name": [
        "from", "supplier", "vendor", "issued by", "seller",
    ],
    "buyer_name": [
        "bill to", "billed to", "invoice to", "customer", "buyer",
        "sold to", "ship to", "deliver to",
    ],
    "payment_terms": [
        "payment terms", "terms", "payment", "net terms",
    ],
    "currency": [
        "currency",
    ],
    "incoterm": [
        "incoterm", "shipping terms", "delivery terms",
    ],
    # Amount labels — keep "subtotal/total/vat" out of the longest-first
    # match risk by listing the most specific phrasings first. parser
    # validates Decimal-ness so labels matching dates won't survive.
    "invoice_amount": [
        "subtotal", "sub total", "sub-total", "net amount", "net total",
        "amount", "amount excl tax", "amount excluding tax",
    ],
    "tax_amount": [
        "vat", "tax", "tax amount", "vat amount", "sales tax", "gst",
        "tax (20%)", "vat (20%)",
    ],
    "tax_percent": [
        "tax rate", "vat rate", "tax %", "vat %",
    ],
    "invoice_total_incl_tax": [
        "total due", "amount due", "balance due", "grand total",
        "total amount", "total incl tax", "total inclusive of tax",
        "invoice total", "total",
    ],
    # Same vocab serves PO + Quote — schema drives which fields query it
    "total_amount": [
        "subtotal", "sub total", "sub-total", "net amount", "net total",
        "amount", "amount excl tax", "amount excluding tax",
    ],
    "total_amount_incl_tax": [
        "total due", "amount due", "balance due", "grand total",
        "total amount", "total incl tax", "total inclusive of tax",
        "order total", "quote total", "total",
    ],
}


@dataclass
class LabelAnchoredLocator(Locator):
    """Generic label-anchored locator. Plug in a parser for type validation.

    Capture mode is controlled by `multi_word`:
        False (default): captures the next single token after the label.
            Right for IDs, dates, amounts.
        True: captures up to end of the same line. Right for names and
            free-form fields where the value spans multiple words.
    """

    field: str                                              # e.g., "invoice_id"
    parser: Callable[[str], Optional[object]]               # parses raw → typed
    name: str = ""
    multi_word: bool = False                                # capture whole line?

    _MIN_CONFIDENCE = 0.65
    _MAX_CONFIDENCE = 0.92

    def __post_init__(self):
        if not self.name:
            self.name = f"label:{self.field}"

    def locate(self, doc: ParsedDocument) -> Optional[LocatorOutput]:
        labels = LABEL_VOCAB.get(self.field, [])
        if not labels:
            return None

        text = doc.full_text or ""
        if not text:
            return None

        # Single-token vs multi-word capture
        value_pat = r"([^\n]{1,120})" if self.multi_word else r"([^\s,;\n]{1,80})"

        for label in sorted(labels, key=len, reverse=True):  # longest-first
            pattern = re.compile(
                r"\b" + re.escape(label) + r"\b\s*[:#]?\s*" + value_pat,
                re.IGNORECASE,
            )
            for m in pattern.finditer(text):
                raw_value = m.group(1).strip().rstrip(".,;:")
                parsed = self.parser(raw_value)
                if parsed is None:
                    continue
                conf = min(
                    self._MIN_CONFIDENCE + 0.04 * len(label.split()),
                    self._MAX_CONFIDENCE,
                )
                return LocatorOutput(
                    value=parsed,
                    confidence=conf,
                    evidence=AnchorRef(
                        char_offset=m.start(1),
                        raw_text=raw_value,
                    ),
                    why=f"matched label {label!r}",
                )
        return None
