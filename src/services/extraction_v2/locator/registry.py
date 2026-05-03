"""Locator registry: per-(doc_type, field) list of strategies.

Wires up the default locator set for invoice / quote / purchase_order
extraction. New strategies are added by registering them here OR via
the vendor-template store (per-vendor cached locators).
"""
from __future__ import annotations

import re
from typing import Optional

from src.services.extraction_v2.locator.base import Locator
from src.services.extraction_v2.locator.strategies.block_label import BlockLabelLocator
from src.services.extraction_v2.locator.strategies.filename import FilenameLocator
from src.services.extraction_v2.locator.strategies.header import HeaderLocator
from src.services.extraction_v2.locator.strategies.format import (
    FormatLocator,
    INVOICE_ID_PATTERN, PO_ID_PATTERN, QUOTE_ID_PATTERN,
)
from src.services.extraction_v2.locator.strategies.label_anchored import LabelAnchoredLocator
from src.services.extraction_v2.locator.strategies.structural_wrapper import StructuralLocator
from src.services.extraction_v2.parsers import parse_amount, parse_date
from src.services.extraction_v2.types import (
    InvalidValue, InvoiceId, PoId, QuoteId,
)


__all__ = ["build_locators"]


def _try(ctor):
    """Wrap a typed constructor so it returns None on failure."""
    def _f(raw):
        try:
            return ctor(raw)
        except (InvalidValue, ValueError, TypeError):
            return None
    return _f


_NAME_REJECT_RE = re.compile(
    r"^\s*("
    r"\d|invoice|bill|due|payment|po|order|quote|valid|delivery|"
    r"subtotal|total|tax|vat|amount|currency|terms|date|number|reference|"
    r"sort\s+code|account|iban|swift|bic"
    r")",
    re.IGNORECASE,
)


def _clean_name(raw: str) -> Optional[str]:
    """Normalize a candidate name. Reject the line if it looks like a label,
    a number, or a bank/payment artefact (e.g. 'Sort Code 40-18-22')."""
    if raw is None:
        return None
    s = raw.strip().rstrip(".,;:")
    if len(s) < 2 or len(s) > 120:
        return None
    if _NAME_REJECT_RE.match(s):
        return None
    return s


# Filename patterns for PK extraction
_PO_FILENAME_RE = re.compile(r"PO[\-\s]?\d{4,}", re.I)
_INVOICE_FILENAME_RE = re.compile(r"INV[\-\s]?[\w\-]+", re.I)
_QUOTE_FILENAME_RE = re.compile(r"(?:QUT|QTE|QUOTE)[\-\s]?[\w\-]+", re.I)

# Vendor name from filename: leading word(s) before any ID prefix or digits.
# Examples: "AQUARIUS INV-25-050.pdf" → "AQUARIUS"
#           "DUNCAN_PO526800.pdf" → "DUNCAN"
#           "invoice_acme.docx" → "acme" (after the doc-type word)
_VENDOR_FILENAME_RE = re.compile(
    r"^(?:invoice[_\-\s]+|po[_\-\s]+|quote[_\-\s]+)?"
    r"([A-Za-z][A-Za-z]{2,})"
    r"(?=[_\-\s]+(?:inv|po|qut|qte|quote|\d))",
    re.IGNORECASE,
)


def _vendor_from_filename(raw: str) -> Optional[str]:
    """Title-case a filename-derived vendor stub. Reject doc-type words."""
    if not raw:
        return None
    s = raw.strip().strip("_-").title()
    if s.upper() in {"INV", "INVOICE", "PO", "QUT", "QTE", "QUOTE", "BILL", "ORDER"}:
        return None
    if len(s) < 3:
        return None
    return s


def build_locators(doc_type: str) -> dict[str, list[Locator]]:
    """Return the default locator set per field for `doc_type`.

    Each field has 2-4 independent strategies. The consensus runner
    needs at least 2 to commit (or 1 with conf ≥ 0.95).
    """
    if doc_type == "Invoice":
        return {
            "invoice_id": [
                StructuralLocator(field="invoice_id", doc_type="Invoice", parser=_try(InvoiceId)),
                LabelAnchoredLocator(field="invoice_id", parser=_try(InvoiceId)),
                FormatLocator(field="invoice_id", pattern=INVOICE_ID_PATTERN, parser=_try(InvoiceId)),
                FilenameLocator(field="invoice_id", pattern=_INVOICE_FILENAME_RE, parser=_try(InvoiceId)),
            ],
            "po_id": [
                StructuralLocator(field="po_id", doc_type="Invoice", parser=_try(PoId)),
                LabelAnchoredLocator(field="po_id", parser=_try(PoId)),
                FormatLocator(field="po_id", pattern=PO_ID_PATTERN, parser=_try(PoId)),
                FilenameLocator(field="po_id", pattern=_PO_FILENAME_RE, parser=_try(PoId)),
            ],
            "invoice_date": [
                StructuralLocator(field="invoice_date", doc_type="Invoice", parser=parse_date),
                LabelAnchoredLocator(field="invoice_date", parser=parse_date),
            ],
            "due_date": [
                StructuralLocator(field="due_date", doc_type="Invoice", parser=parse_date),
                LabelAnchoredLocator(field="due_date", parser=parse_date),
            ],
            "invoice_amount": [
                StructuralLocator(field="invoice_amount", doc_type="Invoice", parser=parse_amount),
                LabelAnchoredLocator(field="invoice_amount", parser=parse_amount),
            ],
            "tax_amount": [
                StructuralLocator(field="tax_amount", doc_type="Invoice", parser=parse_amount),
                LabelAnchoredLocator(field="tax_amount", parser=parse_amount),
            ],
            "invoice_total_incl_tax": [
                StructuralLocator(field="invoice_total_incl_tax", doc_type="Invoice", parser=parse_amount),
                LabelAnchoredLocator(field="invoice_total_incl_tax", parser=parse_amount),
            ],
            "supplier_name": [
                StructuralLocator(field="supplier_name", doc_type="Invoice"),
                LabelAnchoredLocator(field="supplier_name", parser=_clean_name, multi_word=True),
                BlockLabelLocator(field="supplier_name", parser=_clean_name),
                HeaderLocator(field="supplier_name", parser=_clean_name),
                FilenameLocator(field="supplier_name", pattern=_VENDOR_FILENAME_RE, parser=_vendor_from_filename, base_confidence=0.62),
            ],
            "buyer_name": [
                LabelAnchoredLocator(field="buyer_name", parser=_clean_name, multi_word=True),
                BlockLabelLocator(field="buyer_name", parser=_clean_name),
            ],
        }

    if doc_type == "Purchase_Order":
        return {
            "po_id": [
                StructuralLocator(field="po_id", doc_type="Purchase_Order", parser=_try(PoId)),
                LabelAnchoredLocator(field="po_id", parser=_try(PoId)),
                FormatLocator(field="po_id", pattern=PO_ID_PATTERN, parser=_try(PoId)),
                FilenameLocator(field="po_id", pattern=_PO_FILENAME_RE, parser=_try(PoId)),
            ],
            "order_date": [
                StructuralLocator(field="order_date", doc_type="Purchase_Order", parser=parse_date),
                LabelAnchoredLocator(field="order_date", parser=parse_date),
            ],
            "expected_delivery_date": [
                StructuralLocator(field="expected_delivery_date", doc_type="Purchase_Order", parser=parse_date),
                LabelAnchoredLocator(field="expected_delivery_date", parser=parse_date),
            ],
            "total_amount": [
                StructuralLocator(field="total_amount", doc_type="Purchase_Order", parser=parse_amount),
                LabelAnchoredLocator(field="total_amount", parser=parse_amount),
            ],
            "tax_amount": [
                StructuralLocator(field="tax_amount", doc_type="Purchase_Order", parser=parse_amount),
                LabelAnchoredLocator(field="tax_amount", parser=parse_amount),
            ],
            "total_amount_incl_tax": [
                StructuralLocator(field="total_amount_incl_tax", doc_type="Purchase_Order", parser=parse_amount),
                LabelAnchoredLocator(field="total_amount_incl_tax", parser=parse_amount),
            ],
            "supplier_name": [
                StructuralLocator(field="supplier_name", doc_type="Purchase_Order"),
                LabelAnchoredLocator(field="supplier_name", parser=_clean_name, multi_word=True),
                BlockLabelLocator(field="supplier_name", parser=_clean_name),
                HeaderLocator(field="supplier_name", parser=_clean_name),
                FilenameLocator(field="supplier_name", pattern=_VENDOR_FILENAME_RE, parser=_vendor_from_filename, base_confidence=0.62),
            ],
            "buyer_name": [
                LabelAnchoredLocator(field="buyer_name", parser=_clean_name, multi_word=True),
                BlockLabelLocator(field="buyer_name", parser=_clean_name),
            ],
        }

    if doc_type == "Quote":
        return {
            "quote_id": [
                StructuralLocator(field="quote_id", doc_type="Quote", parser=_try(QuoteId)),
                LabelAnchoredLocator(field="quote_id", parser=_try(QuoteId)),
                FormatLocator(field="quote_id", pattern=QUOTE_ID_PATTERN, parser=_try(QuoteId)),
                FilenameLocator(field="quote_id", pattern=_QUOTE_FILENAME_RE, parser=_try(QuoteId)),
            ],
            "quote_date": [
                StructuralLocator(field="quote_date", doc_type="Quote", parser=parse_date),
                LabelAnchoredLocator(field="quote_date", parser=parse_date),
            ],
            "validity_date": [
                StructuralLocator(field="validity_date", doc_type="Quote", parser=parse_date),
                LabelAnchoredLocator(field="validity_date", parser=parse_date),
            ],
            "total_amount": [
                StructuralLocator(field="total_amount", doc_type="Quote", parser=parse_amount),
                LabelAnchoredLocator(field="total_amount", parser=parse_amount),
            ],
            "tax_amount": [
                StructuralLocator(field="tax_amount", doc_type="Quote", parser=parse_amount),
                LabelAnchoredLocator(field="tax_amount", parser=parse_amount),
            ],
            "total_amount_incl_tax": [
                StructuralLocator(field="total_amount_incl_tax", doc_type="Quote", parser=parse_amount),
                LabelAnchoredLocator(field="total_amount_incl_tax", parser=parse_amount),
            ],
            "supplier_name": [
                StructuralLocator(field="supplier_name", doc_type="Quote"),
                LabelAnchoredLocator(field="supplier_name", parser=_clean_name, multi_word=True),
                BlockLabelLocator(field="supplier_name", parser=_clean_name),
                HeaderLocator(field="supplier_name", parser=_clean_name),
                FilenameLocator(field="supplier_name", pattern=_VENDOR_FILENAME_RE, parser=_vendor_from_filename, base_confidence=0.62),
            ],
            "buyer_name": [
                LabelAnchoredLocator(field="buyer_name", parser=_clean_name, multi_word=True),
                BlockLabelLocator(field="buyer_name", parser=_clean_name),
            ],
        }

    return {}
