"""The per-deal content hash the scheduler re-triages on (final review F1)."""
import dataclasses
from decimal import Decimal as D

import pytest

from src.services.triage.fingerprint import deal_content_hash
from src.services.triage.model import DuplicateFlag
from tests.triage.helpers import deal, inv, line, po, quote


def _deal():
    return deal(quote(), po(lines=[line(1), line(2, item="ITEM-2")]),
                inv("INV-1", lines=[line(1), line(2, item="ITEM-2")]),
                inv("INV-2", lines=[line(1)]),
                duplicates=[DuplicateFlag("INV-2", "INV-1", D("144")),
                            DuplicateFlag("INV-1", None, None)])


def test_hash_is_a_sha256_hex_digest():
    h = deal_content_hash(_deal())
    assert len(h) == 64 and int(h, 16) >= 0


def test_list_order_does_not_change_the_hash():
    a, b = _deal(), _deal()
    b.invoices.reverse()
    b.duplicates.reverse()
    for d in (*b.pos, *b.invoices):
        d.lines.reverse()
    assert deal_content_hash(a) == deal_content_hash(b)


DOC_FIELDS = {"supplier_id": "SUP-9", "currency": "EUR", "doc_date": None, "net": D("1"),
              "tax": D("2"), "gross": D("3"), "payment_terms": "Net 90", "po_id": "PO-9",
              "quote_ref": "Q-9", "confidence": 0.5, "fx_to_gbp": D("0.87")}


@pytest.mark.parametrize("field,value", sorted(DOC_FIELDS.items()))
def test_any_document_field_change_changes_the_hash(field, value):
    a, b = _deal(), _deal()
    setattr(b.invoices[0], field, value)
    assert deal_content_hash(a) != deal_content_hash(b)


LINE_FIELDS = {"line_ref": "99", "item_id": "X", "description": "Other", "quantity": D("11"),
               "uom": "box", "unit_price": D("12.01"), "line_amount": D("1"), "po_id": "PO-9",
               "delivery_date": __import__("datetime").date(2026, 3, 1)}


@pytest.mark.parametrize("field,value", sorted(LINE_FIELDS.items()))
def test_any_line_field_change_changes_the_hash(field, value):
    a, b = _deal(), _deal()
    b.invoices[0].lines[0] = dataclasses.replace(b.invoices[0].lines[0], **{field: value})
    assert deal_content_hash(a) != deal_content_hash(b)


def test_duplicate_flag_changes_change_the_hash():
    a, b, c = _deal(), _deal(), _deal()
    b.duplicates.pop()
    c.duplicates[0] = DuplicateFlag("INV-2", "INV-1", D("145"))
    assert len({deal_content_hash(a), deal_content_hash(b), deal_content_hash(c)}) == 3


def test_a_document_added_or_moved_changes_the_hash():
    a, b, c = _deal(), _deal(), _deal()
    b.invoices.append(inv("INV-3"))
    c.invoices[1].doc_id = "INV-2b"
    assert len({deal_content_hash(a), deal_content_hash(b), deal_content_hash(c)}) == 3
