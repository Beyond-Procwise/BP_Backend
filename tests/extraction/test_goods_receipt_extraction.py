"""The goods receipt as a document type, and the price fields it must not have.

Two subjects live here because they are the same promise read at two levels: the
vocabulary says a goods receipt is a transaction document that sits under an
order, and the extraction schema says the thing we read off one carries
quantities and nothing priced. Either one failing alone would let a fabricated
price onto the document that proves delivery.
"""
from __future__ import annotations

import pytest

from src.services.concepts.seed import DOCUMENT_TYPES


def test_the_vocabulary_knows_a_goods_receipt():
    dt = DOCUMENT_TYPES["doctype.goods_receipt"]
    assert dt.pipeline_doc_type == "goods_receipt"
    assert dt.default_parent_type == "doctype.order"
    assert dt.role == "role.transaction"


@pytest.mark.parametrize("alias", ["goods receipt", "grn", "delivery note",
                                   "despatch note", "proof of delivery", "packing slip"])
def test_the_names_a_supplier_actually_uses_resolve(alias):
    assert alias in DOCUMENT_TYPES["doctype.goods_receipt"].aliases
