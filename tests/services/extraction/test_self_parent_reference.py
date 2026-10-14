"""A document is not its own parent, and it does not own its parent's number.

Found by Task 11's LIVE verification on 2026-10-03, not by any fixture. A real
order form reading

    "This Order Form is incorporated into and governed by Framework Agreement
     No. FA-2026-0042"

promoted into proc.bp_contracts under contract_id = 'FA-2026-0042' -- its
FRAMEWORK's identifier. contract_id is the upsert key (ON CONFLICT (contract_id)
DO UPDATE SET <every other column>), so the framework agreement's own row was
overwritten: resolved_doc_type on FA-2026-0042 went from
doctype.framework_agreement to doctype.order_form and the framework stopped
existing as a separate contract. Two documents, one row, silently.

Why it cannot be fixed in the pattern layer: a framework agreement's own first
page writes its own identifier in exactly the same words an order form writes its
pointer. A negative lookbehind on contract_id was tried and it blinded the
framework to its own number. The fact that separates them is the document's own
structure, which is known in dispatch and nowhere earlier.

Offline -- the function is pure.
    CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest \
        tests/services/extraction/test_self_parent_reference.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.services.extraction.dispatch import _drop_self_parent_reference  # noqa: E402

FRAMEWORK = "doctype.framework_agreement"
MASTER = "doctype.master_agreement"
ORDER_FORM = "doctype.order_form"
SOW = "doctype.sow"


def test_a_framework_agreement_does_not_sit_under_a_framework_agreement():
    """Its own number is its contract_id; the pointer is the thing that is wrong."""
    columns = {"contract_id": "FA-2026-0042", "framework_ref": "FA-2026-0042"}
    _drop_self_parent_reference(columns, FRAMEWORK)
    assert columns["contract_id"] == "FA-2026-0042"
    assert columns["framework_ref"] is None


def test_an_order_form_does_not_own_its_frameworks_number():
    """THE regression. Keeping contract_id here is what overwrote the framework."""
    columns = {"contract_id": "FA-2026-0042", "framework_ref": "FA-2026-0042"}
    _drop_self_parent_reference(columns, ORDER_FORM)
    assert columns["contract_id"] is None, (
        "the order form kept its framework's identifier as its own primary key")
    assert columns["framework_ref"] == "FA-2026-0042"


def test_a_master_agreement_does_not_sit_under_a_master_agreement():
    columns = {"contract_id": "MSA-4417", "parent_agreement_ref": "MSA-4417"}
    _drop_self_parent_reference(columns, MASTER)
    assert columns["contract_id"] == "MSA-4417"
    assert columns["parent_agreement_ref"] is None


def test_a_sow_does_not_own_its_masters_number():
    columns = {"contract_id": "MSA-4417", "parent_agreement_ref": "MSA-4417"}
    _drop_self_parent_reference(columns, SOW)
    assert columns["contract_id"] is None
    assert columns["parent_agreement_ref"] == "MSA-4417"


def test_a_document_with_its_own_number_keeps_both():
    """The ordinary, correct case must be untouched: an order form that prints its
    own number AND its framework's keeps both, and they are different values."""
    columns = {"contract_id": "OF-2026-0117", "framework_ref": "FA-2026-0042"}
    _drop_self_parent_reference(columns, ORDER_FORM)
    assert columns == {"contract_id": "OF-2026-0117", "framework_ref": "FA-2026-0042"}


def test_references_that_differ_are_left_alone():
    columns = {"contract_id": "SOW-11", "parent_agreement_ref": "MSA-4417",
               "framework_ref": "FA-9"}
    before = dict(columns)
    _drop_self_parent_reference(columns, SOW)
    assert columns == before


def test_nothing_happens_when_the_structure_is_unknown():
    """type_resolution is None when the resolver raised, and a NULL structure must
    not be read as "not a framework" -- that would delete a real contract_id on a
    document whose type nobody derived."""
    columns = {"contract_id": "FA-2026-0042", "framework_ref": "FA-2026-0042"}
    before = dict(columns)
    _drop_self_parent_reference(columns, None)
    assert columns == before


def test_a_reference_differing_only_in_case_or_spacing_is_the_same_number():
    """'fa 2026 0042' and 'FA-2026-0042' are one identifier everywhere else in
    this layer (contract_hierarchy._norm_ref), so they must be here too."""
    columns = {"contract_id": "fa-2026-0042", "framework_ref": "FA 2026 0042"}
    _drop_self_parent_reference(columns, ORDER_FORM)
    assert columns["contract_id"] is None
    assert columns["framework_ref"] == "FA 2026 0042"


def test_a_missing_reference_is_not_a_match():
    columns = {"contract_id": None, "framework_ref": None}
    _drop_self_parent_reference(columns, ORDER_FORM)
    assert columns == {"contract_id": None, "framework_ref": None}


def test_an_unrelated_structure_is_untouched():
    """An invoice has neither field; the function must not invent one."""
    columns = {"contract_id": "C-1"}
    _drop_self_parent_reference(columns, "doctype.invoice")
    assert columns == {"contract_id": "C-1"}
