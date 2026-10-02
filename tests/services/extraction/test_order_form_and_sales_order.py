"""The two structures Nick named, and the measured reason order form is safe now.

'order form' was dropped as an alias of doctype.call_off_contract on 2026-10-01:
it titled every quote-template workbook, giving 12 false disagreements out of 12
uses and no true positive. It returns here as a structure in its own right,
governed by requires_parent_evidence, which is the build spec's own distinction —
"'order form' means one thing under a framework and another on its own".

Live-only. Run with:
    set -a && . ./.env && set +a
    CUDA_VISIBLE_DEVICES="" PROCWISE_TEST_LIVE_DB=1 ./venv/bin/python -m pytest \
        tests/services/extraction/test_order_form_and_sales_order.py -v
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.services.concepts import routing as R                       # noqa: E402
from src.services.concepts import vocabulary as V                    # noqa: E402
from src.services.extraction.type_resolver import resolve_document_type  # noqa: E402

_LIVE = os.environ.get("PROCWISE_TEST_LIVE_DB", "").strip().lower() in ("1", "true", "yes", "on")
pytestmark = pytest.mark.skipif(not _LIVE, reason="needs PROCWISE_TEST_LIVE_DB=1")


@pytest.fixture()
def vocab():
    V.invalidate()
    v = V.ensure_vocabulary()
    # ensure_vocabulary is deliberately fail-soft: on a failed or empty read it
    # returns the vocabulary it already had, which at process start is the seed
    # (vocabulary.py:329, _keep_current). Without this assertion a database
    # outage would turn every "live" test in this file into a seed test that
    # passes -- an outage looking exactly like agreement.
    assert v.source.startswith("bp_concept@"), (
        f"vocabulary came from {v.source!r}, not the database: these tests would "
        "be asserting against the seed while claiming to read live data"
    )
    return v


def test_both_structures_are_live_in_the_vocabulary(vocab):
    assert "doctype.order_form" in vocab.document_types
    assert "doctype.sales_order" in vocab.document_types


def test_order_form_is_the_only_structure_requiring_parent_evidence(vocab):
    """Nick's ruling, 2026-10-02: the rule governs doctype.order_form alone to start.

    call-off, SOW and schedule are conceptually the same but their behaviour is
    measured at 47 agreed / 0 disagreed and no evidence calls for changing it.
    """
    flagged = sorted(
        code for code, dt in vocab.document_types.items() if dt.requires_parent_evidence
    )
    assert flagged == ["doctype.order_form"], flagged


def test_order_form_carries_phrases_to_recognise_a_parent_by(vocab):
    phrases = vocab.document_types["doctype.order_form"].parent_evidence_phrases
    assert phrases, "a flagged structure with no phrases claims nothing"
    assert "framework" in phrases


def test_order_form_is_not_an_alias_of_the_call_off_contract(vocab):
    """The 2026-10-01 ruling stands: 'order form' must not resolve to a call-off."""
    owners = V.resolve_alias("order form", vocab)
    assert owners == ("doctype.order_form",), owners
    assert "order form" not in vocab.document_types["doctype.call_off_contract"].aliases


def test_a_real_call_off_still_matches_its_own_names(vocab):
    for spelling in ("call-off contract", "call off contract", "call-off"):
        assert V.resolve_alias(spelling, vocab) == ("doctype.call_off_contract",), spelling


def test_sales_order_wins_over_the_bare_order_alias(vocab):
    """Longest match wins: 'sales order' is not doctype.order with a word in front."""
    assert V.resolve_alias("sales order", vocab) == ("doctype.sales_order",)
    assert V.resolve_alias("order", vocab) == ("doctype.order",)
    page = "SALES ORDER\n\nAcknowledgement of your order.\n"
    r = resolve_document_type(declared_concept=None, full_text=page, vocabulary=vocab)
    assert r.evidence_concept == "doctype.sales_order", r


def test_a_sales_order_routes_at_the_purchase_order_pipeline(vocab):
    """Nick's ruling, 2026-10-02: a sales order extracts with the PO schema.

    It is the supplier's mirror of a purchase order and carries lines, quantities
    and a total, so the purchase-order schema is the one that fits it.
    """
    pipeline, code = R.pipeline_for_category("sales order", vocabulary=vocab)
    assert (pipeline, code) == ("purchase_order", "doctype.sales_order")


def test_an_order_form_routes_at_the_contract_pipeline(vocab):
    pipeline, code = R.pipeline_for_category("order form", vocabulary=vocab)
    assert (pipeline, code) == ("contract", "doctype.order_form")


def test_no_quote_workbook_became_a_disagreement(vocab):
    """THE measurement this task exists to protect.

    Thirteen stored quote workbooks carry an 'Order Form' title cell. Adding the
    structure without the parent-evidence rule flips every one of them to
    'disagreed'. None of them names a framework, an order of precedence, an
    incorporation clause or a call-off, so all thirteen stand the structure down.
    """
    from tests.services.extraction.test_classification_baseline import (
        _resolve_stored_documents,
    )

    live = _resolve_stored_documents()
    became = {
        src: out for src, out in live.items()
        if out["evidence_concept"] == "doctype.order_form"
    }
    assert not became, (
        f"{len(became)} stored document(s) now read as an order form: {sorted(became)}"
    )
    disagreed = {s: o for s, o in live.items() if o["agreement"] == "disagreed"}
    assert not disagreed, f"disagreements appeared: {sorted(disagreed)}"


def test_a_supplier_named_incorporated_is_not_parent_evidence(vocab):
    """The one measured decision of this task, pinned against the SEEDED data.

    Bare "incorporated" matches a company name whole-word, so it would let any
    quote workbook titled ORDER FORM from an "... Incorporated" supplier claim the
    page -- the 13-workbook defect through a side door. Task 3 proves the rule on
    a hand-built phrase list; this proves the list we actually seeded.
    """
    page = "ORDER FORM\n\nAcme Incorporated\nQuote Ref Q-1234   Valid Until 2026-12-01\n"
    r = resolve_document_type(declared_concept="doctype.quote", full_text=page,
                              vocabulary=vocab)
    assert r.evidence_concept != "doctype.order_form", r
    assert r.agreement != "disagreed", r


def test_the_seeded_phrases_do_not_include_bare_incorporated(vocab):
    phrases = vocab.document_types["doctype.order_form"].parent_evidence_phrases
    assert "incorporated" not in phrases, (
        "bare 'incorporated' matches a supplier name — use 'incorporated into' "
        "and 'incorporated by reference'; see specs/2026-10-02-contract-structures-design.md §4"
    )
    assert "incorporated into" in phrases
    assert "incorporated by reference" in phrases


def test_an_order_form_naming_its_framework_resolves_through_the_live_vocabulary(vocab):
    """Positive path: over-tightening the seeded list must break something visible."""
    page = "ORDER FORM\n\nMade under Framework Agreement FW-2024-0012\nSupplier: Acme Ltd\n"
    r = resolve_document_type(declared_concept=None, full_text=page, vocabulary=vocab)
    assert r.evidence_concept == "doctype.order_form", r
