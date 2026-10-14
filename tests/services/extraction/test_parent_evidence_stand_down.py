"""A child structure only claims a page that names its parent.

Measured reason this exists: 'order form' titles every quote-template workbook
on this corpus. Added as a plain structure it flips 13 quote documents to
'disagreed' with no true positive. With this rule all 53 documents resolve
exactly as they do today, and a real order form that names its framework still
classifies. See specs/2026-10-02-contract-structures-design.md §4.

Offline — these build a Vocabulary by hand, so they run with no database.
    CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest \
        tests/services/extraction/test_parent_evidence_stand_down.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.services.concepts.seed import DocumentType          # noqa: E402
from src.services.concepts.vocabulary import build_vocabulary          # noqa: E402
from src.services.extraction.type_resolver import resolve_document_type  # noqa: E402

PHRASES = ("framework", "order of precedence", "incorporated into",
           "incorporated by reference", "call off")


def _vocab(*types: DocumentType):
    concepts = [
        {"concept_code": t.concept_code, "domain": "DOCUMENT_TYPE", "definition": "d",
         "not_to_be_confused_with": [], "status": "active", "rejection_reason": None}
        for t in types
    ]
    rows = [
        {"concept_code": t.concept_code, "role": t.role,
         "default_parent_type": t.default_parent_type, "execution_mode": t.execution_mode,
         "aliases": list(t.aliases), "identifiers": list(t.identifiers),
         "structural_signals": list(t.structural_signals),
         "pipeline_doc_type": t.pipeline_doc_type, "status": t.status,
         "requires_parent_evidence": t.requires_parent_evidence,
         "parent_evidence_phrases": list(t.parent_evidence_phrases)}
        for t in types
    ]
    return build_vocabulary(concepts, rows, source="test")


ORDER_FORM = DocumentType(
    "doctype.order_form", "role.master", "doctype.framework_agreement", "exec.bilateral",
    ("order form",), (), (), "contract",
    requires_parent_evidence=True, parent_evidence_phrases=PHRASES,
)
QUOTE = DocumentType(
    "doctype.quote", "role.supporting", None, "exec.unilateral",
    ("quote", "quotation"), (), (), "quote",
)
FRAMEWORK = DocumentType(
    "doctype.framework_agreement", "role.framework", None, "exec.bilateral",
    ("framework agreement", "framework"), (), (), "contract",
)


def test_an_order_form_that_names_no_parent_does_not_claim_the_page():
    page = "ORDER FORM\n\nQuote Ref Q-1234   Valid Until 2026-12-01\n"
    r = resolve_document_type(declared_concept="doctype.quote", full_text=page,
                              vocabulary=_vocab(ORDER_FORM, QUOTE))
    assert r.evidence_concept != "doctype.order_form"
    assert r.agreement != "disagreed", (
        "the quote workbook shape must not become a disagreement again"
    )


def test_an_order_form_that_names_its_framework_does_claim_the_page():
    page = ("ORDER FORM\n\nThis Order Form is made under Framework Agreement "
            "FW-2024-0012.\n")
    r = resolve_document_type(declared_concept=None, full_text=page,
                              vocabulary=_vocab(ORDER_FORM, QUOTE))
    assert r.evidence_concept == "doctype.order_form", r
    assert r.status == "matched"


def test_an_order_of_precedence_clause_is_also_parent_evidence():
    page = ("ORDER FORM\n\n2.1 The documents take effect in the following order of "
            "precedence.\n")
    r = resolve_document_type(declared_concept=None, full_text=page,
                              vocabulary=_vocab(ORDER_FORM, QUOTE))
    assert r.evidence_concept == "doctype.order_form", r


def test_standing_down_also_removes_the_structure_from_tier_two():
    """Not just the title. A body-only mention must not win either.

    Filtering title_concepts alone would leave a stood-down structure winning on
    repeated body mentions, which is the same defect one line further down.
    """
    # No title segment at all, so only tier 2 (body mentions) can name a type.
    page = ("Please sign the order form and return it. The order form "
            "must be returned with the order form cover sheet.\n")
    r = resolve_document_type(declared_concept=None, full_text=page,
                              vocabulary=_vocab(ORDER_FORM, QUOTE))
    assert r.evidence_concept != "doctype.order_form", r
    assert "doctype.order_form" not in r.candidates
    assert all(ev.concept_code != "doctype.order_form" for ev in r.evidence), (
        "a structure that stood down must not appear as evidence either"
    )


def test_a_title_naming_both_stays_unresolved_with_both_candidates():
    """Review Focus 1: the rule must not turn a genuine tie into a confident answer.

    The page names a framework, so the order form does NOT stand down — and then
    two structures claim the same title segment, which is a tie a person settles.
    """
    page = "ORDER FORM / FRAMEWORK AGREEMENT\n\nmade between the parties.\n"
    r = resolve_document_type(declared_concept=None, full_text=page,
                              vocabulary=_vocab(ORDER_FORM, QUOTE, FRAMEWORK))
    assert r.status == "unresolved", r
    assert set(r.candidates) == {"doctype.order_form", "doctype.framework_agreement"}, r
    assert r.evidence_concept is None


def test_an_unflagged_structure_is_never_stood_down():
    page = "QUOTATION\n\nValid Until 2026-12-01\n"
    r = resolve_document_type(declared_concept="doctype.quote", full_text=page,
                              vocabulary=_vocab(ORDER_FORM, QUOTE))
    assert r.evidence_concept == "doctype.quote"
    assert r.agreement == "agreed"


def test_a_flagged_structure_with_no_phrases_claims_nothing():
    """The validation check in Task 2 reports this; the resolver must not guess.

    A structure that must show its parent, with no phrase to recognise one by,
    cannot be satisfied — and inferring a default phrase set is how a classifier
    starts inventing parents.
    """
    muted = DocumentType(
        "doctype.order_form", "role.master", "doctype.framework_agreement",
        "exec.bilateral", ("order form",), (), (), "contract",
        requires_parent_evidence=True, parent_evidence_phrases=(),
    )
    page = ("ORDER FORM\n\nmade under Framework Agreement FW-1 in the following order "
            "of precedence.\n")
    r = resolve_document_type(declared_concept=None, full_text=page,
                              vocabulary=_vocab(muted, QUOTE))
    assert r.evidence_concept != "doctype.order_form", r


def test_parent_evidence_matches_with_alias_normalisation():
    """'Call-Off' on the page satisfies the phrase 'call off'.

    Exercises only the '-'-to-space replacement in the match copy; it does not
    cover whitespace-run agreement (a phrase does not match across a line break).
    """
    page = "ORDER FORM\n\nissued under the Call-Off procedure.\n"
    r = resolve_document_type(declared_concept=None, full_text=page,
                              vocabulary=_vocab(ORDER_FORM, QUOTE))
    assert r.evidence_concept == "doctype.order_form", r


def _claims(page):
    r = resolve_document_type(declared_concept=None, full_text=page,
                              vocabulary=_vocab(ORDER_FORM, QUOTE))
    return r.evidence_concept == "doctype.order_form"


def test_a_company_named_incorporated_is_not_parent_evidence():
    assert not _claims("ORDER FORM\n\nAcme Incorporated\nQuote Ref Q-1\n")


def test_the_plural_frameworks_is_not_parent_evidence():
    assert not _claims("ORDER FORM\n\nour frameworks for delivery\n")


def test_recall_off_site_is_not_a_call_off():
    assert not _claims("ORDER FORM\n\nplease recall off-site stock\n")


def test_incorporated_into_is_parent_evidence():
    assert _claims("ORDER FORM\n\nThe terms are incorporated into this order form.\n")


def test_incorporated_by_reference_is_parent_evidence():
    assert _claims("ORDER FORM\n\nThe master terms are incorporated by reference.\n")

# ---------------------------------------------------------------------------
# Final review, Important 2: standing down must not contradict the uploader.
# ---------------------------------------------------------------------------

ORDER = DocumentType(
    "doctype.order", "role.transaction", "doctype.call_off_contract", "exec.bilateral",
    ("purchase order", "po", "order"), (), (), "purchase_order",
)


def test_an_order_form_that_stood_down_is_not_contradicted_by_the_leftovers():
    """A person uploaded this through the Contracts zone AS an order form.

    The page names no framework, so the structure stands down — correctly. What
    must not then happen is the bare `order` alias inside the heading claiming
    the page and the result asserting the document is a PURCHASE ORDER: a
    `document_type_disagreement` on the exact document class this rule exists to
    recognise, against a type from a different pipeline, raised against a
    declaration that standing down has made impossible to agree with.

    `declared_only` is the honest answer: the declared structure was never
    evaluated, so nothing contradicted it.
    """
    page = ("ORDER FORM\n\nOrder Form No. OF-2026-0118\n\nPrinting and fulfilment "
            "of 20,000 brochures. Total GBP 12,500.\n")
    r = resolve_document_type(declared_concept="doctype.order_form", full_text=page,
                              vocabulary=_vocab(ORDER_FORM, ORDER, QUOTE))
    assert r.agreement != "disagreed", (
        f"the uploader said order form and the result contradicts them with "
        f"{r.evidence_concept}: {r}")
    assert r.agreement == "declared_only", r


def test_a_stood_down_declaration_raises_no_review_item():
    """The agreement value is only half of it: the finding is what a person sees."""
    from src.services.extraction.type_resolver import type_resolution_discrepancies
    page = ("ORDER FORM\n\nOrder Form No. OF-2026-0118\n\nPrinting and fulfilment "
            "of 20,000 brochures.\n")
    r = resolve_document_type(declared_concept="doctype.order_form", full_text=page,
                              vocabulary=_vocab(ORDER_FORM, ORDER, QUOTE))
    assert type_resolution_discrepancies(r) == []


def test_a_declared_type_that_did_not_stand_down_still_disagrees():
    """The narrow fix must not mute genuine disagreements. Nothing about THIS
    page's declaration stood down, so the contradiction stands."""
    page = "PURCHASE ORDER\n\nPurchase Order No. PO-4471\n"
    r = resolve_document_type(declared_concept="doctype.quote", full_text=page,
                              vocabulary=_vocab(ORDER_FORM, ORDER, QUOTE))
    assert r.evidence_concept == "doctype.order", r
    assert r.agreement == "disagreed", r


def test_an_order_form_that_names_its_framework_is_still_agreed():
    """And the positive half is untouched: with parent evidence the structure
    does not stand down, so it claims the page and agrees with the uploader."""
    page = ("ORDER FORM\n\nThis Order Form is incorporated into Framework "
            "Agreement No. FA-2026-0042.\n")
    r = resolve_document_type(declared_concept="doctype.order_form", full_text=page,
                              vocabulary=_vocab(ORDER_FORM, ORDER, QUOTE))
    assert r.evidence_concept == "doctype.order_form", r
    assert r.agreement == "agreed", r
