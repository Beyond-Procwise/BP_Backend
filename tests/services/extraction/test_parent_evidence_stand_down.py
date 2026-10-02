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
