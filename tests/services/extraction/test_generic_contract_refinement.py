"""'Contract' is the zone's name, not a claim about which contract it is.

The Contracts upload zone declares the category 'contract', which resolves to
doctype.contract_unspecified. A document that then names its real structure has
refined the declaration, not contradicted it. Without this, every contract Nick
uploads lands a review item for having been more specific than the dropdown.

Offline — hand-built vocabularies, no database.
    CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest \
        tests/services/extraction/test_generic_contract_refinement.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.services.concepts.seed import DocumentType                  # noqa: E402
from src.services.concepts.vocabulary import build_vocabulary         # noqa: E402
from src.services.extraction.type_resolver import (                   # noqa: E402
    resolve_document_type, type_resolution_discrepancies,
)

GENERIC = DocumentType(
    "doctype.contract_unspecified", "role.master", None, None,
    ("contract", "agreement", "contracts"), (), (), "contract",
)
MSA = DocumentType(
    "doctype.master_agreement", "role.master", None, "exec.bilateral",
    ("master agreement", "msa"), (), (), "contract",
)
SOW = DocumentType(
    "doctype.sow", "role.master", "doctype.master_agreement", "exec.bilateral",
    ("sow", "statement of work"), (), (), "contract",
)
INVOICE = DocumentType(
    "doctype.invoice", "role.transaction", None, "exec.unilateral",
    ("invoice", "tax invoice"), (), (), "invoice",
)
NOTICE = DocumentType(
    # pipeline_doc_type is None: recognised, but nothing ingests it.
    "doctype.notice_general", "role.notice", None, "exec.unilateral",
    ("general notice", "notice"), (), (), None,
)


def _vocab(*types):
    concepts = [
        {"concept_code": t.concept_code, "domain": "DOCUMENT_TYPE", "definition": "d",
         "not_to_be_confused_with": [], "status": "active", "rejection_reason": None}
        for t in types
    ]
    rows = [
        {"concept_code": t.concept_code, "role": t.role,
         "default_parent_type": t.default_parent_type, "execution_mode": t.execution_mode,
         "aliases": list(t.aliases), "identifiers": [], "structural_signals": [],
         "pipeline_doc_type": t.pipeline_doc_type, "status": "active",
         "requires_parent_evidence": False, "parent_evidence_phrases": []}
        for t in types
    ]
    return build_vocabulary(concepts, rows, source="test")


V = _vocab(GENERIC, MSA, SOW, INVOICE, NOTICE)


def test_a_master_agreement_uploaded_as_a_contract_is_a_refinement():
    r = resolve_document_type(
        declared_concept="doctype.contract_unspecified",
        full_text="MASTER AGREEMENT\n\nbetween the parties.\n", vocabulary=V,
    )
    assert (r.agreement, r.evidence_concept) == ("refined", "doctype.master_agreement")
    assert r.status == "matched"


def test_a_refinement_raises_no_review_item():
    r = resolve_document_type(
        declared_concept="doctype.contract_unspecified",
        full_text="STATEMENT OF WORK\n\nDeliverables and milestones.\n", vocabulary=V,
    )
    assert r.agreement == "refined"
    assert type_resolution_discrepancies(r) == []


def test_an_invoice_uploaded_as_a_contract_still_disagrees():
    """Refinement is bounded by the pipeline, so a transaction document is not one."""
    r = resolve_document_type(
        declared_concept="doctype.contract_unspecified",
        full_text="TAX INVOICE\n\nAmount due on receipt.\n", vocabulary=V,
    )
    assert (r.agreement, r.evidence_concept) == ("disagreed", "doctype.invoice")
    items = type_resolution_discrepancies(r)
    assert len(items) == 1
    assert items[0].issue_type == "document_type_disagreement"


def test_a_structure_with_no_pipeline_is_not_a_refinement():
    """doctype.notice_general has pipeline_doc_type NULL.

    'Recognised but nothing ingests it' is not 'a kind of contract', and reading
    it as a refinement would silence a finding on a document the contract
    pipeline cannot actually handle.
    """
    r = resolve_document_type(
        declared_concept="doctype.contract_unspecified",
        full_text="GENERAL NOTICE\n\nFor information only.\n", vocabulary=V,
    )
    assert r.agreement == "disagreed", r
    assert len(type_resolution_discrepancies(r)) == 1


def test_one_specific_structure_declared_against_another_still_disagrees():
    """Only the GENERIC declaration can be refined.

    Uploading as a SOW a document that reads as a master agreement is a real
    contradiction: the uploader made a specific claim and the page disputes it.
    """
    r = resolve_document_type(
        declared_concept="doctype.sow",
        full_text="MASTER AGREEMENT\n\nbetween the parties.\n", vocabulary=V,
    )
    assert (r.agreement, r.evidence_concept) == ("disagreed", "doctype.master_agreement")
    assert len(type_resolution_discrepancies(r)) == 1


def test_a_generic_contract_reading_as_generic_is_agreed_not_refined():
    r = resolve_document_type(
        declared_concept="doctype.contract_unspecified",
        full_text="CONTRACT\n\nnumbered clauses and a signature block.\n", vocabulary=V,
    )
    assert r.agreement == "agreed"


def test_an_unresolved_tie_under_a_generic_declaration_still_reaches_a_human():
    """A refinement must not swallow a tie: two candidates still need a person.

    status, never agreement alone, is what the review-item builder reads.

    NOTE: on a tie evidence_concept is None, so agreement is 'declared_only' and
    the 'refined' branch is unreachable here; a tie can never be refined. What
    this test really guards is the `if status != "unresolved"` clause beside
    that branch: removing the clause turns this red, whereas forcing
    status = "matched" inside the refined branch cannot (nothing reaches it).
    """
    r = resolve_document_type(
        declared_concept="doctype.contract_unspecified",
        full_text="MASTER AGREEMENT / STATEMENT OF WORK\n\nbetween the parties.\n",
        vocabulary=V,
    )
    assert r.status == "unresolved", r
    assert len(type_resolution_discrepancies(r)) == 1
