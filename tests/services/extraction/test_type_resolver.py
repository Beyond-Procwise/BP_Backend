"""What the page says about its own type, alongside what the uploader declared.

The resolver's contract: it reports, it never overrules, and every piece of
evidence it returns is a verbatim substring of the text it was given.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.services.concepts.vocabulary import SEED_VOCABULARY, build_vocabulary  # noqa: E402
from src.services.extraction.type_resolver import (  # noqa: E402
    resolve_document_type,
)

V = SEED_VOCABULARY


def _row(code, aliases, *, status="active", signals=()):
    return {
        "concept_code": code, "role": "role.master", "aliases": list(aliases),
        "structural_signals": list(signals), "status": status,
        "default_parent_type": None, "execution_mode": None,
        "identifiers": [], "pipeline_doc_type": None,
    }

FRAMEWORK_PAGE = (
    "FRAMEWORK AGREEMENT\n"
    "Framework Ref: RM6100\n"
    "This framework agreement sets out the terms under which call-off "
    "contracts may be awarded. It orders no goods or services itself.\n"
)

ORDER_FORM_PAGE = (
    "ORDER FORM\n"
    "Framework Ref: RM6100\n"
    "2.1 The following documents are incorporated into this order form.\n"
    "2.2 In the event of conflict the order of precedence is as follows.\n"
)

INVOICE_PAGE = (
    "TAX INVOICE\n"
    "Invoice No: INV-2026-0001\n"
    "Amount due: 1,250.00\n"
)

BLANK_PAGE = "Dear Sir or Madam,\n\nPlease find attached.\n\nKind regards\n"


def test_agreement_when_the_page_says_what_the_uploader_said():
    r = resolve_document_type(
        declared_concept="doctype.invoice", full_text=INVOICE_PAGE, vocabulary=V,
    )
    assert r.status == "matched"
    assert r.agreement == "agreed"
    assert r.declared_concept == "doctype.invoice"
    assert r.evidence_concept == "doctype.invoice"


def test_evidence_is_always_a_verbatim_substring_of_the_page():
    """The grounding contract the rest of extraction holds. A span that is not
    in the text cannot be shown to a reviewer as the reason."""
    r = resolve_document_type(
        declared_concept="doctype.invoice", full_text=INVOICE_PAGE, vocabulary=V,
    )
    assert r.evidence, "an invoice page should produce evidence"
    for ev in r.evidence:
        assert ev.text in INVOICE_PAGE, f"{ev.text!r} is not in the page"
        assert INVOICE_PAGE[ev.start:ev.start + len(ev.text)] == ev.text


def test_disagreement_is_recorded_and_the_declared_type_is_kept():
    """Review Focus 5. A file uploaded as a quote whose page is plainly an
    invoice must not be reclassified or rerouted by this layer."""
    r = resolve_document_type(
        declared_concept="doctype.quote", full_text=INVOICE_PAGE, vocabulary=V,
    )
    assert r.agreement == "disagreed"
    assert r.declared_concept == "doctype.quote"
    assert r.evidence_concept == "doctype.invoice"
    # The resolver reports. It does not decide.
    assert not hasattr(r, "pipeline_doc_type")


def test_a_more_specific_type_in_the_page_is_reported_not_applied():
    """Declared 'contract', page says 'framework agreement'. That is a
    refinement a human confirms, not one this layer makes on its own."""
    r = resolve_document_type(
        declared_concept="doctype.contract_unspecified",
        full_text=FRAMEWORK_PAGE, vocabulary=V,
    )
    assert r.declared_concept == "doctype.contract_unspecified"
    assert r.evidence_concept == "doctype.framework_agreement"
    assert r.agreement == "disagreed"


def test_a_page_matching_nothing_is_unknown_not_the_nearest_option():
    r = resolve_document_type(
        declared_concept=None, full_text=BLANK_PAGE, vocabulary=V,
    )
    assert r.status == "unknown"
    assert r.evidence_concept is None
    assert r.candidates == ()


def test_a_tie_is_unresolved_and_carries_both_candidates():
    """An invoice heading and a quote heading are equal evidence. Equal evidence
    must not be broken by row order or alphabet."""
    r = resolve_document_type(
        declared_concept=None, full_text="INVOICE\nQUOTE\n", vocabulary=V,
    )
    assert r.status == "unresolved"
    assert r.evidence_concept is None
    assert set(r.candidates) == {"doctype.invoice", "doctype.quote"}


def test_two_concepts_claiming_one_alias_is_a_tie_not_a_choice():
    """resolve_alias returns a tuple; a two-owner tuple is still truthy, so the
    collision must come out as unresolved, never as owners[0]."""
    vocab = build_vocabulary([], [
        _row("doctype.alpha", ["shared heading"]),
        _row("doctype.beta", ["shared heading"]),
    ], source="test")
    r = resolve_document_type(
        declared_concept=None, full_text="SHARED HEADING\n", vocabulary=vocab,
    )
    assert r.status == "unresolved"
    assert r.evidence_concept is None
    assert set(r.candidates) == {"doctype.alpha", "doctype.beta"}


def test_a_title_hit_outweighs_a_body_mention():
    """An invoice that merely cites a purchase order is still an invoice."""
    page = "TAX INVOICE\nInvoice No: 1\nAgainst purchase order PO123456.\n"
    r = resolve_document_type(declared_concept=None, full_text=page, vocabulary=V)
    assert r.evidence_concept == "doctype.invoice"


def test_declared_only_when_the_page_is_silent():
    r = resolve_document_type(
        declared_concept="doctype.invoice", full_text=BLANK_PAGE, vocabulary=V,
    )
    assert r.agreement == "declared_only"
    assert r.declared_concept == "doctype.invoice"
    assert r.evidence_concept is None
    assert r.status == "matched"


def test_structural_signals_are_evidence_but_do_not_decide_alone():
    """A signal that appears on the page as a phrase is evidence, verbatim, but
    a page with only that signal names no type."""
    vocab = build_vocabulary([], [
        _row("doctype.alpha", ["alpha heading"], signals=["order of precedence"]),
    ], source="test")
    page = "Intro\nThe Order of Precedence is as follows.\n"
    r = resolve_document_type(declared_concept="doctype.alpha", full_text=page,
                              vocabulary=vocab)
    sig = [e for e in r.evidence if e.kind == "structural_signal"]
    assert [e.text for e in sig] == ["Order of Precedence"]
    assert page[sig[0].start:sig[0].start + len(sig[0].text)] == sig[0].text
    assert r.evidence_concept is None  # 0.5 is below the naming threshold
    assert r.agreement == "declared_only"


def test_seeded_structural_signals_are_prose_and_never_match_a_page():
    """FINDING: the seed's structural_signals are descriptions of a page
    ('lists incorporated documents'), not phrases on it, so they cannot be
    matched as verbatim substrings. Pinned so that it is noticed if the seed
    ever starts carrying phrases."""
    r = resolve_document_type(
        declared_concept="doctype.call_off_contract",
        full_text=ORDER_FORM_PAGE, vocabulary=V,
    )
    assert "structural_signal" not in {ev.kind for ev in r.evidence}


def test_the_longest_phrase_wins_over_the_words_inside_it():
    for page, want in [
        ("MASTER SERVICE AGREEMENT\n", "doctype.master_agreement"),
        ("NON-DISCLOSURE AGREEMENT\n", "doctype.nda"),
        ("FRAMEWORK AGREEMENT\n", "doctype.framework_agreement"),
        ("PURCHASE ORDER\nPO: 1234\n", "doctype.order"),
    ]:
        r = resolve_document_type(declared_concept=None, full_text=page, vocabulary=V)
        assert (r.status, r.evidence_concept) == ("matched", want), page


def test_evidence_offsets_are_byte_exact_even_when_lowercasing_changes_length():
    """\u0130 lowercases to TWO characters; a naive lower() would shift every
    later offset and the span would no longer sit at its recorded start."""
    page = "\u0130\u0130\u0130 header\nTAX  INVOICE no\nTax Invoice No: 1\n"
    r = resolve_document_type(declared_concept=None, full_text=page, vocabulary=V)
    assert r.evidence
    for e in r.evidence:
        assert page[e.start:e.start + len(e.text)] == e.text, e
    assert any(e.text == "Tax Invoice" for e in r.evidence)


def test_a_declared_type_is_never_replaced_by_the_evidence():
    r = resolve_document_type(
        declared_concept="doctype.quote", full_text=INVOICE_PAGE, vocabulary=V,
    )
    assert r.declared_concept == "doctype.quote"
    assert r.evidence_concept == "doctype.invoice"
    assert r.status == "matched" and r.agreement == "disagreed"


def test_an_empty_page_does_not_crash():
    r = resolve_document_type(declared_concept=None, full_text="", vocabulary=V)
    assert r.status == "unknown"
    assert r.evidence == ()


def test_a_proposed_type_is_never_the_evidence_answer():
    """doctype.policy_document is seeded as proposed, so its alias 'policy'
    must not resolve even when the page says it outright."""
    r = resolve_document_type(
        declared_concept=None, full_text="POLICY\nThis policy applies.\n", vocabulary=V,
    )
    assert r.evidence_concept != "doctype.policy_document"
    assert r.status == "unknown"
    assert "doctype.policy_document" not in r.candidates


def test_a_proposed_row_handed_to_the_builder_still_never_resolves():
    vocab = build_vocabulary([], [
        _row("doctype.pol", ["policy"], status="proposed"),
    ], source="test")
    r = resolve_document_type(
        declared_concept=None, full_text="POLICY\n", vocabulary=vocab,
    )
    assert r.status == "unknown" and r.candidates == ()


def test_a_proposed_type_inside_a_hand_built_vocabulary_never_resolves():
    """Bypasses build_vocabulary's own filter, so this proves the resolver holds
    the rule itself rather than relying on the loader."""
    from dataclasses import replace
    from src.services.concepts.seed import DOCUMENT_TYPES
    proposed = DOCUMENT_TYPES["doctype.policy_document"]
    assert proposed.status == "proposed"
    vocab = replace(
        V,
        document_types={**V.document_types, proposed.concept_code: proposed},
    )
    r = resolve_document_type(
        declared_concept=None, full_text="POLICY\n", vocabulary=vocab,
    )
    assert r.status == "unknown" and r.evidence_concept is None


def test_a_concepts_own_name_does_not_double_count_against_a_shared_alias():
    """doctype.alpha lists 'alpha' AND is named alpha. Counted twice it would
    beat doctype.beta, which claims the same word once: a collision silently
    won by whichever concept happened to repeat itself."""
    vocab = build_vocabulary([], [
        _row("doctype.alpha", ["alpha"]),
        _row("doctype.beta", ["alpha"]),
    ], source="test")
    r = resolve_document_type(
        declared_concept=None, full_text="ALPHA\n", vocabulary=vocab,
    )
    assert r.status == "unresolved"
    assert set(r.candidates) == {"doctype.alpha", "doctype.beta"}
