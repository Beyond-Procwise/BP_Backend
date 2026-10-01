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
        declared_concept=None, full_text="INVOICE / QUOTE\n", vocabulary=V,
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
    """An invoice that merely cites a purchase order is still an invoice. The
    page is longer than the title zone, so the citation is a BODY hit; the
    winning span must be a title_alias, not just the one with more hits."""
    page = ("TAX INVOICE\nInvoice No: 1\n" + "x" * 700
            + "\nAgainst purchase order PO123456. Purchase order again.\n")
    r = resolve_document_type(declared_concept=None, full_text=page, vocabulary=V)
    assert r.evidence_concept == "doctype.invoice"
    top = [e for e in r.evidence if e.concept_code == "doctype.invoice"]
    assert top and top[0].kind == "title_alias" and top[0].start < 600
    order_kinds = {e.kind for e in r.evidence if e.concept_code == "doctype.order"}
    assert order_kinds <= {"body_alias"}


def test_title_chars_moves_the_boundary_between_title_and_body():
    page = "TAX INVOICE\n" + "y" * 50 + "\nsee the quote. the quote again.\n"
    def kinds(title_chars):
        r = resolve_document_type(declared_concept="doctype.quote", full_text=page,
                                  vocabulary=V, title_chars=title_chars)
        return {e.kind for e in r.evidence if e.concept_code == "doctype.quote"}
    assert kinds(600) == {"title_alias"}
    assert kinds(20) == {"body_alias"}


def test_body_repetition_never_buries_the_title():
    """A real contract says 'Agreement' dozens of times."""
    page = "MASTER SERVICE AGREEMENT\n" + "This Agreement governs the parties. " * 40
    r = resolve_document_type(declared_concept=None, full_text=page, vocabulary=V)
    assert r.status == "matched"
    assert r.evidence_concept == "doctype.master_agreement"


def test_one_passing_mention_past_the_title_zone_is_not_a_classification():
    page = "Dear Sir,\n" + "z" * 700 + "\nPlease see the invoice attached.\n"
    r = resolve_document_type(declared_concept=None, full_text=page, vocabulary=V)
    assert r.status == "unknown"
    assert r.evidence_concept is None


def test_two_body_mentions_do_classify():
    page = "Dear Sir,\n" + "z" * 700 + "\nThe invoice is attached. Pay the invoice.\n"
    r = resolve_document_type(declared_concept=None, full_text=page, vocabulary=V)
    assert r.evidence_concept == "doctype.invoice"


def test_agreement_value_for_every_situation():
    cases = [
        ("doctype.invoice", INVOICE_PAGE, "agreed"),
        ("doctype.quote", INVOICE_PAGE, "disagreed"),
        ("doctype.invoice", BLANK_PAGE, "declared_only"),
        (None, INVOICE_PAGE, "evidence_only"),
        (None, BLANK_PAGE, "neither"),
        (None, "INVOICE / QUOTE\n", "neither"),
        ("doctype.sow", "INVOICE / QUOTE\n", "declared_only"),
    ]
    for declared, page, want in cases:
        r = resolve_document_type(declared_concept=declared, full_text=page, vocabulary=V)
        assert r.agreement == want, (declared, page, r)
    tie = resolve_document_type(declared_concept="doctype.sow",
                                full_text="INVOICE / QUOTE\n", vocabulary=V)
    assert tie.status == "unresolved"
    assert set(tie.candidates) == {"doctype.invoice", "doctype.quote"}


def test_declared_only_when_the_page_is_silent():
    r = resolve_document_type(
        declared_concept="doctype.invoice", full_text=BLANK_PAGE, vocabulary=V,
    )
    assert r.agreement == "declared_only"
    assert r.declared_concept == "doctype.invoice"
    assert r.evidence_concept is None
    assert r.status == "matched"


def test_structural_signals_corroborate_but_never_name_a_type_alone():
    """Three phrase-shaped signals and no alias anywhere: still unknown, and the
    signals are not shown as a reason because they decided nothing."""
    vocab = build_vocabulary([], [
        _row("doctype.alpha", ["alpha heading"],
             signals=["order of precedence", "ship to address", "bill to address"]),
    ], source="test")
    page = ("Order of Precedence.\nShip-To Address: 1 High St\n"
            "Bill-To Address: 2 Low St\n")
    r = resolve_document_type(declared_concept="doctype.alpha", full_text=page,
                              vocabulary=vocab)
    assert r.status == "matched" and r.evidence_concept is None
    assert r.agreement == "declared_only"
    assert r.evidence == ()
    undeclared = resolve_document_type(declared_concept=None, full_text=page,
                                       vocabulary=vocab)
    assert undeclared.status == "unknown"


def test_a_signal_alongside_an_alias_is_evidence_verbatim():
    vocab = build_vocabulary([], [
        _row("doctype.alpha", ["alpha heading"], signals=["order of precedence"]),
    ], source="test")
    page = "ALPHA HEADING\nThe Order of Precedence is as follows.\n"
    r = resolve_document_type(declared_concept=None, full_text=page, vocabulary=vocab)
    assert r.evidence_concept == "doctype.alpha"
    sig = [e for e in r.evidence if e.kind == "structural_signal"]
    assert [e.text for e in sig] == ["Order of Precedence"]
    assert page[sig[0].start:sig[0].start + len(sig[0].text)] == sig[0].text


def test_a_signal_can_break_a_tie_between_equal_aliases():
    vocab = build_vocabulary([], [
        _row("doctype.alpha", ["shared"], signals=["order of precedence"]),
        _row("doctype.beta", ["shared"]),
    ], source="test")
    r = resolve_document_type(
        declared_concept=None, vocabulary=vocab,
        full_text="SHARED\nThe order of precedence applies.\n")
    assert (r.status, r.evidence_concept) == ("matched", "doctype.alpha")


def test_the_order_forms_call_off_signals_are_prose_and_do_not_match():
    """The call-off contract's seeded signals ('lists incorporated documents')
    are descriptions, not phrases, so this order-form page yields none. Other
    types' signals ('ship-to address') are phrase-shaped and do match."""
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


def test_a_phrase_shaped_seed_signal_matches_verbatim():
    page = "PURCHASE ORDER\nPO: 1234\nShip-To Address: 1 High St\n"
    r = resolve_document_type(declared_concept=None, full_text=page, vocabulary=V)
    sig = [e for e in r.evidence if e.kind == "structural_signal"]
    assert [e.text for e in sig] == ["Ship-To Address"]
    assert page[sig[0].start:sig[0].start + len(sig[0].text)] == sig[0].text


def test_the_heading_line_outranks_a_citation_on_the_next_line():
    r = resolve_document_type(declared_concept=None,
                              full_text="INVOICE\nQUOTE\n", vocabulary=V)
    assert (r.status, r.evidence_concept) == ("matched", "doctype.invoice")


def _quadratic_survivors(spans):
    """Reference rule: a hit is dropped iff a DIFFERENT span covers it."""
    return [a for a in spans
            if not any(b != a and b[0] <= a[0] and a[1] <= b[1] for b in spans)]


def test_the_single_sweep_matches_the_quadratic_reference_filter():
    """Differential: run the real resolver with every concept eligible and
    compare the surviving heading-tier spans with the reference rule over all
    raw alias occurrences."""
    import random
    from src.services.concepts.vocabulary import fold
    from src.services.extraction.type_resolver import _find_all
    words = ["master service agreement", "service agreement", "agreement",
             "invoice", "tax invoice", "po", "purchase order", "order", "x"]
    rng = random.Random(7)
    for _ in range(300):
        # One line so everything within 120 chars is the heading tier.
        page = " ".join(rng.choice(words) for _ in range(rng.randint(1, 6)))
        raw = set()
        for dt in V.document_types.values():
            for alias in {fold(a) for a in (*dt.aliases, dt.concept_code.split(".", 1)[-1])}:
                for st in _find_all(alias, page.lower()):
                    raw.add((st, st + len(alias)))
        want = {sp for sp in _quadratic_survivors(sorted(raw)) if sp[0] < 120}
        wide = resolve_document_type(
            declared_concept=None, full_text=page, vocabulary=V)
        got = {(e.start, e.start + len(e.text)) for e in wide.evidence
               if e.kind != "structural_signal"}
        # evidence only lists eligible concepts' spans, so it is a subset of the
        # reference survivors, and never contains a covered span.
        assert got <= want, page
        assert not (got & (raw - set(want))), page


def test_a_glancing_header_citation_does_not_bury_body_evidence():
    page = ("Acme Ltd\nRef: your quotation of 1 January\n" + "f" * 600
            + "\n" + "This invoice is due. Pay the invoice. " * 20)
    r = resolve_document_type(declared_concept=None, full_text=page, vocabulary=V)
    assert (r.status, r.evidence_concept) == ("matched", "doctype.invoice")


def test_a_documents_list_of_cross_references_does_not_outrank_its_heading():
    sched = ("The Schedules form part of this Agreement: Schedule 1, Annex A, "
             "Appendix 2, Exhibit B.\n")
    for heading, want in [("FRAMEWORK AGREEMENT", "doctype.framework_agreement"),
                          ("MASTER SERVICE AGREEMENT", "doctype.master_agreement"),
                          ("NON-DISCLOSURE AGREEMENT", "doctype.nda")]:
        page = heading + "\nRef: RM6100\n" + sched
        r = resolve_document_type(declared_concept=want, full_text=page, vocabulary=V)
        assert r.evidence_concept == want, heading
        assert r.agreement == "agreed", heading
    r = resolve_document_type(
        declared_concept=None, vocabulary=V,
        full_text="INVOICE\nSee our Quotation, Estimate, Price Quotation and prior Quotes.\n")
    assert r.evidence_concept == "doctype.invoice"


def test_the_heading_is_bounded_when_there_is_no_newline():
    """Unreliable line breaks must not make the whole page 'the heading'."""
    page = "TAX INVOICE " + "lorem ipsum dolor " * 40 + "purchase order PO1 purchase order"
    r = resolve_document_type(declared_concept=None, full_text=page, vocabulary=V)
    assert (r.status, r.evidence_concept) == ("matched", "doctype.invoice")
    swapped = "TAX INVOICE " + "lorem ipsum dolor " * 40 + "purchase order"
    assert resolve_document_type(declared_concept=None, full_text=swapped,
                                 vocabulary=V).evidence_concept == "doctype.invoice"


def test_evidence_returned_is_only_what_counted():
    page = "Dear Sir,\n" + "z" * 10 + "\nPay the invoice. The invoice is due. Quote.\n"
    r = resolve_document_type(declared_concept=None, full_text=page, vocabulary=V)
    assert r.evidence_concept == "doctype.invoice"
    assert {e.concept_code for e in r.evidence} == {"doctype.invoice"}
    # tier-1 concept: its non-heading mentions did not count and are not shown
    page2 = "INVOICE\nPay the invoice. The invoice is due.\n"
    r2 = resolve_document_type(declared_concept=None, full_text=page2, vocabulary=V)
    assert [e.start for e in r2.evidence] == [0]


def test_a_declared_concept_that_only_glanced_off_the_page_shows_no_evidence():
    page = "INVOICE\nWe also saw the quote once.\n"
    r = resolve_document_type(declared_concept="doctype.quote", full_text=page,
                              vocabulary=V)
    assert r.agreement == "disagreed"
    assert {e.concept_code for e in r.evidence} == {"doctype.invoice"}


def test_a_losing_runner_up_is_not_shown_as_the_reason():
    page = "INVOICE\nPay the invoice.\n" + "x" * 5 + "\nquote quote quote\n"
    r = resolve_document_type(declared_concept=None, full_text=page, vocabulary=V)
    assert r.evidence_concept == "doctype.invoice"
    assert {e.concept_code for e in r.evidence} == {"doctype.invoice"}
