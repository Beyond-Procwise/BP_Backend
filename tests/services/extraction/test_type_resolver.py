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


def _build(rows, *, source="test"):
    """build_vocabulary over ``rows`` with the concepts they need.

    Not optional scaffolding: `status` lives on proc.bp_concept AND
    proc.bp_document_type for the same type, and build_vocabulary drops a type
    whose concept is absent — otherwise a type promoted on one table alone
    resolves and routes with no definition behind it. Concepts are always
    ACTIVE here so that each test's own `status=` on the TYPE row is what it is
    testing.
    """
    concepts = [
        {"concept_code": "role.master", "domain": "RELATIONSHIP_ROLE",
         "definition": "Governs a relationship.", "not_to_be_confused_with": [],
         "status": "active", "rejection_reason": None},
    ]
    concepts += [
        {"concept_code": r["concept_code"], "domain": "DOCUMENT_TYPE",
         "definition": "x", "not_to_be_confused_with": [],
         "status": "active", "rejection_reason": None}
        for r in rows
    ]
    return build_vocabulary(concepts, rows, source=source)

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
    vocab = _build([
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
    vocab = _build([
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
    vocab = _build([
        _row("doctype.alpha", ["alpha heading"], signals=["order of precedence"]),
    ], source="test")
    page = "ALPHA HEADING\nThe Order of Precedence is as follows.\n"
    r = resolve_document_type(declared_concept=None, full_text=page, vocabulary=vocab)
    assert r.evidence_concept == "doctype.alpha"
    sig = [e for e in r.evidence if e.kind == "structural_signal"]
    assert [e.text for e in sig] == ["Order of Precedence"]
    assert page[sig[0].start:sig[0].start + len(sig[0].text)] == sig[0].text


def test_a_signal_can_break_a_tie_between_equal_aliases():
    """Tier 2 only. A tier-1 tie is two types named by the document's own
    title, and no amount of corroboration may pick one of those (rule D)."""
    vocab = _build([
        _row("doctype.alpha", ["shared"], signals=["order of precedence"]),
        _row("doctype.beta", ["shared"]),
    ], source="test")
    r = resolve_document_type(
        declared_concept=None, vocabulary=vocab,
        full_text="Dear Sir\nWe hold the shared and the shared.\n"
                  "The order of precedence applies.\n")
    assert (r.status, r.evidence_concept) == ("matched", "doctype.alpha")


def test_a_signal_cannot_break_a_tie_the_title_itself_states():
    """Rule D: a title naming two types is unresolved, full stop."""
    vocab = _build([
        _row("doctype.alpha", ["shared"], signals=["order of precedence"]),
        _row("doctype.beta", ["shared"]),
    ], source="test")
    r = resolve_document_type(
        declared_concept=None, vocabulary=vocab,
        full_text="SHARED\nThe order of precedence applies.\n")
    assert (r.status, r.evidence_concept) == ("unresolved", None)
    assert set(r.candidates) == {"doctype.alpha", "doctype.beta"}


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
    vocab = _build([
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
    won by whichever concept happened to repeat itself. Stated on a page with
    no title, because that is where counting happens at all."""
    vocab = _build([
        _row("doctype.alpha", ["alpha"]),
        _row("doctype.beta", ["alpha"]),
    ], source="test")
    r = resolve_document_type(
        declared_concept=None, vocabulary=vocab,
        full_text="Dear Sir\nWe hold the alpha thing. The alpha again.\n",
    )
    assert r.status == "unresolved"
    assert set(r.candidates) == {"doctype.alpha", "doctype.beta"}
    titled = resolve_document_type(
        declared_concept=None, full_text="ALPHA\n", vocabulary=vocab,
    )
    assert titled.status == "unresolved"
    assert set(titled.candidates) == {"doctype.alpha", "doctype.beta"}


def test_a_phrase_shaped_seed_signal_matches_verbatim():
    page = "PURCHASE ORDER\nPO: 1234\nShip-To Address: 1 High St\n"
    r = resolve_document_type(declared_concept=None, full_text=page, vocabulary=V)
    sig = [e for e in r.evidence if e.kind == "structural_signal"]
    assert [e.text for e in sig] == ["Ship-To Address"]
    assert page[sig[0].start:sig[0].start + len(sig[0].text)] == sig[0].text


def test_a_heading_line_outranks_a_citation_on_a_prose_line():
    r = resolve_document_type(
        declared_concept=None, vocabulary=V,
        full_text="INVOICE\nPlease see the quote attached.\n")
    assert (r.status, r.evidence_concept) == ("matched", "doctype.invoice")


def test_a_title_naming_two_types_on_one_segment_is_a_tie():
    """Rule D. The tie that stays reachable is a title that says both, in a
    line or in a cell of its own."""
    for page in ("INVOICE / QUOTE\n", "| INVOICE / QUOTE |\n",
                 "Acme Ltd\n## INVOICE / QUOTE ##\nAmount due: 1.00\n"):
        r = resolve_document_type(declared_concept=None, full_text=page, vocabulary=V)
        assert (r.status, r.evidence_concept) == ("unresolved", None), page
        assert set(r.candidates) == {"doctype.invoice", "doctype.quote"}, page
    three = resolve_document_type(declared_concept=None, vocabulary=V,
                                  full_text="INVOICE / QUOTE / ORDER\n")
    assert three.status == "unresolved"
    assert set(three.candidates) == {
        "doctype.invoice", "doctype.quote", "doctype.order"}


def test_the_first_title_in_the_page_wins_and_later_ones_do_not_compete():
    """Rule C, and the answer to the round-3 'INVOICE\\nQUOTE' conflict: the
    second line is a second title-like segment, not a competitor. Ordinal, so
    no character count and no repetition decides it."""
    r = resolve_document_type(declared_concept=None, full_text="INVOICE\nQUOTE\n",
                              vocabulary=V)
    assert (r.status, r.evidence_concept) == ("matched", "doctype.invoice")
    swapped = resolve_document_type(declared_concept=None, full_text="QUOTE\nINVOICE\n",
                                    vocabulary=V)
    assert swapped.evidence_concept == "doctype.quote"


LETTERHEAD_ROWS = [
    ("doctype.order", "Acme Global Ltd\nPURCHASE ORDER\nShip-To Address: 1 High St\n"
                      "Bill-To Address: 2 Low St\nPlease bill monthly.\n",
     "doctype.order"),
    ("doctype.invoice", "Acme Ltd\nTAX INVOICE\nAgainst purchase order PO1 and "
                        "purchase order PO2.\n", "doctype.invoice"),
    ("doctype.invoice", "Acme Ltd, 1 High St\nINVOICE\nThe order of precedence and "
                        "the order of works.\n", "doctype.invoice"),
    ("doctype.invoice", "Acme Ltd\nTAX INVOICE\nAgainst purchase order PO123456.\n",
     "doctype.invoice"),
]


def test_a_heading_after_a_letterhead_still_names_the_document():
    for declared, page, want in LETTERHEAD_ROWS:
        r = resolve_document_type(declared_concept=declared, full_text=page, vocabulary=V)
        assert (r.status, r.evidence_concept, r.agreement) == ("matched", want, "agreed"), page


def test_one_character_of_padding_cannot_invert_the_answer():
    for n in (119, 120, 121):
        page = "x" * n + " tax invoice is due\nAgainst purchase order. purchase order.\n"
        r = resolve_document_type(declared_concept=None, full_text=page, vocabulary=V)
        assert r.evidence_concept == "doctype.order", n  # prose line: volume decides
    answers = {resolve_document_type(
        declared_concept=None, vocabulary=V,
        full_text="x" * n + "\ntax invoice\nAgainst purchase order. purchase order.\n"
    ).evidence_concept for n in (119, 120, 121, 500)}
    assert answers == {"doctype.invoice"}


def test_a_page_with_no_line_breaks_has_no_heading_so_volume_decides():
    flat = "TAX INVOICE " + "lorem ipsum dolor " * 40 + "purchase order PO1 purchase order"
    r = resolve_document_type(declared_concept=None, full_text=flat, vocabulary=V)
    assert (r.status, r.evidence_concept) == ("matched", "doctype.order")
    broken = flat.replace("TAX INVOICE ", "TAX INVOICE\n", 1)
    assert resolve_document_type(declared_concept=None, full_text=broken,
                                 vocabulary=V).evidence_concept == "doctype.invoice"


QUOTE_BODY = "The quote is here. Pay the quote.\n"


def _who(first_line):
    return resolve_document_type(declared_concept=None, vocabulary=V,
                                 full_text=first_line + "\n" + QUOTE_BODY).evidence_concept


#: Every label the reviewer measured against the coverage fraction, with the
#: fraction it scored. The fractions run from 0.667 to 1.000 and 'PURCHASE
#: ORDER' also scores 1.000, so no cut point separates them. Rule A does it by
#: equality instead: none of these IS a type phrase...
MEASURED_LABELS = [
    "Bill To", "Order No", "Contract sum", "Agreement ref", "Invoice To",
    "Contract No", "Quotation to", "PURCHASE ORDER FORM",
    "TAX INVOICE for services rendered in period",
]
#: ...except the four that are a type phrase plus a bare '#'. Markup stripping
#: is what rule A asks for, so as a line of their own these DO name a type —
#: the same accepted class as a lone 'Invoice' footer. Rule B is what retires
#: them, because that is the shape they were measured in: a table cell with its
#: value in the next cell.
HASH_LABELS = {"Contract #": "doctype.contract_unspecified", "PO #": "doctype.order",
               "Invoice #": "doctype.invoice", "Quote #": "doctype.quote"}

#: And what IS a title, by equality, with no fraction involved.
REAL_TITLES = {
    "PURCHASE ORDER": "doctype.order",
    "## INVOICE": "doctype.invoice",
    "## INVOICE ##": "doctype.invoice",
    "**PURCHASE ORDER**": "doctype.order",
    "tax invoice": "doctype.invoice",
    "Schedule 1": "doctype.schedule",
    "Annex A": "doctype.schedule",
    "Appendix 2.1": "doctype.schedule",
    "Part IV": None,          # not a type phrase at all
    "'INVOICE'": "doctype.invoice",
    "INVOICE.": "doctype.invoice",
}


def test_a_segment_is_a_title_only_when_it_IS_the_type_phrase():
    """Rule A. Equality after normalisation, never 'mostly'. A continuous
    coverage fraction cannot make this categorical distinction: 'Bill To'
    scored 0.667, 'Schedule 1' 0.889 and 'PO #' 1.000, the same as 'PURCHASE
    ORDER'."""
    for label in MEASURED_LABELS:
        assert _who(label) == "doctype.quote", label  # the body decides, not the label
    for label, names in HASH_LABELS.items():
        # A line of its own, so accepted: '#' is markup and rule A strips it.
        assert _who(label) == names, label
    for title, want in REAL_TITLES.items():
        r = resolve_document_type(declared_concept=None, vocabulary=V,
                                  full_text=title + "\n" + QUOTE_BODY)
        assert r.evidence_concept == (want or "doctype.quote"), title


def test_a_long_line_that_merely_contains_a_type_word_is_not_a_title():
    """What the deleted 80-character bound and 0.75 coverage fraction were for.
    Equality subsumes both: the line is not the phrase at any length."""
    assert _who("INVOICE") == "doctype.invoice"
    assert _who("INVOICE ab") == "doctype.quote"
    assert _who("INVOICE" + " " * 72 + "bc") == "doctype.quote"
    assert _who("INVOICE" + " " * 73 + "bc") == "doctype.quote"
    # These two tie with the body's quotes in tier 2. A title would have been a
    # decided 'order', so 'not order' is exactly the claim: they are not titles.
    assert _who("Against purchase order PO1 and purchase order PO2.") != "doctype.order"
    assert _who("purchase order purchase order") != "doctype.order"


def test_a_multi_cell_table_row_is_field_data_not_a_title():
    """Rule B, and the reviewer's finding 2. 'PO #' normalises to 'PO', which
    IS an alias, so equality alone would make this page an order. In a row
    with a further non-empty cell to its right a cell is a key whose value that
    cell is, and a key is not a heading."""
    page = ("## Sheet: Billing\n"
            "| Orbis Industrial Supplies Limited, 14 Dock Road, Hull | "
            "TAX INVOICE for services rendered in period |\n"
            "| PO # | 4412 |\n"
            "| Amount due | 1,250.00 |\n")
    r = resolve_document_type(declared_concept="doctype.invoice", full_text=page,
                              vocabulary=V)
    assert r.evidence_concept != "doctype.order"
    assert (r.evidence_concept, r.agreement) == (None, "declared_only")
    for label in [*MEASURED_LABELS, *HASH_LABELS, "Schedule 1", "PURCHASE ORDER"]:
        cell = resolve_document_type(
            declared_concept=None, vocabulary=V,
            full_text="| " + label + " | Smith Ltd |\n" + QUOTE_BODY)
        assert cell.evidence_concept == "doctype.quote", label


def test_the_last_non_empty_cell_of_a_row_can_still_be_the_title():
    """The live invoice/PO/quote workbooks put the title in the last non-empty
    cell of row 1, beside the supplier's name and a logo letter. Nothing sits
    to its right, so it is not a key."""
    for row, want in [
        ("| O | Orbis Platform Solutions Ltd |  | INVOICE |  |", "doctype.invoice"),
        ("| Assurity Ltd |  |  | PURCHASE ORDER |  |", "doctype.order"),
        # Was "Order Form" until that alias was dropped from doctype.call_off_contract
        # (it titled every quote-template workbook; see the rulings doc). The row shape
        # is what this test is about, so the case is kept on a surviving call-off alias.
        ("| A | Aureus Workflow Ltd |  | Call-Off Contract |  |",
         "doctype.call_off_contract"),
        ("| **Assurity Ltd**   | **PURCHASE ORDER**   |", "doctype.order"),
        ("| INVOICE |", "doctype.invoice"),
    ]:
        r = resolve_document_type(
            declared_concept=None, vocabulary=V,
            full_text="## Sheet: Data\n\n" + row + "\n| --- | --- |\n" + QUOTE_BODY)
        assert r.evidence_concept == want, row


def test_a_title_keeps_its_markup_and_its_number_off_the_comparison():
    """Rule A's normalisation, each part of it load-bearing on its own."""
    assert _who("## PURCHASE ORDER") == "doctype.order"        # markdown hashes
    assert _who("**PURCHASE ORDER**") == "doctype.order"       # bold stars
    assert _who("  \tPURCHASE ORDER\t  ") == "doctype.order"   # whitespace
    assert _who("(PURCHASE ORDER)") == "doctype.order"         # punctuation
    assert _who("Schedule 1") == "doctype.schedule"            # trailing number
    assert _who("Annex B") == "doctype.schedule"               # trailing letter
    assert _who("Order No") == "doctype.quote"                 # ...but not 'No'


def test_carriage_returns_and_tabs_give_the_same_answer_as_spaces():
    """The reviewer's finding 4: a tail rule that listed its own characters
    ('rstrip(" *")') let '\\r' invert the answer. Normalisation now ends in
    fold(), which is whitespace-insensitive, so there is no such character
    list to get wrong."""
    answers = {
        ws: resolve_document_type(
            declared_concept=None, vocabulary=V,
            full_text=("Acme Ltd" + ws + "TAX INVOICE" + ws + "Order:" + ws
                       + "Order:" + ws)
        ).evidence_concept
        for ws in ("\n", "\r\n", "\n\t")
    }
    assert set(answers.values()) == {"doctype.invoice"}, answers


def test_a_label_cell_that_merely_contains_a_type_word_is_not_a_heading():
    """Real counter-example from the live PO corpus: the table cell
    'Contract sum' (8 of 11 characters = 0.73) must not be a tier-1 claim that
    ties the PO's own 'PURCHASE ORDER' cell. Faithful to the parsed document,
    separator row and all: in PO-2024-0163_PO.docx the title row is followed by
    '|---|---|' and the 'Contract sum' cell sits in a later table entirely."""
    r = resolve_document_type(
        declared_concept=None, vocabulary=V,
        full_text="| **Assurity Ltd**   | **PURCHASE ORDER**   |\n"
                  "|--------------------|----------------------|\n"
                  "\n**Commercial terms**\n\n"
                  "| Contract sum                  | £1,936,000.00 |\n"
                  "|-------------------------------|---------------|\n")
    assert (r.status, r.evidence_concept) == ("matched", "doctype.order")


def test_a_title_cell_with_a_bare_number_directly_below_it_is_lost():
    """The cost of rule B's vertical form, pinned so it is visible. A title cell
    whose own column holds a bare reference value on the next row is read as that
    value's key and the page loses its title. Honest but lossy, in the same class
    as a title row whose last cell is a date. It costs nothing on the 50 live
    documents, where a parsed table always puts '|---|' under its header row."""
    r = resolve_document_type(
        declared_concept=None, vocabulary=V,
        full_text="| Assurity Ltd | PURCHASE ORDER |\n| Contract sum | 1,936,000.00 |\n")
    assert (r.status, r.evidence_concept) == ("unknown", None)
    # ...and the separator row a real parser emits restores it.
    with_sep = resolve_document_type(
        declared_concept=None, vocabulary=V,
        full_text="| Assurity Ltd | PURCHASE ORDER |\n|---|---|\n"
                  "| Contract sum | 1,936,000.00 |\n")
    assert with_sep.evidence_concept == "doctype.order"


def test_a_title_in_its_own_table_cell_is_a_heading():
    r = resolve_document_type(
        declared_concept=None, vocabulary=V,
        full_text="## Sheet: INVOICE\n| O | Orbis Ltd |  | INVOICE |  |\n"
                  "| Reference: Against PO-2024-0145 | Purchase order cited, "
                  "purchase order again |\n")
    assert r.evidence_concept == "doctype.invoice"


def test_a_line_listing_many_type_words_is_not_a_title():
    """A list of cross-references is not the document's name. Equality does
    this on its own: the line is not any one type phrase."""
    sched = "Schedule 1, Annex A, Appendix 2, Exhibit B."
    r = resolve_document_type(
        declared_concept="doctype.framework_agreement", vocabulary=V,
        full_text="FRAMEWORK AGREEMENT\n" + sched + "\n")
    assert (r.evidence_concept, r.agreement) == ("doctype.framework_agreement", "agreed")
    # A list of one type's own references is not that type's title either. Said
    # with one alias twice, so tier-2 volume cannot answer it instead: a title
    # would be a decided 'schedule', a non-title ties with the body's quotes.
    r2 = resolve_document_type(declared_concept=None, vocabulary=V,
                               full_text="Schedule 1, Schedule 2.\n" + QUOTE_BODY)
    assert (r2.status, r2.evidence_concept) == ("unresolved", None)


def test_title_chars_labels_evidence_and_never_changes_the_outcome():
    page = "Acme\nTAX INVOICE\n" + "q" * 300 + "\nPay the invoice. See purchase order.\n"
    seen = set()
    for tc in (0, 20, 600, 10_000):
        r = resolve_document_type(declared_concept=None, full_text=page,
                                  vocabulary=V, title_chars=tc)
        seen.add((r.status, r.evidence_concept, r.candidates))
    assert len(seen) == 1


# --- enforcement points that earlier rounds left unguarded -------------------

def test_the_minimum_applies_to_aliases_not_to_aliases_plus_signals():
    """One passing mention plus a signal must not become a classification."""
    r = resolve_document_type(
        declared_concept=None, vocabulary=V,
        full_text="Dear Sir\nzzz\nPlease see the purchase order.\nShip-To Address: 1 High St\n")
    assert r.status == "unknown"


def test_a_signal_phrase_does_not_leak_aliases_from_inside_itself():
    """'bill' (an invoice alias) sits inside the signal 'Bill-To Address'."""
    r = resolve_document_type(
        declared_concept=None, vocabulary=V,
        full_text="Delivery Note\nzzzzzzzzzz\nBill-To Address: 1 High St\n"
                  "Bill-To Address: 2 Low St\n")
    assert r.status == "unknown" and r.evidence_concept is None


def test_distinct_aliases_are_counted_once_each_not_per_occurrence():
    """3.0 (two distinct aliases) vs 2.0 (one alias twice) is a decided win;
    counting occurrences as variety would tie them at 3.0."""
    r = resolve_document_type(
        declared_concept=None, vocabulary=V,
        full_text="Dear Sir\n\nWe hold your quote and quotation.\n"
                  "The invoice is due; pay the invoice.\n")
    assert (r.status, r.evidence_concept) == ("matched", "doctype.quote")


def test_three_mentions_against_two_is_noise_and_stays_unresolved():
    r = resolve_document_type(
        declared_concept=None, vocabulary=V,
        full_text="Dear Sir\n\nthe invoice, the invoice, the invoice.\n"
                  "the quote, the quote.\n")
    assert r.status == "unresolved"
    assert set(r.candidates) == {"doctype.invoice", "doctype.quote"}


def test_every_tie_candidate_is_shown_at_least_one_span():
    r = resolve_document_type(declared_concept=None, vocabulary=V,
                              full_text="Dear Sir\n" + "po " * 12 + "quote " * 8)
    assert r.status == "unresolved"
    assert len(r.evidence) <= 12
    assert {e.concept_code for e in r.evidence} >= set(r.candidates)


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
        page = " ".join(rng.choice(words) for _ in range(rng.randint(1, 6)))
        raw = set()
        for dt in V.document_types.values():
            for alias in {fold(a) for a in (*dt.aliases, dt.concept_code.split(".", 1)[-1])}:
                for st in _find_all(alias, page.lower()):
                    raw.add((st, st + len(alias)))
        want = set(_quadratic_survivors(sorted(raw)))
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


def test_a_citation_after_the_title_does_not_compete_with_it():
    """Rule C does what round 3 needed a repeated-phrase rule and a
    sentence/label-ending rule for: whatever follows the title is not it."""
    for after in ("Re purchase order PO1 and purchase order PO2",
                  "Please see the purchase order.",
                  "Order:",
                  "purchase order purchase order",
                  "Order of precedence: see the purchase order."):
        r = resolve_document_type(
            declared_concept=None, vocabulary=V,
            full_text="Acme Ltd\nTAX INVOICE\n" + after + "\n")
        assert r.evidence_concept == "doctype.invoice", after


def test_repetition_plays_no_part_in_naming_the_title():
    """The reviewer's finding 1, which is why rule C exists. Two repeated
    'Schedule n' headings used to outvote the document's own title, which would
    have mislabelled nearly every multi-schedule contract."""
    framework = ("FRAMEWORK AGREEMENT\n"
                 "Framework Ref: RM6100\n"
                 "1. This agreement sets out terms.\n\n"
                 "Schedule 1\nPricing and rates.\n\n"
                 "Schedule 2\nService levels.\n")
    msa = ("MASTER SERVICE AGREEMENT\n"
           "Contract No: MSA-7781\n"
           "1. This agreement governs the parties.\n\n"
           "Annex A\nData processing terms.\n\n"
           "Annex B\nSecurity requirements.\n")
    contents = ("MASTER SERVICE AGREEMENT\nContents\n\n"
                "Schedule 1\nSchedule 2\nSchedule 3\nAnnex A\nAnnex B\n"
                "Appendix 1\nExhibit A\n")
    for page, want in [(framework, "doctype.framework_agreement"),
                       (msa, "doctype.master_agreement"),
                       (contents, "doctype.master_agreement")]:
        r = resolve_document_type(declared_concept=want, full_text=page, vocabulary=V)
        assert (r.status, r.evidence_concept, r.agreement) == (
            "matched", want, "agreed"), page[:40]
        assert "doctype.schedule" not in r.candidates
        # ...and the schedules' spans are not offered as the reason, either.
        assert {e.concept_code for e in r.evidence} == {want}
    # A document that really IS a schedule still says so.
    alone = resolve_document_type(declared_concept=None, vocabulary=V,
                                  full_text="Schedule 1\nPricing and rates.\n")
    assert alone.evidence_concept == "doctype.schedule"


def test_a_contents_page_with_no_title_of_its_own_reads_as_a_schedule():
    """An accepted cost, pinned so it is visible rather than a surprise. 'Schedule
    1' IS a title (rule A strips the number), and a bare 'CONTENTS' heading names
    no type, so the first title segment is the first contents entry. Where the
    contents list sits inside a document that names itself — the real shape —
    rule C answers correctly, which is what the assertions below pin."""
    for page in ("CONTENTS\nSchedule 1\nSchedule 2\n",
                 "TABLE OF CONTENTS\nAnnex A\nAppendix 2\n"):
        bare = resolve_document_type(declared_concept=None, full_text=page,
                                     vocabulary=V)
        assert bare.evidence_concept == "doctype.schedule", page
        titled = resolve_document_type(
            declared_concept="doctype.framework_agreement", vocabulary=V,
            full_text="FRAMEWORK AGREEMENT\n" + page)
        assert titled.evidence_concept == "doctype.framework_agreement", page
        assert titled.agreement == "agreed", page


# --- round 5: rule B's vertical form, '#' references, and the evidence floor --

def test_a_key_above_a_bare_reference_value_is_not_the_title():
    """Rule B's vertical form. (B)'s premise is that a key's value sits in the
    next position the layout provides, and beside-it is only one layout. These
    three shapes each produced a confident wrong 'doctype.order' against a
    declared invoice when only the horizontal form existed."""
    rows = [
        ("Purchase Order\nPO-2024-0145\nINVOICE\nInvoice No: INV-1\n"
         "Amount due: 10.00\n", "doctype.invoice"),
        ("Order\n4412\nTAX INVOICE\nAmount due: 10.00\n", "doctype.invoice"),
        ("## Sheet: INVOICE\n"
         "| Ironbridge Managed IT Ltd |  | INVOICE |  | 15/02/2026 |\n"
         "| Item | Qty | Price | PO |\n| Widget | 2 | 5.00 | PO-1 |\n", None),
    ]
    for page, want in rows:
        r = resolve_document_type(declared_concept="doctype.invoice",
                                  full_text=page, vocabulary=V)
        assert r.evidence_concept != "doctype.order", page[:40]
        if want:
            assert (r.evidence_concept, r.agreement) == (want, "agreed"), page[:40]
    # A column header's value sits BELOW it, in its own column — not wherever
    # document order happens to go next. Here the next segment in document order
    # is 'Widget', which is no kind of value, while the cell under 'Order' is
    # 'PO-1', which is. Only a column-aware lookup sees that.
    header = resolve_document_type(
        declared_concept=None, vocabulary=V,
        full_text="| Line | Description | Order |\n| 1 | Widget | PO-1 |\n"
                  "The quote is here. Pay the quote. See the quotation and "
                  "the estimate.\n")
    assert header.evidence_concept == "doctype.quote"
    # ...and a column whose cell below is NOT a value keeps its title.
    titled = resolve_document_type(
        declared_concept=None, vocabulary=V,
        full_text="| Acme Ltd | PURCHASE ORDER |\n| Widget | Blue widget |\n")
    assert titled.evidence_concept == "doctype.order"
    # A value is ONE token carrying a digit and naming no type. A multi-token
    # line below a title is prose, not a value, so the title stands.
    keeps = resolve_document_type(
        declared_concept=None, vocabulary=V,
        full_text="INVOICE\nInvoice No: INV-1\nAmount due: 10.00\n")
    assert keeps.evidence_concept == "doctype.invoice"
    # ...and a non-numeric token below a title is not a value either.
    letterhead = resolve_document_type(
        declared_concept=None, vocabulary=V,
        full_text="INVOICE\nKestrel Logistics Group Ltd\nAmount due: 10.00\n")
    assert letterhead.evidence_concept == "doctype.invoice"


#: Rule B's vertical form takes a real title with it when the title happens to
#: sit immediately above a bare reference. Pinned as a POSITIVE assertion, not a
#: bug: see the test below.
ADJUDICATED_RESIDUAL_PO_BODY = (
    "Terms: the buyer shall raise a purchase order. A purchase order is "
    "required before delivery. The purchase order number must appear on every "
    "despatch note. No order will be accepted without a purchase order.\n"
)


def test_a_title_directly_above_a_bare_reference_loses_its_title_ACCEPTED():
    """ACCEPTED RESIDUAL, ruled and pinned — do not "fix" this without reading
    the whole docstring, because fixing it provably reopens a shape that is
    worse.

    What it asserts: 'INVOICE' on line 1 with 'INV-2026-0001' on line 2 is read
    by rule B's vertical form as a KEY above its VALUE, so the page has no title
    and a purchase-order-heavy body answers instead. Declared doctype.invoice, it
    returns matched / doctype.order / disagreed. Delete the reference line and it
    returns matched / doctype.invoice / agreed — one line decides it.

    Why it is accepted rather than fixed, verbatim from the ruling:

      * Measured incidence is ZERO across the 63 real documents and the 50
        process_monitor documents examined for this layer.
      * The defect and its fix are THE SAME SHAPE. In the shape rule B's vertical
        form exists to fix ('Purchase Order' above 'PO-2024-0145'), the key is
        ALSO the first segment. The only discriminator between that and this is
        whether a later title-like segment exists — and adopting that rule
        provably returns the third shape rule B was built for (a workbook header
        row ending '| PO |', pinned in
        test_a_key_above_a_bare_reference_value_is_not_the_title) to a confident
        wrong answer.
      * It cannot reach routing. Routing comes from the declared category via
        src/services/concepts/routing.py; this module only reports.

    Consequence of it being wrong: a page whose title sits immediately above a
    bare reference loses its title and may be typed from its body clauses — a
    latent risk, not an observed error. It raises a review item for a human, it
    changes no pipeline.

    A future round that wants to revisit it MUST carry BOTH shapes as tests,
    because fixing either one alone reopens the other.
    """
    page = "INVOICE\nINV-2026-0001\n" + ADJUDICATED_RESIDUAL_PO_BODY
    r = resolve_document_type(declared_concept="doctype.invoice",
                              full_text=page, vocabulary=V)
    assert (r.status, r.evidence_concept, r.agreement) == (
        "matched", "doctype.order", "disagreed")
    # It is the bare reference line alone that does this — nothing else.
    without = resolve_document_type(
        declared_concept="doctype.invoice", vocabulary=V,
        full_text="INVOICE\n" + ADJUDICATED_RESIDUAL_PO_BODY)
    assert (without.status, without.evidence_concept, without.agreement) == (
        "matched", "doctype.invoice", "agreed")
    # And the shape that a "fix" would break, asserted here too so the pair
    # travels together: the workbook header row whose last cell is '| PO |'.
    workbook = resolve_document_type(
        declared_concept="doctype.invoice", vocabulary=V,
        full_text="## Sheet: INVOICE\n"
                  "| Ironbridge Managed IT Ltd |  | INVOICE |  | 15/02/2026 |\n"
                  "| Item | Qty | Price | PO |\n| Widget | 2 | 5.00 | PO-1 |\n")
    assert workbook.evidence_concept != "doctype.order"


def test_a_trailing_colon_means_the_value_follows_not_a_title():
    """Rule B applied consistently to the one mark whose whole meaning is 'the
    value follows'. 'Order Date:' is a real parsed segment in a live quote PDF."""
    for label in ("Order:", "Quotation:", "Order Date:", "Invoice:", "Quote;", "PO,"):
        r = resolve_document_type(
            declared_concept=None, vocabulary=V,
            full_text=label + "\nRavensworth House\nSmith Ltd\n")
        assert r.evidence_concept is None, label
    # The accepted cost: a genuine colon-terminated title falls to tier 2.
    assert _who("INVOICE:") == "doctype.quote"
    # A full stop still goes, because fold() drops it and it means nothing.
    assert _who("INVOICE.") == "doctype.invoice"


def test_a_title_followed_by_a_hash_reference_is_still_a_title():
    """Two real documents turn from confidently wrong to agreed on this one
    normalisation step: QUOTE_WSG100024 and QUOTE_WSG100025 title themselves
    'Quotation  # WSG100024' and were classified by a body clause about
    purchase orders instead."""
    body = ("Terms: the buyer may raise a purchase order. "
            "A purchase order is required.\n")
    for title in ("Quotation  # WSG100024", "Quotation # WSG100025",
                  "QUOTATION  #  WSG100024"):
        r = resolve_document_type(declared_concept="doctype.quote",
                                  full_text=title + "\n" + body, vocabulary=V)
        assert (r.evidence_concept, r.agreement) == ("doctype.quote", "agreed"), title
    assert _who("INVOICE #9920") == "doctype.invoice"
    assert _who("INVOICE #") == "doctype.invoice"
    # A '#' reference with no digit is not a reference; the segment is not a title.
    assert _who("Quotation # FINAL DRAFT") == "doctype.quote"


def test_the_evidence_cap_never_adds_rows_instead_of_trimming_them():
    """`[:_MAX_EVIDENCE - len(chosen)]` is a NEGATIVE slice once the per-candidate
    reservation exceeds the cap, so it added rows. The reservation is a floor:
    every candidate keeps a span, so the bound is max(12, len(candidates)) — but
    nothing beyond the reservation may be added once the cap is reached."""
    codes = ["INVOICE", "QUOTE", "ORDER", "SCHEDULE", "ADDENDUM", "CCN", "NDA",
             "SLA", "SOW", "VARIATION", "CONTRACT", "MSA", "FRAMEWORK AGREEMENT",
             "SERVICE AGREEMENT", "CONSULTING AGREEMENT"]
    for n in range(12, len(codes) + 1):
        page = " / ".join(codes[:n]) + "\n"
        r = resolve_document_type(declared_concept=None, full_text=page, vocabulary=V)
        assert r.status == "unresolved", n
        assert len(r.candidates) == n, n
        assert len(r.evidence) <= max(12, len(r.candidates)), (n, len(r.evidence))
        # exactly one reserved row per candidate, and nothing added on top
        assert len(r.evidence) == n, (n, len(r.evidence))
        assert {e.concept_code for e in r.evidence} == set(r.candidates), n
    # The negative slice only ADDS rows when unreserved rows exist to add, so
    # the page needs both: more than 12 candidates AND spare rows. Two seeded
    # phrase-shaped signals supply the spare rows ('ship-to address' for the
    # order, 'bill-to address' for the invoice), and the reservation is 13.
    page = (" / ".join(codes[:13]) + "\n"
            "Ship-To Address: 1 High St\nBill-To Address: 2 Low St\n")
    r = resolve_document_type(declared_concept=None, full_text=page, vocabulary=V)
    assert len(r.candidates) == 13
    assert {e.kind for e in r.evidence} == {"title_alias"}, (
        "a signal row got in past a cap that had no room left")
    assert len(r.evidence) == 13, len(r.evidence)


def test_every_matched_and_every_candidate_carries_at_least_one_span():
    """fold() collapses internal whitespace but the match copy keeps the page's
    length, so a double-spaced title named a concept with no hit in its own span
    and returned ZERO evidence rows. The title segment is the reason.

    Stated on 'MASTER  SERVICE  AGREEMENT': none of its words is an alias of
    doctype.master_agreement on its own, so the fallback is the only thing that
    can produce a span. 'FRAMEWORK  AGREEMENT' no longer reaches the fallback —
    bare 'framework' was restored as an alias, so it hits inside the title span
    — and that case is kept below for exactly that reason."""
    single = resolve_document_type(
        declared_concept=None, vocabulary=V,
        full_text="MASTER  SERVICE  AGREEMENT\nThis agreement sets out terms.\n")
    assert single.evidence_concept == "doctype.master_agreement"
    assert single.evidence, "a matched result with no reason is unreviewable"
    assert single.evidence[0].text == "MASTER  SERVICE  AGREEMENT"
    assert single.evidence[0].start == 0

    # The restored bare alias supplies its own span, at a real offset.
    framework = resolve_document_type(
        declared_concept=None, vocabulary=V,
        full_text="FRAMEWORK  AGREEMENT\nThis agreement sets out terms.\n")
    assert framework.evidence_concept == "doctype.framework_agreement"
    assert framework.evidence[0].text == "FRAMEWORK"
    assert framework.evidence[0].start == 0

    pair = resolve_document_type(
        declared_concept=None, vocabulary=V,
        full_text="FRAMEWORK  AGREEMENT / MASTER  SERVICE  AGREEMENT\n")
    assert pair.status == "unresolved"
    assert set(pair.candidates) == {"doctype.framework_agreement",
                                    "doctype.master_agreement"}
    # both sides shown, each with its OWN words, each byte-exact
    assert {e.concept_code for e in pair.evidence} == set(pair.candidates)
    by_code = {e.concept_code: e for e in pair.evidence}
    assert by_code["doctype.framework_agreement"].text == "FRAMEWORK"
    assert by_code["doctype.master_agreement"].text == "MASTER  SERVICE  AGREEMENT"
    page = "FRAMEWORK  AGREEMENT / MASTER  SERVICE  AGREEMENT\n"
    for e in pair.evidence:
        assert page[e.start:e.start + len(e.text)] == e.text, e


def test_no_matched_or_unresolved_result_is_ever_evidence_free():
    """The invariant behind the case above, over a wide sweep of titles: a page
    the resolver is willing to answer must be able to show why."""
    import itertools
    words = ["INVOICE", "TAX  INVOICE", "FRAMEWORK  AGREEMENT", "PURCHASE  ORDER",
             "MASTER  SERVICE  AGREEMENT", "Schedule  1", "QUOTE", "## INVOICE",
             "| INVOICE |", "NON-DISCLOSURE  AGREEMENT"]
    pages = [w + "\n" for w in words]
    pages += [a + " / " + b + "\n" for a, b in itertools.combinations(words, 2)]
    pages += ["Acme Ltd\n" + w + "\nAmount due: 1.00\n" for w in words]
    seen_matched = seen_unresolved = 0
    for page in pages:
        r = resolve_document_type(declared_concept=None, full_text=page, vocabulary=V)
        if r.status == "matched" and r.evidence_concept:
            seen_matched += 1
            assert r.evidence, page
        if r.status == "unresolved":
            seen_unresolved += 1
            assert {e.concept_code for e in r.evidence} >= set(r.candidates), page
        for e in r.evidence:
            assert page[e.start:e.start + len(e.text)] == e.text, (page, e)
    # The FLOOR. Both assertions above sit behind an `if`, so a change that made
    # every page come back 'unknown' would leave this sweep green over 65 pages
    # while checking nothing — this plan's recurring failure. Live counts today
    # are 32 matched and 33 unresolved; the floor is deliberately loose (it pins
    # that both branches are reached, not the exact split, which legitimate
    # vocabulary edits move).
    assert len(pages) == 65, len(pages)
    assert seen_matched and seen_unresolved, (seen_matched, seen_unresolved)
    assert seen_matched + seen_unresolved == len(pages), (
        seen_matched, seen_unresolved, len(pages))


def test_order_form_is_not_a_call_off_alias_and_quote_workbooks_stay_agreed():
    """'order form' was dropped from doctype.call_off_contract, deliberately.

    It is a real name for a call-off, which is why it was seeded. On this corpus
    it is also the title cell every quote-template workbook carries, so it
    produced 12 disagreements out of 12 uses and no true positive: measured over
    50 live documents, 38 agreed / 12 disagreed before the drop and 50 agreed /
    0 disagreed after. See specs/2026-10-01-document-relationship-layer-rulings.md.

    Re-adding it needs a way to tell a quote template's 'Order Form' heading from
    a real call-off's, which needs the golden-set documents. If you re-add it,
    this test goes red and the 12 false disagreements come back with it.

    A real call-off still matches: that is the second half of this test.
    """
    from src.services.concepts.vocabulary import SEED_VOCABULARY, resolve_alias

    assert resolve_alias("order form", SEED_VOCABULARY) == (), (
        "'order form' resolves again — see this test's docstring before keeping it"
    )

    workbook = "| A | Aureus Workflow Ltd |  | Order Form |  |\n| --- | --- |\n"
    r = resolve_document_type(
        declared_concept="doctype.quote", full_text=workbook,
        vocabulary=SEED_VOCABULARY)
    assert r.agreement != "disagreed", r
    assert r.evidence_concept != "doctype.call_off_contract", r

    # The type is still reachable by its own names, so dropping the alias cost
    # recognition of the phrase, not of the document type.
    call_off = resolve_document_type(
        declared_concept=None, full_text="Call-Off Contract\nFramework Ref: RM6100\n",
        vocabulary=SEED_VOCABULARY)
    assert call_off.evidence_concept == "doctype.call_off_contract", call_off
