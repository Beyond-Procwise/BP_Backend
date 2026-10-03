"""A contract's parties come from its party clause, or they are NULL.

Measured on the live server 2026-10-03, on the first six contract documents this
product has ever ingested: `supplier_id` held the BUYER's name on five of them and
the sentence fragment "Framework Agreement No" on the sixth, and `buyer_org_id`
held the SAME value as `supplier_id` on all six.

Root cause: `SpacyNERExtractor.produce_candidates` has party-aware branches for
exactly two field NAMES — `supplier_name` (header position + buyer-context filter)
and `buyer_id` (the BILL TO block). The contract schema names its party fields
`supplier_id` and `buyer_org_id`, so both fell through to the default branch, which
emits every entity of the required type for any field. Two fields both requiring
ORG therefore received identical candidate lists:

    supplier_id   'BrightWave Digital Ltd.'      <- the BUYER
    supplier_id   'NexaSpark Marketing Ltd.'     <- the actual supplier, second
    supplier_id   'Framework Agreement No'
    buyer_org_id  'BrightWave Digital Ltd.'      <- identical list
    buyer_org_id  'NexaSpark Marketing Ltd.'
    buyer_org_id  'Framework Agreement No'

and on the real Marketing Agreement the list included `Services`, `LIABILITY`,
`Arbitration`, `SEVERABILITY` and `Bank Transfer to`.

No test caught it because `en_core_web_sm` is installed in `.venv` (what the server
runs) and NOT in `venv` (what pytest runs), so under test `fill_ner_gaps` returns []
and that branch never executes. These tests are deliberately offline and
model-free: they test the deterministic reader that now answers first.

    CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest \
        tests/services/extraction/test_contract_parties.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.services.extraction.engineered.contract_parties import read_parties  # noqa: E402

# The real Marketing Agreement's actual party sentence, as pdftotext reads it.
REAL_MARKETING_AGREEMENT = (
    "MARKETING AGREEMENT\nPARTIES\n"
    "This Marketing Agreement (hereinafter referred to as the “Agreement”) is\n"
    "entered into on June 12, 2025 (‘the Effective Date’) by and between BrightWave\n"
    "Digital Ltd. (hereinafter referred to as the “Client”) with an address of 123\n"
    "Innovation Park, London, UK and NexaSpark Marketing Ltd. (hereinafter referred\n"
    "to as the “Marketer”) with an address of 125 Innovation Park, London, UK\n"
    "(collectively referred to as the “Parties”).\n"
)


def test_the_real_marketing_agreement_reads_both_parties_the_right_way_round():
    """The document this product actually holds. Client = buyer, Marketer = supplier."""
    p = read_parties(REAL_MARKETING_AGREEMENT)
    assert p.supplier == "NexaSpark Marketing Ltd.", p
    assert p.buyer == "BrightWave Digital Ltd.", p


def test_a_labelled_party_block_is_read():
    text = ("ORDER FORM\nOrder Form No. OF-2026-0211\n"
            "Buyer: BrightWave Digital Ltd., 123 Innovation Park, London, UK\n"
            "Supplier: NexaSpark Marketing Ltd., 125 Innovation Park, London, UK\n")
    p = read_parties(text)
    assert p.supplier == "NexaSpark Marketing Ltd.", p
    assert p.buyer == "BrightWave Digital Ltd.", p


def test_the_label_order_does_not_matter():
    text = ("Supplier: NexaSpark Marketing Ltd.\nBuyer: BrightWave Digital Ltd.\n")
    p = read_parties(text)
    assert (p.supplier, p.buyer) == ("NexaSpark Marketing Ltd.", "BrightWave Digital Ltd.")


@pytest.mark.parametrize("label,party", [
    ("Vendor", "supplier"), ("Contractor", "supplier"), ("Service Provider", "supplier"),
    ("Seller", "supplier"), ("Consultant", "supplier"),
    ("Client", "buyer"), ("Customer", "buyer"), ("Purchaser", "buyer"), ("Buyer", "buyer"),
])
def test_each_label_in_the_schemas_own_vocabulary_is_read(label, party):
    """These are the canonical_labels extraction_schemas/contract.yaml already
    declares for supplier_id and buyer_org_id. They were never read by anything."""
    p = read_parties(f"AGREEMENT\n{label}: Helio Print Services Ltd.\n")
    assert getattr(p, party) == "Helio Print Services Ltd.", (label, p)


def test_a_silent_document_yields_nothing():
    """THE no-fabrication guard. Better NULL than a guess -- and 'the first ORG in
    the document' is a guess that was wrong five times out of six."""
    text = ("SERVICES AGREEMENT\nThis Agreement is made between the parties identified\n"
            "in Schedule 1 and is governed by the laws of England and Wales.\n")
    p = read_parties(text)
    assert p.supplier is None and p.buyer is None, p


def test_the_fragment_that_was_stored_as_a_supplier_can_never_be_one():
    """'Framework Agreement No' was stored in supplier_id on a live document."""
    text = ("ORDER FORM\nThis Order Form is incorporated into and governed by "
            "Framework Agreement No. FA-2026-0042 dated 5 January 2026.\n")
    p = read_parties(text)
    assert p.supplier is None, p
    assert p.buyer is None, p


def test_one_party_stated_is_one_party_read():
    p = read_parties("AGREEMENT\nSupplier: Helio Print Services Ltd.\n")
    assert p.supplier == "Helio Print Services Ltd."
    assert p.buyer is None


def test_the_same_name_on_both_sides_is_a_read_error_and_yields_neither():
    """The symptom this exists to prevent: one value in both party fields. A
    document naming the same company as both parties has not been read correctly,
    and two wrong fields are worse than two empty ones."""
    p = read_parties("Supplier: Helio Print Services Ltd.\nBuyer: Helio Print Services Ltd.\n")
    assert p.supplier is None and p.buyer is None, p


def test_an_unknown_role_word_is_not_guessed_at():
    text = ("This Agreement is entered into between Helio Print Services Ltd. "
            "(hereinafter referred to as the “Widget Fairy”) and BrightWave Digital "
            "Ltd. (hereinafter referred to as the “Other One”).\n")
    p = read_parties(text)
    assert p.supplier is None and p.buyer is None, p


def test_a_definition_clause_without_the_hereinafter_wording_is_read():
    """'X ("the Supplier")' is as common as the long form."""
    text = ('This Agreement is between BrightWave Digital Ltd. ("the Client") and '
            'NexaSpark Marketing Ltd. ("the Supplier").\n')
    p = read_parties(text)
    assert p.supplier == "NexaSpark Marketing Ltd.", p
    assert p.buyer == "BrightWave Digital Ltd.", p


def test_a_label_wins_over_a_definition_clause():
    """A labelled field is a statement; a definition clause is prose about one. If
    they disagree, the label is the more direct evidence."""
    text = ("Supplier: Helio Print Services Ltd.\n"
            'This Agreement is between BrightWave Digital Ltd. ("the Client") and '
            'NexaSpark Marketing Ltd. ("the Supplier").\n')
    p = read_parties(text)
    assert p.supplier == "Helio Print Services Ltd.", p
    assert p.buyer == "BrightWave Digital Ltd.", p


def test_an_address_is_not_a_party_name():
    """The label's value stops at the company suffix: a party line usually carries
    the address on the same line, and 'NexaSpark Marketing Ltd., 125 Innovation
    Park, London, UK' is not a supplier name."""
    p = read_parties("Supplier: NexaSpark Marketing Ltd., 125 Innovation Park, London, UK\n")
    assert p.supplier == "NexaSpark Marketing Ltd.", p


def test_a_party_name_with_no_company_suffix_is_still_read():
    """Not every counterparty is a Ltd."""
    p = read_parties("Supplier: Westminster City Council\nBuyer: BrightWave Digital Ltd.\n")
    assert p.supplier == "Westminster City Council", p


def test_candidates_carry_the_fields_the_contract_schema_actually_uses():
    """supplier_name / buyer_id are the INVOICE names. The contract schema says
    supplier_id and buyer_org_id, and getting that wrong is the whole bug."""
    from src.services.extraction.engineered.contract_parties import party_candidates
    cands = party_candidates(REAL_MARKETING_AGREEMENT)
    by_field = {c.field: c.value for c in cands}
    assert by_field == {"supplier_id": "NexaSpark Marketing Ltd.",
                        "buyer_org_id": "BrightWave Digital Ltd."}, by_field


def test_a_party_candidate_outranks_the_ner_sweep_and_loses_to_a_regex_hit():
    from src.services.extraction.engineered.contract_parties import party_candidates
    for c in party_candidates(REAL_MARKETING_AGREEMENT):
        assert c.source == "parties", c
        assert 0.70 < c.confidence < 0.92, c


def test_nothing_is_emitted_for_a_silent_document():
    from src.services.extraction.engineered.contract_parties import party_candidates
    assert party_candidates("AGREEMENT\nGoverned by the laws of England.\n") == []


# ---------------------------------------------------------------------------
# The wiring. Reading the clause is only half the fix: the entity sweep must not
# answer a contract's party fields even when the clause says nothing, or the
# original bug returns on the next silent document.
# ---------------------------------------------------------------------------

def test_a_contracts_party_fields_are_barred_from_the_entity_sweep():
    from src.services.extraction.dispatch import _contract_party_candidates
    cands, barred = _contract_party_candidates("contract", REAL_MARKETING_AGREEMENT)
    assert {c.field for c in cands} == {"supplier_id", "buyer_org_id"}
    assert barred == {"supplier_id", "buyer_org_id"}


def test_the_bar_holds_even_when_the_clause_says_nothing():
    """THE regression guard. A silent contract must leave both fields NULL for the
    context layer to ground or missing_required to flag -- not fall back to 'the
    first ORG in the document', which was wrong five times out of six."""
    from src.services.extraction.dispatch import _contract_party_candidates
    cands, barred = _contract_party_candidates("contract", "AGREEMENT\nGoverned by English law.\n")
    assert cands == []
    assert barred == {"supplier_id", "buyer_org_id"}


def test_an_invoice_is_untouched():
    """supplier_name / buyer_id on an invoice have working, position-aware NER
    branches. Nothing here may change them."""
    from src.services.extraction.dispatch import _contract_party_candidates
    cands, barred = _contract_party_candidates("invoice", REAL_MARKETING_AGREEMENT)
    assert cands == [] and barred == set()


# ---------------------------------------------------------------------------
# The shape the PARSER actually produces, which is not the shape a human types.
# Found by the live re-upload 2026-10-03: docling collapses a contract's whole
# party block onto ONE line, so a line-anchored label never matches and both
# labelled documents read NULL. The colon is the label's real signature, not the
# line start — but then "the Supplier may be asked to provide" must still not be
# read as a label, which is what the line anchor was guarding against.
# ---------------------------------------------------------------------------

#: Byte-for-byte what parse() returns for framework2.pdf (first 420 chars).
REAL_PARSED_LABEL_BLOCK = (
    "## FRAMEWORK AGREEMENT\n\n"
    "Framework Agreement No. FA-2026-0077 Buyer: BrightWave Digital Ltd., 123 "
    "Innovation Park, London, UK Supplier: NexaSpark Marketing Ltd., 125 Innovation "
    "Park, London, UK Effective Date: 5 January 2026 End Date: 4 January 2029\n\n"
    "## 1. PURPOSE\n\nThis Framework Agreement sets out the terms under which the "
    "Supplier may be asked to provide marketing services to the Buyer. It creates "
    "no commitment to purchase.\n"
)


def test_the_parsers_single_line_party_block_is_read():
    p = read_parties(REAL_PARSED_LABEL_BLOCK)
    assert p.supplier == "NexaSpark Marketing Ltd.", p
    assert p.buyer == "BrightWave Digital Ltd.", p


def test_a_label_value_stops_at_the_next_label():
    """No comma to stop at, so only the next 'Label:' can end the value."""
    p = read_parties("AGREEMENT\nSupplier: Westminster City Council Buyer: "
                     "BrightWave Digital Ltd. Effective Date: 5 January 2026\n")
    assert p.supplier == "Westminster City Council", p
    assert p.buyer == "BrightWave Digital Ltd.", p


def test_prose_about_a_party_is_not_a_label():
    """'the Supplier may be asked to provide marketing services to the Buyer' is a
    sentence, not a field. It has no colon, which is the whole distinction."""
    p = read_parties("This Framework Agreement sets out the terms under which the "
                     "Supplier may be asked to provide services to the Buyer.\n")
    assert p.supplier is None and p.buyer is None, p


def test_a_drafting_colon_after_a_party_word_yields_nothing():
    """Legal drafting writes 'If the Supplier: (a) fails to deliver...'. That colon
    is punctuation, and what follows is not a company name."""
    p = read_parties("If the Supplier: (a) fails to deliver, or (b) becomes insolvent, "
                     "the Buyer: may terminate.\n")
    assert p.supplier is None, p
    assert p.buyer is None, p
