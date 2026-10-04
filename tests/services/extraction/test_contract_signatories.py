"""Who signed the contract, read from its signature block.

Measured 2026-10-03: `contract_signatory_name` held **"Email Marketing"** on the
real Marketing Agreement. Same root cause as the party fields (see
test_contract_parties.py) — `SpacyNERExtractor`'s default path emits every entity
of the required type for any field, and for a PERSON-typed field it offered the
first thing spaCy mis-tagged as a person. One field, so no duplicate value made it
obvious the way `supplier_id == buyer_org_id` did.

The document says it plainly, and nothing was reading it:

    SIGNATURE AND DATE
    ... This agreement is demonstrated by their signatures below:
    MARKETER
    Name: John Smith      Signature: ____________  Date: June 12, 2025
    CLIENT
    Name: Sarah Johnson   Signature: ____________  Date: June 12, 2025

THE RULING, because the schema has ONE signatory field and a contract has two.
`proc.bp_contracts.contract_signatory_name` holds the **supplier's** signatory: this
is a procurement system, and the question a single slot has to answer is "who bound
the counterparty". The buyer's signatory is deliberately not stored — the schema has
nowhere to put it, and inventing a column is a bigger change than this is. When the
block names only one signatory and does not say which party they signed for, that
one is taken. When it names several and none can be attributed, nothing is stored:
ambiguity is not a fact.

Offline and model-free.
    CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest \
        tests/services/extraction/test_contract_signatories.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.services.extraction.engineered.contract_signatories import (  # noqa: E402
    read_signatory, signatory_candidates,
)

#: Byte-for-byte the tail of the stored full_text for the real Marketing Agreement,
#: including docling's escaped underscores and the one-line Name/Signature/Date run.
REAL_SIGNATURE_BLOCK = (
    "SEVERABILITY\n\nIn an event when any provision of this Agreement is found to be "
    "void and unenforceable by a court of competent jurisdiction, the remaining "
    "provisions will still be enforced.\n\n"
    "SIGNATURE AND DATE\n\n"
    "The Parties hereby agree to the terms and conditions set forth in this Agreement. "
    "This agreement is demonstrated by their signatures below:\n\n"
    "MARKETER\n\n"
    "Name: John Smith Signature: \\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_ Date: June 12, 2025\n\n"
    "CLIENT\n\n"
    "Name: Sarah Johnson Signature: \\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_\\_ Date: June 12, 2025\n"
)


def test_the_real_contract_yields_the_suppliers_signatory():
    """MARKETER is the supplier side, so John Smith -- not Sarah Johnson, and
    certainly not 'Email Marketing'."""
    s = read_signatory(REAL_SIGNATURE_BLOCK)
    assert s.name == "John Smith", s
    assert s.party == "supplier", s


def test_the_value_that_was_actually_stored_can_no_longer_be_produced():
    """'Email Marketing' is in the document (a service line item). It is not a
    signatory, and nothing in the signature block offers it."""
    text = ("SERVICES PROVIDED\n4. Email Marketing and CRM Integration\n\n"
            + REAL_SIGNATURE_BLOCK)
    s = read_signatory(text)
    assert s.name == "John Smith", s


def test_a_signed_by_label_is_read():
    s = read_signatory("IN WITNESS WHEREOF\nSigned by: Priya Raman\nFor and on behalf of "
                       "NexaSpark Marketing Ltd.\n")
    assert s.name == "Priya Raman", s


@pytest.mark.parametrize("label", ["Signatory", "Authorised Signatory", "Signed By",
                                   "Authorised By"])
def test_each_signatory_label_the_schema_declares_is_read(label):
    s = read_signatory(f"SIGNATURES\n{label}: Priya Raman\n")
    assert s.name == "Priya Raman", (label, s)


def test_a_job_title_beside_the_name_is_read_as_the_role():
    s = read_signatory("SIGNATURES\nSupplier\nName: Priya Raman Title: Managing Director "
                       "Date: 5 January 2026\n")
    assert s.name == "Priya Raman", s
    assert s.role == "Managing Director", s


def test_a_signature_rule_is_not_a_name():
    """docling renders the signature line as escaped underscores."""
    s = read_signatory("SIGNATURES\nName: \\_\\_\\_\\_\\_\\_\\_\\_\\_ Date: 5 January 2026\n")
    assert s.name is None, s


def test_a_date_is_not_a_name():
    s = read_signatory("SIGNATURES\nName: June 12, 2025\n")
    assert s.name is None, s


def test_a_company_is_not_a_signatory():
    """'For and on behalf of NexaSpark Marketing Ltd.' is the party, not the person."""
    s = read_signatory("SIGNATURES\nName: NexaSpark Marketing Ltd.\n")
    assert s.name is None, s


def test_a_contract_with_no_signature_block_yields_nothing():
    s = read_signatory("FRAMEWORK AGREEMENT\nFramework Agreement No. FA-1\n"
                       "This Framework Agreement is governed by English law.\n")
    assert s.name is None and s.role is None, s


def test_a_name_outside_the_signature_block_is_ignored():
    """The buyer's contact in the party clause is not who signed."""
    s = read_signatory("PARTIES\nContact: Sarah Johnson\nSERVICES\nMarketing.\n")
    assert s.name is None, s


def test_two_signatories_with_no_party_attribution_yield_nothing():
    """Two names and no way to tell whose is whose: ambiguity is not a fact."""
    s = read_signatory("SIGNATURES\nName: John Smith Date: 1 June 2025\n"
                       "Name: Sarah Johnson Date: 1 June 2025\n")
    assert s.name is None, s


def test_one_unattributed_signatory_is_taken():
    s = read_signatory("SIGNATURES\nName: John Smith Date: 1 June 2025\n")
    assert s.name == "John Smith", s
    assert s.party is None, s


def test_the_buyers_signatory_alone_is_not_stored_as_the_suppliers():
    """Only the CLIENT side signed this copy. The field means the supplier's
    signatory, so storing Sarah Johnson in it would be a false statement."""
    s = read_signatory("SIGNATURES\nCLIENT\nName: Sarah Johnson Date: 1 June 2025\n")
    assert s.name is None, s
    assert s.buyer_name == "Sarah Johnson", s


def test_the_candidate_carries_the_schemas_field_names():
    """The supplier's signatory lands on contract_signatory_name -- the CONTRACT
    schema's field. (The buyer's now lands on buyer_signatory_name; that is
    test_both_signatories_are_emitted_as_candidates.)"""
    by_field = {c.field: c.value for c in signatory_candidates(REAL_SIGNATURE_BLOCK)}
    assert by_field["contract_signatory_name"] == "John Smith", by_field


def test_the_candidate_outranks_the_entity_sweep():
    for c in signatory_candidates(REAL_SIGNATURE_BLOCK):
        assert c.source == "parties", c
        assert c.confidence > 0.70, c


def test_nothing_is_emitted_without_a_signature_block():
    assert signatory_candidates("AGREEMENT\nGoverned by English law.\n") == []


def test_the_signatory_field_is_barred_from_the_entity_sweep_for_a_contract():
    """Same bar as the party fields, and for the same reason: 'the first PERSON in
    the document' produced 'Email Marketing'."""
    from src.services.extraction.dispatch import _contract_party_candidates
    _cands, barred = _contract_party_candidates("contract", REAL_SIGNATURE_BLOCK)
    assert "contract_signatory_name" in barred


def test_jurisdiction_is_not_barred():
    """It comes from the same default path and it is CORRECT ('United Kingdom' on
    five of six live documents). Barring the path wholesale would lose that."""
    from src.services.extraction.dispatch import _contract_party_candidates
    _cands, barred = _contract_party_candidates("contract", REAL_SIGNATURE_BLOCK)
    assert "jurisdiction" not in barred


# ---------------------------------------------------------------------------
# The backfill decision, same four rules as the party fields.
# ---------------------------------------------------------------------------

def _decide(**kw):
    from src.services.extraction.engineered.contract_signatories import (
        decide_signatory_correction,
    )
    return decide_signatory_correction(**kw)


def test_the_block_corrects_the_value_the_sweep_stored():
    d = _decide(full_text=REAL_SIGNATURE_BLOCK, stored_name="Email Marketing",
                stored_role=None, provenance_source="ner")
    assert d.name == "John Smith"
    assert d.changed is True
    assert "signature block" in d.reason


def test_a_name_the_block_agrees_with_is_left_alone():
    """Both sides must agree for the row to be left alone: since 2026-10-04 an
    empty buyer_signatory_name IS a difference when the block names one."""
    d = _decide(full_text=REAL_SIGNATURE_BLOCK, stored_name="John Smith",
                stored_role=None, provenance_source="parties",
                stored_buyer_name="Sarah Johnson", stored_buyer_role=None)
    assert d.changed is False, d


def test_no_signature_block_and_a_sweep_value_is_cleared():
    d = _decide(full_text="FRAMEWORK AGREEMENT\nGoverned by English law.\n",
                stored_name="Email Marketing", stored_role=None, provenance_source="ner")
    assert d.name is None and d.changed is True
    assert "sweep" in d.reason


def test_no_signature_block_and_a_non_sweep_value_is_left_alone():
    d = _decide(full_text="FRAMEWORK AGREEMENT\nGoverned by English law.\n",
                stored_name="Priya Raman", stored_role=None,
                provenance_source="context_layer")
    assert d.changed is False and d.name == "Priya Raman"


def test_a_human_confirmed_signatory_is_never_touched():
    d = _decide(full_text=REAL_SIGNATURE_BLOCK, stored_name="Someone Else",
                stored_role=None, provenance_source="hitl")
    assert d.changed is False
    assert "human" in d.reason


# ---------------------------------------------------------------------------
# The buyer's signatory now has columns of its own
# (deploy/sql/2026-10-04_contract_buyer_signatory.sql), so it is emitted rather
# than parsed and discarded.
# ---------------------------------------------------------------------------

def test_both_signatories_are_emitted_as_candidates():
    cands = signatory_candidates(REAL_SIGNATURE_BLOCK)
    by_field = {c.field: c.value for c in cands}
    assert by_field == {"contract_signatory_name": "John Smith",
                        "buyer_signatory_name": "Sarah Johnson"}, by_field


def test_the_buyers_role_is_emitted_when_the_block_carries_one():
    text = ("SIGNATURES\nSUPPLIER\nName: Priya Raman Title: Managing Director\n"
            "CLIENT\nName: Sarah Johnson Title: Head of Procurement\n")
    by_field = {c.field: c.value for c in signatory_candidates(text)}
    assert by_field == {
        "contract_signatory_name": "Priya Raman",
        "contract_signatory_role": "Managing Director",
        "buyer_signatory_name": "Sarah Johnson",
        "buyer_signatory_role": "Head of Procurement",
    }, by_field


def test_the_buyers_signatory_alone_is_emitted_on_its_own_field():
    """Only the CLIENT side signed this copy. That is now storable, where before
    it was read and thrown away -- and it must NOT land in the supplier's field."""
    cands = signatory_candidates("SIGNATURES\nCLIENT\nName: Sarah Johnson Date: 1 June 2025\n")
    by_field = {c.field: c.value for c in cands}
    assert by_field == {"buyer_signatory_name": "Sarah Johnson"}, by_field


def test_an_unattributed_single_signatory_does_not_become_the_buyers():
    """One name with no party label is the signatory; nothing says it is the
    buyer's, so the buyer field stays empty."""
    by_field = {c.field: c.value for c in
                signatory_candidates("SIGNATURES\nName: John Smith Date: 1 June 2025\n")}
    assert by_field == {"contract_signatory_name": "John Smith"}, by_field


def test_the_buyer_signatory_fields_are_barred_from_the_entity_sweep():
    from src.services.extraction.dispatch import _contract_party_candidates
    _c, barred = _contract_party_candidates("contract", REAL_SIGNATURE_BLOCK)
    assert "buyer_signatory_name" in barred


def test_the_buyer_fields_exist_in_the_contract_schema():
    """A candidate for a field the schema does not declare is dropped silently by
    dispatch's valid_cols filter, so this is the wiring that makes it reachable."""
    from src.services.extraction.pattern_registry import get_registry
    cols = {f.db_column for f in get_registry("contract").schema.fields if f.db_column}
    assert {"buyer_signatory_name", "buyer_signatory_role"} <= cols


def test_the_buyer_fields_are_not_ner_typed():
    """The sweep is what stored 'Email Marketing' as a person. A PERSON-typed
    field with no party-aware branch falls straight back into it."""
    from src.services.extraction.pattern_registry import get_registry
    for f in get_registry("contract").schema.fields:
        if f.name.startswith("buyer_signatory"):
            assert f.judge.ner_type_check in (None, "none"), f.name


def test_the_correction_carries_the_buyers_signatory_too():
    """The backfill has to fill the new column on rows extracted before it
    existed, or every contract already stored keeps an empty buyer_signatory_name
    that the document could have answered."""
    d = _decide(full_text=REAL_SIGNATURE_BLOCK, stored_name="John Smith",
                stored_role=None, provenance_source="parties",
                stored_buyer_name=None, stored_buyer_role=None)
    assert d.buyer_name == "Sarah Johnson", d
    assert d.changed is True, "the buyer's column is empty and the block names it"


def test_a_row_already_holding_both_signatories_is_left_alone():
    d = _decide(full_text=REAL_SIGNATURE_BLOCK, stored_name="John Smith",
                stored_role=None, provenance_source="parties",
                stored_buyer_name="Sarah Johnson", stored_buyer_role=None)
    assert d.changed is False, d


def test_clearing_a_sweep_value_clears_the_buyer_side_too():
    d = _decide(full_text="FRAMEWORK AGREEMENT\nGoverned by English law.\n",
                stored_name="Email Marketing", stored_role=None,
                provenance_source="ner", stored_buyer_name="Services",
                stored_buyer_role=None)
    assert d.name is None and d.buyer_name is None and d.changed is True
