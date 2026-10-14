"""The references the vocabulary declares must be extractable.

proc.bp_document_type says a call-off points at its framework through
framework_ref. That field did not exist in extraction_schemas/contract.yaml, so
the pointer was declared and never filled.

Offline — the registry compiles from YAML with no database.
    CUDA_VISIBLE_DEVICES="" ./venv/bin/python -m pytest \
        tests/services/extraction/test_contract_parent_reference_fields.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.services.extraction.pattern_extractor import run_pattern_extractor  # noqa: E402
from src.services.extraction.pattern_registry import get_registry           # noqa: E402


@pytest.fixture(scope="module")
def registry():
    return get_registry("contract")


class _Parsed:
    """The only thing run_pattern_extractor needs off a parsed document.

    Its signature is run_pattern_extractor(parsed, doc_type) and it reads
    `parsed.full_text` (pattern_extractor.py:101). It does NOT take a registry
    -- it loads one for the doc_type itself.
    """

    def __init__(self, text: str) -> None:
        self.full_text = text


def _value(text, field):
    """The value the L1 pattern layer reads for one field, or None.

    run_pattern_extractor returns list[Candidate] (src/services/extraction/
    types.py:34) with .field, .value, .span, .source, .pattern_name and
    .confidence. Higher-prior patterns are tried first per field, so the first
    candidate for a field is the highest-prior match.
    """
    for c in run_pattern_extractor(_Parsed(text), "contract"):
        if c.field == field:
            return c.value
    return None


def test_both_fields_exist_and_are_optional(registry):
    names = {f.name for f in registry.schema.fields}
    assert {"framework_ref", "parent_agreement_ref"} <= names, sorted(names)
    for field in registry.schema.fields:
        if field.name in ("framework_ref", "parent_agreement_ref"):
            assert field.required is False, (
                f"{field.name} must be optional: most contracts name no parent, and "
                "a required field would block promotion on every standalone agreement"
            )
            assert field.db_column == field.name


def test_a_framework_reference_is_read(registry):
    text = "ORDER FORM\n\nThis Order Form is made under Framework Agreement No: FW-2024-0012.\n"
    assert _value(text, "framework_ref") == "FW-2024-0012"


def test_the_made_under_phrasing_is_read(registry):
    text = "CALL-OFF CONTRACT\n\nmade under Framework Agreement RM6187.\n"
    assert _value(text, "framework_ref") == "RM6187"


def test_a_parent_agreement_reference_is_read(registry):
    text = "STATEMENT OF WORK\n\nThis SOW is issued under Master Agreement MSA-4417.\n"
    assert _value(text, "parent_agreement_ref") == "MSA-4417"


@pytest.mark.parametrize("placeholder", ["N/A", "NA", "TBC", "TBD", "NONE", "SEE"])
@pytest.mark.parametrize("field,text_template", [
    ("framework_ref",        "ORDER FORM\n\nFramework Agreement No: {ph}\n"),
    ("framework_ref",        "ORDER FORM\n\nmade under Framework Agreement {ph}\n"),
    ("parent_agreement_ref", "STATEMENT OF WORK\n\nMaster Agreement: {ph}\n"),
    ("parent_agreement_ref", "STATEMENT OF WORK\n\nissued under Master Agreement {ph}\n"),
])
def test_a_placeholder_is_refused_in_every_pattern(field, text_template, placeholder):
    """'N/A' is not a reference, in any of the four patterns.

    Each of the four value regexes carries the same rejection branch. Testing one
    of them left three unproven: removing the branch from the other three would
    have kept the suite green. All 24 combinations were measured as rejected
    before this test was written.
    """
    assert _value(text_template.format(ph=placeholder), field) is None


def test_unqualified_under_agreement_is_read_but_prose_is_not(registry):
    """issued_under_agreement's "master" qualifier is deliberately optional.

    A SOW may name its parent as "Agreement MSA-4417" without the word "master",
    so the pattern matches an agreement reference with no qualifier at all. What
    stops ordinary prose matching is the value regex, which needs capitals or four
    or more digits: "the agreement dated 1 March" yields nothing.
    """
    assert _value("SOW\n\nexecuted under Agreement ABC-123\n", "parent_agreement_ref") == "ABC-123"
    assert _value("SOW\n\nentered into pursuant to the agreement X123\n", "parent_agreement_ref") == "X123"
    assert _value("SOW\n\nissued under the agreement dated 1 March\n", "parent_agreement_ref") is None


def test_a_sow_naming_its_msa_does_not_populate_parent_contract_id(registry):
    """Sitting under an agreement is not amending it.

    parent_contract_id anchors on amends/supplements/varies. If a plain "issued
    under" filled it too, the maths could not tell a child from an amendment.
    """
    text = "STATEMENT OF WORK\n\nThis SOW is issued under Master Agreement MSA-4417.\n"
    assert _value(text, "parent_agreement_ref") == "MSA-4417"
    assert _value(text, "parent_contract_id") is None


def test_an_amendment_still_populates_parent_contract_id(registry):
    """The existing behaviour must not move."""
    text = "VARIATION\n\nThis deed amends Contract MSA-4417.\n"
    assert _value(text, "parent_contract_id") == "MSA-4417"


def test_a_contract_naming_no_parent_reads_neither_field(registry):
    text = "MASTER AGREEMENT\n\nbetween Acme Ltd and Beta Ltd, numbered clauses.\n"
    assert _value(text, "framework_ref") is None
    assert _value(text, "parent_agreement_ref") is None


def test_prose_mentioning_agreements_names_no_parent(registry):
    """A page that talks about agreements and frameworks but cites no reference.

    Every anchor phrase appears here, with connectors, so a loosened anchor or
    value regex would pick up an ordinary word.
    """
    text = (
        "MASTER AGREEMENT\n\n"
        "This Master Agreement: the parties agree that each Statement of Work is made\n"
        "under the agreement and is issued under this agreement. The Framework Agreement:\n"
        "see the terms below. Pursuant to the agreement, the supplier shall perform.\n"
        "Nothing in this Framework Agreement - or in any agreement - limits liability.\n"
    )
    assert _value(text, "framework_ref") is None
    assert _value(text, "parent_agreement_ref") is None


# --- the optional "no" connector must not eat the front of a reference ---------
#
# In the connector-less prose patterns the optional `(?:number|no\.?|#)?` token
# used to swallow the leading "NO" of a real reference: "NOVA-123" was stored as
# "VA-123". A clipped value is worse than a missing one downstream -- a wrong
# reference scores CONFLICT, an absent one MISSING.

@pytest.mark.parametrize("field,template", [
    ("framework_ref",        "ORDER FORM\n\nmade under Framework Agreement {ref}\n"),
    ("parent_agreement_ref", "STATEMENT OF WORK\n\nissued under Master Agreement {ref}\n"),
    ("parent_contract_id",   "VARIATION\n\nThis deed amends Contract {ref}\n"),
])
@pytest.mark.parametrize("ref", ["NOVA-123", "NORTH-99"])
def test_a_reference_beginning_no_is_extracted_whole(field, template, ref):
    assert _value(template.format(ref=ref), field) == ref


@pytest.mark.parametrize("field,template", [
    ("framework_ref",        "ORDER FORM\n\nmade under Framework Agreement {c} MSA-4417\n"),
    ("parent_agreement_ref", "STATEMENT OF WORK\n\nissued under Master Agreement {c} MSA-4417\n"),
    ("parent_contract_id",   "VARIATION\n\nThis deed amends Contract {c} MSA-4417\n"),
])
@pytest.mark.parametrize("connector", ["No:", "No.", "No", "Number:", "#"])
def test_an_explicit_connector_still_works(field, template, connector):
    assert _value(template.format(c=connector), field) == "MSA-4417"


def test_a_hash_directly_before_the_reference_still_works():
    text = "STATEMENT OF WORK\n\nissued under Master Agreement #MSA-4417\n"
    assert _value(text, "parent_agreement_ref") == "MSA-4417"


# ---------------------------------------------------------------------------
# Found by Task 11's LIVE verification on a real PDF, 2026-10-03, not by any
# fixture. An order form reading
#
#   "This Order Form is incorporated into and governed by Framework Agreement
#    No. FA-2026-0042 dated 5 January 2026"
#
# produced contract_id = 'FA-2026-0042' and framework_ref = NULL. Two defects in
# one sentence, and together they destroyed data:
#
#   * contract_id's anchors refuse "Parent", "Master", "Principal", "amends",
#     "amendment to", "supplements" and "varies" -- every label naming a
#     DIFFERENT contract -- but not "Framework". So the order form took its
#     framework's identifier as its own.
#   * contract_id is the upsert key for proc.bp_contracts (ON CONFLICT
#     (contract_id) DO UPDATE SET <every other column>), so promoting the order
#     form OVERWROTE the framework agreement's row: resolved_doc_type on
#     FA-2026-0042 changed from doctype.framework_agreement to doctype.order_form
#     and the framework disappeared from the table as a separate contract.
#   * framework_ref missed the same reference because its anchor required a
#     colon or hyphen after "Framework Agreement No", which no real contract
#     writes, and its prose connector list did not include "incorporated into"
#     -- a phrase THIS plan added to parent_evidence_phrases in §4. The rule that
#     recognises the sentence and the extractor that reads it disagreed.
# ---------------------------------------------------------------------------

_LIVE_ORDER_FORM = (
    "ORDER FORM\n\n"
    "Order Form No. OF-2026-0117\n\n"
    "This Order Form is incorporated into and governed by Framework Agreement "
    "No. FA-2026-0042 dated 5 January 2026 between BrightWave Digital Ltd. and "
    "NexaSpark Marketing Ltd.\n"
)


def test_an_order_form_does_not_take_its_frameworks_number_as_its_own():
    """The one that cost a row: contract_id is the upsert key."""
    assert _value(_LIVE_ORDER_FORM, "contract_id") != "FA-2026-0042", (
        "the order form claimed its framework's identifier; promoting it would "
        "overwrite proc.bp_contracts row FA-2026-0042")


def test_the_framework_reference_in_the_live_sentence_is_read():
    assert _value(_LIVE_ORDER_FORM, "framework_ref") == "FA-2026-0042"


@pytest.mark.parametrize("sentence", [
    "This Order Form is incorporated into Framework Agreement No. FA-2026-0042.",
    "This Order Form is governed by Framework Agreement No. FA-2026-0042.",
    "Framework Agreement No. FA-2026-0042 applies to this Order Form.",
    "Framework Agreement Number FA-2026-0042 applies.",
    "Framework Agreement No: FA-2026-0042 applies.",
    "This order is called off under Framework Agreement FA-2026-0042.",
])
def test_the_framework_number_is_read_with_or_without_a_colon(sentence):
    """A colon after "No" is a typesetting accident, not a fact about the
    document. Requiring one is why the live file read NULL."""
    assert _value(sentence, "framework_ref") == "FA-2026-0042", sentence


@pytest.mark.parametrize("sentence", [
    "This Framework Agreement sets out the terms for ad-hoc orders.",
    "This Framework Agreement is between BrightWave Digital Ltd. and NexaSpark.",
    "The Framework Agreement shall commence on 5 January 2026.",
])
def test_a_framework_agreement_describing_itself_is_not_a_reference(sentence):
    """The guard on making the connector optional: prose that merely says
    "Framework Agreement" must not hand the next capitalised word over as a
    reference. A framework agreement's own page says this constantly."""
    assert _value(sentence, "framework_ref") is None, sentence


def test_a_framework_agreement_still_reads_its_own_number_as_contract_id():
    """The guard that stopped a blanket fix. A framework agreement's own first
    page writes its own identifier EXACTLY as an order form writes its pointer:
    "Framework Agreement No. FA-2026-0042". A negative lookbehind on contract_id
    was tried first and it blinded the framework to its own number, so the regex
    layer is left reading both -- and the document's own structure settles which
    is which, in dispatch, where the structure is known. See
    test_a_document_is_not_its_own_parent in
    tests/services/extraction/test_self_parent_reference.py.
    """
    page = ("FRAMEWORK AGREEMENT\n\nFramework Agreement No. FA-2026-0042\n\n"
            "This Framework Agreement is entered into on 5 January 2026.\n")
    assert _value(page, "contract_id") == "FA-2026-0042"
    # Indistinguishable at this layer, and deliberately not "fixed" here:
    assert _value(page, "framework_ref") == "FA-2026-0042"


def test_an_order_form_reads_its_own_number_not_its_frameworks():
    """What the live file could not do: print its own number and be believed."""
    assert _value(_LIVE_ORDER_FORM, "contract_id") == "OF-2026-0117"
    assert _value(_LIVE_ORDER_FORM, "framework_ref") == "FA-2026-0042"


@pytest.mark.parametrize("sentence", [
    "This Statement of Work is made under Master Agreement No. MSA-4417.",
    "Master Agreement No. MSA-4417 governs this SOW.",
    "Master Services Agreement Number MSA-4417 applies.",
    "Master Agreement No: MSA-4417 applies.",
])
def test_the_master_agreement_number_is_read_with_or_without_a_colon(sentence):
    """parent_agreement_ref carried the identical mandatory-colon flaw, and
    "Master Agreement No. MSA-4417" is the commonest shape a SOW has."""
    assert _value(sentence, "parent_agreement_ref") == "MSA-4417", sentence


@pytest.mark.parametrize("sentence", [
    "This Master Agreement sets out the terms between the parties.",
    "This Master Services Agreement is dated 5 January 2026.",
])
def test_a_master_agreement_describing_itself_is_not_a_reference(sentence):
    assert _value(sentence, "parent_agreement_ref") is None, sentence


def test_a_sow_reads_its_masters_number_as_the_parent_reference():
    page = ("STATEMENT OF WORK\n\nSOW No. SOW-2026-11\n\nThis Statement of Work "
            "is made under Master Agreement No. MSA-4417.\n")
    assert _value(page, "parent_agreement_ref") == "MSA-4417"
