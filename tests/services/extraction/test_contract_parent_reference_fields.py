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
